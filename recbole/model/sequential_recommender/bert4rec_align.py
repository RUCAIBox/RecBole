# -*- coding: utf-8 -*-
# Align-only variant of BERT4Rec with optional text embedding alignment

import random
import os

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder


class BERT4RecAlign(SequentialRecommender):
    def __init__(self, config, dataset):
        super(BERT4RecAlign, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.mask_ratio = config["mask_ratio"]

        self.MASK_ITEM_SEQ = config["MASK_ITEM_SEQ"]
        self.POS_ITEMS = config["POS_ITEMS"]
        self.NEG_ITEMS = config["NEG_ITEMS"]
        self.MASK_INDEX = config["MASK_INDEX"]

        self.loss_type = config["loss_type"]
        self.initializer_range = config["initializer_range"]

        # load dataset info
        self.mask_token = self.n_items
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.hidden_size, padding_idx=0
        )  # mask token add 1
        self.position_embedding = nn.Embedding(
            self.max_seq_length, self.hidden_size
        )  # add mask_token at the last
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.output_ffn = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_gelu = nn.GELU()
        self.output_ln = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.output_bias = nn.Parameter(torch.zeros(self.n_items))

        # --- text-alignment settings ---
        self.alignment_weight = config["alignment_weight"] if "alignment_weight" in config else 0.0
        self.temperature = config["temperature"] if "temperature" in config else 0.07
        self.normalize_text = config["normalize_text"] if "normalize_text" in config else True
        self.detach_text_emb = config["detach_text_emb"] if "detach_text_emb" in config else True
        self.item_text_emb_path = config["item_text_emb_path"] if "item_text_emb_path" in config else None

        # Load external item text embeddings (frozen) and projection
        item_text_emb = self._load_text_embeddings(self.item_text_emb_path, self.n_items)
        if item_text_emb is not None and self.normalize_text:
            with torch.no_grad():
                norms = torch.norm(item_text_emb, p=2, dim=1, keepdim=True)
                zero_mask = norms.squeeze(1) <= 0
                item_text_emb = item_text_emb / norms.clamp_min(1e-8)
                item_text_emb[torch.isnan(item_text_emb)] = 0.0
        self.register_buffer(
            "item_text_emb", item_text_emb if item_text_emb is not None else None
        )

        if self._has_item_text():
            item_text_dim = self.item_text_emb.shape[1]
            self.item_text_proj = nn.Linear(item_text_dim, self.hidden_size)
        else:
            self.item_text_proj = None

        # Logging for alignment availability
        if self.alignment_weight > 0.0:
            if not self._has_item_text() or self.item_text_proj is None:
                self.logger.warning(
                    "BERT4RecAlign: alignment_weight>0 but no valid item_text_emb loaded from '%s' (expected rows=%d). Alignment will be disabled.",
                    str(self.item_text_emb_path), self.n_items,
                )
            else:
                try:
                    nan_rows = int(torch.isnan(self.item_text_emb).any(dim=1).sum().item())
                    zero_rows = int((torch.norm(self.item_text_emb, p=2, dim=1) == 0).sum().item())
                except Exception:
                    nan_rows = zero_rows = -1
                self.logger.info(
                    "BERT4RecAlign: loaded item_text_emb shape=%s; projecting to hidden_size=%d. sanitized_nan_rows=%d zero_rows=%d",
                    tuple(self.item_text_emb.shape), self.hidden_size, nan_rows, zero_rows,
                )

        self._align_debug_logged = False

        # we only need compute the loss at the masked position
        try:
            assert self.loss_type in ["BPR", "CE"]
        except AssertionError:
            raise AssertionError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _load_text_embeddings(self, path, expected_rows):
        if path is None or (isinstance(path, str) and path.strip() == ""):
            return None
        if not isinstance(path, str) or not os.path.exists(path):
            return None
        emb = None
        try:
            if path.endswith(".npy"):
                emb_np = np.load(path)
                emb = torch.from_numpy(emb_np).float()
            else:
                loaded = torch.load(path, map_location="cpu")
                if isinstance(loaded, torch.Tensor):
                    emb = loaded.float()
                elif isinstance(loaded, np.ndarray):
                    emb = torch.from_numpy(loaded).float()
                elif isinstance(loaded, dict) and "emb" in loaded:
                    emb = loaded["emb"].float()
        except Exception:
            emb = None
        if emb is None or emb.dim() != 2:
            return None
        if emb.size(0) != expected_rows:
            return None
        return emb

    def _has_item_text(self) -> bool:
        return hasattr(self, "item_text_emb") and self.item_text_emb is not None

    def _info_nce_align(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.size(0) == 0 or b.size(0) == 0:
            return torch.zeros(1, device=a.device)
        assert a.size(0) == b.size(0), f"Batch size mismatch: {a.size(0)} vs {b.size(0)}"
        a = F.normalize(a, dim=1)
        b = F.normalize(b, dim=1)
        logits = torch.matmul(a, b.t()) / self.temperature
        labels = torch.arange(a.size(0), device=a.device)
        return nn.CrossEntropyLoss()(logits, labels)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def reconstruct_test_data(self, item_seq, item_seq_len):
        padding = torch.zeros(
            item_seq.size(0), dtype=torch.long, device=item_seq.device
        )
        item_seq = torch.cat((item_seq, padding.unsqueeze(-1)), dim=-1)
        for batch_id, last_position in enumerate(item_seq_len):
            item_seq[batch_id][last_position] = self.mask_token
        item_seq = item_seq[:, 1:]
        return item_seq

    def forward(self, item_seq):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=True)
        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        ffn_output = self.output_ffn(trm_output[-1])
        ffn_output = self.output_gelu(ffn_output)
        output = self.output_ln(ffn_output)
        return output

    def multi_hot_embed(self, masked_index, max_length):
        masked_index = masked_index.view(-1)
        multi_hot = torch.zeros(
            masked_index.size(0), max_length, device=masked_index.device
        )
        multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
        return multi_hot

    def calculate_loss(self, interaction):
        masked_item_seq = interaction[self.MASK_ITEM_SEQ]
        pos_items = interaction[self.POS_ITEMS]
        neg_items = interaction[self.NEG_ITEMS]
        masked_index = interaction[self.MASK_INDEX]

        seq_output = self.forward(masked_item_seq)
        pred_index_map = self.multi_hot_embed(
            masked_index, masked_item_seq.size(-1)
        )
        pred_index_map = pred_index_map.view(
            masked_index.size(0), masked_index.size(1), -1
        )
        seq_output = torch.bmm(pred_index_map, seq_output)

        if self.loss_type == "BPR":
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = (
                torch.sum(seq_output * pos_items_emb, dim=-1) + self.output_bias[pos_items]
            )
            neg_score = (
                torch.sum(seq_output * neg_items_emb, dim=-1) + self.output_bias[neg_items]
            )
            targets_mask = (masked_index > 0).float()
            loss = -torch.sum(
                torch.log(1e-14 + torch.sigmoid(pos_score - neg_score)) * targets_mask
            ) / torch.sum(targets_mask)
        elif self.loss_type == "CE":
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            test_item_emb = self.item_embedding.weight[: self.n_items]
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) + self.output_bias
            targets_mask = (masked_index > 0).float().view(-1)
            loss = torch.sum(
                loss_fct(logits.view(-1, test_item_emb.size(0)), pos_items.view(-1))
                * targets_mask
            ) / torch.sum(targets_mask)
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        if (
            self.alignment_weight > 0.0
            and self._has_item_text()
            and self.item_text_proj is not None
        ):
            valid_mask = (masked_index > 0).view(-1)
            if valid_mask.any():
                pos_ids_flat = pos_items.view(-1)[valid_mask]
                id_item_e = self.item_embedding(pos_ids_flat)
                txt_emb = self.item_text_emb[pos_ids_flat]
                if self.detach_text_emb:
                    txt_emb = txt_emb.detach()
                txt_item_e = self.item_text_proj(txt_emb)
                align_loss = self._info_nce_align(id_item_e, txt_item_e)
                loss = loss + self.alignment_weight * align_loss

                if not self._align_debug_logged:
                    try:
                        self.logger.info(
                            "BERT4RecAlign: first-step align_loss=%.6f, masked_pos=%d, proj_norm=%.6f",
                            align_loss.item(), int(pos_ids_flat.numel()), float(self.item_text_proj.weight.norm().item()),
                        )
                    except Exception:
                        pass
                    self._align_debug_logged = True

        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
        seq_output = self.forward(item_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_item_emb = self.item_embedding(test_item)
        scores = (torch.mul(seq_output, test_item_emb)).sum(dim=1) + self.output_bias[
            test_item
        ]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
        seq_output = self.forward(item_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        test_items_emb = self.item_embedding.weight[: self.n_items]
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1)) + self.output_bias
        return scores


