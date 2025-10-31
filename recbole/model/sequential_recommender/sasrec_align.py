# -*- coding: utf-8 -*-
# Align-only variant of SASRec with optional text embedding alignment

import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss


class SASRecAlign(SequentialRecommender):
    def __init__(self, config, dataset):
        super(SASRecAlign, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
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

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # --- text-alignment settings ---
        self.alignment_weight = config["alignment_weight"] if "alignment_weight" in config else 0.0
        self.temperature = config["temperature"] if "temperature" in config else 0.07
        self.normalize_text = config["normalize_text"] if "normalize_text" in config else True
        self.detach_text_emb = config["detach_text_emb"] if "detach_text_emb" in config else True
        self.item_text_emb_path = config["item_text_emb_path"] if "item_text_emb_path" in config else None

        item_text_emb = self._load_text_embeddings(self.item_text_emb_path, self.n_items)
        if item_text_emb is not None and self.normalize_text:
            with torch.no_grad():
                norms = torch.norm(item_text_emb, p=2, dim=1, keepdim=True)
                zero_mask = norms.squeeze(1) <= 0
                # avoid divide-by-zero; sanitize NaNs
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

        self._align_debug_logged = False

        # Logging for alignment availability
        if self.alignment_weight > 0.0:
            if not self._has_item_text() or self.item_text_proj is None:
                self.logger.warning(
                    "SASRecAlign: alignment_weight>0 but no valid item_text_emb loaded from '%s' (expected rows=%d). Alignment will be disabled.",
                    str(self.item_text_emb_path), self.n_items,
                )
            else:
                try:
                    nan_rows = int(torch.isnan(self.item_text_emb).any(dim=1).sum().item())
                    zero_rows = int((torch.norm(self.item_text_emb, p=2, dim=1) == 0).sum().item())
                except Exception:
                    nan_rows = zero_rows = -1
                self.logger.info(
                    "SASRecAlign: loaded item_text_emb shape=%s; projecting to hidden_size=%d. sanitized_nan_rows=%d zero_rows=%d",
                    tuple(self.item_text_emb.shape), self.hidden_size, nan_rows, zero_rows,
                )

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

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
        a = F.normalize(a, dim=1)
        b = F.normalize(b, dim=1)
        logits = torch.matmul(a, b.t()) / self.temperature
        labels = torch.arange(a.size(0), device=a.device)
        return nn.CrossEntropyLoss()(logits, labels)

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
        else:  # CE
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)

        # optional alignment loss (align ID item embeddings with text embeddings)
        if (
            self.alignment_weight > 0.0
            and self._has_item_text()
            and self.item_text_proj is not None
        ):
            pos_ids_flat = pos_items.view(-1)
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
                        "SASRecAlign: first-step align_loss=%.6f, batch_pos=%d, proj_norm=%.6f",
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
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores


