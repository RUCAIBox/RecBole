# -*- coding: utf-8 -*-
# Align-only variant of SASRec with optional text embedding alignment

import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, MLPLayers
from recbole.model.loss import BPRLoss


class DCNV2Cross(nn.Module):
    """DCN-V2 cross network (non-mix) over dense features.

    Follows the original implementation in recbole.model.context_aware_recommender.dcnv2
    with the update rule: x_{l+1} = x_l + x_0 ⊙ (W_l x_l + b_l).
    """

    def __init__(self, input_dim: int, num_layers: int = 3):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_layers = int(max(0, num_layers))
        # W: (in_feature_num, in_feature_num) per layer
        self.cross_layer_w = nn.ParameterList(
            nn.Parameter(torch.randn(self.input_dim, self.input_dim))
            for _ in range(self.num_layers)
        )
        # b: (in_feature_num, 1) per layer
        self.bias = nn.ParameterList(
            nn.Parameter(torch.zeros(self.input_dim, 1))
            for _ in range(self.num_layers)
        )

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        if self.num_layers == 0:
            return x0
        # x0: [batch, in_feature_num]
        x0_u = x0.unsqueeze(dim=2)  # [B, D, 1]
        xl = x0_u
        for i in range(self.num_layers):
            xl_w = torch.matmul(self.cross_layer_w[i], xl)  # [B, D, 1]
            xl_w = xl_w + self.bias[i]
            xl_dot = torch.mul(x0_u, xl_w)
            xl = xl_dot + xl
        xl = xl.squeeze(dim=2)  # [B, D]
        return xl


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

        # --- text-alignment settings & feature fusion ---
        self.alignment_weight = config["alignment_weight"] if "alignment_weight" in config else 0.0
        self.temperature = config["temperature"] if "temperature" in config else 0.07
        self.normalize_text = config["normalize_text"] if "normalize_text" in config else True
        self.detach_text_emb = config["detach_text_emb"] if "detach_text_emb" in config else True
        self.use_llm = config["use_llm"] if "use_llm" in config else False
        self.use_cross = config["use_cross"] if "use_cross" in config else False
        self.use_align = config["use_align"] if "use_align" in config else True
        self.text_cross_layer_num = config["text_cross_layer_num"] if "text_cross_layer_num" in config else 3
        # New: simple non-cross enhancements
        self.text_weight = float(config["text_weight"]) if "text_weight" in config else 1.0
        self.text_tail_threshold = int(config["text_tail_threshold"]) if "text_tail_threshold" in config else 0
        # For backward compatibility: accept single path as base
        item_text_emb_path_base = (
            config["item_text_emb_path_base"] if "item_text_emb_path_base" in config else None
        )
        item_text_emb_path_llm = (
            config["item_text_emb_path_llm"] if "item_text_emb_path_llm" in config else None
        )
        if item_text_emb_path_base is None and item_text_emb_path_llm is None:
            # Fallback to legacy key
            item_text_emb_path_base = config["item_text_emb_path"] if "item_text_emb_path" in config else None

        emb_base = self._load_text_embeddings(item_text_emb_path_base, self.n_items)
        emb_llm = self._load_text_embeddings(item_text_emb_path_llm, self.n_items)

        if self.normalize_text:
            with torch.no_grad():
                if emb_base is not None:
                    norms = torch.norm(emb_base, p=2, dim=1, keepdim=True)
                    emb_base = emb_base / norms.clamp_min(1e-8)
                    emb_base[torch.isnan(emb_base)] = 0.0
                if emb_llm is not None:
                    norms = torch.norm(emb_llm, p=2, dim=1, keepdim=True)
                    emb_llm = emb_llm / norms.clamp_min(1e-8)
                    emb_llm[torch.isnan(emb_llm)] = 0.0

        self.register_buffer("item_text_emb_base", emb_base if emb_base is not None else None)
        self.register_buffer("item_text_emb_llm", emb_llm if emb_llm is not None else None)

        # Precompute item popularity for optional tail gating (no extra IO during training)
        pop_counts = None
        try:
            inter_iids = dataset.inter_feat[dataset.iid_field].numpy()
            pop_counts = np.bincount(inter_iids, minlength=self.n_items)
        except Exception:
            pop_counts = np.zeros((self.n_items,), dtype=np.int64)
        self.register_buffer("item_popularity", torch.from_numpy(pop_counts).long())
        if self.text_tail_threshold > 0:
            gate = (self.item_popularity <= int(self.text_tail_threshold)).float()
        else:
            gate = None
        # shape [n_items], 1.0 means enable text for that item
        self.register_buffer("text_item_gate_all", gate)

        # Determine text input formation according to flags and availability
        base_dim = int(self.item_text_emb_base.shape[1]) if self.item_text_emb_base is not None else 0
        llm_dim = int(self.item_text_emb_llm.shape[1]) if self.item_text_emb_llm is not None else 0
        self._text_mode = "none"
        if self.use_llm:
            if base_dim > 0 and llm_dim > 0:
                self._text_mode = "both"
                text_in_dim = base_dim + llm_dim
            elif llm_dim > 0:
                self._text_mode = "llm"
                text_in_dim = llm_dim
            elif base_dim > 0:
                self._text_mode = "base"
                text_in_dim = base_dim
            else:
                text_in_dim = 0
        else:
            if base_dim > 0:
                self._text_mode = "base"
                text_in_dim = base_dim
            else:
                text_in_dim = 0

        # Build text projection/fusion modules
        self.text_cross = None
        self.text_deep = None
        self.text_predictor = None
        self.item_text_proj = None
        self.item_concat_predictor = None
        
        # Item-side fusion modules for integrating text into scoring
        self.item_fusion_cross = None
        self.item_fusion_deep = None
        self.item_fusion_predictor = None
        
        if text_in_dim > 0:
            if self.use_cross:
                self.text_cross = DCNV2Cross(text_in_dim, num_layers=self.text_cross_layer_num)
                # Simple deep tower to the model hidden size
                self.text_deep = MLPLayers([text_in_dim, self.hidden_size], dropout=0.0, bn=False)
                self.text_predictor = nn.Linear(text_in_dim + self.hidden_size, self.hidden_size)
                
                # Item-side fusion: combine item embedding with text features
                # Input: [item_emb(hidden_size), text_features(text_in_dim)]
                fusion_input_dim = self.hidden_size + text_in_dim
                self.item_fusion_cross = DCNV2Cross(fusion_input_dim, num_layers=self.text_cross_layer_num)
                self.item_fusion_deep = MLPLayers([fusion_input_dim, self.hidden_size], dropout=0.0, bn=False)
                self.item_fusion_predictor = nn.Linear(fusion_input_dim + self.hidden_size, self.hidden_size)
            else:
                self.item_text_proj = nn.Linear(text_in_dim, self.hidden_size)
                # Concatenation-based fusion (no-cross): [item_emb, text_proj] -> hidden_size
                self.item_concat_predictor = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self._align_debug_logged = False
        
        # Cache for fused item embeddings to improve efficiency
        self._fused_item_emb_cache = None

        # Logging for alignment availability
        if self.alignment_weight > 0.0:
            if not self._has_item_text() or (
                (self.use_cross and (self.text_cross is None or self.text_deep is None or self.text_predictor is None))
                or (not self.use_cross and self.item_text_proj is None)
            ):
                self.logger.warning(
                    "SASRecAlign: alignment_weight>0 but no valid text modules/embeddings (expected rows=%d). Alignment will be disabled.",
                    self.n_items,
                )
            else:
                try:
                    # Prefer to check the union emb if both are available
                    check_emb = None
                    if self.item_text_emb_llm is not None:
                        check_emb = self.item_text_emb_llm
                    elif self.item_text_emb_base is not None:
                        check_emb = self.item_text_emb_base
                    nan_rows = int(torch.isnan(check_emb).any(dim=1).sum().item()) if check_emb is not None else -1
                    zero_rows = int((torch.norm(check_emb, p=2, dim=1) == 0).sum().item()) if check_emb is not None else -1
                except Exception:
                    nan_rows = zero_rows = -1
                self.logger.info(
                    "SASRecAlign: text mode=%s base_dim=%d llm_dim=%d -> hidden_size=%d; sanitized_nan_rows=%d zero_rows=%d",
                    self._text_mode, base_dim, llm_dim, self.hidden_size, nan_rows, zero_rows,
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
        return (
            hasattr(self, "item_text_emb_base")
            and hasattr(self, "item_text_emb_llm")
            and (self.item_text_emb_base is not None or self.item_text_emb_llm is not None)
        )

    def _gather_text_raw(self, ids_flat: torch.Tensor) -> torch.Tensor:
        parts = []
        if self._text_mode in ("base", "both") and self.item_text_emb_base is not None:
            parts.append(self.item_text_emb_base[ids_flat])
        if self._text_mode in ("llm", "both") and self.item_text_emb_llm is not None:
            parts.append(self.item_text_emb_llm[ids_flat])
        if len(parts) == 0:
            return torch.zeros((ids_flat.size(0), 0), device=ids_flat.device)
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

    def _project_text(self, raw: torch.Tensor) -> torch.Tensor:
        if raw.size(1) == 0:
            return torch.zeros((raw.size(0), self.hidden_size), device=raw.device)
        if self.use_cross and self.text_cross is not None and self.text_deep is not None and self.text_predictor is not None:
            cross_out = self.text_cross(raw)
            deep_out = self.text_deep(raw)
            fused = torch.cat([cross_out, deep_out], dim=1)
            return self.text_predictor(fused)
        elif (not self.use_cross) and self.item_text_proj is not None:
            return self.item_text_proj(raw)
        else:
            return torch.zeros((raw.size(0), self.hidden_size), device=raw.device)

    def _info_nce_align(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.size(0) == 0 or b.size(0) == 0:
            return torch.zeros(1, device=a.device)
        a = F.normalize(a, dim=1)
        b = F.normalize(b, dim=1)
        logits = torch.matmul(a, b.t()) / self.temperature
        labels = torch.arange(a.size(0), device=a.device)
        return nn.CrossEntropyLoss()(logits, labels)
    
    def _get_fused_item_embeddings(self, item_ids: torch.Tensor = None) -> torch.Tensor:
        """Get item embeddings fused with text features.
        
        Args:
            item_ids: Specific item IDs to get embeddings for. If None, returns all items.
            
        Returns:
            Fused item embeddings of shape [n_items, hidden_size] or [batch_size, hidden_size]
        """
        if item_ids is None:
            # Get all item embeddings
            all_ids = torch.arange(self.n_items, device=self.item_embedding.weight.device)
            item_emb = self.item_embedding.weight  # [n_items, hidden_size]
        else:
            all_ids = item_ids
            item_emb = self.item_embedding(item_ids)
        
        # If no text features or fusion modules, return original embeddings
        if (
            (not self._has_item_text())
            or (self.use_cross and self.item_fusion_predictor is None)
            or ((not self.use_cross) and (self.item_text_proj is None or self.item_concat_predictor is None))
        ):
            return item_emb
            
        # Get text features for items
        text_raw = self._gather_text_raw(all_ids)
        if self.detach_text_emb:
            text_raw = text_raw.detach()
        
        if self.use_cross and self.item_fusion_predictor is not None:
            # Apply gating/weighting to text features before cross fusion for stability
            if self.text_item_gate_all is not None:
                if item_ids is None:
                    gate = self.text_item_gate_all
                else:
                    gate = self.text_item_gate_all[all_ids]
                gate = gate.to(item_emb.device).unsqueeze(1)
                scaled_text = (self.text_weight * gate) * text_raw
            else:
                scaled_text = self.text_weight * text_raw

            # Fuse item embeddings with scaled text features using cross network
            fusion_input = torch.cat([item_emb, scaled_text], dim=1)
            cross_out = self.item_fusion_cross(fusion_input)
            deep_out = self.item_fusion_deep(fusion_input)
            fused = torch.cat([cross_out, deep_out], dim=1)
            fused_emb = self.item_fusion_predictor(fused)
        else:
            # Concatenation fusion (no-cross): [item_emb, scaled text_proj] -> predictor -> hidden_size
            text_proj = self._project_text(text_raw)
            if self.text_item_gate_all is not None:
                if item_ids is None:
                    gate = self.text_item_gate_all
                else:
                    gate = self.text_item_gate_all[all_ids]
                gate = gate.to(item_emb.device).unsqueeze(1)
                scaled_text = (self.text_weight * gate) * text_proj
            else:
                scaled_text = self.text_weight * text_proj
            concat = torch.cat([item_emb, scaled_text], dim=1)
            fused_emb = self.item_concat_predictor(concat)
            
        return fused_emb

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
            # Use fused embeddings for positive and negative items
            pos_items_emb = self._get_fused_item_embeddings(pos_items)
            neg_items_emb = self._get_fused_item_embeddings(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
        else:  # CE
            # Use fused embeddings for all items
            test_item_emb = self._get_fused_item_embeddings()
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)

        # optional alignment loss (align ID item embeddings with text embeddings)
        if (
            self.use_align
            and self.alignment_weight > 0.0
            and self._has_item_text()
            and ((self.use_cross and self.text_predictor is not None) or ((not self.use_cross) and self.item_text_proj is not None))
        ):
            pos_ids_flat = pos_items.view(-1)
            id_item_e = self.item_embedding(pos_ids_flat)
            txt_raw = self._gather_text_raw(pos_ids_flat)
            if self.detach_text_emb:
                txt_raw = txt_raw.detach()
            txt_item_e = self._project_text(txt_raw)
            align_loss = self._info_nce_align(id_item_e, txt_item_e)
            loss = loss + self.alignment_weight * align_loss

            if not self._align_debug_logged:
                try:
                    self.logger.info(
                        "SASRecAlign: first-step align_loss=%.6f, batch_pos=%d, proj_norm=%.6f",
                        align_loss.item(), int(pos_ids_flat.numel()), float(
                            (self.text_predictor.weight if (self.use_cross and self.text_predictor is not None) else self.item_text_proj.weight).norm().item()
                        ),
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
        # Use fused embeddings for test items
        test_item_emb = self._get_fused_item_embeddings(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        # Use fused embeddings for all items
        test_items_emb = self._get_fused_item_embeddings()
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores



 # Alias to enable model name 'SASRec_Align' to load this module
SASRec_Align = SASRecAlign
