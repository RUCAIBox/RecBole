# -*- coding: utf-8 -*-
# @Time    : 2026/03/04
# @Author  : Tal Cordova
# @Email   : talcordova56@gmail.com

r"""
DIF-SASRec
################################################

Reference:
    Ruihong Xie et al. "Decoupled Side Information Fusion for Sequential Recommendation."
    In SIGIR 2022.

Reference code:
    https://github.com/AIM-SE/DIF-SR
"""

import copy
import math

import torch
import torch.nn as nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import FeedForward, VanillaAttention
from recbole.model.loss import BPRLoss
from recbole.utils import FeatureType


# ---------------------------------------------------------------------------
# Module-private DIF attention layers
# (promoted to recbole/model/layers.py if adopted more broadly)
# ---------------------------------------------------------------------------


class _DIFMultiHeadAttention(nn.Module):
    """
    Decoupled Information Fusion Multi-Head Self-Attention.

    Computes separate attention logits for:
      - item embeddings  (standard Q / K / V)
      - position embeddings  (separate Q / K, shared V)
      - each optional context tower  (separate Q / K per tower, shared V)

    All logits are fused *before* softmax so each source independently
    shapes the attention distribution.

    Fusion strategies (``fusion_type``):
      - ``'sum'``    element-wise sum of all score matrices (default).
      - ``'concat'`` concatenate along seq axis then linear-project back.
      - ``'gate'``   weighted combination via ``VanillaAttention``.
    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        tower_hidden_sizes,
        hidden_dropout_prob=0.2,
        attn_dropout_prob=0.2,
        layer_norm_eps=1e-12,
        fusion_type="sum",
        max_len=50,
    ):
        super().__init__()

        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads})"
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = hidden_size // n_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.fusion_type = fusion_type
        self.tower_names = list(tower_hidden_sizes.keys())

        # Per-tower head sizes (tower dim may differ from hidden_size)
        self.tower_attention_head_sizes = {}
        self.tower_all_head_sizes = {}
        for name, dim in tower_hidden_sizes.items():
            head_size = max(1, dim // n_heads)
            self.tower_attention_head_sizes[name] = head_size
            self.tower_all_head_sizes[name] = n_heads * head_size

        # Item Q / K / V
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        # Position Q / K  (V is shared from item path)
        self.query_p = nn.Linear(hidden_size, self.all_head_size)
        self.key_p = nn.Linear(hidden_size, self.all_head_size)

        # Per-tower Q / K
        self.query_layers = nn.ModuleDict()
        self.key_layers = nn.ModuleDict()
        for name, dim in tower_hidden_sizes.items():
            out = self.tower_all_head_sizes[name]
            self.query_layers[name] = nn.Linear(dim, out)
            self.key_layers[name] = nn.Linear(dim, out)

        # Fusion mechanism (built only when needed)
        if fusion_type == "concat":
            num_sources = len(self.tower_names) + 2  # towers + item + position
            self.fusion_layer = nn.Linear(max_len * num_sources, max_len)
        elif fusion_type == "gate":
            self.fusion_layer = VanillaAttention(max_len, max_len)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def _transpose(self, x):
        """[B, L, H] → [B, heads, L, head_dim]"""
        s = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*s).permute(0, 2, 1, 3)

    def _transpose_tower(self, x, name):
        """[B, L, tower_H] → [B, heads, L, tower_head_dim]"""
        h = self.tower_attention_head_sizes[name]
        s = x.size()[:-1] + (self.num_attention_heads, h)
        return x.view(*s).permute(0, 2, 1, 3)

    def forward(self, input_tensor, tower_embeddings, position_embedding, attention_mask):
        """
        Args:
            input_tensor (Tensor): item embeddings  ``[B, L, H]``
            tower_embeddings (dict): name → ``[B, L, tower_dim]``
            position_embedding (Tensor): ``[B, L, H]``
            attention_mask (Tensor): causal mask ``[B, 1, L, L]``  (0 / -10000)
        Returns:
            Tensor: ``[B, L, H]``
        """
        # --- item scores ---
        iq = self._transpose(self.query(input_tensor))   # [B, H, L, hd]
        ik = self._transpose(self.key(input_tensor))
        iv = self._transpose(self.value(input_tensor))
        item_scores = torch.matmul(iq, ik.transpose(-1, -2))  # [B, H, L, L]

        # --- position scores ---
        pq = self._transpose(self.query_p(position_embedding))
        pk = self._transpose(self.key_p(position_embedding))
        pos_scores = torch.matmul(pq, pk.transpose(-1, -2))   # [B, H, L, L]

        # --- tower scores ---
        tower_score_list = []
        for name in self.tower_names:
            if name in tower_embeddings:
                t = tower_embeddings[name]
                tq = self._transpose_tower(self.query_layers[name](t), name)
                tk = self._transpose_tower(self.key_layers[name](t), name)
                # [B, heads, L, L] → unsqueeze at dim=-2 → [B, heads, L, 1, L]
                tower_score_list.append(
                    torch.matmul(tq, tk.transpose(-1, -2)).unsqueeze(-2)
                )

        # --- fuse ---
        if tower_score_list:
            # tower_table: [B, heads, L, num_towers, L]
            tower_table = torch.cat(tower_score_list, dim=-2)
            ts = tower_table.shape
            feat_num, attn_size = ts[-2], ts[-1]

            if self.fusion_type == "sum":
                attention_scores = (
                    tower_table.sum(dim=-2) + item_scores + pos_scores
                )
            elif self.fusion_type == "concat":
                flat = tower_table.view(ts[:-2] + (feat_num * attn_size,))
                attention_scores = torch.cat(
                    [flat, item_scores, pos_scores], dim=-1
                )
                attention_scores = self.fusion_layer(attention_scores)
            elif self.fusion_type == "gate":
                all_scores = torch.cat(
                    [
                        tower_table,
                        item_scores.unsqueeze(-2),
                        pos_scores.unsqueeze(-2),
                    ],
                    dim=-2,
                )
                attention_scores, _ = self.fusion_layer(all_scores)
        else:
            attention_scores = item_scores + pos_scores

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)

        ctx = torch.matmul(attention_probs, iv)
        ctx = ctx.permute(0, 2, 1, 3).contiguous()
        ctx = ctx.view(ctx.size()[:-2] + (self.all_head_size,))

        hidden_states = self.dense(ctx)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class _DIFTransformerLayer(nn.Module):
    def __init__(
        self,
        n_heads,
        hidden_size,
        tower_hidden_sizes,
        intermediate_size,
        hidden_dropout_prob=0.2,
        attn_dropout_prob=0.2,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        fusion_type="sum",
        max_len=50,
    ):
        super().__init__()
        self.attention = _DIFMultiHeadAttention(
            n_heads=n_heads,
            hidden_size=hidden_size,
            tower_hidden_sizes=tower_hidden_sizes,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            fusion_type=fusion_type,
            max_len=max_len,
        )
        # Reuse RecBole's existing FeedForward
        self.intermediate = FeedForward(
            hidden_size=hidden_size,
            inner_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
        )

    def forward(self, hidden_states, tower_embeddings, position_embedding, attention_mask):
        attn_out = self.attention(
            hidden_states, tower_embeddings, position_embedding, attention_mask
        )
        return self.intermediate(attn_out)


class _DIFTransformerEncoder(nn.Module):
    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        tower_hidden_sizes=None,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        fusion_type="sum",
        max_len=50,
    ):
        super().__init__()
        tower_hidden_sizes = tower_hidden_sizes or {}
        layer = _DIFTransformerLayer(
            n_heads=n_heads,
            hidden_size=hidden_size,
            tower_hidden_sizes=tower_hidden_sizes,
            intermediate_size=inner_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            fusion_type=fusion_type,
            max_len=max_len,
        )
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(
        self,
        hidden_states,
        tower_embeddings,
        position_embedding,
        attention_mask,
        output_all_encoded_layers=True,
    ):
        all_encoder_layers = []
        for layer_module in self.layers:
            hidden_states = layer_module(
                hidden_states, tower_embeddings, position_embedding, attention_mask
            )
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


# ---------------------------------------------------------------------------
# Public model
# ---------------------------------------------------------------------------


class DIFSASRec(SequentialRecommender):
    r"""
    DIFSASRec extends SASRec by *decoupling* item-content and position attention
    scores so that each information source independently shapes the attention
    distribution without interference.

    When ``context_fields`` is empty (default) the model behaves as a SASRec
    variant with decoupled item / position attention.  When one or more
    ``FLOAT`` or ``FLOAT_SEQ`` item feature fields are listed in
    ``context_fields``, those pre-computed embeddings (e.g. LLM-extracted
    features stored in the ``.item`` file) are added as additional decoupled
    towers in the attention mechanism.

    NOTE:
        Context embeddings are frozen during training — only the lightweight
        compression MLP that maps them into the attention space is learned.
    """

    def __init__(self, config, dataset):
        super(DIFSASRec, self).__init__(config, dataset)

        # -- architecture --
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.dif_fusion_type = config["dif_fusion_type"]
        self.loss_type = config["loss_type"]

        # -- optional context --
        self.context_fields = config.get("context_fields", [])
        self.context_compression_ratio = config.get("context_compression_ratio", 0.5)

        # -- core embeddings --
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        # -- context lookup tables + compression MLPs (one per field) --
        self.context_lookup = nn.ModuleDict()
        self.context_compress = nn.ModuleDict()
        tower_hidden_sizes = {}

        for field in self.context_fields:
            if field not in dataset.field2type:
                raise ValueError(
                    f"context_fields: '{field}' not found in dataset. "
                    f"Available fields: {list(dataset.field2type.keys())}"
                )
            ftype = dataset.field2type[field]
            if ftype not in (FeatureType.FLOAT, FeatureType.FLOAT_SEQ):
                raise ValueError(
                    f"context_fields: '{field}' has type {ftype!r}; "
                    f"expected FLOAT or FLOAT_SEQ."
                )

            feat_data = dataset.item_feat[field].float()  # [n_items] or [n_items, D]
            if feat_data.dim() == 1:
                feat_data = feat_data.unsqueeze(-1)
            feat_dim = feat_data.shape[-1]
            compressed_dim = max(1, int(feat_dim * self.context_compression_ratio))

            # Placeholder embedding — weights overwritten after _init_weights
            self.context_lookup[field] = nn.Embedding(
                self.n_items, feat_dim, padding_idx=0
            )
            self.context_compress[field] = nn.Sequential(
                nn.Linear(feat_dim, compressed_dim),
                nn.ReLU(),
                nn.LayerNorm(compressed_dim),
            )
            tower_hidden_sizes[field] = compressed_dim

        # -- DIF transformer encoder --
        self.dif_encoder = _DIFTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            tower_hidden_sizes=tower_hidden_sizes,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            fusion_type=self.dif_fusion_type,
            max_len=self.max_seq_length,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # General weight init first …
        self.apply(self._init_weights)

        # … then overwrite context lookup weights with pre-computed features and freeze
        for field in self.context_fields:
            feat_data = dataset.item_feat[field].float()
            if feat_data.dim() == 1:
                feat_data = feat_data.unsqueeze(-1)
            self.context_lookup[field].weight.data.copy_(feat_data)
            self.context_lookup[field].weight.requires_grad = False

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _get_tower_embeddings(self, item_seq):
        """Look up and compress context features for each position in the sequence."""
        tower_embeddings = {}
        for field in self.context_fields:
            feat_seq = self.context_lookup[field](item_seq)  # [B, L, feat_dim]
            B, L, D = feat_seq.shape
            compressed = self.context_compress[field](feat_seq.view(B * L, D))
            tower_embeddings[field] = compressed.view(B, L, -1)
        return tower_embeddings

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

        tower_embeddings = self._get_tower_embeddings(item_seq)
        extended_attention_mask = self.get_attention_mask(item_seq)

        encoder_outputs = self.dif_encoder(
            hidden_states=input_emb,
            tower_embeddings=tower_embeddings,
            position_embedding=position_embedding,
            attention_mask=extended_attention_mask,
            output_all_encoded_layers=True,
        )
        output = encoder_outputs[-1]
        seq_output = self.gather_indexes(output, item_seq_len - 1)
        return seq_output  # [B, H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            return self.loss_fct(pos_score, neg_score)
        else:  # CE
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            return self.loss_fct(logits, pos_items)

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores
