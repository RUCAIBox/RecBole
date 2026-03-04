# -*- coding: utf-8 -*-
"""
Standalone smoke tests for DIFSASRec.

Runs without the full RecBole quick_start infrastructure (no ray required).
Covers:
  - forward / calculate_loss (CE and BPR) / predict / full_sort_predict
  - all three fusion types: sum, concat, gate
  - with and without context towers
"""

import unittest

import torch
import torch.nn as nn

from recbole.model.sequential_recommender.difsasrec import DIFSASRec
from recbole.utils import FeatureType


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------

class _MockItemFeat:
    def __init__(self, data: dict):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]


class _MockDataset:
    def __init__(self, n_items=50, context_fields=None):
        self.n_items = n_items
        self.field2type = {}
        feat_data = {}

        if context_fields:
            for field, dim in context_fields.items():
                self.field2type[field] = FeatureType.FLOAT_SEQ
                feat_data[field] = torch.randn(n_items, dim)

        self.item_feat = _MockItemFeat(feat_data)

    def num(self, field):
        return self.n_items


class _MockConfig(dict):
    """dict subclass that supports both config["key"] and config.get("key")."""
    def __init__(self, extra=None):
        defaults = {
            # SequentialRecommender fields
            "USER_ID_FIELD": "user_id",
            "ITEM_ID_FIELD": "item_id",
            "LIST_SUFFIX": "_list",
            "ITEM_LIST_LENGTH_FIELD": "item_length",
            "NEG_PREFIX": "neg_",
            "MAX_ITEM_LIST_LENGTH": 10,
            "device": torch.device("cpu"),
            # DIFSASRec fields
            "n_layers": 2,
            "n_heads": 2,
            "hidden_size": 16,
            "inner_size": 32,
            "hidden_dropout_prob": 0.0,
            "attn_dropout_prob": 0.0,
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-12,
            "initializer_range": 0.02,
            "dif_fusion_type": "sum",
            "loss_type": "CE",
        }
        if extra:
            defaults.update(extra)
        super().__init__(defaults)


class _MockInteraction(dict):
    """dict-like interaction batch."""
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

B, L = 4, 10   # batch size, sequence length
N_ITEMS = 50


def _make_interaction(config, loss_type):
    item_seq = torch.randint(1, N_ITEMS, (B, L))
    item_seq_len = torch.randint(1, L + 1, (B,))
    pos_items = torch.randint(1, N_ITEMS, (B,))

    inter = _MockInteraction({
        config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]: item_seq,
        config["ITEM_LIST_LENGTH_FIELD"]: item_seq_len,
        config["ITEM_ID_FIELD"]: pos_items,
    })
    if loss_type == "BPR":
        neg_items = torch.randint(1, N_ITEMS, (B,))
        inter[config["NEG_PREFIX"] + config["ITEM_ID_FIELD"]] = neg_items
    return inter


def _build_model(fusion_type="sum", loss_type="CE", context_fields_dims=None):
    cfg = _MockConfig({
        "dif_fusion_type": fusion_type,
        "loss_type": loss_type,
        "context_fields": list(context_fields_dims.keys()) if context_fields_dims else [],
    })
    dataset = _MockDataset(n_items=N_ITEMS, context_fields=context_fields_dims)
    model = DIFSASRec(cfg, dataset).eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDIFSASRecFusionTypes(unittest.TestCase):

    def _run_forward_and_loss(self, fusion_type, loss_type):
        model, cfg = _build_model(fusion_type=fusion_type, loss_type=loss_type)
        inter = _make_interaction(cfg, loss_type)

        with torch.no_grad():
            loss = model.calculate_loss(inter)
        self.assertFalse(torch.isnan(loss), f"NaN loss ({fusion_type}, {loss_type})")
        self.assertFalse(torch.isinf(loss), f"Inf loss ({fusion_type}, {loss_type})")

    def test_sum_fusion_CE(self):
        self._run_forward_and_loss("sum", "CE")

    def test_sum_fusion_BPR(self):
        self._run_forward_and_loss("sum", "BPR")

    def test_concat_fusion_CE(self):
        self._run_forward_and_loss("concat", "CE")

    def test_concat_fusion_BPR(self):
        self._run_forward_and_loss("concat", "BPR")

    def test_gate_fusion_CE(self):
        self._run_forward_and_loss("gate", "CE")

    def test_gate_fusion_BPR(self):
        self._run_forward_and_loss("gate", "BPR")


class TestDIFSASRecOutputShapes(unittest.TestCase):

    def test_forward_shape(self):
        model, cfg = _build_model()
        item_seq = torch.randint(1, N_ITEMS, (B, L))
        item_seq_len = torch.randint(1, L + 1, (B,))
        with torch.no_grad():
            out = model.forward(item_seq, item_seq_len)
        self.assertEqual(out.shape, (B, 16))  # hidden_size=16

    def test_predict_shape(self):
        model, cfg = _build_model()
        inter = _make_interaction(cfg, "CE")
        with torch.no_grad():
            scores = model.predict(inter)
        self.assertEqual(scores.shape, (B,))

    def test_full_sort_predict_shape(self):
        model, cfg = _build_model()
        inter = _make_interaction(cfg, "CE")
        with torch.no_grad():
            scores = model.full_sort_predict(inter)
        self.assertEqual(scores.shape, (B, N_ITEMS))


class TestDIFSASRecWithContextTowers(unittest.TestCase):

    def test_single_tower_sum(self):
        model, cfg = _build_model(
            fusion_type="sum",
            context_fields_dims={"feat_a": 32},
        )
        inter = _make_interaction(cfg, "CE")
        with torch.no_grad():
            loss = model.calculate_loss(inter)
        self.assertFalse(torch.isnan(loss))

    def test_two_towers_concat(self):
        model, cfg = _build_model(
            fusion_type="concat",
            context_fields_dims={"feat_a": 32, "feat_b": 64},
        )
        inter = _make_interaction(cfg, "CE")
        with torch.no_grad():
            loss = model.calculate_loss(inter)
        self.assertFalse(torch.isnan(loss))

    def test_two_towers_gate(self):
        model, cfg = _build_model(
            fusion_type="gate",
            context_fields_dims={"feat_a": 32, "feat_b": 64},
        )
        inter = _make_interaction(cfg, "CE")
        with torch.no_grad():
            loss = model.calculate_loss(inter)
        self.assertFalse(torch.isnan(loss))

    def test_context_weights_frozen(self):
        """Context lookup embedding weights must not accumulate gradients."""
        model, cfg = _build_model(
            fusion_type="sum",
            context_fields_dims={"feat_a": 32},
        )
        inter = _make_interaction(cfg, "CE")
        loss = model.calculate_loss(inter)
        loss.backward()
        self.assertIsNone(
            model.context_lookup["feat_a"].weight.grad,
            "context_lookup weights should be frozen (requires_grad=False)",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
