"""Unit tests for Mixed Hybrid list mixing / deduplication."""

import unittest

import pandas as pd

from mixed_hybrid import evaluate_recommendations, mix_recommendations


def _recs(model_rows):
    """Build a source frame from {user: [item, ...]} preserving list order as rank."""
    rows = []
    for user_id, items in model_rows.items():
        for rank, item_id in enumerate(items, start=1):
            rows.append(
                {
                    "user_id": str(user_id),
                    "item_id": str(item_id),
                    "rank": rank,
                    "score": float(len(items) - rank + 1),
                }
            )
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rank", "score"])


class TestMixing(unittest.TestCase):
    def test_duplicates_removed(self):
        sources = {
            "A": _recs({"u1": ["A", "B", "C"]}),
            "B": _recs({"u1": ["B", "D", "E"]}),
            "C": _recs({"u1": ["A", "F", "G"]}),
        }
        mixed = mix_recommendations(sources, ["A", "B", "C"], top_k=10)
        items = mixed["item_id"].tolist()
        # Round-robin order with per-user dedupe (skip already-seen, keep filling).
        self.assertEqual(items, ["A", "B", "F", "C", "D", "G", "E"])
        self.assertEqual(len(items), len(set(items)))

    def test_quota_affects_order(self):
        sources = {
            "ItemKNN": _recs({"u1": ["A", "B", "C", "D", "E"]}),
            "BPR": _recs({"u1": ["F", "G", "H", "I", "J"]}),
            "ContentBased": _recs({"u1": ["K", "L", "M", "N", "O"]}),
        }
        quotas = {"ItemKNN": 2, "BPR": 1, "ContentBased": 1}
        mixed = mix_recommendations(sources, list(quotas), quotas, top_k=8)
        self.assertEqual(
            mixed["item_id"].tolist(),
            ["A", "B", "F", "K", "C", "D", "G", "L"],
        )
        self.assertEqual(
            mixed["source"].tolist(),
            [
                "ItemKNN",
                "ItemKNN",
                "BPR",
                "ContentBased",
                "ItemKNN",
                "ItemKNN",
                "BPR",
                "ContentBased",
            ],
        )

    def test_user_isolation(self):
        sources = {
            "A": _recs({"u1": ["X", "Y"], "u2": ["X", "Z"]}),
            "B": _recs({"u1": ["X", "Z"], "u2": ["Y", "X"]}),
        }
        mixed = mix_recommendations(sources, ["A", "B"], top_k=3)
        u1 = mixed[mixed["user_id"] == "u1"]["item_id"].tolist()
        u2 = mixed[mixed["user_id"] == "u2"]["item_id"].tolist()
        # u1: A->X, B skips X then takes Z, A->Y
        self.assertEqual(u1, ["X", "Z", "Y"])
        # u2: A->X, B->Y, A->Z
        self.assertEqual(u2, ["X", "Y", "Z"])

    def test_missing_source_for_user(self):
        sources = {
            "A": _recs({"u1": ["A1", "A2", "A3"]}),
            "B": _recs({}),  # no recommendations at all
            "C": _recs({"u1": ["C1", "C2"]}),
        }
        mixed = mix_recommendations(sources, ["A", "B", "C"], top_k=4)
        self.assertEqual(mixed["item_id"].tolist(), ["A1", "C1", "A2", "C2"])

    def test_top_k_cap(self):
        sources = {
            "A": _recs({"u1": list("ABCDEFGHIJ")}),
            "B": _recs({"u1": list("KLMNOPQRST")}),
        }
        mixed = mix_recommendations(sources, ["A", "B"], top_k=5)
        self.assertEqual(len(mixed), 5)
        self.assertEqual(mixed["rank"].tolist(), [1, 2, 3, 4, 5])

    def test_determinism(self):
        sources = {
            "A": _recs({"u1": ["A", "B", "C"]}),
            "B": _recs({"u1": ["D", "E", "A"]}),
            "C": _recs({"u1": ["F", "G", "H"]}),
        }
        quotas = {"A": 2, "B": 1, "C": 1}
        first = mix_recommendations(sources, list(quotas), quotas, top_k=6)
        second = mix_recommendations(sources, list(quotas), quotas, top_k=6)
        pd.testing.assert_frame_equal(first, second)

    def test_exhausted_source_continues(self):
        sources = {
            "A": _recs({"u1": ["A1", "A2"]}),
            "B": _recs({"u1": ["B1", "B2", "B3", "B4", "B5", "B6"]}),
        }
        mixed = mix_recommendations(sources, ["A", "B"], {"A": 5, "B": 1}, top_k=6)
        # A only has 2 unique items; mixer should keep filling from B.
        self.assertEqual(len(mixed), 6)
        self.assertIn("A1", mixed["item_id"].tolist())
        self.assertIn("B1", mixed["item_id"].tolist())


class TestEvaluation(unittest.TestCase):
    def test_evaluate_simple(self):
        recs = _recs({"1": ["a", "b", "c"]})
        truth = pd.DataFrame({"user_id": ["1", "1"], "item_id": ["b", "z"]})
        metrics = evaluate_recommendations(recs, truth, k=3)
        self.assertAlmostEqual(metrics["Hit@3"], 1.0)
        self.assertAlmostEqual(metrics["Recall@3"], 0.5)
        self.assertAlmostEqual(metrics["MRR@3"], 0.5)


if __name__ == "__main__":
    unittest.main()
