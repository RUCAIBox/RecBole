"""Mixed Hybrid: combine ranked recommendation lists from multiple sources.

This is list-level mixing (quota round-robin / simple round-robin), not score
regression. Raw scores from different models are never treated as comparable.

Sources are the top-N lists written by run_model.py (outputs/<model>/recs_<stage>.tsv,
columns user_id, item_id, rank, score). Those lists already hide previously seen items:
    validation -> training items hidden
    test       -> training + validation items hidden
Deduplication across sources is per user.
"""

import math
import os
from typing import Dict, List, Mapping, Optional, Sequence

import pandas as pd

PRIMARY_METRIC = "NDCG@10"  # same as valid_metric in configs/base.yaml
TOP_K = 10

QuotaDict = Mapping[str, int]
Recommendations = Mapping[str, pd.DataFrame]


def load_sources(model_names: Sequence[str], stage: str) -> Dict[str, pd.DataFrame]:
    """Top-N lists of every model for stage 'valid' or 'test'."""
    from common import load_recs  # lazy: the mixing logic and its tests do not need RecBole

    return {name: load_recs(name, stage) for name in model_names}


# --------------------------------------------------------------------------- #
# Mixing
# --------------------------------------------------------------------------- #
def _ranked_item_lists(frame: pd.DataFrame) -> Dict[str, List[str]]:
    if frame is None or frame.empty:
        return {}
    # Ties in rank are broken by item_id, so the result is deterministic.
    ordered = frame.sort_values(["user_id", "rank", "item_id"], kind="mergesort")
    return {
        str(user_id): group["item_id"].astype(str).tolist()
        for user_id, group in ordered.groupby("user_id", sort=False)
    }


def _score_lookup(frame: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if frame is None or frame.empty:
        return {}
    return {
        str(user_id): dict(zip(group["item_id"].astype(str), group["score"].astype(float)))
        for user_id, group in frame.groupby("user_id", sort=False)
    }


def mix_recommendations(
    recommendations: Recommendations,
    model_order: Sequence[str],
    quotas: Optional[QuotaDict] = None,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """Mix ranked lists into one list of top_k unique items per user.

    In each round, take up to quotas[model] not-yet-chosen items from each model's list
    (in model_order), and repeat until top_k items are chosen or every list is used up.
    quotas=None is plain round-robin: one item per model per round.

    Returns user_id, item_id, rank, source, score; score is the source model's own score
    and is not comparable across sources.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    lists_by_model = {name: _ranked_item_lists(recommendations.get(name)) for name in model_order}
    scores_by_model = {name: _score_lookup(recommendations.get(name)) for name in model_order}
    users = sorted(set().union(*(lists.keys() for lists in lists_by_model.values())))

    rows: List[dict] = []
    for user_id in users:
        pointers = {name: 0 for name in model_order}
        seen_items: set = set()
        chosen: List[tuple] = []  # (item_id, source, score)

        while len(chosen) < top_k:
            progressed = False
            for name in model_order:
                take = 1 if quotas is None else max(0, int(quotas.get(name, 0)))
                user_list = lists_by_model[name].get(user_id, [])
                taken = 0
                while taken < take and pointers[name] < len(user_list) and len(chosen) < top_k:
                    item_id = user_list[pointers[name]]
                    pointers[name] += 1
                    if item_id in seen_items:
                        continue
                    seen_items.add(item_id)
                    score = scores_by_model[name].get(user_id, {}).get(item_id, float("nan"))
                    chosen.append((item_id, name, score))
                    taken += 1
                    progressed = True
            if not progressed:
                break

        for rank, (item_id, source, score) in enumerate(chosen, start=1):
            rows.append({"user_id": user_id, "item_id": item_id, "rank": rank, "source": source, "score": score})

    return pd.DataFrame(rows, columns=["user_id", "item_id", "rank", "source", "score"])


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _dcg(hits: Sequence[int]) -> float:
    return sum(hit / math.log2(idx + 2) for idx, hit in enumerate(hits))


def evaluate_recommendations(recommendations: pd.DataFrame, ground_truth: pd.DataFrame, k: int = TOP_K) -> Dict[str, float]:
    """Recall, MRR, NDCG, Hit, Precision @k, averaged over the users in ground_truth (as RecBole does)."""
    truth = (
        ground_truth[["user_id", "item_id"]].astype(str).drop_duplicates()
        .groupby("user_id", sort=False)["item_id"].apply(set).to_dict()
    )
    ordered = recommendations.sort_values(["user_id", "rank", "item_id"], kind="mergesort")
    preds = {
        str(user_id): group["item_id"].astype(str).tolist()
        for user_id, group in ordered.groupby("user_id", sort=False)
    }

    recalls, mrrs, ndcgs, hits, precisions = [], [], [], [], []
    for user_id, relevant in truth.items():
        flags = [1 if item in relevant else 0 for item in preds.get(str(user_id), [])[:k]]
        n_hits = sum(flags)
        recalls.append(n_hits / len(relevant))
        precisions.append(n_hits / k)
        hits.append(1.0 if n_hits else 0.0)
        mrrs.append(1.0 / (flags.index(1) + 1) if n_hits else 0.0)
        ndcgs.append(_dcg(flags) / _dcg([1] * min(len(relevant), k)))

    mean = lambda values: float(sum(values) / len(values)) if values else 0.0  # noqa: E731
    return {
        f"Recall@{k}": mean(recalls),
        f"MRR@{k}": mean(mrrs),
        f"NDCG@{k}": mean(ndcgs),
        f"Hit@{k}": mean(hits),
        f"Precision@{k}": mean(precisions),
    }


def format_metrics(metrics: Mapping[str, float], digits: int = 4) -> str:
    return "  ".join(f"{name}={value:.{digits}f}" for name, value in metrics.items())


def quotas_to_label(quotas: Optional[QuotaDict]) -> str:
    if quotas is None:
        return "RoundRobin"
    return " ".join(f"{name}={int(value)}" for name, value in quotas.items())


def save_recommendations(frame: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)
