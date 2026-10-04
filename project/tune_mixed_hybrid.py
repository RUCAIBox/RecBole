"""Tune Mixed Hybrid quotas on the VALIDATION recommendation lists.

Never uses the test set for selection. The primary metric is NDCG@10 (same as
configs/base.yaml). A round-robin baseline and several quota configurations are
evaluated; the best one is written to configs/mixed_hybrid_best.yaml.

Prerequisite:
    python run_model.py ItemKNN BPR ContentBased

Example:
    python tune_mixed_hybrid.py
"""

import os

import pandas as pd
import yaml

from common import CONFIG_DIR, OUTPUT_DIR, load_split
from mixed_hybrid import (
    PRIMARY_METRIC,
    evaluate_recommendations,
    load_sources,
    mix_recommendations,
    quotas_to_label,
    save_recommendations,
)

# Neighbourhood CF / matrix factorisation / content similarity.
MODELS = ["ItemKNN", "BPR", "ContentBased"]

# Items taken per round from each source; None is plain round-robin (1 each).
CANDIDATE_QUOTAS = [
    None,
    {"ItemKNN": 5, "BPR": 3, "ContentBased": 2},
    {"ItemKNN": 4, "BPR": 4, "ContentBased": 2},
    {"ItemKNN": 3, "BPR": 5, "ContentBased": 2},
    {"ItemKNN": 5, "BPR": 2, "ContentBased": 3},
    {"ItemKNN": 4, "BPR": 3, "ContentBased": 3},
    {"ItemKNN": 3, "BPR": 4, "ContentBased": 3},
    {"ItemKNN": 2, "BPR": 4, "ContentBased": 4},
    {"ItemKNN": 2, "BPR": 3, "ContentBased": 5},
    {"ItemKNN": 3, "BPR": 3, "ContentBased": 4},
]


def main():
    _, valid, _ = load_split()
    sources = load_sources(MODELS, stage="valid")

    print(f"Primary metric: {PRIMARY_METRIC}")
    print(f"{'Configuration':<40} {'Recall@10':>10} {'MRR@10':>10} {'NDCG@10':>10}")
    rows, best = [], None
    for quotas in CANDIDATE_QUOTAS:
        mixed = mix_recommendations(sources, MODELS, quotas)
        metrics = evaluate_recommendations(mixed, valid)
        label = quotas_to_label(quotas)
        rows.append({"configuration": label, **metrics})
        print(f"{label:<40} {metrics['Recall@10']:>10.4f} {metrics['MRR@10']:>10.4f} {metrics['NDCG@10']:>10.4f}")
        if best is None or metrics[PRIMARY_METRIC] > best["metrics"][PRIMARY_METRIC]:
            best = {"quotas": quotas, "label": label, "metrics": metrics, "recs": mixed}

    out_dir = os.path.join(OUTPUT_DIR, "MixedHybrid")
    os.makedirs(out_dir, exist_ok=True)
    table_path = os.path.join(out_dir, "tuning_valid.csv")
    pd.DataFrame(rows).sort_values(PRIMARY_METRIC, ascending=False).to_csv(table_path, index=False)
    save_recommendations(best["recs"], os.path.join(out_dir, "recs_valid.tsv"))

    best_path = os.path.join(CONFIG_DIR, "mixed_hybrid_best.yaml")
    payload = {
        "models": MODELS,
        "quotas": best["quotas"],
        "primary_metric": PRIMARY_METRIC,
        "valid_metrics": best["metrics"],
    }
    with open(best_path, "w", encoding="utf-8") as f:
        f.write(f"# Best by valid {PRIMARY_METRIC} = {best['metrics'][PRIMARY_METRIC]:.4f} (written by tune_mixed_hybrid.py)\n")
        yaml.safe_dump(payload, f, sort_keys=False)

    print()
    print(f"Best configuration: {best['label']}")
    print(f"Wrote {table_path}")
    print(f"Wrote {best_path}")


if __name__ == "__main__":
    main()
