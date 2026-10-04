"""Run the frozen Mixed Hybrid configuration once on the TEST set.

Loads configs/mixed_hybrid_best.yaml (written by tune_mixed_hybrid.py), mixes each
model's test lists, and compares the hybrid with the standalone models on the same
test ground truth. Never retunes on test.

Example:
    python run_mixed_hybrid.py
"""

import json
import os

import yaml

from common import CONFIG_DIR, OUTPUT_DIR, load_split
from mixed_hybrid import (
    evaluate_recommendations,
    format_metrics,
    load_sources,
    mix_recommendations,
    quotas_to_label,
    save_recommendations,
)


def main():
    config_path = os.path.join(CONFIG_DIR, "mixed_hybrid_best.yaml")
    if not os.path.exists(config_path):
        raise SystemExit(f"Missing {config_path}. Run tune_mixed_hybrid.py first.")
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    models, quotas = cfg["models"], cfg["quotas"]

    _, _, test = load_split()
    sources = load_sources(models, stage="test")

    print("--- Standalone test metrics ---")
    standalone = {}
    for name in models:
        standalone[name] = evaluate_recommendations(sources[name], test)
        print(f"[{name}] {format_metrics(standalone[name])}")

    mixed = mix_recommendations(sources, models, quotas)
    hybrid = evaluate_recommendations(mixed, test)
    print("--- Mixed Hybrid test metrics ---")
    print(f"Configuration: {quotas_to_label(quotas)} (selected on validation by {cfg['primary_metric']})")
    print(format_metrics(hybrid))

    out_dir = os.path.join(OUTPUT_DIR, "MixedHybrid")
    recs_path = os.path.join(out_dir, "recs_test.tsv")
    save_recommendations(mixed, recs_path)
    metrics_path = os.path.join(out_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({**cfg, "standalone_test": standalone, "mixed_hybrid_test": hybrid}, f, indent=2)

    print()
    print(f"Wrote {recs_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
