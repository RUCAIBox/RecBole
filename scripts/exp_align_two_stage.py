#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-stage training for SASRecAlign:
  - Stage 1: freeze backbone, train projection head + cross for few epochs
  - Stage 2: unfreeze (small LR) joint fine-tuning
Also supports grid over temperature (tau) and alignment_weight, with optional
Top-K neighbor exclusion in InfoNCE.
"""
import argparse
import glob
import os
from datetime import datetime

# Ensure project root is on sys.path when running this script directly
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from recbole.quick_start import run_recbole, load_data_and_model
from recbole.utils import get_trainer
import shutil
# Reduce CUDA fragmentation issues (use new env var; old one deprecated)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "cuda_malloc_backend:expandable_segments")


def ensure_timestamp_dataset(dataset_name: str, data_root: str = "dataset") -> str:
    """Ensure the .inter file of the dataset has a 'timestamp' field.
    If missing, create a derived dataset '<name>_tsfix' with synthetic per-user timestamps
    assigned in original row order.
    Returns the dataset name to use (original or derived).
    """
    src_dir = os.path.join(data_root, dataset_name)
    inter_src = os.path.join(src_dir, f"{dataset_name}.inter")
    if not os.path.exists(inter_src):
        # no file to inspect; just return original
        return dataset_name
    with open(inter_src, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
    cols = header.split("\t")
    if "timestamp" in cols:
        return dataset_name

    # build derived dataset with synthetic timestamps
    derived_name = f"{dataset_name}_tsfix"
    dst_dir = os.path.join(data_root, derived_name)
    os.makedirs(dst_dir, exist_ok=True)

    # copy auxiliary files if exist
    for ext in (".item", ".user"):
        src_file = os.path.join(src_dir, f"{dataset_name}{ext}")
        if os.path.exists(src_file):
            dst_file = os.path.join(dst_dir, f"{derived_name}{ext}")
            shutil.copy2(src_file, dst_file)

    # rewrite inter with timestamp
    inter_dst = os.path.join(dst_dir, f"{derived_name}.inter")
    uid_idx = cols.index("user_id") if "user_id" in cols else None
    if uid_idx is None:
        # cannot synthesize without user_id; fallback to original
        return dataset_name
    new_header = header + "\ttimestamp"
    user_counter = {}
    with open(inter_src, "r", encoding="utf-8") as fin, open(inter_dst, "w", encoding="utf-8") as fout:
        fout.write(new_header + "\n")
        _ = fin.readline()  # skip original header already read
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            uid = parts[uid_idx]
            cnt = user_counter.get(uid, 0) + 1
            user_counter[uid] = cnt
            fout.write(f"{line}\t{cnt}\n")
    return derived_name


def latest_checkpoint(path_dir: str) -> str:
    files = sorted(
        glob.glob(os.path.join(path_dir, "*.pth")),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    return files[0] if files else None


def main():
    parser = argparse.ArgumentParser(description="Two-stage SASRecAlign alignment experiments")
    parser.add_argument("--dataset", default="Amazon_Beauty")
    parser.add_argument("--mode", choices=["base", "llm"], default="base", help="Which text embeddings to use")
    parser.add_argument("--stage1_epochs", type=int, default=8)
    parser.add_argument("--stage2_epochs", type=int, default=50)
    parser.add_argument("--stage2_lr", type=float, default=3e-5)
    parser.add_argument("--exclude_topk", type=int, default=0, help="Exclude Top-K neighbors in InfoNCE negatives")
    parser.add_argument("--checkpoint_root", default="results/two_stage_checkpoints")
    parser.add_argument("--base_emb", default="dataset/Amazon_Beauty/item_text_emb.base.npy")
    parser.add_argument("--llm_emb", default="dataset/Amazon_Beauty/item_text_emb.qwen3.npy")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.05, 0.07])
    parser.add_argument("--weights", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    args = parser.parse_args()

    os.makedirs(args.checkpoint_root, exist_ok=True)

    # Ensure dataset has timestamp; if not, synthesize a derived dataset with timestamps
    dataset_to_use = ensure_timestamp_dataset(args.dataset, data_root="dataset")

    combos = [(t, w) for t in args.temperatures for w in args.weights]
    for tau, align_w in combos:
        tag = f"{dataset_to_use}-{args.mode}-tau{tau}-w{align_w}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ckpt_dir = os.path.join(args.checkpoint_root, tag)
        os.makedirs(ckpt_dir, exist_ok=True)

        # Common config overrides
        common_cfg = {
            "model": "SASRec_Align",
            "dataset": dataset_to_use,
            "field_separator": "\t",
            "TIME_FIELD": "timestamp",
            "seq_separator": " ",
            "load_col": {
                "inter": ["user_id", "item_id", "timestamp"],
                "item": ["item_id"],
            },
            # memory-friendly evaluation/training batch sizes for large item catalogs
            "train_batch_size": 256,
            "eval_batch_size": 16,
            "topk": [10],
            "metrics": ["Hit", "Recall", "NDCG", "MRR"],
            "eval_args": {
                "group_by": "user",
                "order": "TO",
                "split": {"LS": "valid_and_test"},
                "mode": {"valid": "uni100", "test": "full"},
            },
            "train_neg_sample_args": None,
            "loss_type": "CE",
            "checkpoint_dir": ckpt_dir,
            "use_align": True,
            "alignment_weight": float(align_w),
            "temperature": float(tau),
            "align_exclude_topk": int(args.exclude_topk),
            "use_cross": True,
            "disable_text_feature": False,
        }
        if args.mode == "base":
            common_cfg["use_llm"] = False
            common_cfg["item_text_emb_path_base"] = args.base_emb
        else:
            common_cfg["use_llm"] = True
            common_cfg["item_text_emb_path_llm"] = args.llm_emb

        # ---------------- Stage 1: Freeze backbone ----------------
        stage1_cfg = dict(common_cfg)
        stage1_cfg.update(
            {
                "freeze_backbone": True,
                "epochs": int(args.stage1_epochs),
            }
        )
        print(f"[Stage1] {tag}  tau={tau}  align_w={align_w}  freeze_backbone=True  epochs={args.stage1_epochs}")
        res1 = run_recbole(
            model="SASRec_Align",
            dataset=dataset_to_use,
            config_dict=stage1_cfg,
            saved=True,
        )
        print(f"[Stage1] best_valid={res1.get('best_valid_result')} test={res1.get('test_result')}")

        # find latest checkpoint
        model_file = latest_checkpoint(ckpt_dir)
        if not model_file:
            print(f"[ERROR] No checkpoint found in {ckpt_dir}")
            continue

        # ---------------- Stage 2: Unfreeze & small LR ----------------
        config2, model2, dataset2, train_data2, valid_data2, test_data2 = load_data_and_model(model_file=model_file)
        # make sure model name is correct and freeze is lifted
        config2["model"] = "SASRec_Align"
        if hasattr(model2, "set_freeze"):
            model2.set_freeze(False)
        # small LR & stage2 epochs
        config2["learning_rate"] = float(args.stage2_lr)
        config2["epochs"] = int(args.stage2_epochs)
        # keep memory-friendly eval settings
        config2["eval_batch_size"] = 16
        config2["topk"] = [10]
        config2["metrics"] = ["Hit", "Recall", "NDCG", "MRR"]
        # keep alignment settings
        config2["alignment_weight"] = float(align_w)
        config2["temperature"] = float(tau)
        config2["align_exclude_topk"] = int(args.exclude_topk)
        # ensure evaluation protocol
        config2["eval_args"] = {
            "group_by": "user",
            "order": "TO",
            "split": {"LS": "valid_and_test"},
            "mode": {"valid": "uni100", "test": "full"},
        }

        trainer2 = get_trainer(config2["MODEL_TYPE"], config2["model"])(config2, model2)
        print(f"[Stage2] {tag}  small_lr={args.stage2_lr}  epochs={args.stage2_epochs}")
        best_valid_score2, best_valid_result2 = trainer2.fit(train_data2, valid_data2, saved=True, show_progress=config2["show_progress"])
        test_result2 = trainer2.evaluate(test_data2, load_best_model=True, show_progress=config2["show_progress"])
        print(f"[Stage2] best_valid={best_valid_result2}")
        print(f"[Stage2] test={test_result2}")
        # brief metrics
        for k in ["Hit@10", "Recall@10", "NDCG@10", "MRR@10"]:
            if k in test_result2:
                print(f"[Stage2][test]{k}: {test_result2[k]:.6f}")


if __name__ == "__main__":
    main()


