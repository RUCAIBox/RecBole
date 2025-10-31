#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export RecBole internal item id -> external token mapping and optionally validate
an item text embedding matrix (.npy).

Outputs CSV with columns: internal_item_id, item_token, and optionally title if available.

Usage:
  python tools/export_internal_item_mapping.py \
    --dataset Amazon_Beauty \
    --config recbole/properties/model/GRU4RecCPR.yaml \
    --output dataset/Amazon_Beauty/item_index_mapping.csv \
    --check_emb dataset/Amazon_Beauty/item_text_emb.base.npy

This helps verify that your saved `item_text_emb.npy` is aligned with RecBole's
internal id order (row i corresponds to item internal id i; row 0 is [PAD]).
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

# Make project importable
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from recbole.config.configurator import Config
from recbole.data.utils import create_dataset


def _detect_item_file(dataset) -> Optional[str]:
    dataset_dir = getattr(dataset, "dataset_path", None)
    dataset_name = getattr(dataset, "dataset_name", None)
    if not dataset_dir or not dataset_name:
        return None
    candidate = os.path.join(dataset_dir, f"{dataset_name}.item")
    return candidate if os.path.exists(candidate) else None


def _find_col_by_base(df: pd.DataFrame, base_names):
    cols = list(df.columns)
    for name in base_names:
        if name in cols:
            return name
    for name in base_names:
        for c in cols:
            if isinstance(c, str) and c.split(":")[0] == name:
                return c
    return None


def _choose_title_field(df: pd.DataFrame) -> Optional[str]:
    return _find_col_by_base(df, ["title", "item_title", "name", "item_name", "product_title", "product_name"])


def export_mapping(dataset_name: str, config_files, output_csv, check_emb: Optional[str] = None) -> None:
    if not dataset_name:
        raise KeyError("--dataset is required (e.g., --dataset Amazon_Beauty)")
    cfg = Config(model="BPR", dataset=dataset_name, config_file_list=config_files)
    dataset = create_dataset(cfg)

    iid_field = dataset.iid_field
    n_items = dataset.num(iid_field)
    ids = np.arange(n_items, dtype=np.int64)
    tokens = dataset.id2token(iid_field, ids)
    tokens = tokens.astype(str)

    df = pd.DataFrame({
        "internal_item_id": ids,
        "item_token": tokens,
    })

    # Attach title if possible
    item_file = _detect_item_file(dataset)
    if item_file is not None:
        try:
            item_df = pd.read_csv(item_file, sep="\t")
            title_col = _choose_title_field(item_df)
            item_id_col = _find_col_by_base(item_df, [cfg["ITEM_ID_FIELD"], "item_id", "item", "iid"])
            if item_id_col is not None and title_col is not None:
                join_df = item_df[[item_id_col, title_col]].copy()
                join_df[item_id_col] = join_df[item_id_col].astype(str)
                df = df.merge(join_df, left_on="item_token", right_on=item_id_col, how="left")
                df.rename(columns={title_col: "title"}, inplace=True)
                df.drop(columns=[c for c in [item_id_col] if c in df.columns], inplace=True)
        except Exception as e:
            print(f"[WARN] Failed to attach title from .item file: {e}")

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved mapping to: {os.path.abspath(output_csv)}  rows={len(df)}")

    # Optional embedding validation
    if check_emb is not None:
        if not os.path.exists(check_emb):
            print(f"[ERROR] Embedding file not found: {check_emb}")
            return
        emb = np.load(check_emb, mmap_mode="r")
        ok = True
        if emb.shape[0] != n_items:
            print(f"[MISMATCH] emb rows={emb.shape[0]} vs n_items={n_items}")
            ok = False
        if emb.shape[0] > 0:
            pad_norm = float(np.linalg.norm(emb[0]))
            if pad_norm > 1e-6:
                print(f"[WARN] Row-0 (PAD) norm={pad_norm:.6f} (expected ~0.0)")
        print(f"Embedding dim={emb.shape[1]}  dtype={emb.dtype}")
        if ok:
            print("Embedding alignment check: OK (row count matches dataset)")


def parse_args():
    p = argparse.ArgumentParser(description="Export internal item id mapping and validate embeddings")
    p.add_argument("--dataset", required=True, help="Dataset name, e.g., Amazon_Beauty")
    p.add_argument("--config", nargs="+", required=False, default=[], help="Optional YAML configs for building dataset")
    p.add_argument("--output", required=True, help="Output CSV path for mapping file")
    p.add_argument("--check_emb", default=None, help="Optional .npy embedding file to validate")
    return p.parse_args()


def main():
    args = parse_args()
    export_mapping(args.dataset, args.config, args.output, args.check_emb)


if __name__ == "__main__":
    main()


