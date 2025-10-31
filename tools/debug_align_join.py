#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Debug tool: verify that item text embeddings are correctly joined and used by *_Align models.

It loads a model + dataset with your yaml, grabs one training batch, and prints a small
sample of items with:
  - internal item_id and token
  - ID embedding norm
  - projected text embedding norm
  - cosine similarity(ID, projected_text)
  - whether the raw text row is all-zeros (to detect missing join)

Usage examples:
  python tools/debug_align_join.py \
    --model SASRecAlign \
    --dataset Amazon_Beauty \
    --config_files sasrec_align_base.yaml \
    --sample 20 --device cuda:0

  python tools/debug_align_join.py \
    --model BERT4RecAlign \
    --dataset Amazon_Beauty \
    --config_files bert4rec_align_qwen3.yaml \
    --sample 20 --device cuda:0
"""

import argparse
import os
import sys
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

# Make project importable when running from repo root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from recbole.config.configurator import Config
from recbole.data.utils import create_dataset, data_preparation
from recbole.utils import get_model


def parse_args():
    p = argparse.ArgumentParser(description="Inspect alignment join on a sample batch")
    p.add_argument("--model", required=True, help="Model name, e.g., SASRecAlign or BERT4RecAlign")
    p.add_argument("--dataset", required=True, help="Dataset name, e.g., Amazon_Beauty")
    p.add_argument("--config_files", nargs="+", required=True, help="YAML config files")
    p.add_argument("--sample", type=int, default=20, help="Number of items to print")
    p.add_argument("--device", default=None, help="cuda:0 or cpu; default auto")
    return p.parse_args()


def select_device(dev: str | None) -> torch.device:
    if dev:
        return torch.device(dev)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_items_from_batch(batch) -> torch.Tensor:
    # Prefer POS_ITEMS if available (BERT4Rec), else ITEM_ID (SASRec)
    cols = set(batch.columns)
    if "pos_items" in cols:
        items = batch["pos_items"]  # [B, mask_len]
        items = items.view(-1)
    elif "item_id" in cols:
        items = batch["item_id"]  # [B]
        items = items.view(-1)
    else:
        # fallback: try any key that ends with 'item_id'
        key = next((c for c in cols if c.endswith("item_id")), None)
        if key is None:
            return torch.zeros(0, dtype=torch.long)
        items = batch[key].view(-1)
    # remove padding (<=0) and deduplicate
    items = items[items > 0]
    if items.numel() == 0:
        return items
    items = torch.unique(items)
    return items


def main():
    args = parse_args()
    device = select_device(args.device)

    cfg = Config(model=args.model, dataset=args.dataset, config_file_list=args.config_files)
    dataset = create_dataset(cfg)
    train_data, valid_data, test_data = data_preparation(cfg, dataset)

    ModelClass = get_model(cfg["model"])
    model = ModelClass(cfg, dataset).to(device)
    model.eval()

    print(f"Model: {cfg['model']}")
    print(f"item_text_emb_path: {getattr(model, 'item_text_emb_path', None)}")
    has_text = hasattr(model, "item_text_emb") and model.item_text_emb is not None
    print(f"has_item_text: {has_text}")
    if has_text:
        emb_buf = model.item_text_emb
        print(f"item_text_emb tensor shape: {tuple(emb_buf.shape)} dtype={emb_buf.dtype} device={emb_buf.device}")
        try:
            norms_all = torch.norm(emb_buf, p=2, dim=1)
            total = emb_buf.size(0)
            zero_rows = int((norms_all == 0).sum().item())
            nan_rows = int(torch.isnan(emb_buf).any(dim=1).sum().item())
            print(f"global zero_rows={zero_rows}/{total} nan_rows={nan_rows}/{total}")
        except Exception as e:
            print(f"global stats error: {e}")

    # Fetch one training batch
    batch = next(iter(train_data))
    items = pick_items_from_batch(batch)
    if items.numel() == 0:
        print("No positive items found in first batch; try rerun or check config.")
        return

    # Sample N items
    if items.numel() > args.sample:
        items = items[: args.sample]

    items = items.to(device)
    id_e = model.item_embedding(items)
    id_norm = torch.norm(id_e, dim=1).detach().cpu().numpy()

    def safe_normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
        n = torch.norm(x, p=2, dim=dim, keepdim=True)
        x = x / n.clamp_min(eps)
        x[torch.isnan(x)] = 0.0
        return x

    text_norm = np.full_like(id_norm, fill_value=np.nan, dtype=np.float64)
    cos_sim = np.full_like(id_norm, fill_value=np.nan, dtype=np.float64)
    text_zero = np.full(id_norm.shape, False, dtype=bool)

    if has_text and hasattr(model, "item_text_proj") and model.item_text_proj is not None:
        with torch.no_grad():
            txt_raw = model.item_text_emb[items]
            text_zero = (txt_raw.abs().sum(dim=1) == 0).cpu().numpy()
            raw_norm = torch.norm(txt_raw, dim=1).detach().cpu().numpy()
            txt_proj = model.item_text_proj(txt_raw.to(device))
            proj_norm = torch.norm(txt_proj, dim=1)
            id_e_n = safe_normalize(id_e, dim=1)
            txt_proj_n = safe_normalize(txt_proj, dim=1)
            cos = torch.sum(id_e_n * txt_proj_n, dim=1)
            cos_sim = cos.detach().cpu().numpy()
            text_norm = proj_norm.detach().cpu().numpy()
    else:
        print("Text alignment not active (no embedding or projection).")

    # Print rows
    iid_field = dataset.iid_field
    print("\nSampled items (internal_id, token, id_norm, text_norm(proj), cos, text_zero):")
    for i in range(items.size(0)):
        iid = int(items[i].item())
        token = dataset.id2token(iid_field, iid)
        tn = text_norm[i] if not np.isnan(text_norm[i]) else float('nan')
        cs = cos_sim[i] if not np.isnan(cos_sim[i]) else float('nan')
        print(f"  {iid:8d}  {str(token)[:40]:40s}  {id_norm[i]:8.4f}  {tn:8.4f}  {cs:7.4f}  {text_zero[i]}")

    # Summary
    valid_mask = ~np.isnan(cos_sim)
    if valid_mask.any():
        print(f"\nMean cosine (valid): {float(np.nanmean(cos_sim)):.4f}")
        print(f"Text rows zero (count/total): {int(text_zero.sum())}/{len(text_zero)}")
    else:
        print("No valid cosine values - text alignment likely disabled.")


if __name__ == "__main__":
    main()


