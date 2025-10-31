#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inspect SASRec / SASRecAlign training samples to verify data sanity and alignment.

What this checks (first N batches):
  - Sequence length stats: mean/median/min/max, zero-padding ratio
  - Positive item presence and basic validity (non-zero)
  - Negative sample sanity (if available): not equal to pos; not in history (rate)
  - For Align models: text embedding availability, norms, and cosine(ID vs projected text)
  - Prints a few example rows with last K history tokens and target

Usage examples:
  python tools/debug_sasrec_samples.py \
    --model SASRec \
    --dataset Amazon_Beauty \
    --config_files recbole/sasrec_align_base.yaml \
    --batches 2 --sample_rows 10 --device cuda:0

  python tools/debug_sasrec_samples.py \
    --model SASRecAlign \
    --dataset Amazon_Beauty \
    --config_files recbole/sasrec_align_qwen3.yaml \
    --batches 1 --sample_rows 15 --device cuda:0
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect SASRec/SASRecAlign sample batches")
    p.add_argument("--model", required=True, help="Model class name, e.g., SASRec or SASRecAlign")
    p.add_argument("--dataset", required=True, help="Dataset name, e.g., Amazon_Beauty")
    p.add_argument("--config_files", nargs="+", required=True, help="YAML config files")
    p.add_argument("--batches", type=int, default=1, help="Number of training batches to scan")
    p.add_argument("--sample_rows", type=int, default=12, help="Rows to print from first batch")
    p.add_argument("--device", default=None, help="cuda:0 or cpu; default auto")
    p.add_argument("--print_last_k", type=int, default=5, help="Print last-K history tokens in examples")
    return p.parse_args()


def select_device(dev: Optional[str]) -> torch.device:
    if dev:
        return torch.device(dev)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_field(batch, *candidates: str) -> Optional[torch.Tensor]:
    cols = set(batch.columns)
    for name in candidates:
        if name in cols:
            return batch[name]
    return None


def l2_norm(x: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    n = torch.norm(x, p=2, dim=dim, keepdim=True)
    x = x / n.clamp_min(eps)
    x[torch.isnan(x)] = 0.0
    return x


def tensor_to_cpu_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def id_list_to_tokens(dataset, field: str, ids: List[int]) -> List[str]:
    if len(ids) == 0:
        return []
    arr = np.array(ids, dtype=np.int64)
    tokens = dataset.id2token(field, arr)
    # ensure str
    return [str(t) for t in tokens.tolist()]


def summarize_batches(model, dataset, train_data, batches: int, sample_rows: int, device: torch.device, print_last_k: int) -> None:
    iid_field = dataset.iid_field
    uid_field = getattr(dataset, "uid_field", None)

    # Alignment info if available
    has_text = hasattr(model, "item_text_emb") and model.item_text_emb is not None
    if has_text:
        buf = model.item_text_emb
        try:
            norms = torch.norm(buf, p=2, dim=1)
            zero_rows = int((norms == 0).sum().item())
            nan_rows = int(torch.isnan(buf).any(dim=1).sum().item())
            print(f"Align: item_text_emb shape={tuple(buf.shape)} zero_rows={zero_rows} nan_rows={nan_rows}")
        except Exception as e:
            print(f"Align: failed global stats: {e}")
    else:
        print("Align: no item_text_emb (OK for SASRec)")

    # Aggregate statistics
    seq_lens: List[int] = []
    pos_nonzero = 0
    total_rows = 0
    neg_eq_pos = 0
    neg_in_history = 0
    neg_checked_rows = 0

    # Use first iterator fresh (train_data is Iterable)
    it = iter(train_data)
    cached_first_batch = None
    for b in range(max(1, batches)):
        try:
            batch = next(it)
        except StopIteration:
            break
        if b == 0:
            cached_first_batch = batch

        item_seq = get_field(batch, "item_seq", "item_id_list", "item_list")
        item_seq_len = get_field(batch, "item_seq_len", "item_length", "item_list_length")
        pos_items = get_field(batch, "pos_items", "item_id", "pos_item_id")
        neg_items = get_field(batch, "neg_items", "neg_item_id")
        users = get_field(batch, uid_field) if uid_field else None

        if item_seq is None or item_seq_len is None or pos_items is None:
            print("[WARN] Missing expected fields in batch; available columns:", list(batch.columns))
            continue

        # Flatten shapes consistently
        item_seq = item_seq.to(device)
        item_seq_len = item_seq_len.to(device).view(-1)
        pos_items = pos_items.to(device)
        if pos_items.dim() > 1:
            pos_items = pos_items.view(-1)
        B, L = item_seq.size(0), item_seq.size(1)

        total_rows += B
        seq_lens.extend(tensor_to_cpu_np(item_seq_len).tolist())
        pos_nonzero += int((pos_items > 0).sum().item())

        # Negative sanity (if exists and 1-per-row)
        if neg_items is not None:
            if neg_items.dim() > 1:
                # If multiple negatives per row, we only check the first column to keep it simple
                neg_items_chk = neg_items[:, 0].to(device)
            else:
                neg_items_chk = neg_items.to(device)
            if neg_items_chk.numel() == pos_items.numel():
                neg_checked_rows += B
                neg_eq_pos += int((neg_items_chk == pos_items).sum().item())

                # Build history mask per row to test membership
                # Determine per-row history set up to seq_len
                seq_cpu = item_seq.detach().cpu().numpy()
                len_cpu = item_seq_len.detach().cpu().numpy()
                neg_cpu = neg_items_chk.detach().cpu().numpy()
                in_hist = 0
                for i in range(B):
                    li = int(len_cpu[i])
                    hist = set(int(x) for x in seq_cpu[i, : max(li, 0)] if int(x) > 0)
                    if int(neg_cpu[i]) in hist:
                        in_hist += 1
                neg_in_history += in_hist

    # Print aggregate
    if len(seq_lens) > 0:
        arr = np.array(seq_lens, dtype=np.int32)
        print(
            f"SeqLens: mean={float(arr.mean()):.2f} median={float(np.median(arr)):.2f} min={int(arr.min())} max={int(arr.max())}"
        )
    print(f"Rows scanned: {total_rows}; pos>0 rows: {pos_nonzero}")
    if neg_checked_rows > 0:
        print(
            f"Neg checks: rows={neg_checked_rows}  neg==pos: {neg_eq_pos}  neg_in_history: {neg_in_history}"
        )

    # Detailed examples from first batch
    if cached_first_batch is None:
        print("No batch cached for examples.")
        return

    print("\nExample rows (from first batch):")
    batch = cached_first_batch
    item_seq = get_field(batch, "item_seq", "item_id_list", "item_list")
    item_seq_len = get_field(batch, "item_seq_len", "item_length", "item_list_length")
    pos_items = get_field(batch, "pos_items", "item_id", "pos_item_id")
    neg_items = get_field(batch, "neg_items", "neg_item_id")
    users = get_field(batch, uid_field) if uid_field else None

    if item_seq is None or item_seq_len is None or pos_items is None:
        print("[WARN] Missing expected fields in first batch; available columns:", list(batch.columns))
        return

    # Move to CPU for easy numpy processing
    item_seq = item_seq.detach().cpu()
    item_seq_len = item_seq_len.detach().cpu().view(-1)
    if pos_items.dim() > 1:
        pos_items = pos_items.view(-1)
    pos_items = pos_items.detach().cpu()
    if neg_items is not None:
        if neg_items.dim() > 1:
            neg_items = neg_items[:, 0]
        neg_items = neg_items.detach().cpu()

    B, L = item_seq.size(0), item_seq.size(1)
    rows_to_print = min(B, sample_rows)

    # For alignment metrics, compute once for printed pos items
    pos_ids_unique = torch.unique(pos_items[:rows_to_print])
    id_e = model.item_embedding(pos_ids_unique.to(device))
    id_e_n = l2_norm(id_e, dim=1)

    cos_dict: Dict[int, Tuple[float, float, float, bool]] = {}
    if has_text and getattr(model, "item_text_proj", None) is not None:
        with torch.no_grad():
            txt_raw = model.item_text_emb[pos_ids_unique]
            text_zero_mask = (txt_raw.abs().sum(dim=1) == 0).detach().cpu().numpy()
            proj = model.item_text_proj(txt_raw.to(device))
            proj_norm = torch.norm(proj, dim=1).detach().cpu().numpy()
            proj_n = l2_norm(proj, dim=1)
            cos = torch.sum(id_e_n * proj_n, dim=1).detach().cpu().numpy()
            id_norm_vals = torch.norm(id_e, dim=1).detach().cpu().numpy()
        for idx, iid in enumerate(pos_ids_unique.detach().cpu().numpy().tolist()):
            cos_dict[int(iid)] = (
                float(id_norm_vals[idx]),
                float(proj_norm[idx]),
                float(cos[idx]),
                bool(text_zero_mask[idx]),
            )

    # Pretty print
    print("idx  uid  seq_len  last_history_tokens  pos_token  neg_token  pos_in_hist  [id_norm text_norm cos zero]")
    for i in range(rows_to_print):
        li = int(item_seq_len[i].item())
        hist = [int(x) for x in item_seq[i, : max(0, li)].tolist() if int(x) > 0]
        last_k = hist[-print_last_k:] if len(hist) > 0 else []

        pos = int(pos_items[i].item())
        neg = int(neg_items[i].item()) if neg_items is not None else 0
        pos_in_hist = pos in set(hist)

        # tokens
        last_tokens = id_list_to_tokens(dataset, iid_field, last_k)
        pos_tok = dataset.id2token(iid_field, np.array([pos], dtype=np.int64)).tolist()[0] if pos > 0 else "0"
        neg_tok = dataset.id2token(iid_field, np.array([neg], dtype=np.int64)).tolist()[0] if neg > 0 else "0"

        extra = ""
        if pos in cos_dict:
            idn, tnn, cs, tz = cos_dict[pos]
            extra = f"  [{idn:.4f} {tnn:.4f} {cs:.4f} {tz}]"

        uid_disp = int(users[i].item()) if users is not None else 0
        print(
            f"{i:3d}  {uid_disp:4d}  {li:7d}  {str(last_tokens)}  {str(pos_tok)}  {str(neg_tok)}  {str(pos_in_hist):>5s}{extra}"
        )


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
    loss_type = cfg['loss_type'] if 'loss_type' in cfg else 'unknown'
    hidden_size = cfg['hidden_size'] if 'hidden_size' in cfg else 'NA'
    max_seq_length = cfg['MAX_ITEM_LIST_LENGTH'] if 'MAX_ITEM_LIST_LENGTH' in cfg else 'NA'
    n_users = dataset.num(dataset.uid_field) if hasattr(dataset, 'uid_field') else 'NA'
    n_items = dataset.num(dataset.iid_field)
    print(f"loss_type: {loss_type}  hidden_size: {hidden_size}")
    print(f"n_users: {n_users}  n_items: {n_items}  max_seq_length: {max_seq_length}")

    # Alignment header
    item_text_path = getattr(model, "item_text_emb_path", None)
    has_text = hasattr(model, "item_text_emb") and model.item_text_emb is not None
    print(f"item_text_emb_path: {item_text_path}  has_item_text: {has_text}")

    summarize_batches(model, dataset, train_data, args.batches, args.sample_rows, device, args.print_last_k)


if __name__ == "__main__":
    main()


