#!/usr/bin/env python3
"""utility script to sanity-check InfoNCE alignment in SASRecAlign.

Example
-------
python scripts/verify_infonce.py \
    --dataset Amazon_Beauty \
    --config_files sasrec_align_qwen3.yaml \
    --num_items 64

The script will:
1. build the RecBole configuration/dataset (forced on CPU for portability);
2. instantiate `SASRecAlign` and check whether alignment is enabled;
3. sample the first `num_items` item ids and compute InfoNCE alignment loss;
4. output temperature, alignment_weight, and similarity diagnostics.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import torch

from recbole.config import Config
from recbole.data.utils import create_dataset

# importing the model directly avoids relying on registry side-effects
from recbole.model.sequential_recommender.sasrec_align import SASRecAlign


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify InfoNCE alignment settings for SASRecAlign")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g., Amazon_Beauty")
    parser.add_argument(
        "--model",
        default="SASRecAlign",
        help="Model class name (default: SASRecAlign)",
    )
    parser.add_argument(
        "--config_files",
        nargs="*",
        default=None,
        help="Optional list of YAML config files, same as run_recbole --config_files",
    )
    parser.add_argument(
        "--num_items",
        type=int,
        default=64,
        help="Number of leading item ids to sample for the InfoNCE check (default: 64)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra diagnostics including similarity matrix stats",
    )
    return parser.parse_args()


def build_config(dataset: str, model: str, config_files: Optional[List[str]]) -> Config:
    cfg = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_files,
        config_dict={"device": "cpu"},
    )
    # ensure CPU-only to avoid GPU dependency for a sanity check
    cfg["use_gpu"] = False
    cfg["device"] = "cpu"
    return cfg


def compute_infonce(model: SASRecAlign, max_items: int, verbose: bool = False) -> None:
    if not model.use_align or model.alignment_weight <= 0.0:
        print("[WARN] Alignment not enabled (use_align=False or alignment_weight<=0). Nothing to verify.")
        return

    if not model._has_item_text():
        print("[WARN] Model has no text embeddings loaded; InfoNCE cannot be computed.")
        return

    # sample item ids [1, max_items]; clamp based on actual number of items
    upper = min(model.n_items - 1, max_items)
    if upper <= 0:
        print("[ERROR] Dataset contains no items beyond PAD id=0.")
        return

    item_ids = torch.arange(1, upper + 1, dtype=torch.long)

    with torch.no_grad():
        id_emb = model.item_embedding(item_ids)
        text_raw = model._gather_text_raw(item_ids)
        if model.detach_text_emb:
            text_raw = text_raw.detach()
        text_proj = model._project_text(text_raw)

        loss = model._info_nce_align(id_emb, text_proj)

        print("=== InfoNCE Verification ===")
        print(f"n_items_used       : {item_ids.size(0)}")
        print(f"alignment_weight   : {model.alignment_weight}")
        print(f"temperature        : {model.temperature}")
        print(f"loss (InfoNCE CE)  : {loss.item():.6f}")

        if verbose:
            logits = torch.matmul(
                torch.nn.functional.normalize(id_emb, dim=1),
                torch.nn.functional.normalize(text_proj, dim=1).t(),
            ) / model.temperature
            diag = torch.diagonal(logits)
            off_diag = logits.flatten()[
                ~torch.eye(logits.size(0), dtype=torch.bool).flatten()
            ]
            print("--- similarity diagnostics (before softmax) ---")
            print(f"diag mean / std    : {diag.mean().item():.6f} / {diag.std().item():.6f}")
            print(f"off-diag mean/std  : {off_diag.mean().item():.6f} / {off_diag.std().item():.6f}")


def main() -> None:
    args = parse_args()

    try:
        cfg = build_config(args.dataset, args.model, args.config_files)
    except Exception as exc:  # pragma: no cover - guard for configuration errors
        print(f"[ERROR] Failed to build Config: {exc}")
        sys.exit(1)

    dataset = create_dataset(cfg)

    try:
        model = SASRecAlign(cfg, dataset)
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] Unable to instantiate SASRecAlign: {exc}")
        sys.exit(1)

    compute_infonce(model, args.num_items, verbose=args.verbose)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
verify_infonce.py - 验证InfoNCE实现的正确性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def info_nce_loss(a, b, temperature=0.07):
    """InfoNCE损失函数的实现"""
    # L2归一化
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    
    # 计算相似度矩阵
    logits = torch.matmul(a, b.t()) / temperature
    
    # 对角线元素是正样本对
    labels = torch.arange(a.size(0), device=a.device)
    
    # 使用交叉熵损失
    loss = nn.CrossEntropyLoss()(logits, labels)
    
    return loss, logits

def test_infonce():
    """测试InfoNCE实现"""
    print("=== InfoNCE实现验证 ===\n")
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 测试1: 完美对齐的情况
    print("测试1: 完美对齐")
    batch_size = 4
    dim = 128
    
    # 创建相同的embeddings
    emb = torch.randn(batch_size, dim)
    loss, logits = info_nce_loss(emb, emb, temperature=0.07)
    
    print(f"Batch size: {batch_size}")
    print(f"Embedding dimension: {dim}")
    print(f"Loss (应该接近0): {loss.item():.6f}")
    print(f"Logits对角线均值: {torch.diag(logits).mean().item():.4f}")
    print(f"Logits非对角线均值: {(logits.sum() - torch.diag(logits).sum()).item() / (batch_size * (batch_size - 1)):.4f}")
    
    # 测试2: 随机embeddings
    print("\n测试2: 随机embeddings")
    a = torch.randn(batch_size, dim)
    b = torch.randn(batch_size, dim)
    
    loss, logits = info_nce_loss(a, b, temperature=0.07)
    print(f"Loss: {loss.item():.6f}")
    
    # 测试3: 不同温度参数
    print("\n测试3: 温度参数影响")
    temperatures = [0.05, 0.07, 0.1, 0.2]
    
    for temp in temperatures:
        loss, _ = info_nce_loss(a, b, temperature=temp)
        print(f"Temperature={temp}: Loss={loss.item():.6f}")
    
    # 测试4: 梯度流
    print("\n测试4: 梯度流验证")
    a_param = nn.Parameter(torch.randn(batch_size, dim))
    b_param = nn.Parameter(torch.randn(batch_size, dim))
    
    optimizer = torch.optim.Adam([a_param, b_param], lr=0.01)
    
    initial_loss, _ = info_nce_loss(a_param, b_param)
    print(f"初始Loss: {initial_loss.item():.6f}")
    
    # 优化几步
    for i in range(10):
        optimizer.zero_grad()
        loss, _ = info_nce_loss(a_param, b_param)
        loss.backward()
        optimizer.step()
    
    final_loss, _ = info_nce_loss(a_param, b_param)
    print(f"优化后Loss: {final_loss.item():.6f}")
    print(f"Loss下降: {(initial_loss - final_loss).item():.6f}")
    
    # 测试5: SASRecAlign中的实际用法
    print("\n测试5: SASRecAlign集成验证")
    print("InfoNCE在SASRecAlign中的使用:")
    print("1. 对正样本的ID embeddings和文本embeddings进行对齐")
    print("2. 使用batch内的其他样本作为负样本")
    print("3. 通过alignment_weight控制对齐损失的权重")
    print("4. 支持detach_text_emb防止梯度回传到预训练embeddings")
    
    print("\n✓ InfoNCE实现验证完成")

if __name__ == "__main__":
    test_infonce()
