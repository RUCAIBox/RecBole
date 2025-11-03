#!/usr/bin/env python3
"""检查item数量不匹配问题"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from recbole.config import Config
from recbole.data.utils import create_dataset

print("=== 检查Item数量不匹配 ===\n")

# 1. 检查embedding文件
emb_path = "dataset/Amazon_Beauty/item_text_emb.base.npy"
if os.path.exists(emb_path):
    emb = np.load(emb_path)
    print(f"Embedding文件: {emb_path}")
    print(f"  形状: {emb.shape}")
    print(f"  Item数: {emb.shape[0]}")

# 2. 检查映射文件
mapping_path = "dataset/Amazon_Beauty/item_index_mapping.csv"
if os.path.exists(mapping_path):
    df = pd.read_csv(mapping_path)
    print(f"\n映射文件: {mapping_path}")
    print(f"  行数: {len(df)}")
    print(f"  最大internal_item_id: {df['internal_item_id'].max()}")

# 3. 检查RecBole数据集
print(f"\nRecBole数据集:")
config = Config(
    model="SASRec",
    dataset="Amazon_Beauty",
    config_dict={"device": "cpu"}
)
dataset = create_dataset(config)
print(f"  n_items: {dataset.num(dataset.iid_field)}")

# 4. 检查原始数据文件
inter_path = "dataset/Amazon_Beauty/Amazon_Beauty.inter"
if os.path.exists(inter_path):
    inter_df = pd.read_csv(inter_path, sep='\t')
    print(f"\n原始交互文件: {inter_path}")
    print(f"  唯一items: {inter_df['item_id'].nunique()}")
    print(f"  总交互数: {len(inter_df)}")

# 5. 建议
print("\n=== 分析 ===")
if emb.shape[0] != dataset.num(dataset.iid_field):
    print("⚠️  Embedding维度与数据集不匹配!")
    print(f"   差异: {dataset.num(dataset.iid_field) - emb.shape[0]} items")
    print("\n可能的原因:")
    print("1. 数据集在生成embedding后被过滤了")
    print("2. 使用了不同版本的数据集")
    print("\n解决方案:")
    print("1. 重新生成embedding文件")
    print("2. 或修改_load_text_embeddings方法，允许部分匹配")
