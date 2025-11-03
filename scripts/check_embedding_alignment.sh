#!/bin/bash
# 使用现有的 export_internal_item_mapping.py 工具检查 embedding 对齐

BASE_DIR="/home/charlie/project/RecBole"
cd $BASE_DIR

echo "=== 检查 Embedding 对齐情况 ==="
echo ""

# 检查 Base embeddings
echo "1. 检查 Base embeddings:"
python tools/export_internal_item_mapping.py \
  --dataset Amazon_Beauty \
  --output dataset/Amazon_Beauty/item_index_mapping_check.csv \
  --check_emb dataset/Amazon_Beauty/item_text_emb.base.npy

echo ""
echo "2. 检查 Qwen3 embeddings:"
python tools/export_internal_item_mapping.py \
  --dataset Amazon_Beauty \
  --output dataset/Amazon_Beauty/item_index_mapping_check.csv \
  --check_emb dataset/Amazon_Beauty/item_text_emb.qwen3.npy

# 清理临时文件
rm -f dataset/Amazon_Beauty/item_index_mapping_check.csv

echo ""
echo "=== 分析 ==="
echo "如果看到 MISMATCH，说明 embedding 文件的维度与当前数据集不匹配"
echo "需要重新生成 embedding 文件"
