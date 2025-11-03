#!/bin/bash
# 01_prepare_embeddings.sh - 准备item映射和Base embeddings

BASE_DIR="/home/charlie/project/RecBole"
cd $BASE_DIR

echo "=== Step 1: 生成item索引映射文件 ==="
python tools/export_internal_item_mapping.py \
  --dataset Amazon_Beauty \
  --config recbole/properties/model/SASRecAlign.yaml \
  --output dataset/Amazon_Beauty/item_index_mapping.csv

if [ $? -eq 0 ]; then
    echo "✓ Item索引映射文件生成成功"
else
    echo "✗ Item索引映射文件生成失败"
    exit 1
fi

echo ""
echo "=== Step 2: 生成Base (TF-IDF+SVD) embeddings ==="
python tools/build_item_text_emb_base.py \
  --dataset Amazon_Beauty \
  --config recbole/properties/model/SASRecAlign.yaml \
  --output dataset/Amazon_Beauty/item_text_emb.base.npy \
  --svd_dim 256 \
  --ngram_min 1 \
  --ngram_max 2 \
  --dtype float16

if [ $? -eq 0 ]; then
    echo "✓ Base embeddings生成成功"
    echo "文件位置: dataset/Amazon_Beauty/item_text_emb.base.npy"
else
    echo "✗ Base embeddings生成失败"
    exit 1
fi

echo ""
echo "=== 准备工作完成 ==="
echo "请确保Qwen3 embeddings已在GPU机器上生成完成"
echo "期望文件: dataset/Amazon_Beauty/item_text_emb.qwen3.npy"
