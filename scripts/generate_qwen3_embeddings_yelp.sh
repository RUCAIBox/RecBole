#!/bin/bash
# 在GPU机器上为Yelp生成Qwen3 embeddings

BASE_DIR="/home/charlie/project/RecBole"
cd $BASE_DIR

echo "=== 生成Yelp Qwen3 Embeddings ==="
mkdir -p logs

# 前置检查：映射是否匹配Base
python tools/export_internal_item_mapping.py \
  --dataset Yelp \
  --config yelp_sasrec_base_plain.yaml \
  --output dataset/Yelp/item_index_mapping_check.csv \
  --check_emb dataset/Yelp/item_text_emb.base.npy || true

# Qwen3模型路径（按需修改）
QWEN_MODEL_PATH="/home/charlie/project/qwen/Model"

nohup python tools/build_item_text_emb_qwen3_hf.py \
  --mapping dataset/Yelp/item_index_mapping.csv \
  --model_name_or_path ${QWEN_MODEL_PATH} \
  --output dataset/Yelp/item_text_emb.qwen3.npy \
  --batch_size 8 \
  --max_length 128 \
  --dtype float16 \
  --project_dim 256 \
  --dataset Yelp \
  --config yelp_sasrec_base_plain.yaml \
  --prompt_template "[TITLE] {text}" \
  --device_map auto \
  > logs/generate_qwen3_yelp_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "已启动后台生成，日志见 logs/generate_qwen3_yelp_*.log"


