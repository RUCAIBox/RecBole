#!/bin/bash
# 检查文本embeddings文件是否存在

BASE_DIR="/home/charlie/project/RecBole"
cd $BASE_DIR

echo "=== 检查文本Embeddings文件 ==="
echo "当前目录: $(pwd)"
echo ""

# 检查base embeddings
BASE_FILE="dataset/Amazon_Beauty/item_text_emb.base.npy"
if [ -f "$BASE_FILE" ]; then
    echo "✓ Base embeddings 存在: $BASE_FILE"
    python -c "import numpy as np; e=np.load('$BASE_FILE'); print(f'  形状: {e.shape}, 类型: {e.dtype}')"
else
    echo "✗ Base embeddings 不存在: $BASE_FILE"
    echo "  需要运行: bash scripts/01_prepare_embeddings.sh"
fi

echo ""

# 检查Qwen3 embeddings
QWEN3_FILE="dataset/Amazon_Beauty/item_text_emb.qwen3.npy"
if [ -f "$QWEN3_FILE" ]; then
    echo "✓ Qwen3 embeddings 存在: $QWEN3_FILE"
    python -c "import numpy as np; e=np.load('$QWEN3_FILE'); print(f'  形状: {e.shape}, 类型: {e.dtype}')"
else
    echo "⚠ Qwen3 embeddings 不存在: $QWEN3_FILE"
    echo "  需要在GPU机器上运行生成脚本"
fi

echo ""

# 检查映射文件
MAPPING_FILE="dataset/Amazon_Beauty/item_index_mapping.csv"
if [ -f "$MAPPING_FILE" ]; then
    echo "✓ 映射文件存在: $MAPPING_FILE"
    head -5 $MAPPING_FILE
else
    echo "✗ 映射文件不存在: $MAPPING_FILE"
fi
