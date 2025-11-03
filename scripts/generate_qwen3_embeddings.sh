#!/bin/bash
# generate_qwen3_embeddings.sh - 在GPU机器上生成Qwen3 embeddings
# 这个脚本应该在有GPU的机器上运行

BASE_DIR="/home/charlie/project/RecBole"
cd $BASE_DIR

echo "=== 生成Qwen3 Embeddings ==="
echo "开始时间: $(date)"
echo "注意: 此脚本需要在GPU机器上运行"
echo ""

# 检查GPU
echo "--- GPU检查 ---"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv
    echo ""
else
    echo "警告: 未检测到nvidia-smi，可能没有GPU"
    echo "继续执行可能会很慢..."
    echo ""
fi

# 检查必要文件
echo "--- 检查输入文件 ---"
if [ ! -f "dataset/Amazon_Beauty/item_index_mapping.csv" ]; then
    echo "错误: 找不到item_index_mapping.csv"
    echo "请先运行: bash scripts/01_prepare_embeddings.sh"
    exit 1
fi
echo "✓ 输入文件存在"

# 设置Qwen模型路径
QWEN_MODEL_PATH="/home/charlie/project/qwen/Model"
echo ""
echo "--- 检查Qwen模型 ---"
echo "模型路径: $QWEN_MODEL_PATH"

if [ ! -d "$QWEN_MODEL_PATH" ]; then
    echo "错误: 找不到Qwen模型目录"
    echo "请确保已下载Qwen模型到: $QWEN_MODEL_PATH"
    exit 1
fi

# 运行生成脚本
echo ""
echo "--- 开始生成Embeddings ---"
echo "这可能需要一些时间，取决于数据集大小和GPU性能"

nohup python tools/build_item_text_emb_qwen3_hf.py \
  --mapping dataset/Amazon_Beauty/item_index_mapping.csv \
  --model_name_or_path ${QWEN_MODEL_PATH} \
  --output dataset/Amazon_Beauty/item_text_emb.qwen3.npy \
  --batch_size 8 \
  --max_length 128 \
  --dtype float16 \
  --project_dim 256 \
  --dataset Amazon_Beauty \
  --config recbole/properties/model/SASRecAlign.yaml \
  --prompt_template "[TITLE] {text}" \
  --device_map auto \
  > logs/generate_qwen3_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID=$!
echo "进程ID: $PID"
echo "日志文件: logs/generate_qwen3_$(date +%Y%m%d_%H%M%S).log"

# 等待几秒检查是否正常启动
sleep 5

if ps -p $PID > /dev/null; then
    echo ""
    echo "✓ 生成进程已启动"
    echo ""
    echo "提示:"
    echo "- 使用 'tail -f logs/generate_qwen3_*.log' 查看进度"
    echo "- 使用 'ps -p $PID' 检查进程状态"
    echo "- 完成后检查: dataset/Amazon_Beauty/item_text_emb.qwen3.npy"
else
    echo ""
    echo "✗ 进程启动失败"
    echo "查看最新日志:"
    tail -20 logs/generate_qwen3_*.log
fi
