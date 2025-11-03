#!/bin/bash
# 00_quick_test.sh - 快速测试环境和配置是否正确

BASE_DIR="/home/charlie/project/RecBole"
cd $BASE_DIR

echo "=== RecBole SASRec 环境测试 ==="
echo "当前目录: $(pwd)"
echo "时间: $(date)"
echo ""

# 检查Python环境
echo "--- Python环境检查 ---"
python --version
echo ""

# 检查必要的文件
echo "--- 文件检查 ---"
files_to_check=(
    "run_recbole.py"
    "recbole/model/sequential_recommender/sasrec.py"
    "recbole/model/sequential_recommender/sasrec_align.py"
    "sasrec_align_base.yaml"
    "sasrec_align_qwen3.yaml"
    "dataset/Amazon_Beauty/Amazon_Beauty.inter"
)

all_good=true
for file in "${files_to_check[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file (缺失)"
        all_good=false
    fi
done

if [ "$all_good" = false ]; then
    echo ""
    echo "错误: 部分必要文件缺失，请检查路径"
    exit 1
fi

# 检查embeddings
echo ""
echo "--- Embeddings文件检查 ---"
if [ -f "dataset/Amazon_Beauty/item_text_emb.base.npy" ]; then
    echo "✓ Base embeddings 存在"
    python -c "import numpy as np; e=np.load('dataset/Amazon_Beauty/item_text_emb.base.npy'); print(f'  Shape: {e.shape}, Dtype: {e.dtype}')"
else
    echo "✗ Base embeddings 缺失"
fi

if [ -f "dataset/Amazon_Beauty/item_text_emb.qwen3.npy" ]; then
    echo "✓ Qwen3 embeddings 存在"
    python -c "import numpy as np; e=np.load('dataset/Amazon_Beauty/item_text_emb.qwen3.npy'); print(f'  Shape: {e.shape}, Dtype: {e.dtype}')"
else
    echo "⚠ Qwen3 embeddings 缺失 (LLM实验需要)"
fi

# 运行最小测试
echo ""
echo "--- 运行最小训练测试 ---"
echo "使用少量epoch测试配置是否正确..."

mkdir -p results/test

python run_recbole.py \
    --model SASRec \
    --dataset Amazon_Beauty \
    --config_dict "epochs=1,eval_step=1,stopping_step=1,hidden_size=64" \
    > results/test/quick_test.log 2>&1

if [ $? -eq 0 ]; then
    echo "✓ 基础SASRec测试通过"
    
    # 测试SASRecAlign
    echo ""
    echo "测试SASRecAlign模型..."
    python run_recbole.py \
        --model SASRecAlign \
        --dataset Amazon_Beauty \
        --config_files sasrec_align_base.yaml \
        --config_dict "epochs=1,eval_step=1,stopping_step=1" \
        > results/test/quick_test_align.log 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✓ SASRecAlign测试通过"
        echo ""
        echo "=== 所有测试通过 ✓ ==="
        echo "环境配置正确，可以开始正式实验"
    else
        echo "✗ SASRecAlign测试失败"
        echo "查看日志: results/test/quick_test_align.log"
        tail -20 results/test/quick_test_align.log
    fi
else
    echo "✗ 基础测试失败"
    echo "查看日志: results/test/quick_test.log"
    tail -20 results/test/quick_test.log
fi
