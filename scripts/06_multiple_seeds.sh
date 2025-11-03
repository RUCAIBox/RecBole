#!/bin/bash
# 06_multiple_seeds.sh - 多随机种子实验用于计算方差

BASE_DIR="/home/charlie/project/RecBole"
cd $BASE_DIR

LOG_DIR="results/sasrec_experiments/seeds"
mkdir -p $LOG_DIR

echo "=== 多随机种子实验 ==="
echo "开始时间: $(date)"

# 定义种子列表
SEEDS=(2020 2021 2022 2023 2024)
echo "随机种子: ${SEEDS[@]}"

# 选择最佳配置进行多种子验证
# 这里使用 LLM + Cross + Align 配置
CONFIG="sasrec_align_qwen3.yaml"
echo "使用配置: ${CONFIG}"
echo ""

# 运行实验
results=()
for seed in "${SEEDS[@]}"; do
    echo "--- 种子 ${seed} ---"
    
    python run_recbole.py \
        --model SASRecAlign \
        --dataset Amazon_Beauty \
        --config_files ${CONFIG} \
        --config_dict "seed=${seed}" \
        > ${LOG_DIR}/seed_${seed}.log 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✓ 完成"
        
        # 提取MRR@10指标
        mrr=$(grep -oP 'MRR@10\s*:\s*\K[\d.]+' ${LOG_DIR}/seed_${seed}.log | head -1)
        if [ ! -z "$mrr" ]; then
            results+=($mrr)
            echo "MRR@10: $mrr"
        fi
    else
        echo "✗ 失败"
    fi
    echo ""
done

echo "=== 实验完成 ==="
echo "结束时间: $(date)"

# 计算统计信息
if [ ${#results[@]} -gt 0 ]; then
    echo ""
    echo "=== 统计结果 ==="
    python -c "
import numpy as np

results = [${results[@]}]
mean = np.mean(results)
std = np.std(results)
ci95 = 1.96 * std / np.sqrt(len(results))

print(f'样本数: {len(results)}')
print(f'平均值: {mean:.4f}')
print(f'标准差: {std:.4f}')
print(f'95% 置信区间: {mean:.4f} ± {ci95:.4f}')
print(f'范围: [{mean-ci95:.4f}, {mean+ci95:.4f}]')
"
fi
