#!/bin/bash
# 05_hyperparameter_search.sh - 对齐损失的超参数网格搜索

BASE_DIR="/home/charlie/project/RecBole"
cd $BASE_DIR

LOG_DIR="results/sasrec_experiments/hyperparam"
mkdir -p $LOG_DIR

echo "=== 超参数网格搜索 ==="
echo "开始时间: $(date)"

# 定义参数网格
ALIGN_WEIGHTS=(0.05 0.1 0.2)
TEMPERATURES=(0.05 0.07)

echo "参数网格:"
echo "- alignment_weight: ${ALIGN_WEIGHTS[@]}"
echo "- temperature: ${TEMPERATURES[@]}"
echo ""

# 计数器
total_exp=0
success_exp=0

for align_weight in "${ALIGN_WEIGHTS[@]}"; do
    for temp in "${TEMPERATURES[@]}"; do
        ((total_exp++))
        
        echo "--- 实验 ${total_exp}: alignment_weight=${align_weight}, temperature=${temp} ---"
        
        # Base + Cross + Align 配置
        echo "运行 Base 配置..."
        python run_recbole.py \
            --model SASRec_Align \
            --dataset Amazon_Beauty \
            --config_files sasrec_align_base.yaml \
            --config_dict "alignment_weight=${align_weight},temperature=${temp}" \
            > ${LOG_DIR}/base_aw${align_weight}_t${temp}.log 2>&1
        
        if [ $? -eq 0 ]; then
            echo "✓ Base配置完成"
            ((success_exp++))
        else
            echo "✗ Base配置失败"
        fi
        
        # LLM + Cross + Align 配置  
        echo "运行 LLM 配置..."
        python run_recbole.py \
            --model SASRec_Align \
            --dataset Amazon_Beauty \
            --config_files sasrec_align_qwen3.yaml \
            --config_dict "alignment_weight=${align_weight},temperature=${temp}" \
            > ${LOG_DIR}/llm_aw${align_weight}_t${temp}.log 2>&1
        
        if [ $? -eq 0 ]; then
            echo "✓ LLM配置完成"
            ((success_exp++))
        else
            echo "✗ LLM配置失败"
        fi
        
        echo ""
    done
done

echo "=== 超参数搜索完成 ==="
echo "结束时间: $(date)"
echo "总实验数: ${total_exp}"
echo "成功完成: ${success_exp}"

# 创建结果汇总
echo ""
echo "=== 创建结果汇总 ==="
python -c "
import os
import re

log_dir = '${LOG_DIR}'
results = []

for f in os.listdir(log_dir):
    if f.endswith('.log'):
        with open(os.path.join(log_dir, f), 'r') as file:
            content = file.read()
            match = re.search(r'MRR@10\s*:\s*([\d.]+)', content)
            if match:
                mrr = float(match.group(1))
                results.append((f, mrr))

results.sort(key=lambda x: x[1], reverse=True)

print('Top 5 configurations by MRR@10:')
for i, (config, mrr) in enumerate(results[:5], 1):
    print(f'{i}. {config}: MRR@10 = {mrr:.4f}')
"
