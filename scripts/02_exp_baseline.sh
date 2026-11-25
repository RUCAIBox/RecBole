#!/bin/bash
# 02_exp_baseline.sh - 运行基线SASRec实验

BASE_DIR="/home/charlie/project/RecBole"
cd $BASE_DIR

# 创建结果目录
mkdir -p results/sasrec_experiments
LOG_DIR="results/sasrec_experiments"

echo "=== 实验1: Baseline SASRec (无文本特征) ==="
echo "开始时间: $(date)"

python run_recbole.py \
    --model SASRec \
    --dataset Amazon_Beauty \
    --config_dict "hidden_size=256,eval_args.mode.test='full'" \
    > ${LOG_DIR}/exp1_baseline_sasrec.log 2>&1

if [ $? -eq 0 ]; then
    echo "✓ 基线实验完成"
    echo "日志文件: ${LOG_DIR}/exp1_baseline_sasrec.log"
    
    # 提取关键指标
    echo ""
    echo "测试结果预览:"
    grep -A 5 "test result" ${LOG_DIR}/exp1_baseline_sasrec.log | tail -6
else
    echo "✗ 基线实验失败"
    echo "请检查日志: ${LOG_DIR}/exp1_baseline_sasrec.log"
    exit 1
fi

echo ""
echo "结束时间: $(date)"
