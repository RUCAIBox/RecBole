#!/bin/bash
# 实验1: Yelp2018 Baseline SASRec (无文本特征)

BASE_DIR="/home/charlie/project/RecBole"
cd $BASE_DIR

EXP_NAME="yelp_exp1_baseline_sasrec"
MODEL="SASRec"
DATASET="yelp2018"
CONFIG_FILE="yelp_sasrec_baseline.yaml"

LOG_DIR="results/yelp_experiments"
mkdir -p $LOG_DIR

echo "=========================================="
echo "实验: $EXP_NAME"
echo "模型: $MODEL"
echo "数据集: $DATASET"
echo "配置文件: $CONFIG_FILE"
echo "开始时间: $(date)"
echo "=========================================="

python run_recbole.py \
    --model $MODEL \
    --dataset $DATASET \
    --config_files $CONFIG_FILE \
    > ${LOG_DIR}/${EXP_NAME}.log 2>&1

if [ $? -eq 0 ]; then
    echo "✓ 实验完成"
    echo ""
    echo "测试结果:"
    grep -A 5 "test result" ${LOG_DIR}/${EXP_NAME}.log | tail -6
else
    echo "✗ 实验失败"
fi

echo ""
echo "日志文件: ${LOG_DIR}/${EXP_NAME}.log"
echo "结束时间: $(date)"


