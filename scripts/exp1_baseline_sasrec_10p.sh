#!/bin/bash
# 实验1: Baseline SASRec (Beauty 10%)

BASE_DIR="/Users/lvchao0428/project/ownRecBole/RecBole"
cd $BASE_DIR

# 准备 10% Beauty 数据集
echo "准备 Beauty 10% 子集..."
bash scripts/make_beauty_10p.sh
if [ $? -ne 0 ]; then
    echo "✗ 数据集准备失败"
    exit 1
fi

# 实验配置
EXP_NAME="exp1_baseline_sasrec_beauty_10p"
MODEL="SASRec"
CONFIG_FILES="sasrec_baseline.yaml,sasrec_baseline_beauty_10p.yaml"
DATASET_NAME="Amazon_Beauty_10p"

# 输出目录
LOG_DIR="results/sasrec_experiments"
mkdir -p $LOG_DIR

echo "=========================================="
echo "实验: $EXP_NAME"
echo "模型: $MODEL"
echo "数据集: $DATASET_NAME"
echo "配置文件: $CONFIG_FILES"
echo "开始时间: $(date)"
echo "=========================================="

# 运行实验
python run_recbole.py \
    --model $MODEL \
    --config_files $CONFIG_FILES \
    > ${LOG_DIR}/${EXP_NAME}.log 2>&1

# 检查结果
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



