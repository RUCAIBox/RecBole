#!/bin/bash
# 实验0: SASRecAlign 关闭文本分支（等价于原始 SASRec）

BASE_DIR="/home/charlie/project/RecBole"
cd $BASE_DIR

EXP_NAME="exp0_sasrecalign_text_off"
MODEL="SASRec_Align"
DATASET="Amazon_Beauty"
CONFIG_FILES="sasrec_align_text_off.yaml"

LOG_DIR="results/sasrec_experiments"
mkdir -p $LOG_DIR

echo "=========================================="
echo "实验: $EXP_NAME"
echo "模型: $MODEL"
echo "数据集: $DATASET"
echo "配置文件: $CONFIG_FILES"
echo "开始时间: $(date)"
echo "=========================================="

python run_recbole.py \
    --model $MODEL \
    --dataset $DATASET \
    --config_files $CONFIG_FILES \
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


