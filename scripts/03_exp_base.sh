#!/bin/bash
# 03_exp_base.sh - Base embeddings相关实验

BASE_DIR="/home/charlie/project/RecBole"
cd $BASE_DIR

LOG_DIR="results/sasrec_experiments"
mkdir -p $LOG_DIR

echo "=== Base Embeddings实验系列 ==="
echo "开始时间: $(date)"

# 实验2: SASRec + Base (无交叉，无对齐)
echo ""
echo "--- 实验2: SASRec + Base ---"
python run_recbole.py \
    --model SASRecAlign \
    --dataset Amazon_Beauty \
    --config_files sasrec_align_base.yaml \
    --config_dict "use_llm=False,use_cross=False,use_align=False" \
    > ${LOG_DIR}/exp2_sasrec_base.log 2>&1

echo "状态: $([[ $? -eq 0 ]] && echo '✓ 完成' || echo '✗ 失败')"

# 实验3: SASRec + Base + Cross (有交叉，无对齐)
echo ""
echo "--- 实验3: SASRec + Base + Cross ---"
python run_recbole.py \
    --model SASRecAlign \
    --dataset Amazon_Beauty \
    --config_files sasrec_align_base.yaml \
    --config_dict "use_llm=False,use_cross=True,use_align=False" \
    > ${LOG_DIR}/exp3_sasrec_base_cross.log 2>&1

echo "状态: $([[ $? -eq 0 ]] && echo '✓ 完成' || echo '✗ 失败')"

# 实验4: SASRec + Base + Cross + Align (全部启用)
echo ""
echo "--- 实验4: SASRec + Base + Cross + Align ---"
python run_recbole.py \
    --model SASRecAlign \
    --dataset Amazon_Beauty \
    --config_files sasrec_align_base.yaml \
    > ${LOG_DIR}/exp4_sasrec_base_cross_align.log 2>&1

echo "状态: $([[ $? -eq 0 ]] && echo '✓ 完成' || echo '✗ 失败')"

echo ""
echo "=== Base实验系列完成 ==="
echo "结束时间: $(date)"

# 显示所有Base实验的关键结果
echo ""
echo "=== 结果摘要 ==="
for exp in exp2_sasrec_base exp3_sasrec_base_cross exp4_sasrec_base_cross_align; do
    echo ""
    echo "--- ${exp} ---"
    if [ -f "${LOG_DIR}/${exp}.log" ]; then
        grep "test result" ${LOG_DIR}/${exp}.log -A 3 | head -4
    fi
done
