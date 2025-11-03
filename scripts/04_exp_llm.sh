#!/bin/bash
# 04_exp_llm.sh - LLM (Qwen3) embeddings相关实验

BASE_DIR="/home/charlie/project/RecBole"
cd $BASE_DIR

LOG_DIR="results/sasrec_experiments"
mkdir -p $LOG_DIR

echo "=== LLM (Qwen3) Embeddings实验系列 ==="
echo "开始时间: $(date)"

# 检查Qwen3 embeddings文件是否存在
if [ ! -f "dataset/Amazon_Beauty/item_text_emb.qwen3.npy" ]; then
    echo "✗ 错误: 找不到Qwen3 embeddings文件"
    echo "期望路径: dataset/Amazon_Beauty/item_text_emb.qwen3.npy"
    echo "请先在GPU机器上生成Qwen3 embeddings"
    exit 1
fi

# 实验5: SASRec + LLM (无交叉，无对齐)
echo ""
echo "--- 实验5: SASRec + LLM ---"
python run_recbole.py \
    --model SASRecAlign \
    --dataset Amazon_Beauty \
    --config_files sasrec_align_qwen3.yaml \
    --config_dict "use_llm=True,use_cross=False,use_align=False" \
    > ${LOG_DIR}/exp5_sasrec_llm.log 2>&1

echo "状态: $([[ $? -eq 0 ]] && echo '✓ 完成' || echo '✗ 失败')"

# 实验6: SASRec + LLM + Cross (有交叉，无对齐)
echo ""
echo "--- 实验6: SASRec + LLM + Cross ---"
python run_recbole.py \
    --model SASRecAlign \
    --dataset Amazon_Beauty \
    --config_files sasrec_align_qwen3.yaml \
    --config_dict "use_llm=True,use_cross=True,use_align=False" \
    > ${LOG_DIR}/exp6_sasrec_llm_cross.log 2>&1

echo "状态: $([[ $? -eq 0 ]] && echo '✓ 完成' || echo '✗ 失败')"

# 实验7: SASRec + LLM + Cross + Align (全部启用)
echo ""
echo "--- 实验7: SASRec + LLM + Cross + Align ---"
python run_recbole.py \
    --model SASRecAlign \
    --dataset Amazon_Beauty \
    --config_files sasrec_align_qwen3.yaml \
    > ${LOG_DIR}/exp7_sasrec_llm_cross_align.log 2>&1

echo "状态: $([[ $? -eq 0 ]] && echo '✓ 完成' || echo '✗ 失败')"

echo ""
echo "=== LLM实验系列完成 ==="
echo "结束时间: $(date)"

# 显示所有LLM实验的关键结果
echo ""
echo "=== 结果摘要 ==="
for exp in exp5_sasrec_llm exp6_sasrec_llm_cross exp7_sasrec_llm_cross_align; do
    echo ""
    echo "--- ${exp} ---"
    if [ -f "${LOG_DIR}/${exp}.log" ]; then
        grep "test result" ${LOG_DIR}/${exp}.log -A 3 | head -4
    fi
done
