#!/bin/bash
# 快速Ablation：在 SASRecAlign 上扫 text_weight / 对齐 / 尾部门控 / cross / dual_path
#
# 用法：
#   bash scripts/ablation_dual_path.sh
# 可修改下方 DATASET / CONFIG / 搜索空间。

set -e

# 仓库根目录
BASE_DIR="$(cd "$(dirname "$0")/.."; pwd)"
cd "$BASE_DIR"

EXP_NAME_PREFIX="ablation_dual_path"
MODEL="SASRec_Align"
DATASET="${DATASET:-Amazon_Beauty}"
CONFIG_FILE="${CONFIG_FILE:-sasrec_base_plain.yaml}"
LOG_DIR="results/${EXP_NAME_PREFIX}"
mkdir -p "$LOG_DIR"

# 搜索空间（可按需缩小）
TEXT_WEIGHTS=(0.1 0.3 1.0)
ALIGN_ON=(false true)
TAIL_THRESHOLDS=(0 10)
USE_CROSS=(false true)
DUAL_PATH=(true)

# 其他固定设置（如需要可以注释掉或改为阵列）
ALIGN_WEIGHT=0.2
TEMPERATURE=0.07

total=0
for tw in "${TEXT_WEIGHTS[@]}"; do
  for align in "${ALIGN_ON[@]}"; do
    for tt in "${TAIL_THRESHOLDS[@]}"; do
      for cross in "${USE_CROSS[@]}"; do
        for dual in "${DUAL_PATH[@]}"; do
          total=$((total+1))
          tag="${EXP_NAME_PREFIX}_tw${tw}_al${align}_tt${tt}_cr${cross}_dp${dual}"
          echo "[$total] ${tag}"

          # 若不开启对齐，将权重置0以避免无效损失
          aw="$ALIGN_WEIGHT"
          if [ "$align" = "false" ]; then
            aw="0.0"
          fi

          # 运行
          python run_recbole.py \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --config_files "$CONFIG_FILE" \
            --config_dict "use_cross=${cross},use_align=${align},text_tail_threshold=${tt},text_weight=${tw},alignment_weight=${aw},temperature=${TEMPERATURE},dual_path_scoring=${dual},text_score_weight=1.0,score_temperature=1.0" \
            > "${LOG_DIR}/${tag}.log" 2>&1 || true

          # 提取 best valid 与 test 简要行
          echo "---- ${tag} (tail) ----" | tee -a "${LOG_DIR}/SUMMARY.txt"
          grep -E "best valid|test result" -n "${LOG_DIR}/${tag}.log" | tail -2 | tee -a "${LOG_DIR}/SUMMARY.txt"
          echo "" | tee -a "${LOG_DIR}/SUMMARY.txt"
        done
      done
    done
  done
done

echo "完成，共 ${total} 次实验。汇总见：${LOG_DIR}/SUMMARY.txt"


