#!/bin/bash
# make_beauty_10p.sh - Create a 10% user-level subset for Amazon_Beauty (or another dataset)

set -e

# Resolve project root
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

# ---------------------- Configurable parameters ----------------------
# Dataset name (folder name under dataset/, and file prefix). Examples:
#   Amazon_Beauty, Amazon_Beauty-example
DATASET=${DATASET:-Amazon_Beauty}

# Sample ratio (fraction of users)
RATIO=${RATIO:-0.1}

# Random seed for sampling
SEED=${SEED:-42}

# Min interactions per user to be eligible
MIN_INTERACTIONS=${MIN_INTERACTIONS:-3}

# Item filter mode: 'used' (keep only items appearing in sampled inter) or 'all'
ITEM_FILTER=${ITEM_FILTER:-used}

# Output dataset name and directory
OUTPUT_NAME=${OUTPUT_NAME:-${DATASET}_10p}
OUTPUT_DIR=${OUTPUT_DIR:-dataset}

# Whether to prepare base (TF-IDF+SVD) item embeddings for the subset
PREPARE_BASE=${PREPARE_BASE:-true}

# Config used when generating mapping/embeddings
CONFIG=${CONFIG:-sasrec_base_plain.yaml}

# --------------------------- Run sampling ----------------------------
echo "=== Create ${RATIO} user subset for ${DATASET} ==="
python tools/subsample_users.py \
  --input_dir "${OUTPUT_DIR}/${DATASET}" \
  --dataset_name "${DATASET}" \
  --sample_ratio "${RATIO}" \
  --seed "${SEED}" \
  --min_interactions "${MIN_INTERACTIONS}" \
  --item_filter "${ITEM_FILTER}" \
  --output_dir "${OUTPUT_DIR}" \
  --output_name "${OUTPUT_NAME}" \
  --overwrite

echo "✓ Subset created at ${OUTPUT_DIR}/${OUTPUT_NAME}"

# --------------------- Prepare base embeddings (opt) -----------------
if [ "$PREPARE_BASE" = "true" ]; then
  echo ""
  echo "=== Generate item_index_mapping for ${OUTPUT_NAME} ==="
  python tools/export_internal_item_mapping.py \
    --dataset "${OUTPUT_NAME}" \
    --config "${CONFIG}" \
    --output "${OUTPUT_DIR}/${OUTPUT_NAME}/item_index_mapping.csv"

  echo ""
  echo "=== Generate Base (TF-IDF+SVD) embeddings for ${OUTPUT_NAME} ==="
  python tools/build_item_text_emb_base.py \
    --dataset "${OUTPUT_NAME}" \
    --config "${CONFIG}" \
    --output "${OUTPUT_DIR}/${OUTPUT_NAME}/item_text_emb.base.npy" \
    --svd_dim 256 \
    --ngram_min 1 \
    --ngram_max 2 \
    --dtype float16
  echo "✓ Base embeddings ready: ${OUTPUT_DIR}/${OUTPUT_NAME}/item_text_emb.base.npy"
fi

echo ""
echo "=== Next steps ==="
echo "Run baseline on subset:"
echo "  python run_recbole.py --model SASRec --dataset ${OUTPUT_NAME} --config sasrec_baseline.yaml"


