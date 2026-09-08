#!/bin/bash
set -euo pipefail

# Fair five-fold comparison on the same dataset/support/seed.
# Usage: bash run_spen_5fold.sh [organ] [modality] [label_set] [pipeline_mode]
# pipeline_mode: none (full ProtoSAM) or coarse_only (coarse matcher only)

ORGAN=${1:-liver}
MODALITY=${2:-mri}
LABEL_SET=${3:-1}
PIPELINE_MODE=${4:-none}
LOG_ROOT="fivefold_logs/${MODALITY}_${ORGAN}_${PIPELINE_MODE}"
mkdir -p "$LOG_ROOT"

# Isolate QLPE from ALPG: grid+QLPE tests OT on the baseline prototypes, while
# ALPG-multi shows the cost/benefit of replacing those prototypes.
for MATCHER in baseline spen_qlpe_multi spen_alpg_multi
do
    echo "===== ${MATCHER}: ${MODALITY}/${ORGAN}, five folds, pipeline=${PIPELINE_MODE} ====="
    bash run_ablation.sh \
        "$MATCHER" "$ORGAN" "$MODALITY" "$LABEL_SET" "" "" "$PIPELINE_MODE" \
        2>&1 | tee "$LOG_ROOT/${MATCHER}.log"
done

python3 summarize_spen_5fold.py \
    --root "test_${MODALITY}" \
    --organ "$ORGAN" \
    --pipeline-mode "$PIPELINE_MODE" \
    --output "$LOG_ROOT/summary.csv"
