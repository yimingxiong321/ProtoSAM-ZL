#!/bin/bash
set -euo pipefail

# Fast coarse-only screening before committing to all five folds.
# Usage: bash run_spen_multiproto_screen.sh [organ] [modality] [label_set]

ORGAN=${1:-liver}
MODALITY=${2:-mri}
LABEL_SET=${3:-1}
LOG_ROOT="fivefold_logs/${MODALITY}_${ORGAN}_coarse_only/multiproto_screen"
mkdir -p "$LOG_ROOT"

# Fold 0 was the hardest SPEN fold; fold 2 was the only previous win.
for FOLD in 0 2
do
    for MATCHER in baseline spen_qlpe_multi spen_alpg_multi
    do
        echo "===== ${MATCHER}: fold ${FOLD}, coarse_only ====="
        bash run_ablation.sh \
            "$MATCHER" "$ORGAN" "$MODALITY" "$LABEL_SET" "" "$FOLD" coarse_only \
            2>&1 | tee "$LOG_ROOT/${MATCHER}_fold${FOLD}.log"
    done
done

python3 summarize_spen_5fold.py \
    --root "test_${MODALITY}" \
    --organ "$ORGAN" \
    --pipeline-mode coarse_only \
    --output "$LOG_ROOT/summary.csv"
