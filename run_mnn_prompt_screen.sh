#!/bin/bash
set -e

# P2: change only SAM positive-point selection. Coarse logits, bbox, centroid,
# and the alpha=0.5 soft-mask prompt remain unchanged.
FOLDS=(0 2)
MATCHERS=(baseline dense_fg_hard_attn)
PIPELINE_MODE=soft_prompt_a05_mnn

for fold in "${FOLDS[@]}"; do
    for matcher in "${MATCHERS[@]}"; do
        ./run_ablation.sh "$matcher" liver mri 1 "" "$fold" "$PIPELINE_MODE"
    done
done

python3 summarize_spen_5fold.py \
    --root test_mri \
    --organ liver \
    --pipeline-mode "$PIPELINE_MODE" \
    --output fivefold_logs/mri_liver_soft_prompt_a05_mnn_screen/summary.csv
