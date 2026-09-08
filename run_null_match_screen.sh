#!/bin/bash
set -e

# P1: isolate background-aware rejection before adding spatial/MNN modules.
# Fold 0 is the favourable Dense case; fold 2 is the near-tie hard case.
FOLDS=(0 2)
PIPELINES=(coarse_only soft_prompt_a05)
MATCHER=dense_fg_hard_attn_null

for pipeline_mode in "${PIPELINES[@]}"; do
    for fold in "${FOLDS[@]}"; do
        ./run_ablation.sh "$MATCHER" liver mri 1 "" "$fold" "$pipeline_mode"
    done

    python3 summarize_spen_5fold.py \
        --root test_mri \
        --organ liver \
        --pipeline-mode "$pipeline_mode" \
        --output "fivefold_logs/mri_liver_${pipeline_mode}_null_screen/summary.csv"
done
