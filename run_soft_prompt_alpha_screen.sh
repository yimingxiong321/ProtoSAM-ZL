#!/bin/bash
set -e

# Fast two-fold screen.  Fold 0 previously showed a large gain, while fold 2
# was approximately tied, so together they expose both favourable and hard cases.
FOLDS=(0 2)
PROMPT_MODES=(soft_prompt_a025 soft_prompt_a05 soft_prompt_a1)
MATCHERS=(baseline dense_fg_hard_attn)

for prompt_mode in "${PROMPT_MODES[@]}"; do
    for fold in "${FOLDS[@]}"; do
        for matcher in "${MATCHERS[@]}"; do
            ./run_ablation.sh "$matcher" liver mri 1 "" "$fold" "$prompt_mode"
        done
    done

    python3 summarize_spen_5fold.py \
        --root test_mri \
        --organ liver \
        --pipeline-mode "$prompt_mode" \
        --output "fivefold_logs/mri_liver_${prompt_mode}_screen/summary.csv"
done
