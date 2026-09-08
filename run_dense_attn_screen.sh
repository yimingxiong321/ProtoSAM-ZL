#!/bin/bash
set -e

# Test whether the normalized-LSE mass penalty caused the dense matcher to
# under-segment. Fold 0 is the known failure case; fold 2 checks that its small
# positive result is retained. Baseline is rerun so all boundary metrics exist.
for FOLD in 0 2; do
    ./run_ablation.sh baseline liver mri 1 "" "$FOLD" coarse_only
    ./run_ablation.sh dense_fg_hard_attn liver mri 1 "" "$FOLD" coarse_only
    ./run_ablation.sh dense_fg_soft_attn liver mri 1 "" "$FOLD" coarse_only
done

python3 summarize_spen_5fold.py \
    --root test_mri \
    --organ liver \
    --pipeline-mode coarse_only \
    --output fivefold_logs/mri_liver_coarse_only/dense_attn_screen/summary.csv
