#!/bin/bash
set -e

# Locked P0/P1/P2 five-fold evaluation. Avoid changing temperature or threshold
# after looking at these results.
for MATCHER in baseline dense_fg_hard dense_fg_soft; do
    ./run_ablation.sh "$MATCHER" liver mri 1 "" "" coarse_only
done

python3 summarize_spen_5fold.py \
    --root test_mri \
    --organ liver \
    --pipeline-mode coarse_only \
    --output fivefold_logs/mri_liver_coarse_only/dense_5fold/summary.csv
