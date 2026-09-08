#!/bin/bash
set -e

# Fast P0/P1/P2 screen on the same MRI-liver folds used by earlier matcher
# screens. These are development results; run all five folds only after fixing
# hyperparameters.
for FOLD in 0 2; do
    ./run_ablation.sh baseline liver mri 1 "" "$FOLD" coarse_only
    ./run_ablation.sh dense_fg_hard liver mri 1 "" "$FOLD" coarse_only
    ./run_ablation.sh dense_fg_soft liver mri 1 "" "$FOLD" coarse_only
done

python3 summarize_spen_5fold.py \
    --root test_mri \
    --organ liver \
    --pipeline-mode coarse_only \
    --output fivefold_logs/mri_liver_coarse_only/dense_screen/summary.csv
