#!/bin/bash
set -e

# Cheapest causal validity check: fold 0 twice, with identical full-slice
# protocol.  The only delta is candidate_audit=False -> True.
./run_ablation.sh \
    dense_fg_hard_attn liver mri 1 "" 0 soft_prompt_a05 False True
./run_ablation.sh \
    dense_fg_hard_attn liver mri 1 "" 0 soft_prompt_a05 True False

mapfile -t CONTROL < <(find test_mri -path '*soft_prompt_a05_candidate_audit_control*' -name prediction_manifest.csv | sort)
mapfile -t AUDIT < <(find test_mri -path '*soft_prompt_a05_candidate_audit_fold_0*' -name prediction_manifest.csv | sort)
if [ "${#CONTROL[@]}" -ne 1 ] || [ "${#AUDIT[@]}" -ne 1 ]; then
    echo "expected one control and one audit prediction manifest" >&2
    exit 1
fi
python3 compare_prediction_manifests.py "${CONTROL[0]}" "${AUDIT[0]}"
