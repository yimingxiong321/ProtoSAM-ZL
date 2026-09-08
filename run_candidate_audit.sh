#!/bin/bash
set -e

# P1 candidate-generation audit on the locked mainline.  The final prediction
# still uses the historical single-component CCA and soft-prompt alpha=0.5.
for FOLD in 0 1 2 3 4; do
    mapfile -t EXISTING < <(find test_mri -path "*soft_prompt_a05_candidate_audit_fold_${FOLD}*" -name proposal_slices.csv 2>/dev/null | sort)
    if [ "${#EXISTING[@]}" -eq 1 ]; then
        echo "reusing completed candidate audit fold ${FOLD}: ${EXISTING[0]}"
    elif [ "${#EXISTING[@]}" -eq 0 ]; then
        ./run_ablation.sh \
            dense_fg_hard_attn liver mri 1 "" "$FOLD" soft_prompt_a05 True
    else
        echo "multiple candidate-audit outputs found for fold ${FOLD}; resolve ambiguity first" >&2
        exit 1
    fi
done

mapfile -t AUDIT_FILES < <(find test_mri -path '*soft_prompt_a05_candidate_audit*' -name proposal_slices.csv | sort)
if [ "${#AUDIT_FILES[@]}" -ne 5 ]; then
    echo "expected 5 fold proposal_slices.csv files, found ${#AUDIT_FILES[@]}" >&2
    exit 1
fi
python3 audit_candidate_proposals.py "${AUDIT_FILES[@]}" \
    --output fivefold_logs/mri_liver_candidate_audit/candidate_audit_summary.csv
