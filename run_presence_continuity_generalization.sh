#!/bin/bash
set -euo pipefail

# Validate the frozen support-conditioned presence pipeline on additional
# anatomical targets before integrating the gate into ProtoSAM/SAM.
#
# Usage:
#   ./run_presence_continuity_generalization.sh screen all
#   ./run_presence_continuity_generalization.sh full mri
#   ./run_presence_continuity_generalization.sh full ct
#   ./run_presence_continuity_generalization.sh full all
#
# Environment overrides:
#   MATCHER=dense_fg_hard_attn   frozen coarse matcher
#   TARGET_SENSITIVITY=0.975     calibration sensitivity used by LOVO
#   ABSOLUTE_MIN=0.1             all-absent volume fail-safe
#   RERUN_INFERENCE=missing      missing: reuse existing and run only absent folds
#                                1: always rerun; 0: never run inference
#   OUTPUT_ROOT=...              evaluation output directory

STAGE=${1:-screen}
SCOPE=${2:-all}
MATCHER=${MATCHER:-dense_fg_hard_attn}
TARGET_SENSITIVITY=${TARGET_SENSITIVITY:-0.975}
ABSOLUTE_MIN=${ABSOLUTE_MIN:-0.1}
RERUN_INFERENCE=${RERUN_INFERENCE:-missing}
OUTPUT_ROOT=${OUTPUT_ROOT:-presence_continuity_generalization}

case "$STAGE" in
    screen) FOLDS=(0 2) ;;
    full) FOLDS=(0 1 2 3 4) ;;
    *) echo "stage must be screen or full"; exit 1 ;;
esac

if [ "$SCOPE" != "all" ] && [ "$SCOPE" != "mri" ] && [ "$SCOPE" != "ct" ]; then
    echo "scope must be all, mri, or ct"
    exit 1
fi

latest_presence_csv () {
    local experiment_dir=$1
    if [ ! -d "$experiment_dir" ]; then
        return 1
    fi
    local latest
    latest=$(find "$experiment_dir" -type f -name presence_scores.csv -printf '%T@ %p\n' \
        | sort -nr \
        | sed -n '1{s/^[^ ]* //;p;}')
    if [ -z "$latest" ]; then
        return 1
    fi
    printf '%s\n' "$latest"
}

run_task () {
    local modality=$1
    local organ=$2
    local label_set=$3
    local dataset
    if [ "$modality" = "mri" ]; then
        dataset="CHAOST2"
    else
        dataset="SABS"
    fi

    for fold in "${FOLDS[@]}"; do
        echo
        echo "=== ${dataset} ${organ} fold ${fold}: ${MATCHER} + LR + P4 ==="
        local experiment_dir="test_${modality}/dinov2_l14_${modality}_cca_grid_8_res_672_${organ}_${MATCHER}_presence_score_fold_${fold}"
        local existing_csv=""
        existing_csv=$(latest_presence_csv "$experiment_dir") || true
        if [ "$RERUN_INFERENCE" = "1" ] || { [ "$RERUN_INFERENCE" = "missing" ] && [ -z "$existing_csv" ]; }; then
            ./run_ablation.sh \
                "$MATCHER" "$organ" "$modality" "$label_set" "" "$fold" presence_score
        fi

        local csv_path
        csv_path=$(latest_presence_csv "$experiment_dir") || {
            echo "No presence_scores.csv found below: $experiment_dir"
            echo "Run with RERUN_INFERENCE=1 or check the experiment name."
            exit 1
        }

        local output_dir="${OUTPUT_ROOT}/${modality}/${organ}/fold${fold}"
        python3 evaluate_presence_z_continuity.py \
            "$csv_path" \
            --score fg_bg_likelihood_ratio \
            --target-sensitivity "$TARGET_SENSITIVITY" \
            --absolute-min "$ABSOLUTE_MIN" \
            --output-dir "$output_dir"
    done
}

# CHAOST2 liver has already been validated.  The MRI generalization stage
# intentionally covers the three remaining organs.
if [ "$SCOPE" = "all" ] || [ "$SCOPE" = "mri" ]; then
    run_task mri spleen 1
    run_task mri rk 0
    run_task mri lk 0
fi

# Validate every target on the modality shift from MRI to CT.
if [ "$SCOPE" = "all" ] || [ "$SCOPE" = "ct" ]; then
    run_task ct liver 1
    run_task ct spleen 1
    run_task ct rk 0
    run_task ct lk 0
fi

python3 summarize_presence_continuity_generalization.py \
    "$OUTPUT_ROOT" \
    --expected-folds "${#FOLDS[@]}"
