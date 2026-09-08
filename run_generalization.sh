#!/bin/bash
set -e

# Generalization protocol for the current validated ablations.
#
# Usage:
#   ./run_generalization.sh screen all    # coarse-only, folds/episodes 0 and 2
#   ./run_generalization.sh screen mri_other # MRI organs except completed liver
#   ./run_generalization.sh full ct       # one dataset family, all five
#   ./run_generalization.sh full all      # every dataset/configuration
#
# For MRI/CT, fold means the official five-fold split. For Polyp there is no
# anatomical five-fold split, so fold ids denote five deterministic support
# episodes with seeds 4200..4204. Baseline and Dense share the same seed.

STAGE=${1:-screen}
SCOPE=${2:-all}
MATCHERS=(baseline dense_fg_hard_attn)

if [ "$SCOPE" != "all" ] && [ "$SCOPE" != "mri" ] && [ "$SCOPE" != "mri_other" ] && [ "$SCOPE" != "ct" ] && [ "$SCOPE" != "polyp" ]; then
    echo "scope must be all, mri, mri_other, ct, or polyp"
    exit 1
fi

case "$STAGE" in
    screen)
        FOLDS=(0 2)
        PIPELINES=(coarse_only)
        ;;
    full)
        FOLDS=(0 1 2 3 4)
        PIPELINES=(coarse_only soft_prompt_a05)
        ;;
    *)
        echo "stage must be screen or full"
        exit 1
        ;;
esac

run_anatomical_dataset () {
    local modality=$1
    local organ=$2
    local label_set=$3
    for pipeline in "${PIPELINES[@]}"; do
        for fold in "${FOLDS[@]}"; do
            for matcher in "${MATCHERS[@]}"; do
                ./run_ablation.sh \
                    "$matcher" "$organ" "$modality" "$label_set" "" "$fold" "$pipeline"
            done
        done
    done
}

for modality in mri ct; do
    if [ "$SCOPE" == "all" ] || [ "$SCOPE" == "$modality" ] || { [ "$SCOPE" == "mri_other" ] && [ "$modality" == "mri" ]; }; then
        if [ "$SCOPE" != "mri_other" ]; then
            run_anatomical_dataset "$modality" liver 1
        fi
        run_anatomical_dataset "$modality" spleen 1
        run_anatomical_dataset "$modality" rk 0
        run_anatomical_dataset "$modality" lk 0
    fi
done

# Polyp support images are randomly drawn by the loader. Convert the five
# episode ids into deterministic seeds, paired across matcher/pipeline.
if [ "$SCOPE" == "all" ] || [ "$SCOPE" == "polyp" ]; then
    for pipeline in "${PIPELINES[@]}"; do
        for fold in "${FOLDS[@]}"; do
            episode_seed=$((4200 + fold))
            for matcher in "${MATCHERS[@]}"; do
                PROTOSAM_SEED=$episode_seed ./run_ablation.sh \
                    "$matcher" spleen polyp 1 "" "$fold" "$pipeline"
            done
        done
    done
fi

python3 summarize_generalization.py \
    --roots test_mri test_ct test_polyp \
    --output-dir "generalization_logs/${STAGE}_${SCOPE}"
