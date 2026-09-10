#!/bin/bash
set -e
# Select the physical GPU without editing this runbook.  CUDA remaps the
# selected device to logical cuda:0 inside the Python process.
GPUID1=${PROTOSAM_GPU_ID:-0}
export CUDA_VISIBLE_DEVICES=$GPUID1

# ============================================================
# Ablation experiment runner for ProtoSAM
#
# Usage:
#   ./run_ablation.sh <mode> <organ> <modality> <label_set> [fixes] [fold] [pipeline_mode] [candidate_audit] [audit_control]
#
# Arguments:
#   mode      - baseline, dense_fg_hard, dense_fg_soft,
#               dense_fg_hard_attn, dense_fg_soft_attn,
#               dense_fg_hard_attn_null,
#               spen_qlpe_multi, spen_alpg_multi, spen_full_multi,
#               none, oracle_coarse, oracle_prompt, coarse_only, oracle_mask
#   organ     - rk, lk, liver, spleen
#   modality  - mri, ct, or polyp
#   label_set - 0 (kidneys tested) or 1 (liver+spleen tested)
#   fixes     - optional comma-separated: multimask_score,neg_points,area_filter,weighted_agg
#   fold      - optional single fold number (0-4). If omitted, runs all 5 folds.
#   pipeline_mode - optional ProtoSAM ablation. Relevant values here are none,
#                   coarse_only, presence_score, soft_prompt (legacy), soft_prompt_a025,
#                   soft_prompt_a05, soft_prompt_a05_mnn, and soft_prompt_a1.
#                   This is independent
#                   from the coarse matcher mode.
#
# Examples:
#   # Toy experiment: oracle_prompt, liver, MRI, fold 0 only (~1 min)
#   ./run_ablation.sh oracle_prompt liver mri 1 "" 0
#
#   # Full 5-fold: oracle_coarse on liver MRI
#   ./run_ablation.sh oracle_coarse liver mri 1
#
#   # Baseline with area_filter fix, fold 0 only
#   ./run_ablation.sh baseline liver mri 1 area_filter 0
# ============================================================

MODE=${1:-none}
ORGAN=${2:-liver}
MODALITY=${3:-mri}
LABEL_SET=${4:-1}
FIXES=${5:-""}
SINGLE_FOLD=${6:-""}
PIPELINE_MODE=${7:-none}
CANDIDATE_AUDIT=${8:-False}
AUDIT_CONTROL=${9:-False}

# Configs
MODEL_NAME='dinov2_l14'
PROTOSAM_SAM_VER="sam_h"
INPUT_SIZE=672
PROTO_GRID=8
SEED=${PROTOSAM_SEED:-42}

# Determine which folds to run
if [ -n "$SINGLE_FOLD" ]; then
    ALL_EV=( $SINGLE_FOLD )
    echo ">>> TOY MODE: running fold $SINGLE_FOLD only"
else
    ALL_EV=( 0 1 2 3 4 )
    echo ">>> FULL MODE: running 5-fold cross validation"
fi

if [ "$MODALITY" != "ct" ] && [ "$MODALITY" != "mri" ] && [ "$MODALITY" != "polyp" ]; then
    echo "modality must be ct, mri, or polyp"
    exit 1
fi

if [ "$MODALITY" == "ct" ]; then
    DATASET='SABS_Superpix'
fi
if [ "$MODALITY" == "mri" ]; then
    DATASET='CHAOST2_Superpix'
fi
if [ "$MODALITY" == "polyp" ]; then
    DATASET='polyps'
    # curr_cls is unused by the Polyp loader, but the historical pipeline uses
    # spleen as a harmless placeholder in experiment names/configuration.
    ORGAN='spleen'
fi

if [ "$INPUT_SIZE" -gt 256 ]; then
    DATASET=${DATASET}'_672'
fi

NWORKER=4
LORA=0
RELOAD_PATH="None"
DO_CCA="True"
ALL_SCALE=( "MIDDLE" )

# Build experiment name prefix
CPT="${MODEL_NAME}_${MODALITY}"
if [ "$DO_CCA" = "True" ]; then
    CPT="${CPT}_cca"
fi
CPT="${CPT}_grid_${PROTO_GRID}_res_${INPUT_SIZE}_${ORGAN}"

if [ "$MODE" != "none" ] && [ "$MODE" != "baseline" ]; then
    CPT="${CPT}_${MODE}"
fi
if [ "$PIPELINE_MODE" != "none" ]; then
    CPT="${CPT}_${PIPELINE_MODE}"
fi

if [ -n "$FIXES" ]; then
    FIXES_NAME=$(echo "$FIXES" | tr ',' '_')
    CPT="${CPT}_${FIXES_NAME}"
fi

if [ "$CANDIDATE_AUDIT" == "True" ]; then
    CPT="${CPT}_candidate_audit"
elif [ "$AUDIT_CONTROL" == "True" ]; then
    CPT="${CPT}_candidate_audit_control"
fi

# Polyp: DINO cosine-sim 1-shot (PAPSP). Override with SUPPORT_SELECT_MODE=random.
if [ "$MODALITY" == "polyp" ]; then
    SUPPORT_SELECT_MODE="${SUPPORT_SELECT_MODE:-dino_sim}"
else
    SUPPORT_SELECT_MODE="${SUPPORT_SELECT_MODE:-random}"
fi
if [ "$SUPPORT_SELECT_MODE" != "random" ]; then
    CPT="${CPT}_${SUPPORT_SELECT_MODE}"
fi

CPT="${CPT}_fold"
SUPP_ID='[4]'

# Keep the coarse matcher (grid/SPEN) independent from ProtoSAM ablations.
case "$MODE" in
    spen_*|dense_fg_*)
        CLSNAME="$MODE"
        ABLATION_ARG="ablation_mode=${PIPELINE_MODE}"
        ;;
    baseline|none)
        CLSNAME="grid_proto"
        ABLATION_ARG="ablation_mode=${PIPELINE_MODE}"
        ;;
    *)
        CLSNAME="grid_proto"
        ABLATION_ARG="ablation_mode=${MODE}"
        ;;
esac

# Build ablation_fixes argument for Sacred using bash native string replacement
# a,b,c -> ['a','b','c']
FIXES_ARG=""
if [ -n "$FIXES" ]; then
    FIXED="${FIXES//,/\',\'}"
    FIXES_ARG="ablation_fixes=['${FIXED}']"
fi

# P0/P1 target-presence evaluation must retain target-absent slices.  The
# presence_score mode returns the unchanged coarse mask and writes per-slice
# score evidence; it does not apply a gate or choose a threshold.
PRESENCE_ARGS=()
if [ "$PIPELINE_MODE" == "presence_score" ]; then
    # npart=1 selects the middle positive slice (the existing 1-part protocol)
    # once and uses it for every query slice in the episode.  This removes the
    # query-dependent three-part support selection from presence evaluation.
    PRESENCE_ARGS=("skip_no_organ_slices=False" "task.npart=1")
fi

# P1 is a read-only branch before CCA.  It records every thresholded coarse
# component but leaves DO_CCA, prompt construction, and SAM output unchanged.
CANDIDATE_AUDIT_ARGS=()
if [ "$CANDIDATE_AUDIT" == "True" ]; then
    CANDIDATE_AUDIT_ARGS=(
        "candidate_audit=True"
        "candidate_audit_thresholds=[0.3,0.4,0.5]"
        "record_prediction_manifest=True"
        "skip_no_organ_slices=False"
    )
elif [ "$AUDIT_CONTROL" == "True" ]; then
    # Sham/no-op control: same full slice set and prediction hashing, but no
    # pre-CCA candidate/LR computation.  This isolates candidate_audit itself.
    CANDIDATE_AUDIT_ARGS=(
        "record_prediction_manifest=True"
        "skip_no_organ_slices=False"
    )
fi

echo "==================================="
echo "Ablation: mode=$MODE organ=$ORGAN modality=$MODALITY seed=$SEED fixes=$FIXES folds=${ALL_EV[*]}"
echo "==================================="
if [ -n "$FIXES_ARG" ]; then
    echo "  Sacred arg: $FIXES_ARG"
fi

for ((i=0; i<${#ALL_EV[@]}; i++))
do
    EVAL_FOLD=${ALL_EV[i]}
    CPT_W_FOLD="${CPT}_${EVAL_FOLD}"
    echo $CPT_W_FOLD on GPU $GPUID1
    for SUPERPIX_SCALE in "${ALL_SCALE[@]}"
    do
        PREFIX="test_vfold${EVAL_FOLD}"
        LOGDIR="./test_${MODALITY}/${CPT_W_FOLD}"

        if [ ! -d "$LOGDIR" ]; then
            mkdir -p "$LOGDIR"
        fi

        python3 validation_protosam.py with \
            "modelname=$MODEL_NAME" \
            "base_model=alpnet" \
            "coarse_pred_only=False" \
            "protosam_sam_ver=$PROTOSAM_SAM_VER" \
            "curr_cls=$ORGAN" \
            'usealign=True' \
            'optim_type=sgd' \
            reload_model_path=$RELOAD_PATH \
            num_workers=$NWORKER \
            scan_per_load=-1 \
            'use_wce=True' \
            exp_prefix=$PREFIX \
            "clsname=$CLSNAME" \
            eval_fold=$EVAL_FOLD \
            dataset=$DATASET \
            label_sets=$LABEL_SET \
            proto_grid_size=$PROTO_GRID \
            min_fg_data=1 seed=$SEED \
            save_snapshot_every=25000 \
            superpix_scale=$SUPERPIX_SCALE \
            path.log_dir=$LOGDIR \
            support_idx=$SUPP_ID \
            lora=$LORA \
            do_cca=$DO_CCA \
            "support_select_mode=$SUPPORT_SELECT_MODE" \
            "input_size=($INPUT_SIZE, $INPUT_SIZE)" \
            "$ABLATION_ARG" \
            "${PRESENCE_ARGS[@]}" \
            "${CANDIDATE_AUDIT_ARGS[@]}" \
            $FIXES_ARG
    done
done
