#!/usr/bin/env bash
# run_all.sh — full synthetic-dataset comparison via EndoLMSPEC (Linux / DGX)
#
# Bash port of run_all.ps1. Compares YOUR diffusion synthetic dataset vs ENDO4IE
# by training EndoLMSPEC on each and cross-evaluating on held-out test frames.
# Mode B (size-matched): each dataset uses its OWN frames; ENDO4IE's training set
# is capped to match the diffusion set's size so the result reflects data QUALITY,
# not quantity. Both datasets are split with the same leakage-safe stem-hash, so
# any shared frame always lands in the same split.
#
# Both datasets use the SAME standardized flat layout:
#     <dataset>/{Normal, Overexposed, Underexposed}
# pairing by file stem within each exposure type.
#
# Run from anywhere:  bash dataset_compare/run_all.sh
# Re-running is safe; comment out blocks you've already completed to resume.

set -euo pipefail

# ---------------------------------------------------------------- paths / config
# Resolve script location so the repo paths work regardless of cwd.
DC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # dataset_compare
LMSPEC="$(dirname "$DC")"                            # EndoLMSPEC-main
EDM="$(dirname "$LMSPEC")"                           # edm (repo root)

# >>> EDIT THESE TWO for the DGX (or export them before calling the script) <<<
# Point at the dataset roots you uploaded; each must contain
# Normal/ Overexposed/ Underexposed/ subfolders.
SYN="${SYN:-$EDM/syntethic_images/Endo4IE2_diffusion}"   # diffusion set
ENDO="${ENDO:-$EDM/syntethic_images/Endo4IE}"            # ENDO4IE set

DATA="$DC/compare_data"        # prepared exposure_dataset trees
CKPT="$DC/checkpoints"         # trained models
RES="$DC/results"              # metrics
SUMMARY="$RES/summary.csv"

GPU="${GPU:-0}"                                  # CUDA device index
LW=(0.25 0.25 1.0 0.25 0.0)                      # paper's best loss weights
EPOCHS=(40 30)                                   # [128px, 256px]; e.g. (2 2) for a smoke test

export WANDB_MODE="${WANDB_MODE:-disabled}"      # skip the wandb login prompt during training

echo "config: SYN=$SYN"
echo "        ENDO=$ENDO"
echo "        GPU=$GPU  EPOCHS=${EPOCHS[*]}  LW=${LW[*]}"

train_count() {  # number of training GT frames for a prepared tag
    find "$DATA/$1/training/gt_images" -maxdepth 1 -type f | wc -l
}

# ============================================================ 1. PREPARE DATA
echo -e "\n=== STEP 1: prepare data ==="
cd "$DC"

# prepare YOUR (smaller) datasets first, then cap ENDO4IE training to match.
python prepare_data.py --name mine --exposure over \
    --normal_dir "$SYN/Normal" --exposed_dir "$SYN/Overexposed"  --out_root "$DATA"
python prepare_data.py --name mine --exposure under \
    --normal_dir "$SYN/Normal" --exposed_dir "$SYN/Underexposed" --out_root "$DATA"

CAP_OVER="$(train_count mine_over)"
CAP_UNDER="$(train_count mine_under)"
echo "size-matching ENDO4IE training to mine: over=$CAP_OVER under=$CAP_UNDER"

python prepare_data.py --name endo4ie --exposure over \
    --normal_dir "$ENDO/Normal" --exposed_dir "$ENDO/Overexposed"  --out_root "$DATA" --cap_train "$CAP_OVER"
python prepare_data.py --name endo4ie --exposure under \
    --normal_dir "$ENDO/Normal" --exposed_dir "$ENDO/Underexposed" --out_root "$DATA" --cap_train "$CAP_UNDER"

# ============================================================ 2-3. PATCHES + TRAIN
TAGS=(mine_over endo4ie_over mine_under endo4ie_under)
for tag in "${TAGS[@]}"; do
    tree="$DATA/$tag"
    echo -e "\n=== STEP 2: patches for $tag ==="
    cd "$LMSPEC"
    python patches_extraction.py --exposure_dataset "$tree" --patches_size_pow 7 8

    echo -e "\n=== STEP 3: train $tag ==="
    python main_training.py --exposure_dataset "$tree" \
        --checkpoint_dir "$CKPT/$tag" \
        --GPU "$GPU" --loss_weights "${LW[@]}" --epochs_list "${EPOCHS[@]}"
done

# ============================================================ 4. CROSS-EVALUATE
echo -e "\n=== STEP 4: cross-evaluation ==="
[ -f "$SUMMARY" ] && rm -f "$SUMMARY"
cd "$DC"
for exp in over under; do
    for train in "mine_$exp" "endo4ie_$exp"; do
        for test in "mine_$exp" "endo4ie_$exp"; do
            model="$CKPT/$train/main_net/model_256.pth"
            python evaluate_pairs.py \
                --model "$model" \
                --test_dir "$DATA/$test/test" \
                --train_tag "$train" --test_tag "$test" \
                --out_csv  "$RES/${train}__on__${test}.csv" \
                --summary_csv "$SUMMARY" --gpu "$GPU"
        done
    done
done

# ============================================================ 5. RESULTS TABLE
echo -e "\n=== STEP 5: results matrix ==="
python make_matrix.py --summary_csv "$SUMMARY"
