#!/usr/bin/env bash
# setup_ablations.sh — One-time prep for the ablation study.
#
# Reads every path/throttle from `ablation.env` (next to this script). Override
# the env-file path with `ABLATION_ENV=/path/to/your.env`.
#
# Uses the PRODUCTION scripts from "code _to_generate_pairs/" (HADepth depth +
# hybrid luminance/depth masking) — the root copies are older snapshots that
# don't support those backends.
#
# Steps:
#   1. Train-set depth cache         (HADepth, one .npy per train image)
#   2. Both pair sets:
#        a) hybrid    — for BASELINE / LAB_FULL / loss variants
#        b) luminance — for the NO_DEPTH variant (pair targets contain NO depth)
#   3. Deterministic N-image test subset  (default N=200, seed=42)
#   4. Test-subset depth cache
#   5. Test-subset GT pairs               (for PSNR/SSIM/LPIPS)
#   6. Per-variant config YAMLs            (12 files)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ABLATION_ENV:-$SCRIPT_DIR/ablation.env}"
if [ ! -f "$ENV_FILE" ]; then
  echo "[error] env file not found: $ENV_FILE" >&2
  exit 1
fi
echo "[env] sourcing $ENV_FILE"
set -a; . "$ENV_FILE"; set +a

for v in NORMAL_TRAIN_DIR NORMAL_TEST_DIR PROD_CODE_DIR DEPTH_BACKEND; do
  if [ -z "${!v:-}" ]; then echo "[error] $v not set in $ENV_FILE" >&2; exit 1; fi
done
PROD="$PROD_CODE_DIR"
if [ ! -d "$PROD" ]; then
  echo "[error] PROD_CODE_DIR '$PROD' does not exist (must be the folder that contains" >&2
  echo "        the production depth_estimator.py / generate_pairs.py / depth_augment.py)." >&2
  exit 1
fi
mkdir -p "$ABL_ROOT/logs"

# HADepth-only sanity checks
if [ "$DEPTH_BACKEND" = "hadepth" ]; then
  for v in HADEPTH_REPO HADEPTH_CKPT; do
    if [ -z "${!v:-}" ]; then echo "[error] $v required when DEPTH_BACKEND=hadepth" >&2; exit 1; fi
  done
fi

# Build the depth-estimator argument list once.
depth_args=(--backend "$DEPTH_BACKEND")
if [ "$DEPTH_BACKEND" = "hadepth" ]; then
  depth_args+=(--hadepth_repo "$HADEPTH_REPO" --hadepth_ckpt "$HADEPTH_CKPT")
fi

# Common pair-generator arguments (gammas come from env, overridable per call).
common_pair_args=(
  --gamma_over   "$GAMMA_OVER"
  --gamma_under  "$GAMMA_UNDER"
  --num_variants "$NUM_VARIANTS"
)

echo "[1/6] Training-set depth cache → $DEPTH_DIR  (backend=$DEPTH_BACKEND)"
if [ ! -d "$DEPTH_DIR" ] || [ -z "$(ls -A "$DEPTH_DIR" 2>/dev/null || true)" ]; then
  python "$PROD/depth_estimator.py" \
    "${depth_args[@]}" \
    --input_dir "$NORMAL_TRAIN_DIR" \
    --output_dir "$DEPTH_DIR" \
    2>&1 | tee "$ABL_ROOT/logs/setup_depth_train.log"
else
  echo "    [skip] $DEPTH_DIR already populated"
fi

echo "[2/6] Pair set — $MASK_STRATEGY_BASELINE (baseline) → $PAIRS_BASELINE"
if [ ! -d "$PAIRS_BASELINE/normal" ]; then
  python "$PROD/generate_pairs.py" \
    --normal_dir   "$NORMAL_TRAIN_DIR" \
    --depth_dir    "$DEPTH_DIR" \
    --output_dir   "$PAIRS_BASELINE" \
    --mask_strategy "$MASK_STRATEGY_BASELINE" \
    --hybrid_blend "$HYBRID_BLEND" \
    "${common_pair_args[@]}" \
    2>&1 | tee "$ABL_ROOT/logs/setup_pairs_baseline.log"
else
  echo "    [skip] $PAIRS_BASELINE already exists"
fi

echo "[3/6] Pair set — $MASK_STRATEGY_NO_DEPTH (NO_DEPTH variant) → $PAIRS_NO_DEPTH"
if [ ! -d "$PAIRS_NO_DEPTH/normal" ]; then
  python "$PROD/generate_pairs.py" \
    --normal_dir   "$NORMAL_TRAIN_DIR" \
    --depth_dir    "$DEPTH_DIR" \
    --output_dir   "$PAIRS_NO_DEPTH" \
    --mask_strategy "$MASK_STRATEGY_NO_DEPTH" \
    "${common_pair_args[@]}" \
    2>&1 | tee "$ABL_ROOT/logs/setup_pairs_no_depth.log"
else
  echo "    [skip] $PAIRS_NO_DEPTH already exists"
fi

echo "[4/6] Subsample test set ($TEST_MAX_IMAGES images, seed=$TEST_SAMPLE_SEED)"
python "$SCRIPT_DIR/subsample_test_set.py" \
  --src "$NORMAL_TEST_DIR" \
  --dst "$TEST_NORMAL_SUBSET" \
  --n "$TEST_MAX_IMAGES" \
  --seed "$TEST_SAMPLE_SEED" \
  2>&1 | tee "$ABL_ROOT/logs/setup_subset.log"

echo "[5/6] Test-subset depth cache → $DEPTH_TEST_DIR"
if [ ! -d "$DEPTH_TEST_DIR" ] || [ -z "$(ls -A "$DEPTH_TEST_DIR" 2>/dev/null || true)" ]; then
  python "$PROD/depth_estimator.py" \
    "${depth_args[@]}" \
    --input_dir "$TEST_NORMAL_SUBSET" \
    --output_dir "$DEPTH_TEST_DIR" \
    2>&1 | tee "$ABL_ROOT/logs/setup_depth_test.log"
else
  echo "    [skip] $DEPTH_TEST_DIR already populated"
fi

echo "[6a/6] Test-subset GT pairs (mask=$MASK_STRATEGY_BASELINE) → $TEST_PAIRS"
if [ ! -d "$TEST_PAIRS/normal" ]; then
  python "$PROD/generate_pairs.py" \
    --normal_dir   "$TEST_NORMAL_SUBSET" \
    --depth_dir    "$DEPTH_TEST_DIR" \
    --output_dir   "$TEST_PAIRS" \
    --mask_strategy "$MASK_STRATEGY_BASELINE" \
    --hybrid_blend "$HYBRID_BLEND" \
    --gamma_over   "$GAMMA_OVER" \
    --gamma_under  "$GAMMA_UNDER" \
    --num_variants 1 \
    2>&1 | tee "$ABL_ROOT/logs/setup_pairs_test.log"
else
  echo "    [skip] $TEST_PAIRS already exists"
fi

echo "[6b/6] Writing per-variant configs → ablations/configs/{under,over}/"
python -m ablations.generate_configs \
  --base diffusion_config.yaml \
  --pairs_baseline "$PAIRS_BASELINE" \
  --pairs_no_depth "$PAIRS_NO_DEPTH" \
  --depth_dir "$DEPTH_DIR" \
  --epochs "$EPOCHS" \
  --out ablations/configs \
  2>&1 | tee "$ABL_ROOT/logs/setup_configs.log"

echo "[done] setup_ablations.sh complete."
echo
echo "Next:"
echo "  bash ablations/scripts/train_all.sh"
echo "  bash ablations/scripts/eval_all.sh"
echo "  python -m ablations.aggregate_report --root $ABL_ROOT"
