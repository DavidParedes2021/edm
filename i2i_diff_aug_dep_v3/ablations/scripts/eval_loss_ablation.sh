#!/usr/bin/env bash
# eval_loss_ablation.sh — Score the 6 loss-ablation checkpoints on the same
# held-out test subset built by setup_ablations.sh, then aggregate the report.
# Mirrors eval_all.sh but iterates the loss-composition variants.
#
# Run AFTER train_loss_ablation.sh, from the project root:
#   bash ablations/scripts/eval_loss_ablation.sh
# Re-run safe: skips any variant whose eval/summary.json already exists.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ABLATION_ENV:-$SCRIPT_DIR/ablation.env}"
if [ ! -f "$ENV_FILE" ]; then
  echo "[error] env file not found: $ENV_FILE" >&2
  exit 1
fi
echo "[env] sourcing $ENV_FILE"
set -a; . "$ENV_FILE"; set +a

for v in REAL_OVER_DIR REAL_UNDER_DIR TEST_NORMAL_SUBSET; do
  if [ -z "${!v:-}" ]; then echo "[error] $v not set in $ENV_FILE" >&2; exit 1; fi
done
if [ ! -d "$TEST_NORMAL_SUBSET" ]; then
  echo "[error] test subset not found at $TEST_NORMAL_SUBSET — run setup_ablations.sh first." >&2
  exit 1
fi

VARIANTS=(BASELINE VANILLA_DDPM NO_SOBEL NO_L1 NO_EXTREME NO_DGRAD)
DOMAINS=(under over)

mkdir -p "$ABL_ROOT/logs"

for domain in "${DOMAINS[@]}"; do
  [ -n "${ONLY_DOMAIN:-}" ] && [ "$domain" != "$ONLY_DOMAIN" ] && continue
  if [ "$domain" = "under" ]; then
    domain_long="underexposed"; real_dir="$REAL_UNDER_DIR"
  else
    domain_long="overexposed"; real_dir="$REAL_OVER_DIR"
  fi
  for v in "${VARIANTS[@]}"; do
    [ -n "${ONLY_VARIANT:-}" ] && [ "$v" != "$ONLY_VARIANT" ] && continue
    cfg="ablations/configs/$domain/${v,,}.yaml"
    ckpt="./outputs/ablations/$domain/$v/checkpoints/$domain_long/best.pt"
    out="$ABL_ROOT/$domain/$v/eval"
    log="$ABL_ROOT/logs/eval_${domain}_${v}.log"
    if [ ! -f "$ckpt" ]; then
      echo "[skip] $domain/$v — no checkpoint at $ckpt"
      continue
    fi
    if [ -f "$out/summary.json" ]; then
      echo "[skip] $domain/$v — summary.json already exists at $out"
      continue
    fi
    echo "[eval] $domain / $v   →   log: $log"
    extra=()
    [ -n "${MAX_IMAGES:-}" ] && extra+=(--max_images "$MAX_IMAGES")
    python -m ablations.evaluate_ablations \
      --config "$cfg" \
      --checkpoint "$ckpt" \
      --test_normal "$TEST_NORMAL_SUBSET" \
      --test_pairs "$TEST_PAIRS" \
      --test_depth "$DEPTH_TEST_DIR" \
      --real_dir "$real_dir" \
      --output_dir "$out" \
      --domain "$domain_long" \
      --variant_name "$v" \
      "${extra[@]}" \
      2>&1 | tee "$log"
  done
done

echo "[aggregate] writing comparison report"
python -m ablations.aggregate_report \
  --root "$ABL_ROOT" \
  --out_md  ablations/RESULTS_loss.md \
  --out_csv ablations/RESULTS_loss.csv \
  2>&1 | tee "$ABL_ROOT/logs/aggregate_loss.log"

echo "[done] eval_loss_ablation.sh complete."
