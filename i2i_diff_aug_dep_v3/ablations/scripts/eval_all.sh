#!/usr/bin/env bash
# eval_all.sh — Score every trained ablation checkpoint on the same held-out
# test subset and aggregate the comparison report.
#
# All paths come from ablation.env. The held-out subset is the one
# setup_ablations.sh built deterministically (TEST_MAX_IMAGES frames).
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

VARIANTS=(BASELINE NO_DEPTH LAB_FULL NO_SOBEL)
DOMAINS=(under over)

mkdir -p "$ABL_ROOT/logs"

for domain in "${DOMAINS[@]}"; do
  [ -n "${ONLY_DOMAIN:-}" ] && [ "$domain" != "$ONLY_DOMAIN" ] && continue
  if [ "$domain" = "under" ]; then
    domain_long="underexposed"
    real_dir="$REAL_UNDER_DIR"
  else
    domain_long="overexposed"
    real_dir="$REAL_OVER_DIR"
  fi
  for v in "${VARIANTS[@]}"; do
    [ -n "${ONLY_VARIANT:-}" ] && [ "$v" != "$ONLY_VARIANT" ] && continue
    cfg="ablations/configs/$domain/${v,,}.yaml"
    ckpt="$../../outputs/ablations/$domain/$v/checkpoints/$domain_long/best.pt"
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
    # MAX_IMAGES in the env file caps a per-eval smoke run; the deterministic
    # subset built by setup_ablations.sh is the *actual* test set.
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
  --out_md  ablations/RESULTS.md \
  --out_csv ablations/RESULTS.csv \
  2>&1 | tee "$ABL_ROOT/logs/aggregate.log"

echo "[done] eval_all.sh complete."
