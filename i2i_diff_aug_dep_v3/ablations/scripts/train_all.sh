#!/usr/bin/env bash
# train_all.sh — Sequentially trains all 12 ablation variants.
#
# All paths come from ablation.env. Re-run safe: any variant whose best.pt
# already exists is skipped. Filter with ONLY_DOMAIN / ONLY_VARIANT in the
# env file (or pass them inline, e.g. `ONLY_VARIANT=BASELINE bash train_all.sh`).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ABLATION_ENV:-$SCRIPT_DIR/ablation.env}"
if [ ! -f "$ENV_FILE" ]; then
  echo "[error] env file not found: $ENV_FILE" >&2
  exit 1
fi
echo "[env] sourcing $ENV_FILE"
set -a; . "$ENV_FILE"; set +a

VARIANTS=(BASELINE NO_DEPTH LAB_FULL MSE_ONLY NO_L1 NO_SOBEL)
DOMAINS=(under over)

mkdir -p "$ABL_ROOT/logs"

for domain in "${DOMAINS[@]}"; do
  [ -n "${ONLY_DOMAIN:-}" ] && [ "$domain" != "$ONLY_DOMAIN" ] && continue
  if [ "$domain" = "under" ]; then domain_long="underexposed"; else domain_long="overexposed"; fi
  for v in "${VARIANTS[@]}"; do
    [ -n "${ONLY_VARIANT:-}" ] && [ "$v" != "$ONLY_VARIANT" ] && continue
    cfg="ablations/configs/$domain/${v,,}.yaml"
    ckpt_dir="$ABL_ROOT/$domain/$v/checkpoints"
    best="$ckpt_dir/$domain_long/best.pt"
    log="$ABL_ROOT/logs/train_${domain}_${v}.log"
    if [ -f "$best" ]; then
      echo "[skip] $domain/$v — best.pt already at $best"
      continue
    fi
    echo "[train] $domain / $v   →   log: $log"
    python diffusion_train.py \
      --config "$cfg" \
      --domain "$domain_long" \
      --output_subdir "$domain_long" \
      2>&1 | tee "$log"
  done
done

echo "[done] train_all.sh complete."
