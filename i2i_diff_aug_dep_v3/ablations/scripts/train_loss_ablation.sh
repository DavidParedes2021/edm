#!/usr/bin/env bash
# train_loss_ablation.sh — Train the 6 loss-composition ablation variants, with
# TRUE resume-from-interruption (survives running out of disk mid-run).
#
#   BASELINE  VANILLA_DDPM  NO_SOBEL  NO_L1  NO_EXTREME  NO_DGRAD
#
# All 6 use the SAME depth-aware pairs (pairs_baseline); only the loss weights
# differ. mse_weight stays 1.0 in every variant — the auxiliary x0-space losses
# (l1 / edge / extreme / depth-grad) are the only thing ablated.
#
# Resume model (unlike train_all.sh, which restarts a half-trained variant
# because best.pt exists from epoch 1):
#   * a variant is DONE only when diffusion_train.py wrote a COMPLETED sentinel
#     (i.e. the full epoch loop finished) → skipped.
#   * if latest.pt exists but no COMPLETED → resumed via --resume latest.pt.
#   * otherwise → trained from scratch.
#   * if a training call fails (e.g. disk full) the script STOPS immediately;
#     free space and re-run — it picks up exactly where it left off.
#
# Run from the project root (i2i_diff_aug_dep_v3/):
#   bash ablations/scripts/train_loss_ablation.sh
# Filter with ONLY_DOMAIN / ONLY_VARIANT (env file or inline):
#   ONLY_DOMAIN=over ONLY_VARIANT=NO_EXTREME bash ablations/scripts/train_loss_ablation.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ABLATION_ENV:-$SCRIPT_DIR/ablation.env}"
if [ ! -f "$ENV_FILE" ]; then
  echo "[error] env file not found: $ENV_FILE" >&2
  exit 1
fi
echo "[env] sourcing $ENV_FILE"
set -a; . "$ENV_FILE"; set +a

VARIANTS=(BASELINE VANILLA_DDPM NO_SOBEL NO_L1 NO_EXTREME NO_DGRAD)
DOMAINS=(under over)

LOG_DIR="./outputs/ablations/logs"
mkdir -p "$LOG_DIR"

for domain in "${DOMAINS[@]}"; do
  [ -n "${ONLY_DOMAIN:-}" ] && [ "$domain" != "$ONLY_DOMAIN" ] && continue
  if [ "$domain" = "under" ]; then domain_long="underexposed"; else domain_long="overexposed"; fi
  for v in "${VARIANTS[@]}"; do
    [ -n "${ONLY_VARIANT:-}" ] && [ "$v" != "$ONLY_VARIANT" ] && continue

    cfg="ablations/configs/$domain/${v,,}.yaml"
    if [ ! -f "$cfg" ]; then
      echo "[error] missing config $cfg — run generate_configs first (see header of this script's docs)." >&2
      exit 1
    fi

    # Must match diffusion_train.py: ckpt_dir = <output.checkpoints_dir>/<output_subdir>.
    # generate_configs.py sets output.checkpoints_dir = ./outputs/ablations/<domain>/<v>/checkpoints
    ckpt_dir="./outputs/ablations/$domain/$v/checkpoints/$domain_long"
    marker="$ckpt_dir/COMPLETED"
    latest="$ckpt_dir/latest.pt"
    log="$LOG_DIR/train_loss_${domain}_${v}.log"

    if [ -f "$marker" ]; then
      echo "[skip] $domain/$v — COMPLETED ($(cat "$marker" 2>/dev/null | tr -d '\n'))"
      continue
    fi

    resume_args=()
    if [ -f "$latest" ]; then
      echo "[resume] $domain/$v — continuing from $latest"
      resume_args=(--resume "$latest")
    else
      echo "[train]  $domain/$v — fresh start"
    fi
    echo "         log: $log"

    if python diffusion_train.py \
        --config "$cfg" \
        --domain "$domain_long" \
        --output_subdir "$domain_long" \
        "${resume_args[@]}" \
        2>&1 | tee "$log"; then
      echo "[ok] $domain/$v finished"
    else
      echo "[STOP] training failed for $domain/$v (out of disk?). Free space, then" >&2
      echo "       re-run this script — it resumes $domain/$v from latest.pt." >&2
      exit 1
    fi
  done
done

echo "[done] train_loss_ablation.sh complete — all requested variants COMPLETED."
