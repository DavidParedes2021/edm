#!/usr/bin/env bash
# cleanup_train_pairs.sh — Free disk space after training is finished.
#
# Deletes the training pair sets (pairs_baseline, pairs_no_depth) AND the
# training depth cache (depth_train). These are no longer needed once every
# variant's best.pt is on disk; evaluation only uses test_pairs/ + depth_test/
# + test_normal_subset/.
#
# Safety check: refuses to delete unless every expected best.pt exists. Pass
# --force to skip the check (e.g. if you're abandoning incomplete variants).
#
# Usage:
#   bash ablations/scripts/cleanup_train_pairs.sh         # safe (requires all variants trained)
#   bash ablations/scripts/cleanup_train_pairs.sh --force # skip the check
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ABLATION_ENV:-$SCRIPT_DIR/ablation.env}"
if [ ! -f "$ENV_FILE" ]; then
  echo "[error] env file not found: $ENV_FILE" >&2
  exit 1
fi
echo "[env] sourcing $ENV_FILE"
set -a; . "$ENV_FILE"; set +a

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

VARIANTS=(BASELINE NO_DEPTH LAB_FULL NO_SOBEL)
DOMAINS=(under over)

missing=()
for domain in "${DOMAINS[@]}"; do
  # Honour ONLY_DOMAIN so the safety check doesn't insist on a direction the
  # user explicitly skipped (e.g. ONLY_DOMAIN=over while under is deferred).
  [ -n "${ONLY_DOMAIN:-}" ] && [ "$domain" != "$ONLY_DOMAIN" ] && continue
  if [ "$domain" = "under" ]; then domain_long="underexposed"; else domain_long="overexposed"; fi
  for v in "${VARIANTS[@]}"; do
    ckpt="$ABL_ROOT/$domain/$v/checkpoints/$domain_long/best.pt"
    [ -f "$ckpt" ] || missing+=("$domain/$v")
  done
done

if [ "${#missing[@]}" -gt 0 ] && [ "$FORCE" -ne 1 ]; then
  echo "[abort] not all variants have a best.pt yet:"
  for m in "${missing[@]}"; do echo "    - $m"; done
  echo "Re-run with --force to delete anyway."
  exit 1
fi

# Compute the sizes BEFORE deletion so the user sees what they reclaimed.
size_of() { du -sh "$1" 2>/dev/null | awk '{print $1}' || echo "?"; }

for target in "$PAIRS_NO_DEPTH" "$PAIRS_BASELINE" "$DEPTH_DIR"; do
  if [ -e "$target" ]; then
    sz="$(size_of "$target")"
    echo "[rm] $target  ($sz)"
    # Delete pairs_no_depth FIRST so its symlinks into pairs_baseline are
    # removed before pairs_baseline goes — avoids any dangling-symlink moment.
    rm -rf -- "$target"
  else
    echo "[skip] $target (already gone)"
  fi
done

echo "[done] training pair sets removed. Eval only needs:"
echo "    $TEST_NORMAL_SUBSET   (test images)"
echo "    $DEPTH_TEST_DIR       (test depth)"
echo "    $TEST_PAIRS           (GT for PSNR/SSIM/LPIPS)"
