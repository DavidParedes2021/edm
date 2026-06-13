# run_all.ps1 — full synthetic-dataset comparison via EndoLMSPEC (Windows PowerShell)
#
# Compares YOUR diffusion synthetic dataset vs ENDO4IE by training EndoLMSPEC on
# each and cross-evaluating on held-out test frames. Mode B (size-matched): each
# dataset uses its OWN frames, and ENDO4IE's training set is capped to match the
# diffusion set's size so the result reflects data QUALITY, not quantity. (Mode A —
# a common-frame design — is infeasible: the corpora share too few frames. See
# README.) Both datasets are split with the same leakage-safe stem-hash, so any
# shared frame always lands in the same split.
#
# Both datasets use the SAME standardized flat layout:
#     <dataset>/{Normal, Overexposed, Underexposed}
# pairing by file stem within each exposure type.
#
# Run from the EndoLMSPEC-main folder:  .\dataset_compare\run_all.ps1
# Comment out blocks you've already completed to resume.

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------- paths / config
$LMSPEC = $PSScriptRoot | Split-Path            # EndoLMSPEC-main
$EDM    = $LMSPEC | Split-Path                  # edm (repo root)
$DC     = $PSScriptRoot                          # dataset_compare
$SYN    = Join-Path $EDM "syntethic_images\Endo4IE2_diffusion"   # diffusion set
$ENDO   = Join-Path $EDM "syntethic_images\Endo4IE"              # ENDO4IE set

$DATA   = Join-Path $DC "compare_data"          # prepared exposure_dataset trees
$CKPT   = Join-Path $DC "checkpoints"           # trained models
$RES    = Join-Path $DC "results"               # metrics
$SUMMARY = Join-Path $RES "summary.csv"

$GPU    = 0
$LW     = @("0.25","0.25","1.0","0.25","0.0")   # paper's best loss weights
$EPOCHS = @("40","30")                          # [128px, 256px]; lower for a quick smoke test

# ============================================================ 1. PREPARE DATA
# Mode B: full datasets, ENDO4IE training capped to match your set's size.
# (The two corpora share almost no frames -- yours is EAD2020/EDD2020, ENDO4IE
#  is mostly Kvasir -- so a common-frame design isn't viable. See README.)
Write-Host "`n=== STEP 1: prepare data ===" -ForegroundColor Cyan
Push-Location $DC

function Train-Count($tag) {
    (Get-ChildItem (Join-Path $DATA "$tag\training\gt_images") -File).Count
}

# prepare YOUR (smaller) datasets first, then cap ENDO4IE training to match.
# Standardized layout: both datasets share one Normal/ dir per set; the exposure
# type only changes which exposed dir is paired against it.
python prepare_data.py --name mine --exposure over `
    --normal_dir "$SYN\Normal" --exposed_dir "$SYN\Overexposed"  --out_root "$DATA"
python prepare_data.py --name mine --exposure under `
    --normal_dir "$SYN\Normal" --exposed_dir "$SYN\Underexposed" --out_root "$DATA"

$capOver  = Train-Count "mine_over"
$capUnder = Train-Count "mine_under"
Write-Host "size-matching ENDO4IE training to mine: over=$capOver under=$capUnder" -ForegroundColor Yellow

python prepare_data.py --name endo4ie --exposure over `
    --normal_dir "$ENDO\Normal" --exposed_dir "$ENDO\Overexposed"   --out_root "$DATA" --cap_train $capOver
python prepare_data.py --name endo4ie --exposure under `
    --normal_dir "$ENDO\Normal" --exposed_dir "$ENDO\Underexposed" --out_root "$DATA" --cap_train $capUnder
Pop-Location

# ============================================================ 2-3. PATCHES + TRAIN
$tags = @("mine_over","endo4ie_over","mine_under","endo4ie_under")
foreach ($tag in $tags) {
    $tree = Join-Path $DATA $tag
    Write-Host "`n=== STEP 2: patches for $tag ===" -ForegroundColor Cyan
    Push-Location $LMSPEC
    python patches_extraction.py --exposure_dataset "$tree" --patches_size_pow 7 8

    Write-Host "`n=== STEP 3: train $tag ===" -ForegroundColor Cyan
    python main_training.py --exposure_dataset "$tree" `
        --checkpoint_dir (Join-Path $CKPT $tag) `
        --GPU $GPU --loss_weights $LW --epochs_list $EPOCHS
    Pop-Location
}

# ============================================================ 4. CROSS-EVALUATE
Write-Host "`n=== STEP 4: cross-evaluation ===" -ForegroundColor Cyan
if (Test-Path $SUMMARY) { Remove-Item $SUMMARY }
Push-Location $DC
foreach ($exp in @("over","under")) {
    foreach ($train in @("mine_$exp","endo4ie_$exp")) {
        foreach ($test in @("mine_$exp","endo4ie_$exp")) {
            $model = Join-Path $CKPT "$train\main_net\model_256.pth"
            python evaluate_pairs.py `
                --model "$model" `
                --test_dir (Join-Path $DATA "$test\test") `
                --train_tag $train --test_tag $test `
                --out_csv  (Join-Path $RES "${train}__on__${test}.csv") `
                --summary_csv "$SUMMARY" --gpu $GPU
        }
    }
}

# ============================================================ 5. RESULTS TABLE
Write-Host "`n=== STEP 5: results matrix ===" -ForegroundColor Cyan
python make_matrix.py --summary_csv "$SUMMARY"
Pop-Location
