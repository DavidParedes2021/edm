# Dataset comparison: your synthetic data vs ENDO4IE, judged by EndoLMSPEC

## The question, made precise

Both datasets contain **synthetic** over/under-exposed frames (real normal frames
corrupted by two *different* synthesis methods — your depth-aware diffusion
pipeline vs ENDO4IE's). So "which dataset is better?" cannot mean "which has more
realistic-looking corruptions" measured directly. It means:

> **Which dataset, used as training data, produces a better exposure-correction
> model?** — measured on a *held-out test set of frames neither model trained on*.

This is the standard **Train-on-Synthetic / Test protocol**. EndoLMSPEC is the
fixed "judge": same architecture, same hyper-parameters, only the training data
changes.

### Important: the two corpora barely overlap

Measured on disk: your synthetic set is **EAD2020 / EDD2020** endoscopy frames
(504), while ENDO4IE is **mostly Kvasir** (1063/1231 over-exposure frames are
Kvasir; only 53 are EAD2020). They share only **47 frames** (over) / **33**
(under). So a "train both on the *same* frames, vary only the corruption" design
is impossible — too few shared frames to train on.

Consequences for the protocol:

1. **Leakage-safe split** (`common_split.py`): each frame's train/val/test bucket
   is a hash of its filename stem, so on the handful of shared frames a frame is
   *always* in the same split for both datasets — never train-here / test-there.
2. **Mode B — full datasets, size-matched** (default in `run_all.ps1`): each
   dataset uses all its own frames; ENDO4IE's *training* set is capped to the same
   count as yours so the result reflects data **quality**, not quantity. Test sets
   are each dataset's own held-out frames.
3. **Disclose the confound.** Because the two test sets are different scenes
   (EAD/EDD vs Kvasir), the *absolute* cross-test numbers blend two effects:
   corruption-synthesis quality **and** source-content domain shift. That is
   actually a fair reading of "which dataset is better" — a dataset's value
   includes its content diversity — but you must state it. The **robustness drop**
   (each model's diagonal minus its off-diagonal, printed by `make_matrix.py`)
   partly normalises out content and isolates transferability.

## What gets measured: the cross matrix

For each exposure type (over, under) you train 2 models and test each on both
test sets — a 2×2 matrix of SSIM / PSNR / LPIPS:

```
                 tested on:  mine_over   endo4ie_over
trained on mine_over           [A]          [B]
trained on endo4ie_over        [C]          [D]
```

- **[A], [D] diagonal** = in-distribution (test corruption == train corruption). Easy; reference only.
- **[B], [C] off-diagonal** = generalisation to the *other* corruption distribution. **This is the headline.**
- **Verdict (absolute):** your dataset is the better training source if **[B] > [C]**
  — a model trained on your data transfers to ENDO4IE's test better than the reverse.
- **Verdict (robustness):** smaller generalisation drop (diagonal − off-diagonal)
  means more transferable training data; this is less sensitive to the EAD-vs-Kvasir
  content difference between the two test sets.
- Strongest claim: **[B] > [C]** *and* your model's drop < ENDO4IE's drop.

`make_matrix.py` prints these tables and the verdict automatically.

## Files

| File | Purpose |
|---|---|
| `common_split.py` | deterministic, dataset-independent stem→split (no leakage) |
| `prepare_data.py` | source frames → EndoLMSPEC `exposure_dataset` layout + held-out `test/` |
| `evaluate_pairs.py` | run a trained model on a paired test set → SSIM/PSNR/LPIPS CSV |
| `make_matrix.py` | summary.csv → train×test matrices + verdict |
| `run_all.ps1` | runs the whole thing end-to-end (Windows) |

Dataset paths (edit at the top of `run_all.ps1` if they move). Both datasets use
the SAME standardized flat layout — `{Normal, Overexposed, Underexposed}` — paired
by file stem within each exposure type:
- diffusion: `edm/syntethic_images/Endo4IE2_diffusion/{Normal, Overexposed, Underexposed}`
- ENDO4IE:   `edm/syntethic_images/Endo4IE/{Normal, Overexposed, Underexposed}`

## Prerequisites

```powershell
# in the endo_lmspec conda env (see EndoLMSPEC-main/README.md), plus:
pip install scikit-image
pip install lpips        # optional but recommended (perceptual metric)
```

## Run it (one command)

```powershell
cd C:\Users\omarw\Documents\MsC\MSC\Endoscopy\edm\EndoLMSPEC-main
.\dataset_compare\run_all.ps1
```

That executes the 5 steps below. To run them manually instead:

### Step 1 — prepare data (Mode B, size-matched)
```powershell
cd .\dataset_compare
# your set first (it's the smaller one):
python prepare_data.py --name mine --exposure over `
  --normal_dir  "...\edm_aug_diff_synthetic_dataset\normal_frames" `
  --exposed_dir "...\edm_aug_diff_synthetic_dataset\overexposed" --out_root ".\compare_data"
# then ENDO4IE, capping training to your set's training count (see printed count, e.g. ~350):
python prepare_data.py --name endo4ie --exposure over `
  --normal_dir  "...\endo4ie\over\normal" --exposed_dir "...\endo4ie\over\over" `
  --out_root ".\compare_data" --cap_train 350
# ...repeat both with --exposure under
```
(`run_all.ps1` reads the actual count automatically — prefer it over hand-typing the cap.)

### Step 2 — extract patches (per tree, training+validation only)
```powershell
cd ..    # EndoLMSPEC-main
python patches_extraction.py --exposure_dataset ".\dataset_compare\compare_data\mine_over" --patches_size_pow 7 8
```

### Step 3 — train EndoLMSPEC (per tree)
```powershell
python main_training.py --exposure_dataset ".\dataset_compare\compare_data\mine_over" `
  --checkpoint_dir ".\dataset_compare\checkpoints\mine_over" `
  --GPU 0 --loss_weights 0.25 0.25 1.0 0.25 0.0 --epochs_list 40 30
```
Final weights land at `...\checkpoints\mine_over\main_net\model_256.pth`.

> First training run pulls a wandb login prompt. Run `wandb offline` (or
> `wandb disabled`) beforehand to skip it.

### Step 4 — cross-evaluate (fills the matrix)
```powershell
cd .\dataset_compare
python evaluate_pairs.py `
  --model ".\checkpoints\mine_over\main_net\model_256.pth" `
  --test_dir ".\compare_data\endo4ie_over\test" `
  --train_tag mine_over --test_tag endo4ie_over `
  --out_csv ".\results\mine_over__on__endo4ie_over.csv" `
  --summary_csv ".\results\summary.csv" --gpu 0
# repeat for all 4 (train x test) combos per exposure type
```

### Step 5 — read the verdict
```powershell
python make_matrix.py --summary_csv ".\results\summary.csv"
```

## Validate cheap before the long runs

Do one fast end-to-end pass with tiny epochs to confirm wiring, then scale up:
```powershell
# in run_all.ps1 set:  $EPOCHS = @("2","2")
```
Confirm `model_256.pth` is produced and `evaluate_pairs.py` prints metrics, then
restore `@("40","30")` and run for real.

## Variants / extensions

- **No size cap:** drop `--cap_train` to test "the dataset as delivered" (ENDO4IE
  is ~2× larger). Report it *alongside* the capped result, never instead — the
  uncapped numbers confound data quality with quantity.
- **Combined (augmentation gain):** prepare a third tree whose training set is
  `mine ∪ endo4ie`, train `combined_*`, and add it as a third row/column. If
  `combined > endo4ie` on the real-frame test, your data adds value as augmentation.
- **Realism cross-check (no model needed):** if you have any *real* clinically
  over/under-exposed frames, compute FID/KID between each synthetic exposed set
  and those real frames — a direct measure of corruption realism that complements
  the downstream-task result above.

## Reporting

For the thesis: report the 2×2 SSIM and PSNR matrices (over and under separately),
state the split is leakage-safe and frame-matched (Mode A), and run ≥3 seeds
(`--salt` changes the partition; or vary the training seed) to attach mean±std and
a significance test to the [B] vs [C] gap.
