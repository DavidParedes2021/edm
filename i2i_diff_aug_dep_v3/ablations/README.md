# Ablation Study — Depth-Aware Endoscopy Diffusion

Three ablations, all on **150 epochs**, **same seed**, **same data**, **both
directions** (`underexposed`, `overexposed`):

| Group | Variant | What changes from BASELINE |
|---|---|---|
| — | `BASELINE` | Full pipeline (`mask_strategy=hybrid` pairs, depth in UNet + losses + post-proc) |
| **A1 Depth** | `NO_DEPTH` | `mask_strategy=luminance` pairs (no depth in targets), `in_channels=2`, depth-grad loss off, focal-blend off, depth-chroma boost off |
| **A2 L-vs-LAB** | `LAB_FULL` | Diffusion learns full LAB (`in_channels=7`, `out_channels=3`) — AB no longer trivially preserved from source |
| **A3 Loss** | `MSE_ONLY` | `l1=edge=extreme=dg=0` |
| **A3 Loss** | `NO_L1` | `l1_weight=0` |
| **A3 Loss** | `NO_SOBEL` | `edge_weight=0` |

The depth signal lives in TWO places in production: (1) the pair-generation
mask (`hybrid` = 50/50 luminance + depth), and (2) the UNet input + losses +
inference post-processing. The A1 ablation removes it from *both* by switching
the pair generator to `--mask_strategy luminance` AND disabling every
downstream depth consumer.

## Production scripts

The ablation framework calls the **production** pair-generation and depth-
estimation scripts in `code _to_generate_pairs/` (HADepth backend, `--mask_
strategy hybrid` by default). The root copies of those files are older
snapshots that don't support HADepth or the mask-strategy flag — they are
*not* used by the ablation pipeline.

Configure backend / strategy in [ablations/scripts/ablation.env](scripts/ablation.env):

```sh
DEPTH_BACKEND=hadepth                    # or dav2_endo / midas
HADEPTH_REPO=./HADepth
HADEPTH_CKPT=./HADepth/HADepth_fullmodel
MASK_STRATEGY_BASELINE=hybrid            # luminance | depth | hybrid
MASK_STRATEGY_NO_DEPTH=luminance         # the A1 contrast
HYBRID_BLEND=0.5                         # used when mask_strategy=hybrid
```

= 6 variants × 2 directions × 150 epochs = 12 training runs.

## Metrics

| Family | Metric | Purpose |
|---|---|---|
| Reference (vs synthetic GT) | PSNR, SSIM, LPIPS | Pixel / structural / perceptual fidelity |
| Distribution (vs real unpaired) | FID, KID | "Does the augmentation look real?" |
| Strength | `delta_L_mask` | Mean ‖L_out − L_src‖ inside depth-expected region |
| Placement (A1) | `depth_mask_pearson` | Pearson(ΔL, depth-mask). Tests whether the effect lands where depth says it should. |
| Color preservation (A2) | `delta_E00_AB` | CIEDE2000 between source AB and output AB. Zero by construction for `L`-only variants. |
| Texture preservation (A2, A3) | `hf_pearson`, `sobel_l1` | High-frequency correlation / Sobel-edge L1 vs source |

## Workflow (inside the DGX docker container)

All three launcher scripts read paths and throttles from a single
`ablations/scripts/ablation.env` file — no `export` lines needed in the
shell. Edit that file once, then run:

```bash
# 0. edit ablations/scripts/ablation.env (paths to data, EPOCHS, etc.)
#    The four REQUIRED keys at the top are NORMAL_TRAIN_DIR,
#    NORMAL_TEST_DIR, REAL_OVER_DIR, REAL_UNDER_DIR.
#    TEST_MAX_IMAGES controls the test-subset size (default 200).

# 1. one-time prep:
#      train-set depth + both pair sets
#      subsample NORMAL_TEST_DIR → TEST_NORMAL_SUBSET (deterministic, seeded)
#      test-subset depth + GT pairs
#      write the 12 per-variant configs
bash ablations/scripts/setup_ablations.sh

# 2. train all 12 variants sequentially
bash ablations/scripts/train_all.sh
#    re-run safe: skips any variant whose best.pt already exists
#    restrict to one: ONLY_DOMAIN=under ONLY_VARIANT=BASELINE bash ...
#    (or set ONLY_DOMAIN / ONLY_VARIANT in ablation.env)

# 3. evaluate every checkpoint + aggregate the comparison report
bash ablations/scripts/eval_all.sh
#    smoke test: MAX_IMAGES=20 bash ablations/scripts/eval_all.sh

# 4. open the report
cat ablations/RESULTS.md
```

### Pointing at a different env file
```bash
ABLATION_ENV=/path/to/custom.env bash ablations/scripts/setup_ablations.sh
```

### Test-set subsampling
`NORMAL_TEST_DIR` typically has thousands of frames; scoring all of them per
ablation is wasteful and slow. `setup_ablations.sh` copies
`TEST_MAX_IMAGES` frames (default 200) into `TEST_NORMAL_SUBSET` using
`random.sample` with `TEST_SAMPLE_SEED` so every evaluator sees the same
subset. Bump `TEST_MAX_IMAGES` if you need tighter FID/KID estimates
(≥ 200 frames is recommended; below that KID is more reliable than FID).

## Output layout

```
outputs/ablations/
├── data/
│   ├── depth_train/        # one-time depth cache
│   ├── depth_test/
│   ├── pairs_baseline/     # depth-aware pairs (5 variants use these)
│   ├── pairs_no_depth/     # cluster-only pairs (NO_DEPTH uses these)
│   └── test_pairs/         # held-out GT for PSNR/SSIM/LPIPS
├── under/
│   ├── BASELINE/
│   │   ├── checkpoints/underexposed/best.pt
│   │   └── eval/
│   │       ├── generated/            # all generated PNGs (used by FID/KID)
│   │       ├── per_image.jsonl
│   │       └── summary.json
│   ├── NO_DEPTH/...
│   └── ...
├── over/
│   ├── BASELINE/...
│   └── ...
└── logs/                              # one file per train/eval run
```

## What the ablation evaluator does NOT do

`evaluate_ablations.py` runs **minimal inference**: DDIM denoising +
texture reinjection only. It deliberately skips the production
post-processing (`analytic_floor`, `focal_blend`, `chroma_attenuation`,
`l_localize`) because most of those rely on depth and would unfairly
penalise the `NO_DEPTH` variant. The numbers reflect what each model
actually learned to output.

If you want the *visual* quality of a variant under the full production
post-processing, run the existing `diffusion_inference.py` against that
checkpoint as a separate step — the ablation framework doesn't touch
that script.

## Caveats

- KID is reported even on small test sets, but FID is unreliable below
  ~500 generated frames. If your test split is smaller, trust KID more
  than FID.
- The LAB-full model's AB output is *not* clipped against the source AB
  during inference — color drift is exactly what `delta_E00_AB` is
  designed to surface. Don't be alarmed by a non-zero number; that's
  the point.
- The depth-mask Pearson metric uses `gamma=2.0` to build the expected
  mask, matching the underexposed pair generator's default `gamma_under`.
  The metric is direction-symmetric: for `overexposed` it correlates
  against `depth^2` instead.
