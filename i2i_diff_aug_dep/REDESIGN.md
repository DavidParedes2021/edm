# Depth-Aware Paired Luminance Diffusion — Redesign

## 1. Diagnosis of the current pipeline

The current approach has four interacting problems, three of which share the same fix (depth conditioning + chroma-aware post-processing).

| # | Symptom | Root cause |
|---|---|---|
| 1 | Underexposure looks brownish, not black | AB channels are kept untouched while L drops. In LAB, `(L=15, A=+20, B=+25)` is a dark saturated brown — that is the chromatically correct mapping. Real low-light imagery desaturates (rod vision + sensor SNR collapse + Purkinje shift). No component of the pipeline models this. |
| 2 | Unnatural shapes | `find_seed_clusters` builds the exposure mask from luminance *percentiles*. The mask follows whatever is currently bright/dark in the 2D projection. In endoscopy, illumination is physically governed by distance to the scope tip (inverse square), which is a property of scene *geometry*, not pixel intensity. The cluster mask is decoupled from the cause. |
| 3 | Underexposure is too weak | Three stacked causes: (a) L1/MSE regresses toward the dataset mean, and most pixels in the target are mid-L; (b) `apply_exposure_shift` only hits `clip_target_under` where `exposure_map > 0.85`, which rarely covers large regions; (c) the high-frequency texture re-injection adds visible wiggle inside near-black regions, so the eye reads them as "dark textured" rather than black. |
| 4 | 1100 images × class label alone is not enough signal | With only a binary domain label, the UNet has to rediscover the depth→illumination relationship from 1100 frames. Classifier-free guidance can't help because there isn't enough variance between the two "classes" as currently labelled. |

The single lever that moves three of these four problems is **an explicit depth map fed to the pipeline at every stage** — during pair generation, as a UNet input channel, and as a modulator for chroma attenuation at inference.

## 2. Recommended components

### Depth estimator: Depth Anything V2 (Small)
Checkpoint: `depth-anything/Depth-Anything-V2-Small-hf` (~25 M params).

Chosen because:
- **Zero-shot monocular** — trained on 595K labeled + 62M unlabeled images; works well on medical imagery despite not being trained on it.
- **Relative inverse-depth output** — returns a dense map where larger values = closer. After min-max normalization this maps cleanly to "1 = near tissue candidate for overexposure, 0 = far cavity candidate for underexposure". Absolute metric depth is not needed.
- **Small variant is sufficient** for your 1100 images — inference ~40 ms/frame on a 16 GB GPU, ~2 min total for the whole dataset. No reason to pay for Base or Large.
- **Stable HuggingFace integration** via `transformers.pipeline("depth-estimation", ...)` — no custom repo cloning.

Alternatives considered:
- **MiDaS v3.1** — older, less robust on medical. Use only as fallback.
- **HADepth** (mentioned in the brief) — haze-aware, optimized for outdoor transparency effects. Not a natural fit for opaque mucosa.
- **EndoDAC / Endo-FM** — endoscopy-specific. Better domain match in principle, but harder to install, and our use case only needs relative depth, not metric accuracy. Use if you see residual artifacts that you can trace to depth-estimator errors.

Depth maps are **precomputed once** and cached to disk. Don't run depth estimation inside the training loop — it wastes GPU.

### Diffusion backbone
Keep `diffusers.UNet2DModel`. Only change:
- `in_channels = 3` (was 2) — concat depth as the third channel.
- Keep `num_class_embeds = 2` for domain.

You don't need ControlNet for 1100 images. ControlNet's value is leveraging a large frozen base model; here you're training from scratch on a narrow domain, and doubling the parameter count with a ControlNet branch would overfit. Plain concatenation is the right level of mechanism.

Optional enhancement (documented in the train script): a small depth-statistics MLP whose output is added to the time embedding inside the UNet. This is a lightweight FiLM-style global modulation. Off by default; enable via `model.depth_film: true` in the config.

### Conditioning strategy
- **Per-pixel**: concat depth as input channel (handled by `in_channels=3`).
- **Per-image global**: depth summary statistics (`[mean, std, p10, p90]`) projected and added to the class embedding.
- **Domain**: class embedding (over=0, under=1).

### Chroma model
Post-hoc, deterministic, physically motivated. Implemented in `chroma_attenuation.py`. Detailed in §5.

## 3. Pipeline — step by step

```
┌─────────────────────────────────────────────────────────────────┐
│ ONE-TIME PREPROCESSING (run once)                               │
├─────────────────────────────────────────────────────────────────┤
│ 1. depth_estimator.py                                           │
│      for each normal frame:                                     │
│          D = DepthAnythingV2(img)                               │
│          save D as float16 .npy at full res                     │
│                                                                 │
│ 2. generate_pairs.py  (now depth-aware)                         │
│      for each normal frame × N variants:                        │
│          load image + depth                                     │
│          L_target = depth_aware_augment(img, depth,             │
│                       mode, strength, shift_magnitude)          │
│          save (normal_LAB, target_L, depth) tuple               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ TRAINING (diffusion_train.py)                                   │
├─────────────────────────────────────────────────────────────────┤
│ For each step:                                                  │
│     x = target_L (normalised to [-1,1])                         │
│     c = (source_L, depth, class_label)                          │
│     t ~ Uniform(1..T)                                           │
│     ε ~ N(0, I)                                                 │
│     x_t = sqrt(α̅_t) x + sqrt(1-α̅_t) ε                          │
│     model_input = cat([x_t, source_L, depth], dim=1) → (B,3,H,W)│
│     ε_pred = UNet(model_input, t, class=class_label)            │
│     x0_hat = (x_t - sqrt(1-α̅_t) ε_pred) / sqrt(α̅_t)            │
│                                                                 │
│     Loss = mse(ε_pred, ε)                        [base]         │
│          + λ_edge · sobel_loss(x0_hat, x)        [sharpness]    │
│          + λ_l1 · l1(x0_hat, x)                  [pixel]        │
│          + λ_dark · darkness_weighted_l1(x0_hat, x, depth)      │
│          + λ_dg · depth_gradient_consistency(x0_hat - x, depth) │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ INFERENCE (diffusion_inference.py)                              │
├─────────────────────────────────────────────────────────────────┤
│ 1. Load normal RGB, compute or load depth                       │
│ 2. Run DDIM: x_T → x_0 = L_pred (at 256×256)                    │
│ 3. Upsample L_pred to original resolution                       │
│ 4. Texture reinjection (unchanged):                             │
│      L_final = low_pass(L_pred) + high_pass(L_orig)             │
│ 5. Chroma attenuation (NEW):                                    │
│      ΔL = L_final - L_orig                                      │
│      s = sat_scale(L_final, ΔL, depth, mode)                    │
│      A' = A × s                                                 │
│      B' = B × s + cool_shift(L_final, mode)                     │
│ 6. LAB(L_final, A', B') → RGB                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 4. Depth-aware augmentation (pair generation)

The new `depth_augment.py` replaces the cluster-based seed mask with a **depth-driven exposure mask**, retaining cluster-based variation as a secondary modulation for stochasticity.

### Overexposed target
```
mask_over = normalize(depth) ** γ_over
# γ_over = 1.5 concentrates effect on nearest regions
# Retain 15% weight from the luminance-cluster mask for texture-locked highlights
mask_over = 0.85 * mask_over + 0.15 * cluster_mask_over
```

### Underexposed target (this is where the changes matter most)
```
far_mask = (1 - normalize(depth)) ** γ_under
# γ_under = 2.0 — strongly concentrates darkness on far regions
mask_under = 0.85 * far_mask + 0.15 * cluster_mask_under

# Aggressive shift:
L_target = L_low - shift_magnitude * mask_under
# For very-far pixels (depth < 0.1), force hard clip to L ≤ 5
core = far_mask > 0.8
L_target[core] = clip_target_under   # 0
```

### Resolution-invariance
Depth is spatially aligned with the source image. All masks are computed at full source resolution, then resized to 256×256 for training. At inference, depth is resized to 256×256 for the UNet and kept at full resolution for the chroma-attenuation post-processing.

## 5. Fixing "brownish" underexposure — chroma attenuation

**The key insight.** `(L=10, A=+20, B=+25)` is physically a dark brown. To get a dark *black*, you need `(L=10, A≈0, B≈0)`. Real low-light images look that way because of three independent physical effects, all of which collapse chroma as luminance drops:

1. **Rod vision is achromatic.** Below ~3 cd/m² the rods take over from cones; humans literally can't see color in dim light (the Purkinje effect).
2. **Sensor shot noise collapses chroma SNR.** In photon-starved regions, the per-channel signal goes below the read noise, and any color information is dominated by noise — which denoising/ISP algorithms then smooth away.
3. **Slight blue-shift in scotopic vision.** Not critical but adds realism.

The fix is a deterministic post-process applied *after* the diffusion produces `L_final`. Given original chroma `(A, B)`, new luminance `L_new`, original luminance `L_orig`, and depth `D`:

```
# Saturation multiplier: 1.0 at normal L, fades toward 0 as L drops
# Smooth C¹ transition with knee at L=40
s(L_new) = clip( (L_new - 5) / 35 , 0.1, 1.0 )

# Depth modulation: far regions desaturate more (they're in shadow, noise-limited)
s *= 1 - 0.3 * (1 - D)    # D=0 (far) cuts another 30% of chroma

# Overexposure has its own issue: highlight clipping desaturates
# (saturated light bleaches the color channels too)
if mode == "overexposed":
    overshoot = clip((L_new - 90) / 10, 0, 1)
    s *= 1 - 0.6 * overshoot

A_new = A * s
B_new = B * s

# Subtle cool shift in very dark pixels (Purkinje)
B_shift = -1.5 * clip((20 - L_new) / 20, 0, 1)    # up to -1.5 in B
B_new = B_new + B_shift
```

This is deterministic, fast, and applied at full resolution — no model needed. Tune the constants in the config.

## 6. Architectural options for depth conditioning — what I picked and why

| Option | Mechanism | Verdict |
|---|---|---|
| **A. Channel concatenation** (chosen) | Input = `cat(noisy_L, source_L, depth)` → 3 channels | Works by default, costs ~30 extra params in the first conv. Proven in Palette, InstructPix2Pix, and many image-to-image diffusion papers. |
| B. ControlNet branch | Parallel encoder copies, depth feeds the copy, features injected into decoder | Designed for reusing large frozen models. You're training from scratch on 1100 images — doubles parameter count, strictly worse for your sample size. |
| C. Cross-attention | Depth patches as tokens, attended by the UNet | Expensive at 256×256; rarely beats concat for dense, pixel-aligned conditions. |
| **D. FiLM on global depth stats** (optional, chosen as add-on) | 4-D depth descriptor → MLP → added to time embedding | Cheap (~50 K params), gives the model a global "how much depth variance is there" signal orthogonal to the per-pixel channel. Off by default, toggleable via config. |

## 7. Loss redesign

Current losses: `mse(ε)` + `l1(x0_hat, x)` + `sobel(x0_hat, x)`.

Add two, and do not add perceptual (VGG) loss — it pulls toward "natural-looking" texture, which here actively fights true underexposure.

### Darkness-weighted L1 (`λ_dark = 0.15`)
Upweight pixel error where the *target* is dark. This counteracts the mean-regression bias.

```
w = 1 + α · exp(-target_L / τ)    # α=3.0, τ=0.3 (on [-1,1] scale)
loss_dark = mean(w · |x0_hat - target|)
```

At the extreme, a pixel that should be pure black (target ≈ -1) gets ~3× the loss of a mid-grey pixel.

### Depth-gradient consistency (`λ_dg = 0.05`)
The exposure shift `ΔL = x0_hat - source` should vary smoothly along depth isocontours and sharply across depth discontinuities. Penalize inconsistency via gradient alignment:

```
∇ΔL   = Sobel(x0_hat - source_L)
∇D    = Sobel(depth)
# For underexposed: gradient of ΔL should point roughly toward increasing depth
# (magnitudes) — i.e., |∇ΔL| should correlate with |∇D|.
loss_dg = mean(relu( |∇D|_normalized - |∇ΔL|_normalized - margin ))
```

This is a soft, one-sided hinge — only penalizes when the exposure field is too smooth at a depth edge. It does not force the shift to match depth exactly, only to respect its discontinuities.

### What to remove
- **No VGG perceptual loss.** VGG was trained on natural images and rewards natural-looking texture. Its gradient in pure-black regions pulls toward mid-grey. Don't add it.
- **No histogram-matching loss.** Tempting but it compares marginal L distributions globally, which is orthogonal to where the darkness appears. The depth-gradient loss handles spatial placement better.

## 8. Augmentation & normalization hygiene

1. **Joint spatial augmentation** — flips and rotations must be applied identically to `source_L`, `target_L`, and `depth`. Currently done for source/target; add depth. Already handled in the updated `diffusion_dataset.py`.
2. **Depth normalization** — per-image min-max to [0, 1] at save time. Do NOT standardize across the dataset; relative depth is what matters, and endoscopy frames have very different depth ranges.
3. **Gamma jitter on source_L** — small random gamma (0.9–1.1) applied identically to source and target at training time. Simulates light-source intensity variation across frames.
4. **Depth noise** (optional) — add small Gaussian noise to depth at training time (σ=0.02). Makes the model robust to depth-estimator errors at inference.

## 9. Inference-time details

### DDIM steps
Keep 50. More steps do not help at this model size.

### Texture σ
Unchanged: `3.0 × max(H, W) / 512`.

### Chroma constants (config)
```yaml
chroma:
  sat_knee: 5.0           # L value at which chroma = 0.1
  sat_full: 40.0          # L value at which chroma = 1.0
  sat_floor: 0.1          # minimum chroma multiplier (never fully 0 to avoid banding)
  depth_chroma_boost: 0.3 # additional desat for far regions
  highlight_desat: 0.6    # chroma reduction for clipped highlights
  purkinje_b_shift: 1.5   # max blue shift in dark regions
```

All constants are tuneable from the YAML without retraining.

### Do not forget
The AB channels **bypass the UNet entirely**. They come from the original image, are modified by the chroma attenuator, and are then recombined with `L_final`. The UNet never sees or touches them.

## 10. File map — what changes

| File | Change |
|---|---|
| `depth_estimator.py` | **NEW** — precompute depth maps once, save as `.npy` |
| `chroma_attenuation.py` | **NEW** — the brownish→black fix |
| `depth_augment.py` | **NEW** — depth-aware augmentation (wraps `exposure_augment`) |
| `generate_pairs.py` | UPDATED — uses `depth_augment`, saves depth alongside pairs |
| `diffusion_dataset.py` | UPDATED — loads depth, applies it as third channel |
| `diffusion_train.py` | UPDATED — 3-channel input, new losses, optional FiLM |
| `diffusion_inference.py` | UPDATED — loads/computes depth, applies chroma attenuation |
| `diffusion_config.yaml` | UPDATED — `in_channels: 3`, chroma params, new loss weights |
| `losses.py` | UPDATED — adds `DarknessWeightedL1Loss`, `DepthGradientConsistencyLoss` |
| `exposure_augment.py` | UNCHANGED — still the deterministic baseline, called by `depth_augment` |
| `evaluate.py` | UNCHANGED |
| `dataset.py` | UNCHANGED — legacy, kept for compatibility |

## 11. Minimal run plan

```bash
# 1. Install new dependency
pip install transformers accelerate

# 2. Precompute depth maps for all normal frames  (one-time, ~2 min)
python depth_estimator.py \
    --input_dir ./data/normal \
    --output_dir ./data/depth \
    --model_id depth-anything/Depth-Anything-V2-Small-hf

python depth_estimator.py --input_dir "../../../../data/datasets/edm_consolidated_dataset/consolidated_classified_dataset/normal_frames" --output_dir "../../../../projects/i2i_diff_aug_dep/outputs/depts"  --model_id depth-anything/Depth-Anything-V2-Small-hf

# 3. Regenerate paired training data (now depth-aware)
python generate_pairs.py \
    --normal_dir ./data/normal \
    --depth_dir ./data/depth \
    --output_dir ./data/pairs \
    --num_variants 3

python generate_pairs.py --normal_dir "../../../data/datasets/edm_consolidated_dataset/consolidated_classified_dataset/normal_frames" --depth_dir "../../../projects/i2i_diff_aug_dep/outputs/depts" --output_dir "../../../projects/i2i_diff_aug_dep/outputs/pairs" --num_variants 3

# 4. Train (config already points to the new pairs dir)
python diffusion_train.py --config diffusion_config.yaml

# 5. Generate
python diffusion_inference.py \
    --config diffusion_config.yaml \
    --checkpoint ./output_diffusion/checkpoints/best.pt
```

## 12. Expected improvements vs. current

| Metric | Current (measured/reported) | Expected |
|---|---|---|
| Underexposed mean L shift | ~2-5 | ~35-45 |
| Dark-region chroma (mean \|A\|+\|B\|) | ≈ source (30-50) | ≤ 10 in deep regions |
| Texture correlation to source | 0.55 (SDEdit), ~0.95 (current paired) | ≥ 0.95 (unchanged — texture still reinjected) |
| Visible cavity darkening | weak/absent | ≥ 70% of frames |

If you don't see these numbers after training, the most common culprits are: (a) depth normalization drift between preprocessing and inference, (b) `λ_dark` too low, (c) `sat_knee`/`sat_full` in the chroma block not tuned for your tissue.

## 13. Perceptual-realism checklist

1. Dark regions should read as **black or near-black**, not brown. If not — raise `sat_floor` reduction, lower `sat_knee`.
2. Near-cavity pixels should darken *more* than mid-depth pixels. If depth-gradient is inverted — you probably didn't invert Depth Anything's output (it returns inverse depth).
3. No sharp exposure discontinuities inside continuous tissue. If present — `falloff_sigma` too small, or depth map is noisy (try `Depth-Anything-V2-Base-hf`).
4. Highlights in overexposure should **desaturate**, not just brighten. This is handled by the `highlight_desat` term in the chroma block.
5. Texture is identical to source. If not — sigma for texture decomp is wrong, or you accidentally denoised the high-pass.
