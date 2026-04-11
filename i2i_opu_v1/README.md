# Luminance Diffusion — Endoscopy Exposure Augmentation

## 1. Architecture Design

**Approach: Conditional DDPM on the L channel with SDEdit inference.**

The model is a `UNet2DModel` (from `diffusers`) that operates exclusively on the **L channel** of LAB colour space. It is trained with **class conditioning** (overexposed=0, underexposed=1, unconditional=2) and **classifier-free guidance (CFG)** dropout during training.

At inference, we use **SDEdit** (Meng et al., 2022): we take the normal image's L channel, add a controlled amount of noise, then denoise with target-domain conditioning and high CFG scale. The original A and B channels are preserved unchanged, so the final RGB image has identical chrominance to the input.

### Why this works better than alternatives

| Alternative | Problem |
|---|---|
| Full-image diffusion (RGB) | Learns texture/colour changes, produces blurry outputs |
| CycleGAN on RGB | Mode collapse, checkerboard artifacts, weak exposure changes |
| Pix2Pix (paired) | Requires paired data we don't have |
| DDPM from pure noise | Destroys structure → blurry results |

**SDEdit** starts from the original structure and only modifies it, which is fundamentally why it preserves sharpness.

---

## 2. Why Previous Experiments Failed

### Problem 1: Blurry outputs
**Root cause:** Denoising from pure noise (t=T) forces the model to hallucinate all structure. With limited data (~1100 images), the model averages over modes → blur.

**Fix:** SDEdit starts denoising from a partially-noised version of the real image (noise_strength=0.3–0.6). Structure is baked into the latent from the start.

### Problem 2: Overexposed and underexposed outputs looked identical
**Root cause:** Without strong conditioning signal, the model learns a single average transformation. Weak classifier guidance or no guidance at all means the domain signal is ignored.

**Fix:**
1. **Classifier-free guidance** with scale 10–15 amplifies the domain-specific signal.
2. **Training on L channel only** means the model's entire capacity is devoted to learning luminance distributions per domain, not wasting parameters on colour.
3. **Cosine noise schedule** (`squaredcos_cap_v2`) preserves more signal at high timesteps than linear, giving the model better gradient signal for exposure patterns.

---

## 3. Luminance-Only Approach — Validation

**This is sound and is the recommended approach.** Here's why:

- Overexposure and underexposure are **purely luminance phenomena** in endoscopy. The light source creates bright/dark regions; colour (hue, saturation) of tissue doesn't change.
- Operating on L channel only means: **1 input channel instead of 3** → model is 3× smaller, trains faster, uses less VRAM.
- **Zero colour shift by construction** — A and B channels are untouched.
- **Sharpness preservation** — the model only needs to learn one channel's distribution, so it can allocate all capacity to getting luminance patterns right.

The LAB colour space (not YCbCr) is preferred because LAB's L channel is perceptually uniform — a Δ10 in L looks the same magnitude regardless of the base value.

---

## 4. Training Strategy

### Data handling (unpaired)
- We train a standard DDPM on L channels sampled from overexposed and underexposed folders.
- Each sample carries a domain label (0 or 1). With 10% probability the label is replaced with the null token (2) for CFG.
- The model learns: "what does the L channel distribution look like for overexposed endoscopy?" and separately for underexposed.
- Normal images are **only used at inference time** as the SDEdit starting point.

### Balancing
The dataset oversamples the minority domain (281 overexposed vs 817 underexposed) so each epoch sees roughly equal counts.

### Loss functions
| Loss | Weight | Purpose |
|---|---|---|
| MSE (ε-prediction) | 1.0 | Standard diffusion objective |
| Sobel edge | 0.02 | Penalises blurred edges in the x₀ estimate |
| VGG perceptual | 0.05 | Ensures perceptually consistent denoising |

### Other techniques
- **EMA** (decay 0.9999) for stable generation quality
- **Mixed precision** (fp16) for VRAM efficiency
- **Gradient clipping** (norm 1.0) for stability

---

## 5. Evaluation Metrics

| Metric | What it measures | Expected behaviour |
|---|---|---|
| **SSIM** (L channels, normal vs generated) | Structural preservation | Should be 0.7–0.9 (high but not 1.0) |
| **PSNR** (L channels) | Pixel-level deviation | Moderate (20–30 dB) — change is intentional |
| **Brightness shift** | Mean L change | Positive for overexposed, negative for underexposed |
| **Extreme pixel ratio** | Fraction of L > 90 (over) or L < 15 (under) | Should increase significantly vs normal |
| **Histogram KL** | Distribution match with real over/under images | Lower is better |

---

## 6. Running the Pipeline

### Setup
```bash
pip install pyyaml  # only additional dependency beyond requirements
```

### Directory structure
```
project/
├── config.yaml
├── train.py
├── inference.py
├── evaluate.py
├── dataset.py
├── losses.py
├── data/
│   ├── normal/        (504 images)
│   ├── overexposed/   (281 images)
│   └── underexposed/  (817 images)
└── output/
    ├── checkpoints/
    ├── samples/
    └── generated/
```

### Train (RTX 3050, 4 GB VRAM — smoke test)
```bash
# Edit config.yaml:
#   image.size: 128        (reduce for 4 GB)
#   training.batch_size: 2
#   training.epochs: 5     (just to verify it runs)
python train.py --config config.yaml
```

### Train (DGX, 16 GB VRAM — full run)
```bash
# Edit config.yaml:
#   image.size: 256
#   training.batch_size: 16
#   training.epochs: 300
python train.py --config config.yaml
```

### Generate paired dataset
```bash
python inference.py --config config.yaml --checkpoint output/checkpoints/best.pt

# or with custom params:
python inference.py --config config.yaml --checkpoint output/checkpoints/best.pt \
    --domain overexposed --noise_strength 0.5 --guidance_scale 12
```

### Evaluate
```bash
python evaluate.py \
    --normal_dir ./data/normal \
    --generated_dir ./output/generated/overexposed \
    --reference_dir ./data/overexposed
```

---

## 7. Key Hyperparameters to Tune

| Parameter | Default | Effect of increasing |
|---|---|---|
| `noise_strength` | 0.45 | More exposure change, less structure preservation |
| `guidance_scale` | 10.0 | Stronger domain-specific effect |
| `image.size` | 256 | Higher quality, more VRAM |
| `training.epochs` | 300 | Better quality (diminishing returns past ~200) |

**Start with defaults.** If exposure changes are still too subtle, increase `guidance_scale` to 15. If structure is lost, decrease `noise_strength` to 0.3.
