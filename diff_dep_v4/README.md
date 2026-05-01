# YCLDI — YCbCr-Conditioned Luminance Diffusion

A diffusion-based image-to-image pipeline that, given a **normal** endoscopy frame,
generates a **focalized overexposed** and a **focalized underexposed** version of
the same frame — yielding a synthetic paired dataset
`(normal → overexposed, underexposed)`.

The pipeline is designed around five hard requirements drawn from the failure
modes of prior attempts: outputs must stay **sharp**, exposure changes must be
**strong** and **focal** (not dispersed), underexposure must be **true near-black**
(not brownish), and over- and under-exposure must appear in **separate** images.
The architecture maps each failure mode to a specific design choice — see
[Why this design](#why-this-design).

---

## Quick start

### 1. Install dependencies

The pinned set is compatible with `torchvision==0.12.0+cu113`:

```bash
pip install -r requirements.txt
```

Optional but recommended on the DGX:

```bash
pip install xformers
```

### 2. Configure paths

Edit `config.yaml` and set the three input directories and the output root:

```yaml
paths:
  normal_dir: /data/endoscopy/normal
  over_dir:   /data/endoscopy/overexposed
  under_dir:  /data/endoscopy/underexposed

output:
  root: ./runs/ycldi_v1
```

Everything else (checkpoints, sample grids, logs, generated images) is derived
as `<output.root>/{checkpoints,samples,logs,generated}`.

### 3. Train

```bash
# DGX (single 16 GB GPU)
python train.py --config config.yaml

# Resume an interrupted run from the last checkpoint
python train.py --config config.yaml --resume

# 4 GB local smoke test: drop batch size, image size, enable checkpointing
python train.py --config config.yaml \
    --batch_size 2 --image_size 128 --gradient_checkpointing
```

The trainer:
- saves only `checkpoint-last.pt` (always overwritten) and `checkpoint-best.pt` (on val-loss improvement);
- writes a per-epoch sample grid to `<root>/samples/`;
- streams metrics to W&B if `wandb.enabled: true` in the config.

### 4. Generate

After training, run inference on a directory (or single file) of normal frames:

```bash
python infer.py --config config.yaml --input /data/endoscopy/normal --ckpt best
```

Useful overrides:

```bash
python infer.py --config config.yaml --input frames/ --output gen/ \
    --cfg_scale 7.5    # stronger effect (tradeoff: less faithful to input)
    --strength 0.6     # more SDEdit noise = more change, less preservation
    --steps 50         # DDIM inference steps
```

For each input frame the script writes three files to the output directory:

```
<stem>_normal.png         # cropped + resized original (paired-dataset anchor)
<stem>_overexposed.png    # focalized synthetic overexposure
<stem>_underexposed.png   # focalized synthetic underexposure
```

If `inference.save_debug: true` it also dumps the pseudo-depth map and the
two vulnerability masks under `<output>/debug/` for visual inspection.

---

## Why this design

A single class-conditional UNet that diffuses **only the Y (luminance) channel**
in YCbCr space, trained on the unpaired union of all three classes, then sampled
with **SDEdit + step-wise RePaint masking** under **classifier-free guidance**
that uses *normal* as the negative class. Every component below maps to a
specific prior failure:

| Prior failure | Design choice that addresses it |
|---|---|
| Blurry outputs | Generative diffusion sampling (no regression-to-the-mean as in deterministic I2I); SDEdit preserves high-frequency detail because the input itself supplies the structure to be denoised; step-wise RePaint keeps the unmasked region bit-identical to the original. |
| Weak / imperceptible exposure changes | (a) AdaGN class conditioning at every ResNet block via `resnet_time_scale_shift="scale_shift"`, (b) **contrastive CFG** with `negative_class=normal`: the network is pushed *away from normal* AND *toward the target exposure*, which amplifies the difference far more than null-class CFG, (c) tunable `cfg_scale` and `sdedit_strength`. |
| Brownish underexposure (chrominance drift) | Diffusion runs **only on Y**; Cb/Cr are passed through unchanged. True near-black is reachable by construction; no color shift is possible. |
| Dispersed / non-focal effect | A **vulnerability mask** computed from luminance × pseudo-depth is enforced as a hard RePaint constraint at every denoising step, plus a final blend `y = mask·y_gen + (1-mask)·y_in`. Outside the focal cluster the output is mathematically identical to the input. |
| Over- and under-exposure in the same image | The two effects are produced by **separate inference passes**, each with its own class label and its own mask. Contrastive CFG further suppresses the unwanted direction within a single pass. |

### Architectural details

- **Backbone** — `diffusers.UNet2DModel`, 1-channel in/out (Y only), 5 levels,
  channels `(128, 128, 256, 256, 512)`, attention placed only at the deepest
  non-bottleneck level (16×16 down, 32×32 up) plus the mid block. This keeps
  attention VRAM bounded and fits a batch of 8 at 256×256 in fp16 on a 16 GB GPU.
- **Conditioning** — `num_class_embeds=4` for `{normal, over, under, null}`;
  `resnet_time_scale_shift="scale_shift"` enables AdaGN-style modulation, so
  `(timestep + class)` produces a per-block scale/shift on GroupNorm features.
- **Schedules** — `DDPMScheduler(scaled_linear, T=1000)` for training,
  `DDIMScheduler` (50 steps) for inference. `alphas_cumprod` and
  `final_alpha_cumprod` are explicitly moved to the GPU at construction.
- **Training objective** — vanilla ε-prediction MSE on Y. We deliberately do
  **not** add auxiliary losses (L1 / VGG / gradient / histogram). Those were
  compensating for regression-to-mean of the prior deterministic approach;
  diffusion does not need them and they bias the model toward dataset
  averages, which is exactly the failure mode we are trying to escape.
- **Class dropout 10 %** — replaces the label with `CLASS_NULL` 10 % of the
  time, enabling both standard CFG (`negative_class=null`) and the contrastive
  variant (`negative_class=normal`) at inference. Default is contrastive
  because it produces a substantially stronger effect on this dataset.
- **EMA decay 0.9999** — sampling always uses the EMA weights; the live model
  keeps receiving gradients.
- **AMP (fp16)** with an fp32 cast at the loss boundary — avoids the NaN
  spikes seen in the earlier deterministic trainer.

### Inference algorithm (per frame, per class)

1. Load RGB → square center-crop on the shorter side → bicubic resize to
   `image_size`.
2. Split YCbCr; keep Y in `[-1, 1]` for diffusion, Cb/Cr in `[0, 1]` set aside.
3. Compute pseudo-depth from RGB luminance (dark = far, bright = close).
   Endoscopy is co-located with its own light source, so this proxy is
   physically grounded. Drop in MiDaS by replacing `depth_from_rgb` in
   `mask.py` if you want metric depth.
4. Build the focal masks:
   `mask_over ∝ Y · (1 − depth)`, `mask_under ∝ (1 − Y) · depth`,
   Gaussian-blurred and percentile-normalized, then `**gamma` to sharpen.
5. SDEdit: noise Y to `t = strength · T`.
6. DDIM denoise loop. At each step:
    - `ε_pos = unet(x, t, target_class)`
    - `ε_neg = unet(x, t, negative_class)`     # "normal" by default
    - `ε_cfg = ε_neg + cfg_scale · (ε_pos − ε_neg)`
    - `x_denoised = scheduler.step(ε_cfg, t, x).prev_sample`
    - RePaint: `x = mask · x_denoised + (1 − mask) · noise(y_in, t_next)`
7. Final blend `y_final = mask · y_gen + (1 − mask) · y_in`.
8. Recombine `Y_final` with the original `Cb/Cr` → save as PNG.

### Tuning knobs (all in `config.yaml` → `inference:`)

| Knob | Effect | Default |
|---|---|---|
| `cfg_scale` | Larger → stronger, more saturated effect; too large → artifacts | 6.0 |
| `cfg_negative_class` | `normal` (contrastive, stronger) vs `null` (standard CFG, milder) | `normal` |
| `sdedit_strength` | Larger → more change, less faithful to the input | 0.55 |
| `mask_gamma` | `>1` sharpens the focal cluster (less halo) | 1.5 |
| `mask_blend_strength` | Caps the mask; lower = softer focal blend | 0.9 |
| `ddim_steps` | More steps → smoother but slower | 50 |

---

## File layout

```
ycldi/
├── config.yaml         # single source of truth
├── requirements.txt
├── colorspace.py       # RGB <-> YCbCr (BT.601), device-safe
├── dataset.py          # YCbCrEndoscopyDataset (Y, label) pairs from 3 dirs
├── ema.py              # EMA wrapper
├── checkpoint.py       # atomic last/best checkpoint manager
├── mask.py             # pseudo-depth + focal vulnerability masks
├── model.py            # 1-channel class-conditional UNet (AdaGN)
├── utils.py            # seeding, config, output paths, sample grids
├── train.py            # ε-prediction MSE training with EMA + CFG dropout
└── infer.py            # SDEdit + RePaint + contrastive CFG generation
```

---

## Troubleshooting

- **`alphas_cumprod` device mismatch** — both schedulers move
  `alphas_cumprod` (and `final_alpha_cumprod` for DDIM) to the GPU at
  construction time. If you replace the scheduler, mirror that.
- **OOM on the 4 GB local card** — drop `data.image_size` to 128 and
  `train.batch_size` to 1–2; enable `train.gradient_checkpointing: true`.
  Smoke test only — do full training on the DGX.
- **Effect is too weak** — raise `cfg_scale` (try 7.5–9), keep
  `cfg_negative_class: normal`, raise `sdedit_strength` toward 0.65.
- **Effect bleeds outside the cluster** — raise `mask_gamma` (e.g., 2.0),
  raise `mask_blend_strength` toward 1.0, or lower `inference.blur_sigma`
  in `mask.py` (currently 16.0; smaller = tighter clusters).
- **Underexposure looks brown again** — sanity-check that nothing in the
  pipeline is touching Cb/Cr. The whole point of Y-only diffusion is that
  the chrominance is a passthrough; if you see a hue shift it is a bug, not
  a tuning issue.
- **`save_debug: true`** dumps Y, depth, and both masks alongside each
  generated triple — your fastest debugging tool when results are off.
