# Illumination Diffusion — LDM I2I Pipeline

Generates synthetic **over-exposed** and **under-exposed** frames from Normal
input frames using a **Latent Diffusion Model** (LDM) conditioned on a
**continuous EV scalar** (stops).

---

## File structure

```
illumination_diffusion/
├── config.py              # All hyperparameters, auto-adapts to GPU tier
├── dataset.py             # IlluminationDataset with histogram pseudo-pairing
├── model.py               # EVEmbedding, UNet builder, LPIPS/SSIM/Hist losses
├── train.py               # Training loop
├── inference.py           # Generate synthetic pairs from a trained checkpoint
├── evaluate.py            # FID / LPIPS / SSIM / histogram metrics
├── generate_dummy_data.py # Create synthetic test images (no real data needed)
├── requirements.txt
└── data/
    ├── normal/
    ├── overexposed/
    └── underexposed/
```

---

## Quick start — laptop smoke test (4 GB GPU)

```bash
# 1. Install dependencies
pip install torch==1.12.1+cu113 torchvision==0.12.0+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

# 2. Generate dummy data (no real images needed for smoke test)
python generate_dummy_data.py --n 20 --size 128

# 3. Run smoke-test training (200 steps, batch=1, 128×128)
python train.py --smoke --no-vae

# With VAE (downloads ~335 MB on first run):
python train.py --smoke
```

---

## Full training on DGX (16 GB GPU slice)

```bash
# Pull from git, then:
python generate_dummy_data.py  # or replace data/ with your real dataset

# Training auto-detects 16 GB and uses:
#   image_size=512, batch=2, grad_accum=4, fp16, gradient_checkpointing
python train.py

# Resume from checkpoint:
python train.py --resume checkpoints/step_0010000
```

---

## Data layout

Place your images in:
- `data/normal/`       — Domain A (correctly exposed)
- `data/overexposed/`  — Domain B (blown highlights)
- `data/underexposed/` — Domain C (dark + noisy)

Images do **not** need to be paired. The pipeline uses histogram
specification to generate geometry-consistent pseudo-pairs automatically.

---

## Inference — generate a synthetic dataset

```bash
# Overexposed (+2.0 EV):
python inference.py \
    --input     data/normal \
    --output    synthetic/overexposed \
    --ev        2.0 \
    --checkpoint checkpoints/step_0050000

# Underexposed (−2.5 EV):
python inference.py \
    --input     data/normal \
    --output    synthetic/underexposed \
    --ev        -2.5 \
    --checkpoint checkpoints/step_0050000

# Sweep multiple EV values at once:
python inference.py \
    --input     data/normal \
    --output    synthetic \
    --ev-sweep  "2.0,2.5,3.0,-1.5,-2.0,-2.5" \
    --checkpoint checkpoints/step_0050000
```

---

## Evaluation

```bash
python evaluate.py \
    --generated  synthetic/overexposed \
    --normal     data/normal \
    --real       data/overexposed \
    --domain     over
```

Outputs: FID, LPIPS (geometry), 1−SSIM (structure), ΔHistogram, exposure
accuracy %.

---

## GPU tier auto-detection

| Tier  | VRAM  | image_size | batch | grad_accum | eff. batch |
|-------|-------|-----------|-------|-----------|-----------|
| smoke | ≤4 GB | 128       | 1     | 2         | 2         |
| low   | ≤8 GB | 256       | 2     | 4         | 8         |
| mid   | ≤16 GB| 512       | 2     | 4         | 8         |
| high  | >16 GB| 512       | 8     | 1         | 8         |

Force smoke mode for laptop testing: `python train.py --smoke`

---

## Key design decisions (fixes for your prior issues)

| Problem | Fix |
|---|---|
| **Blurry outputs** | Moved to latent-space diffusion (frozen SD VAE). The VAE decoder reconstructs fine detail the pixel-space UNet cannot. |
| **Imperceptible exposure shift** | Replaced binary 2-class embedding with a **continuous EV sinusoidal scalar embedding** — the model now receives a precise target EV (e.g. +2.5 stops), not just a 1-bit "overexpose" flag. |
| **Device mismatch errors** | Every module has `device` pinned in its constructor. `Accelerator.prepare()` wraps the UNet; VAE/losses are explicitly `.to(device)`. |
| **OOM errors** | Gradient checkpointing + mixed precision + gradient accumulation + VAE micro-batching + tier-adaptive batch/resolution. |

---

## WandB

Set `USE_WANDB = True` in `config.py` and optionally `wandb login` before
training. Pass `--no-wandb` to disable at run time.
