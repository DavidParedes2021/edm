# YCLDI -- YCbCr-Conditioned Luminance Diffusion (Underexposure)

A diffusion-based image-to-image pipeline that takes a "normal" endoscopy
frame and synthesizes its underexposed counterpart by **operating only on
the Y (luminance) channel of YCbCr**, then re-attaching the original Cb/Cr.

## Why luminance-only

Under BT.601, `Cb` and `Cr` are *differences* between R, G, B. When
illumination scales R, G, B roughly proportionally -- which is the case for
white-LED endoscopy -- Y drops but Cb/Cr stay fixed. Substituting only the
generated Y back into the original Cb/Cr therefore yields:

- Sharp output (no diffusion-induced texture drift in chroma).
- Identical color balance and saturation as the source.
- A 1-channel diffusion problem -> smaller model, faster training,
  comfortably fits a 16 GB GPU.

The trade-off: real underexposed sensors do exhibit chroma-side artifacts
(noise floor, demosaicing). For *clean synthetic labels* used to supervise a
downstream illumination-correction model, that is a feature -- you get
exposure changes only, with everything else held fixed.

## Layout

```
ycldi_under/
├── config.yaml      single source of truth
├── color.py         RGB <-> YCbCr (BT.601 full-range, fp32-clean inverse)
├── data.py          UnpairedYCbCrDataset, balanced loader
├── model.py         ConditionalYUNet (~31 M params)
├── diffusion.py     DDPMScheduler + DDIM sampler + CFG closure
├── losses.py        eps-MSE + L1 + grad-diff + VGG + sorted-W1
├── checkpoint.py    atomic CheckpointManager + EMA
├── train.py         training entry point
└── infer.py         batch translation entry point
```

## Setup

The project is pinned to your environment (diffusers 0.14.0, torchvision
0.12.0+cu113). The only direct extra is PyYAML and tqdm, which are already
transitively pulled in by the pinned stack.

```bash
pip install pyyaml tqdm
```

Edit `config.yaml` to point `data.normal_dir` and `data.underexposed_dir`
at your folders.

## Train

```bash
python -m ycldi_under.train --config config.yaml
# resume from checkpoint-last
python -m ycldi_under.train --config config.yaml --resume
```

Sample grids land in `runs/ycldi_under/samples/epoch_*.png` every
`output.sample_every_epochs` epochs. Each grid has three rows:

```
row 1: original normal RGB
row 2: generated UNDEREXPOSED (target class = 1)
row 3: generated NORMAL        (target class = 0; sanity check identity)
```

If row 3 doesn't reconstruct row 1, the model is over-conditioning on the
class label or under-conditioning on `y_cond`.

## Inference -- build a paired dataset

```bash
python -m ycldi_under.infer \
    --checkpoint runs/ycldi_under/checkpoints/checkpoint-best/state.pt \
    --input  /data/endoscopy/normal \
    --output /data/endoscopy/synth_under \
    --target under \
    --cfg-scale 2.5 \
    --steps 50 \
    --save-pairs
```

`--save-pairs` writes side-by-side `input | output` images next to each
generated frame for manual QA. Output filenames mirror the input tree, so
you get a clean parallel dataset.

## Knobs that matter

| Knob | Where | Effect |
| --- | --- | --- |
| `cfg_scale` | inference | Higher = more pronounced underexposure but more chance of artifacts in bright regions. Start at 2.0-3.0. |
| `lambda_grad` | training | If outputs look soft, raise toward 0.2-0.3. |
| `lambda_hist` | training | If the *distribution* of darkness doesn't match real underexposed images, raise toward 0.1-0.2. |
| Condition perturbation in `train.perturb_condition` | training | The random gamma range is the most important hyperparameter. If the model ignores the class label (under == normal), the gamma range is too narrow. If reconstructions of class=normal drift, it's too wide. |
| `attention_resolutions` | model | Adding `[32, 16]` means a deeper UNet (channel_mults `[1,2,3,4,4]`). Helps quality, costs VRAM. |

## Why self-conditional + perturbed `y_cond` for unpaired data

The model is trained as: *given clean Y target and a perturbed copy as
condition, predict the noise added to the target, conditioned on the
target's class label*.

- Random gamma in the perturbation breaks the identity shortcut -- the
  model cannot read the target's exposure level off `y_cond`.
- The class label becomes the only reliable source of "which exposure
  regime", which is exactly the lever we want at inference.
- At inference, `y_cond` is a clean normal Y and the class label is set to
  "under". CFG amplifies the class-driven shift while `y_cond` anchors
  structure. Because the perturbation distribution at training time
  *includes* the identity (gamma=1, blur=0), clean inputs are in
  distribution.

## Validating the run

Beyond the visual grids, a useful quick check after a few hundred steps:

```python
# In a notebook
from infer import translate_batch, build_model_from_ckpt
import torch
state = torch.load('runs/ycldi_under/checkpoints/checkpoint-last/state.pt')
model, sched = build_model_from_ckpt(state, 'cuda')
# translate a batch with cfg_scale=1.0 (no guidance) -- output should look
# nearly identical to the input (the conditional branch alone). With
# cfg_scale=3.0, the underexposure should be visible. The gap between
# these two is what CFG buys you.
```
