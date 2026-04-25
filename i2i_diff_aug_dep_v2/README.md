Architecture Design — single decisive proposal
The decision in one sentence
Train two specialist diffusion models that perform CIE-LAB L*-only, depth-mask-conditioned inpainting of the focal cluster of a normal frame, leaving the original (a*, b*) and the L* outside the mask bit-identical.

Why this exact design (mapped to your previous failures)
Failure you observed	Root cause	Cure baked into this design
Outputs blurry	Diffusing/encoding RGB → upsampled chroma + VAE-style smearing; loss averaged over the whole image	Diffuse only L*, 1 channel, no VAE, no resampling of chroma. (a*, b*) is never touched
Over- and under-exposed look identical / brownish	A single conditional model with weak class signal collapses both modes; pushing RGB warmward looks "brownish" instead of saturated	Two independent specialists (one per artifact). Class is implicit in which checkpoint you load, not a soft label
Effect dispersed over whole image	No spatial prior — model learns the marginal	A soft binary mask of the focal cluster is concatenated into the UNet input + a mask-weighted MSE (×6 inside) drives gradients only into that cluster
Effect too weak	Loss averaging hides the rare strong-change pixels	Mask-weighted loss + classifier-free guidance at sampling (w=3.5) amplifies the conditional delta
Both artifacts in one image	Single conditional model leaks modes	Two separate models — structurally impossible
Whole-image luminance shift looks blurred	Diffusion changed L globally, including textured regions	Hard substitution at the end: outside the mask, L is bit-identical to the input. Plus RePaint-style noised-known mixing during sampling for a clean boundary
Unnatural shapes	Mask was unconstrained noise	Masks come from the largest connected component of the depth-driven score map → always one focal blob, plus dilation + Gaussian softening for natural edges
Why luminance-only diffusion in LAB
LAB perceptually decorrelates luminance (L*) from chrominance (a*, b*). Endoscopy exposure artifacts are primarily a luminance phenomenon (saturation; shadow). Touching a*, b* is what produced the "brownish overexposure" complaint.
A 1-channel model is small (~50 M params), fits 16 GB DGX with batch 8 + AMP, and starts on a 4 GB RTX 3050 with batch 2 / image 192.
"Sharpness" is not a property the network has to learn — outside the mask, the original L* is reused exactly. Even inside the mask we substitute the predicted L* into the original (a*, b*), so all the high-frequency chromatic detail (vessels, fluid sheen) is preserved.
Why a photometric depth proxy (not MiDaS)
The endoscope's light source travels with the camera. Under inverse-square illumination falloff, brightness is monotone with 1/depth². So a Gaussian-blurred −log L* is a smooth inverse-depth proxy. This:

Avoids dragging in MiDaS / heavy depth nets that may not pip-install cleanly under your torch 1.11.0+cu113 / torchvision 0.12.0 constraint.
Is the very signal that drives natural exposure artifacts in the data, so it's directly causally aligned with the task.
Drives focalized masks: brightest cluster (lowest depth) → overexposure ROI; darkest cluster (highest depth) → underexposure ROI. Always one largest connected component, then dilated and softened.
Why this is unsupervised-but-aligned (the data is unpaired)
Conventional unpaired translation (CycleGAN / DDIB) is exactly what produces dispersed, weak, blurry results. Instead we reformulate the unpaired problem as conditional inpainting of the artifact's L* on its own context:

For the over-exposure model, training data are the 281 over-exposed frames. The mask = the bright focal cluster of that frame. Loss = noise-prediction on L* inside the mask, conditioned on (L* outside, mask, depth proxy).
At inference on a normal frame, the mask comes from the normal frame's depth proxy. The "outside L*" is normal-tissue L* — which is distributionally indistinguishable from the periphery of the artifact frames seen at training. The model paints in the trained artifact L* distribution.
Same recipe for the 817 under-exposed frames.
This is unsupervised in the sense that no paired (normal, artifact) pair is ever required; it's "supervised" only in the sense that every pixel has a target — its own L* value.

UNet + scheduler choices
diffusers.UNet2DModel, 5 resolutions, base 64, mults [1,2,2,4,4], attention at the two deepest resolutions, GroupNorm-32. ~50 M parameters.
4 input channels: [L_noisy(t), L_known·(1-mask), mask, depth]. 1 output channel: ε (noise).
DDPM (linear β, T=1000, ε-prediction) for training; DDIM (50 steps) for sampling.
EMA (decay 0.999), AMP-FP16, grad-clip 1.0, AdamW 1e-4. Mask dropout 10% during training enables CFG at sampling.
Sampling
DDIM 50 steps with CFG w=3.5: eps = eps_uncond + 3.5·(eps_cond − eps_uncond). CFG is the lever for "make the artifact stronger" without retraining.
RePaint-style consistency: at every step, replace the outside-mask region with q(L_known, t_next) so the boundary is well-defined.
Hard substitution at the end so only-inside-mask pixels actually change. Combined with lab2rgb(L_pred, a*_orig, b*_orig), this guarantees zero color drift outside the mask and zero chrominance drift anywhere.
Files written

code_250426/
├── config.yaml              # all paths + all hyperparameters
├── requirements.txt
├── train.py                 # CLI entry point
├── sample.py                # CLI entry point
└── illum_diff/
    ├── __init__.py
    ├── config.py            # YAML loader
    ├── color.py             # LAB <-> RGB (skimage), L scaling
    ├── depth.py             # photometric depth proxy + focal-cluster masks
    ├── dataset.py           # ArtifactInpaintingDataset, NormalSampleDataset
    ├── model.py             # UNet2DModel builder
    ├── ema.py               # EMA shadow weights
    ├── trainer.py           # full training loop (AMP, EMA, ckpts, periodic samples)
    └── sampler.py           # DDIM + CFG + RePaint inpainting
How to run

# Install
pip install -r requirements.txt

# 1) RTX 3050 (4GB) smoke test — confirms the pipeline starts, then Ctrl-C
python train.py --config config.yaml --artifact overexposure \
    --batch_size 2 --image_size 192 --num_steps 50 --num_workers 0

# 2) DGX (16GB) — full runs (one specialist at a time)
python train.py --config config.yaml --artifact overexposure
python train.py --config config.yaml --artifact underexposure

# 3) Generate the synthetic paired dataset
python sample.py --config config.yaml --artifact overexposure \
    --ckpt outputs/checkpoints/overexposure_best.pt --num_samples 200
python sample.py --config config.yaml --artifact underexposure \
    --ckpt outputs/checkpoints/underexposure_best.pt --num_samples 200
Each sampling run writes triplets sample_XX_normal.png / sample_XX_<artifact>.png / sample_XX_mask.png / sample_XX_depth.png so you can audit the focal mask alongside the generation.

Knobs you'll likely want to tune
Effect strength: sample.cfg_scale (3.5 default). Push to 5.0–7.0 if changes still feel weak; drop to 2.0 if oversaturated.
Effect locality / size: mask.cluster_fraction (0.18 default). Smaller → tighter blob.
Boundary softness: mask.blur_sigma, mask.dilation_iters.
Inside-vs-outside loss balance: train.mask_loss_weight_inside (6.0 default).
What is guaranteed by construction (not just hoped for)
Chrominance is bit-identical to the input — (a*, b*) are never modified.
Pixels outside the mask have bit-identical L* to the input (hard substitution).
Each generated image carries either over- or under-exposure, never both — they are produced by two separate models with separate weights.
The artifact is always a single connected blob (largest CC), placed where depth says the real cause-of-artifact is.