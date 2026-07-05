#!/usr/bin/env python3
"""
depth_estimator.py — Precompute monocular depth maps for the normal-frames set.

Backends
--------
  hadepth               *BEST quality on endoscopy.* HADepth (Highlight-Aware
                        depth, Signal/Image/Video Processing) — DA-V2 ViT-B/14
                        backbone + DoRA-4 + DPT head, fine-tuned on SCARED +
                        ColonDepth. Beats EndoDAC / AF-SfMLearner / Depth
                        Anything on published benchmarks. Requires the HADepth/
                        repo to be cloned locally (it is, as a sibling of this
                        folder) plus the pretrained depth_model.pth.
  dav2_endo             Depth Anything V2 (HF) with endoscopy-aware pre/post-
                        processing. Falls back here if HADepth isn't available.
                        Vignette inpainted before forward pass; output
                        normalised within the visible-tissue region; vignette
                        pixels forced to depth=1 (near). Needs transformers >=
                        4.42.
  midas                 torch.hub MiDaS DPT_Large. Natural-image prior; without
                        endoscopy preprocessing, the vignette confuses it.
                        Inverse depth, higher = closer.
  midas_hybrid          Faster, slightly lower-quality MiDaS DPT_Hybrid.
  depth_anything_v2_hf  Plain DA V2 via HF pipeline (no endoscopy preprocessing).

Output
------
  One float16 .npy per input image, per-image min-max normalised to [0, 1]:
      1.0 = nearest surface
      0.0 = farthest surface
  Plus a greyscale PNG preview in <o>/previews/ for quick sanity checks.

Usage
-----
  # BEST — HADepth, fine-tuned on endoscopy
  python depth_estimator.py --backend hadepth \\
      --input_dir  .../normal_frames \\
      --output_dir .../depth
  # (defaults: --hadepth_repo ../HADepth, --hadepth_ckpt ../HADepth/HADepth_fullmodel)

  # Endoscopy-aware DA V2 (no extra repo needed)
  python depth_estimator.py --backend dav2_endo \\
      --input_dir  .../normal_frames \\
      --output_dir .../depth

  # Plain backends without endoscopy preprocessing
  python depth_estimator.py --backend midas --input_dir ... --output_dir ...
  python depth_estimator.py --backend depth_anything_v2_hf --input_dir ... --output_dir ...
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ═══════════════════════════════════════════════════════════════════════════
# Backend loaders
# ═══════════════════════════════════════════════════════════════════════════

def _load_midas(model_type: str, device: torch.device):
    """Load a MiDaS model via torch.hub. Returns (model, transform_fn)."""
    print(f"[depth] loading MiDaS ({model_type}) via torch.hub …")
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model = model.to(device).eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type in ("DPT_Large", "DPT_Hybrid"):
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform
    return model, transform


def _load_depth_anything_v2_hf(device: torch.device):
    """Load Depth Anything V2 Small via HF transformers pipeline.
    Raises a clear error if the installed transformers is too old."""
    try:
        import transformers
    except ImportError:
        sys.exit("[depth] transformers is not installed.")

    # DA V2 was added to transformers on 2024-07-05 → need >= 4.42
    try:
        parts = [int(p) for p in transformers.__version__.split(".")[:2]]
        major, minor = parts[0], parts[1]
    except Exception:
        major, minor = 0, 0
    if (major, minor) < (4, 42):
        sys.exit(
            f"[depth] transformers {transformers.__version__} is too old for "
            f"Depth Anything V2 (need >= 4.42).\n"
            f"        Either upgrade (`pip install -U 'transformers>=4.44'`) or "
            f"use --backend midas."
        )

    from transformers import pipeline
    print("[depth] loading Depth Anything V2 Small via HF pipeline …")
    pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=0 if device.type == "cuda" else -1,
    )
    return pipe, None


def _load_hadepth(repo_path: str, ckpt_path: str, device: torch.device):
    """Load the HADepth endoscopy depth model.

    HADepth: DA-V2 ViT-B/14 backbone + DoRA-4 + DPT head, fine-tuned on
    SCARED + ColonDepth. Construction must mirror the eval script's
    hyperparameters or the checkpoint won't match.

    Args:
        repo_path: path to the HADepth/ folder on disk. Added to sys.path so
                   `models.hadepth` and `models.backbones.*` import correctly.
        ckpt_path: path to depth_model.pth, OR to the folder containing it
                   (typically HADepth_fullmodel/).
    """
    repo = Path(repo_path).expanduser().resolve()
    if not repo.is_dir():
        sys.exit(
            f"[depth] HADepth repo not found at {repo}\n"
            f"        Pass --hadepth_repo /path/to/HADepth"
        )
    # Insert at index 0 so HADepth's `models/` package shadows any other.
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    # The HADepth repo's models/backbones/layers/utils.py contains a leftover
    # `from .MAB import *` but MAB.py is not shipped. The symbols utils.py
    # actually uses (LayerNorm, ResBottleneckBlock) are all defined locally,
    # so the bad import is functionally a no-op. Pre-register an empty stub
    # module under the expected dotted name so Python's import machinery
    # finds it in sys.modules and skips the file lookup. Keeps the user's
    # HADepth repo untouched.
    import types as _types
    _stub_name = "models.backbones.layers.MAB"
    if _stub_name not in sys.modules:
        sys.modules[_stub_name] = _types.ModuleType(_stub_name)

    try:
        from models.hadepth import hadepth as hadepth_class  # type: ignore
    except ImportError as e:
        sys.exit(
            f"[depth] failed to import HADepth model from {repo}: {e}\n"
            f"        Make sure {repo}/models/ and its dependencies are intact.\n"
            f"        Required Python deps: torch, fvcore, (optional: xFormers)."
        )

    # Mirror evaluate_depth.py construction. pretrained_path=None skips loading
    # the original DA-V2 weights — the HADepth ckpt below contains everything.
    model = hadepth_class(
        backbone_size="base",
        r=4,
        lora_type="dora",
        image_shape=(224, 280),
        pretrained_path=None,
        residual_block_indexes=[2, 5, 8, 11],
        include_cls_token=True,
    )

    ck = Path(ckpt_path).expanduser().resolve()
    if ck.is_dir():
        ck = ck / "depth_model.pth"
    if not ck.is_file():
        sys.exit(f"[depth] HADepth checkpoint not found: {ck}")

    state = torch.load(str(ck), map_location="cpu")
    model_dict = model.state_dict()
    matched = {k: v for k, v in state.items() if k in model_dict and v.shape == model_dict[k].shape}
    n_total = len(model_dict)
    n_match = len(matched)
    n_extra = len([k for k in state if k not in model_dict])
    n_skip_shape = len(state) - n_match - n_extra
    model.load_state_dict(matched, strict=False)
    print(
        f"[depth] HADepth loaded: matched {n_match}/{n_total} keys "
        f"(extra {n_extra}, shape-mismatch {n_skip_shape}) from {ck}"
    )

    model = model.to(device).eval()
    return model, None  # no separate transform; preprocessing is done inline


@torch.no_grad()
def _infer_hadepth(model, _unused, img_pil: Image.Image,
                   device: torch.device,
                   vignette_threshold: float = 5.0,
                   visible_pct_lo: float = 2.0,
                   visible_pct_hi: float = 98.0) -> np.ndarray:
    """HADepth inference + endoscopy post-processing.

    Steps:
      1. Convert RGB to a [0, 1] tensor (no ImageNet normalisation; matches
         HADepth's mono_dataset which only does ToTensor).
      2. Forward pass — the model internally interpolates to 224x280.
      3. Take outputs[("disp", 0)] (sigmoid, [0, 1], higher = closer) — already
         the right convention for our pipeline.
      4. Upsample to original (H, W).
      5. Normalise within the visible-tissue region (percentile-clipped, robust
         to specular highlights).
      6. Force vignette pixels to depth=1.0 (near) so under-augmentation never
         peaks on the camera mask.
    """
    img_np = np.array(img_pil.convert("RGB"))
    H, W = img_np.shape[:2]

    x = torch.from_numpy(img_np.astype(np.float32) / 255.0)
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)

    out = model(x)
    disp = out[("disp", 0)]  # (1, 1, h', w'), in [0, 1], higher = closer
    disp = torch.nn.functional.interpolate(
        disp, size=(H, W), mode="bicubic", align_corners=False
    )
    d = disp.squeeze().detach().cpu().numpy().astype(np.float32)

    visible = _detect_visible_mask(img_np, threshold=vignette_threshold)
    if visible.any():
        dv = d[visible]
        lo = float(np.percentile(dv, visible_pct_lo))
        hi = float(np.percentile(dv, visible_pct_hi))
        if hi > lo + 1e-6:
            d_norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
        else:
            d_norm = np.full_like(d, 0.5)
    else:
        d_norm = np.full_like(d, 0.5)

    d_norm[~visible] = 1.0
    return d_norm.astype(np.float32)


def _load_dav2_endo(hf_model: str, device: torch.device):
    """Load DA V2 (HF AutoModel) for the endoscopy-aware backend.

    Uses the AutoModel API rather than the HF pipeline so we can run our own
    pre/post-processing (vignette inpainting, visible-region normalisation,
    vignette → near). Returns (model, processor).
    """
    try:
        import transformers  # noqa: F401
    except ImportError:
        sys.exit("[depth] transformers is not installed.")
    try:
        parts = [int(p) for p in transformers.__version__.split(".")[:2]]
        major, minor = parts[0], parts[1]
    except Exception:
        major, minor = 0, 0
    if (major, minor) < (4, 42):
        sys.exit(
            f"[depth] transformers {transformers.__version__} is too old for "
            f"Depth Anything V2 (need >= 4.42)."
        )
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor
    print(f"[depth] loading {hf_model} via AutoModelForDepthEstimation …")
    model = AutoModelForDepthEstimation.from_pretrained(hf_model)
    model = model.to(device).eval()
    processor = AutoImageProcessor.from_pretrained(hf_model)
    return model, processor


# ═══════════════════════════════════════════════════════════════════════════
# Per-image inference
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _infer_midas(model, transform, img_pil: Image.Image,
                 device: torch.device) -> np.ndarray:
    """Run MiDaS on a PIL image → (H, W) float32, min-max normalised."""
    img_np = np.array(img_pil.convert("RGB"))
    H, W = img_np.shape[:2]
    # MiDaS transforms expect a numpy uint8 RGB array, return a tensor
    batch = transform(img_np).to(device)
    pred = model(batch)
    # resize to original resolution
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False
    ).squeeze()
    d = pred.detach().cpu().numpy().astype(np.float32)
    return _normalize_01(d)


@torch.no_grad()
def _infer_da_v2_hf(pipe, _unused, img_pil: Image.Image,
                    device: torch.device) -> np.ndarray:
    """Run HF DA V2 pipeline → (H, W) float32, min-max normalised."""
    out = pipe(img_pil)
    # HF depth-estimation pipeline returns {'depth': PIL image, 'predicted_depth': tensor}
    # 'predicted_depth' is inverse depth in the model's native resolution.
    pred = out["predicted_depth"].to(device)
    if pred.ndim == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
    elif pred.ndim == 3:
        pred = pred.unsqueeze(1)
    W, H = img_pil.size  # PIL gives (W, H)
    pred = torch.nn.functional.interpolate(
        pred, size=(H, W), mode="bicubic", align_corners=False
    ).squeeze()
    d = pred.detach().cpu().numpy().astype(np.float32)
    return _normalize_01(d)


def _normalize_01(d: np.ndarray) -> np.ndarray:
    """Per-image min-max normalise to [0, 1]. 1.0 = nearest, 0.0 = farthest."""
    mn, mx = float(d.min()), float(d.max())
    if mx - mn < 1e-6:
        return np.full_like(d, 0.5, dtype=np.float32)
    return ((d - mn) / (mx - mn)).astype(np.float32)


def _detect_visible_mask(img_rgb: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    """Boolean mask: True where the frame shows real tissue, False on the camera vignette.

    Uses Rec.709 luma scaled to [0, 100] and thresholds. Robust to JPEG noise
    around the vignette edge because the camera mask is essentially black.
    """
    Y = 0.2126 * img_rgb[..., 0] + 0.7152 * img_rgb[..., 1] + 0.0722 * img_rgb[..., 2]
    L = (Y.astype(np.float32) / 255.0) * 100.0  # match L convention used downstream
    return L > float(threshold)


@torch.no_grad()
def _infer_dav2_endo(model, processor, img_pil: Image.Image,
                     device: torch.device,
                     vignette_threshold: float = 5.0,
                     visible_pct_lo: float = 2.0,
                     visible_pct_hi: float = 98.0) -> np.ndarray:
    """Endoscopy-aware DA V2 inference.

    Steps:
      1. Detect the visible-tissue mask from the input frame.
      2. Inpaint the vignette with the mean tissue colour so the model doesn't
         interpret the camera mask as far-away background.
      3. Run DA V2 forward pass.
      4. Resize to original resolution.
      5. Normalise *within the visible region only* (percentile clipping is
         robust to specular highlights that would otherwise pin max-near).
      6. Force vignette pixels to depth=1 (near). This is the critical fix:
         under-augmentation peaks at min depth, so the camera mask must not be
         the deepest part of the scene.
    """
    img_np = np.array(img_pil.convert("RGB"))
    H, W = img_np.shape[:2]

    visible = _detect_visible_mask(img_np, threshold=vignette_threshold)
    if visible.any() and not visible.all():
        mean_rgb = img_np[visible].mean(axis=0).astype(np.uint8)
    else:
        mean_rgb = np.array([128, 128, 128], dtype=np.uint8)
    img_filled = img_np.copy()
    img_filled[~visible] = mean_rgb
    img_pil_filled = Image.fromarray(img_filled)

    inputs = processor(images=img_pil_filled, return_tensors="pt").to(device)
    out = model(**inputs)
    pred = out.predicted_depth
    if pred.ndim == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
    elif pred.ndim == 3:
        pred = pred.unsqueeze(1)
    pred = torch.nn.functional.interpolate(
        pred, size=(H, W), mode="bicubic", align_corners=False
    ).squeeze()
    d = pred.detach().cpu().numpy().astype(np.float32)

    # DA V2 outputs higher = closer; we want 1=near, 0=far in [0, 1]. Same direction.
    if visible.any():
        dv = d[visible]
        lo = float(np.percentile(dv, visible_pct_lo))
        hi = float(np.percentile(dv, visible_pct_hi))
        if hi > lo + 1e-6:
            d_norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
        else:
            d_norm = np.full_like(d, 0.5)
    else:
        d_norm = np.full_like(d, 0.5)

    d_norm[~visible] = 1.0  # vignette → near (excluded from "far cavity" peak)
    return d_norm.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Driver
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--backend", default="hadepth",
                        choices=["hadepth", "dav2_endo", "midas",
                                 "midas_hybrid", "depth_anything_v2_hf"],
                        help="hadepth (best for endoscopy) is default; "
                             "dav2_endo is the no-extra-repo fallback.")
    parser.add_argument("--hf_model", type=str,
                        default="depth-anything/Depth-Anything-V2-Small-hf",
                        help="HF model id for the dav2_endo backend.")
    parser.add_argument("--hadepth_repo", type=str, default=None,
                        help="path to the HADepth/ folder. Defaults to "
                             "`<this file>/../HADepth`.")
    parser.add_argument("--hadepth_ckpt", type=str, default=None,
                        help="path to HADepth's depth_model.pth (or the folder "
                             "containing it). Defaults to "
                             "`<hadepth_repo>/HADepth_fullmodel`.")
    parser.add_argument("--vignette_threshold", type=float, default=5.0,
                        help="L (0-100) below which a pixel is treated as camera "
                             "vignette. Used by hadepth and dav2_endo for "
                             "masking and post-fix.")
    parser.add_argument("--save_previews", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true",
                        help="re-run even if .npy already exists")
    args = parser.parse_args()

    in_dir  = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prev_dir = out_dir / "previews"
    if args.save_previews:
        prev_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[depth] device = {device}")

    # ── load backend ────────────────────────────────────────────────────
    if args.backend == "hadepth":
        # Default: ../HADepth relative to this script.
        repo = args.hadepth_repo or str(
            Path(__file__).resolve().parent.parent / "HADepth"
        )
        ckpt = args.hadepth_ckpt or str(Path(repo) / "HADepth_fullmodel")
        model, transform = _load_hadepth(repo, ckpt, device)
        vt = float(args.vignette_threshold)
        def infer(model, transform, img, dev):
            return _infer_hadepth(model, transform, img, dev,
                                  vignette_threshold=vt)
    elif args.backend == "dav2_endo":
        model, transform = _load_dav2_endo(args.hf_model, device)
        vt = float(args.vignette_threshold)
        # closure over vignette_threshold so the iterate-loop signature stays the same
        def infer(model, transform, img, dev):
            return _infer_dav2_endo(model, transform, img, dev,
                                    vignette_threshold=vt)
    elif args.backend == "midas":
        model, transform = _load_midas("DPT_Large", device)
        infer = _infer_midas
    elif args.backend == "midas_hybrid":
        model, transform = _load_midas("DPT_Hybrid", device)
        infer = _infer_midas
    else:  # depth_anything_v2_hf
        model, transform = _load_depth_anything_v2_hf(device)
        infer = _infer_da_v2_hf

    # ── iterate ─────────────────────────────────────────────────────────
    paths = sorted(p for p in in_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        sys.exit(f"[depth] no images found in {in_dir}")

    print(f"[depth] processing {len(paths)} images …")
    for p in tqdm(paths):
        out_npy = out_dir / f"{p.stem}.npy"
        if out_npy.exists() and not args.overwrite:
            continue

        img = Image.open(p).convert("RGB")
        d = infer(model, transform, img, device)  # (H, W) float32 in [0, 1]
        np.save(out_npy, d.astype(np.float16))

        if args.save_previews:
            prev = (d * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(prev, mode="L").save(prev_dir / f"{p.stem}.png")

    print(f"[depth] done — {len(paths)} maps written to {out_dir}")


if __name__ == "__main__":
    main()