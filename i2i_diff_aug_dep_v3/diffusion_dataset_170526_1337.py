"""
diffusion_dataset.py — Paired L-channel dataset for conditional diffusion
with depth conditioning.

Single-domain (split-model) version:
  - PairedLuminanceDataset is now created for ONE domain at a time
    ('overexposed' or 'underexposed'). It returns 3-tuples
    (source_L, target_L, depth) — no class label, because each model
    is trained to do exactly one direction.
  - Joint spatial augmentation applied IDENTICALLY to source_L, target_L,
    and depth.
  - NormalInferenceDataset is unchanged: it loads RGB + depth and returns
    everything inference needs.
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from exposure_augment import rgb_to_lab


# --------------------------------------------------------------------------- #
# Training dataset: paired (normal L, depth) → target L
# --------------------------------------------------------------------------- #

class PairedLuminanceDataset(Dataset):
    """Single-domain dataset.

    Each item returns:
        source_L : (1, H, W) float32 in [-1, 1]
        target_L : (1, H, W) float32 in [-1, 1]
        depth    : (1, H, W) float32 in [-1, 1]  (rescaled from [0,1])
    """

    _ALLOWED_DOMAINS = ("overexposed", "underexposed")

    def __init__(
        self,
        pairs_dir: str,
        domain: str,
        image_size: int = 256,
        augment: bool = True,
        gamma_jitter: float = 0.1,        # ± range for random gamma on L
        depth_noise_sigma: float = 0.02,  # Gaussian noise on depth (robustness)
    ):
        if domain not in self._ALLOWED_DOMAINS:
            raise ValueError(
                f"domain must be one of {self._ALLOWED_DOMAINS}, got {domain!r}"
            )
        self.domain = domain
        self.image_size = image_size
        self.augment = augment
        self.gamma_jitter = gamma_jitter
        self.depth_noise_sigma = depth_noise_sigma

        pairs = Path(pairs_dir)
        normal_dir = pairs / "normal"
        depth_dir = pairs / "depth"
        target_dir = pairs / domain

        if not depth_dir.is_dir():
            raise RuntimeError(
                f"Depth directory missing: {depth_dir}. "
                f"Run depth_estimator.py and regenerate pairs."
            )
        if not target_dir.is_dir():
            raise RuntimeError(
                f"Target directory missing for domain={domain}: {target_dir}. "
                f"Run generate_pairs.py first."
            )

        self.samples = []  # (normal_lab_path, depth_path, target_L_path)
        for npy in sorted(normal_dir.glob("*.npy")):
            stem = npy.stem
            dpath = depth_dir / f"{stem}.npy"
            if not dpath.exists():
                continue
            for t in sorted(target_dir.glob(f"{stem}*.npy")):
                self.samples.append((str(npy), str(dpath), str(t)))

        if not self.samples:
            raise RuntimeError(
                f"No paired samples in {pairs_dir} for domain={domain}. "
                f"Run generate_pairs.py first."
            )

        random.shuffle(self.samples)
        print(f"[PairedDataset] domain={domain}  pairs={len(self.samples)}")

    # ──────────────────────────────────────────────────────────────────── #
    @staticmethod
    def _normalise_L(L):
        return (L / 50.0) - 1.0

    @staticmethod
    def _normalise_depth(d):
        # [0, 1] → [-1, 1]  to match the UNet input scale
        return d * 2.0 - 1.0

    @staticmethod
    def _resize_2d(arr, size):
        img = Image.fromarray(arr.astype(np.float32), mode="F")
        img = img.resize((size, size), Image.LANCZOS)
        return np.array(img, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────── #
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        normal_path, depth_path, target_path = self.samples[idx]

        normal_lab = np.load(normal_path).astype(np.float32)  # (H, W, 3)
        source_L = normal_lab[..., 0]

        target_L = np.load(target_path).astype(np.float32)
        depth = np.load(depth_path).astype(np.float32)

        # resize all to model resolution
        source_L = self._resize_2d(source_L, self.image_size)
        target_L = self._resize_2d(target_L, self.image_size)
        depth = self._resize_2d(depth, self.image_size)
        depth = np.clip(depth, 0.0, 1.0)  # preserve domain

        # ── joint spatial augmentation (SAME transform on all three) ──
        if self.augment:
            if random.random() > 0.5:
                source_L = np.flip(source_L, axis=1).copy()
                target_L = np.flip(target_L, axis=1).copy()
                depth    = np.flip(depth,    axis=1).copy()
            if random.random() > 0.5:
                source_L = np.flip(source_L, axis=0).copy()
                target_L = np.flip(target_L, axis=0).copy()
                depth    = np.flip(depth,    axis=0).copy()
            if random.random() > 0.5:
                k = random.choice([1, 2, 3])
                source_L = np.rot90(source_L, k).copy()
                target_L = np.rot90(target_L, k).copy()
                depth    = np.rot90(depth,    k).copy()

            # gamma jitter on L channels — simulate light-source intensity variation.
            # Apply IDENTICALLY to source and target so the residual is preserved.
            if self.gamma_jitter > 0:
                g = 1.0 + random.uniform(-self.gamma_jitter, self.gamma_jitter)
                # map to [0,1], exponentiate, map back
                s01 = np.clip(source_L / 100.0, 0.0, 1.0)
                t01 = np.clip(target_L / 100.0, 0.0, 1.0)
                source_L = (np.power(s01, g) * 100.0).astype(np.float32)
                target_L = (np.power(t01, g) * 100.0).astype(np.float32)

            # depth noise — robustness to depth-estimator errors
            if self.depth_noise_sigma > 0:
                depth = depth + np.random.normal(0.0, self.depth_noise_sigma, depth.shape).astype(np.float32)
                depth = np.clip(depth, 0.0, 1.0)

        # normalise to [-1, 1]
        source_L = self._normalise_L(source_L)
        target_L = self._normalise_L(target_L)
        depth = self._normalise_depth(depth)

        source_tensor = torch.from_numpy(source_L).unsqueeze(0)
        target_tensor = torch.from_numpy(target_L).unsqueeze(0)
        depth_tensor = torch.from_numpy(depth).unsqueeze(0)

        return source_tensor, target_tensor, depth_tensor


# --------------------------------------------------------------------------- #
# Inference dataset: loads normal RGB + depth at original and model resolution
# --------------------------------------------------------------------------- #

class NormalInferenceDataset(Dataset):
    """Returns:
        source_L   : (1, H_model, W_model)  normalised L, model resolution
        depth_s    : (1, H_model, W_model)  normalised depth, model resolution
        AB         : (H_orig, W_orig, 2)    original-resolution chroma
        L_orig     : (H_orig, W_orig)       original-resolution L
        depth_full : (H_orig, W_orig)       original-resolution depth  (for chroma fix)
        path       : str
        orig_hw    : (int, int)             original height, width
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def __init__(
        self,
        normal_dir: str,
        image_size: int = 256,
        depth_dir: str = None,                 # if None, depth is computed on-the-fly
        depth_backend: str = "hadepth",        # 'hadepth' | 'midas' | 'midas_hybrid' | 'depth_anything_v2_hf'
        hadepth_repo: str = None,              # path to HADepth/ folder (used by 'hadepth')
        hadepth_ckpt: str = None,              # path to depth_model.pth or its parent dir
        hadepth_vignette_threshold: float = 5.0,
    ):
        self.image_size = image_size
        folder = Path(normal_dir)
        self.paths = sorted(
            p for p in folder.iterdir() if p.suffix.lower() in self.EXTENSIONS
        )
        print(f"[InferenceDataset] {len(self.paths)} images from {normal_dir}")

        self.depth_dir = Path(depth_dir) if depth_dir else None
        self._depth_backend = depth_backend
        self._hadepth_repo = hadepth_repo
        self._hadepth_ckpt = hadepth_ckpt
        self._hadepth_vt = float(hadepth_vignette_threshold)
        self._depth_model = None       # lazy — may be pipe (HF) or (model, transform) tuple (MiDaS/HADepth)

    def _ensure_depth_model(self):
        """Lazy-init the on-the-fly depth model. Prefers cached .npy on disk;
        this only fires when depth_dir is missing a file."""
        if self._depth_model is not None:
            return

        import torch as _t
        dev = _t.device("cuda" if _t.cuda.is_available() else "cpu")
        backend = self._depth_backend

        if backend == "hadepth":
            # HADepth (endoscopy-specialised, used to generate the training
            # pairs). Mirrors the loader in
            # `code _to_generate_pairs/depth_estimator.py` so on-the-fly
            # depth matches the training-time convention exactly.
            import sys as _sys
            import types as _types
            repo = self._hadepth_repo or str(
                Path(__file__).resolve().parent / "HADepth"
            )
            ckpt = self._hadepth_ckpt or str(Path(repo) / "HADepth_fullmodel")
            repo_p = Path(repo).expanduser().resolve()
            if not repo_p.is_dir():
                raise RuntimeError(
                    f"[InferenceDataset] HADepth repo not found at {repo_p}. "
                    f"Pass hadepth_repo=... or precompute depth into depth_dir."
                )
            if str(repo_p) not in _sys.path:
                _sys.path.insert(0, str(repo_p))
            # HADepth's models/backbones/layers/utils.py references MAB.py
            # which isn't shipped; stub it so the import succeeds.
            _stub_name = "models.backbones.layers.MAB"
            if _stub_name not in _sys.modules:
                _sys.modules[_stub_name] = _types.ModuleType(_stub_name)
            from models.hadepth import hadepth as hadepth_class  # type: ignore
            m = hadepth_class(
                backbone_size="base",
                r=4,
                lora_type="dora",
                image_shape=(224, 280),
                pretrained_path=None,
                residual_block_indexes=[2, 5, 8, 11],
                include_cls_token=True,
            )
            ck = Path(ckpt).expanduser().resolve()
            if ck.is_dir():
                ck = ck / "depth_model.pth"
            if not ck.is_file():
                raise RuntimeError(
                    f"[InferenceDataset] HADepth checkpoint not found: {ck}"
                )
            state = _t.load(str(ck), map_location="cpu")
            model_dict = m.state_dict()
            matched = {
                k: v for k, v in state.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            m.load_state_dict(matched, strict=False)
            print(
                f"[InferenceDataset] lazy-loaded HADepth: "
                f"matched {len(matched)}/{len(model_dict)} keys from {ck}"
            )
            m = m.to(dev).eval()
            self._depth_model = ("hadepth", m, None, dev)

        elif backend in ("midas", "midas_hybrid"):
            mtype = "DPT_Large" if backend == "midas" else "DPT_Hybrid"
            print(f"[InferenceDataset] lazy-loading MiDaS {mtype} via torch.hub")
            m = _t.hub.load("intel-isl/MiDaS", mtype).to(dev).eval()
            tfs = _t.hub.load("intel-isl/MiDaS", "transforms")
            tf = tfs.dpt_transform if mtype in ("DPT_Large", "DPT_Hybrid") else tfs.small_transform
            self._depth_model = ("midas", m, tf, dev)

        elif backend == "depth_anything_v2_hf":
            # Guard against stale transformers (DA V2 support lands in 4.42)
            import transformers
            try:
                parts = [int(p) for p in transformers.__version__.split(".")[:2]]
                if (parts[0], parts[1]) < (4, 42):
                    raise RuntimeError(
                        f"transformers {transformers.__version__} is too old for "
                        f"Depth Anything V2 (need >= 4.42). "
                        f"Either upgrade or switch depth_backend to 'midas'."
                    )
            except (ValueError, IndexError):
                pass
            from transformers import pipeline
            print("[InferenceDataset] lazy-loading DA V2 Small via HF pipeline")
            pipe = pipeline(
                task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=0 if dev.type == "cuda" else -1,
            )
            self._depth_model = ("hf", pipe, None, dev)
        else:
            raise ValueError(f"unknown depth_backend: {backend}")

    @staticmethod
    def _detect_visible_mask(img_rgb: np.ndarray, threshold: float = 5.0) -> np.ndarray:
        """Boolean mask: True on tissue, False on the camera vignette.
        Same convention used by HADepth in the pair-generation pipeline."""
        Y = (0.2126 * img_rgb[..., 0] + 0.7152 * img_rgb[..., 1]
             + 0.0722 * img_rgb[..., 2])
        L = (Y.astype(np.float32) / 255.0) * 100.0
        return L > float(threshold)

    def _load_or_compute_depth(self, path: Path, img_pil: Image.Image) -> np.ndarray:
        # 1. try disk cache
        if self.depth_dir is not None:
            candidate = self.depth_dir / f"{path.stem}.npy"
            if candidate.exists():
                d = np.load(str(candidate)).astype(np.float32)
                return np.clip(d, 0.0, 1.0)

        # 2. compute inline
        self._ensure_depth_model()
        kind, model_or_pipe, transform, dev = self._depth_model
        import torch as _t

        if kind == "hadepth":
            img_np = np.array(img_pil.convert("RGB"))
            H, W = img_np.shape[:2]
            x = (_t.from_numpy(img_np.astype(np.float32) / 255.0)
                 .permute(2, 0, 1).unsqueeze(0).to(dev))
            with _t.no_grad():
                out = model_or_pipe(x)
                disp = out[("disp", 0)]
                disp = _t.nn.functional.interpolate(
                    disp, size=(H, W), mode="bicubic", align_corners=False
                )
            d = disp.squeeze().detach().cpu().numpy().astype(np.float32)
            visible = self._detect_visible_mask(img_np, threshold=self._hadepth_vt)
            if visible.any():
                dv = d[visible]
                lo = float(np.percentile(dv, 2.0))
                hi = float(np.percentile(dv, 98.0))
                if hi > lo + 1e-6:
                    d = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
                else:
                    d = np.full_like(d, 0.5)
            else:
                d = np.full_like(d, 0.5)
            d[~visible] = 1.0  # vignette → near, matches training-time depth
            return d.astype(np.float32)

        if kind == "midas":
            img_np = np.array(img_pil.convert("RGB"))
            H, W = img_np.shape[:2]
            with _t.no_grad():
                batch = transform(img_np).to(dev)
                pred = model_or_pipe(batch)
                pred = _t.nn.functional.interpolate(
                    pred.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False
                ).squeeze()
            d = pred.detach().cpu().numpy().astype(np.float32)
        else:  # kind == "hf"
            out = model_or_pipe(img_pil)
            dt = out["predicted_depth"]
            if dt.ndim == 2:
                dt = dt.unsqueeze(0).unsqueeze(0)
            elif dt.ndim == 3:
                dt = dt.unsqueeze(0)
            W, H = img_pil.size
            dt = _t.nn.functional.interpolate(
                dt, size=(H, W), mode="bicubic", align_corners=False
            )
            d = dt.squeeze().detach().cpu().numpy().astype(np.float32)

        mn, mx = float(d.min()), float(d.max())
        if mx - mn < 1e-6:
            return np.full_like(d, 0.5, dtype=np.float32)
        return ((d - mn) / (mx - mn)).astype(np.float32)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img_pil = Image.open(path).convert("RGB")
        img_np = np.array(img_pil)
        lab = rgb_to_lab(img_np)

        L_orig = lab[..., 0].copy()
        AB = lab[..., 1:].copy()

        depth_full = self._load_or_compute_depth(path, img_pil)  # (H, W)

        # ── resized model-input tensors ──
        L_s = np.array(
            Image.fromarray(L_orig.astype(np.float32), mode="F").resize(
                (self.image_size, self.image_size), Image.LANCZOS
            ),
            dtype=np.float32,
        )
        L_s = (L_s / 50.0) - 1.0

        D_s = np.array(
            Image.fromarray(depth_full.astype(np.float32), mode="F").resize(
                (self.image_size, self.image_size), Image.LANCZOS
            ),
            dtype=np.float32,
        )
        D_s = np.clip(D_s, 0.0, 1.0) * 2.0 - 1.0  # [-1, 1]

        source_tensor = torch.from_numpy(L_s).unsqueeze(0)
        depth_tensor = torch.from_numpy(D_s).unsqueeze(0)

        return (
            source_tensor,
            depth_tensor,
            AB,
            L_orig,
            depth_full,
            str(path),
            img_np.shape[:2],
        )