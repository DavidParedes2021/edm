"""
model.py — Model components for the Illumination Diffusion pipeline.

Contains:
  • EVEmbedding         — sinusoidal embedding for the continuous EV scalar
                          with learnable null embedding for Classifier-Free Guidance
  • IlluminationUNet    — UNet2DConditionModel configured for LDM I2I
  • VAEWrapper          — frozen SD VAE for encode/decode (pixel ↔ latent)
  • EMA                 — Exponential Moving Average of UNet weights
  • LPIPSLoss           — VGG perceptual loss (torchvision 0.12 compatible)
  • SSIMLoss            — differentiable SSIM on the luminance channel
  • HistogramLoss       — soft Wasserstein-1 histogram matching

All forward passes keep tensors on a single device — device pinning is
enforced in __init__ for every sub-module.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from copy import deepcopy

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
)


# ──────────────────────────────────────────────────────────────────────────────
# EV sinusoidal embedding  (with CFG null embedding)
# ──────────────────────────────────────────────────────────────────────────────

class EVEmbedding(nn.Module):
    """
    Maps a scalar EV value (float, in stops) to a dense embedding vector
    using sinusoidal positional encoding followed by a two-layer MLP.

    Design principles:
      1. Continuous control: any EV in [-4, +4] can be requested at inference.
      2. Monotone inductive bias: neighbouring EV values produce similar
         embeddings, so the model generalises between seen training EVs.
      3. CFG null embedding: a learnable null token replaces the EV embedding
         during unconditional training steps, enabling Classifier-Free Guidance.

    BUGFIX vs original: EV is normalised to [0, 1000] before the sinusoidal
    encoding (just like timestep embeddings in DDPM which use t ∈ [0, 1000]).
    The original code used raw EV * 1000, but because EV ∈ [-4, +4], the
    highest-frequency sinusoid would complete thousands of cycles per EV stop,
    destroying the smooth interpolation property we need.

    Output shape: (B, embed_dim)
    """

    # EV physical range for normalisation
    EV_MIN = -4.0
    EV_MAX =  4.0

    def __init__(self, embed_dim: int = 256, max_period: float = 10_000.0):
        super().__init__()
        self.embed_dim  = embed_dim
        self.max_period = max_period

        half = embed_dim // 2
        self.mlp = nn.Sequential(
            nn.Linear(half * 2, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        # Learnable null embedding used for Classifier-Free Guidance dropout.
        # Initialised to zeros so unconditional denoising starts as the mean.
        self.null_embedding = nn.Parameter(torch.zeros(embed_dim))

    def _sinusoidal(self, ev: torch.Tensor) -> torch.Tensor:
        """
        ev: (B,)  →  (B, embed_dim)

        Normalise EV from [EV_MIN, EV_MAX] to [0, 1000] so the frequency bands
        are well-matched to the input range — identical to how DDPM handles
        timestep t ∈ [0, 1000].
        """
        half   = self.embed_dim // 2
        device = ev.device
        # Frequency bands: log-spaced between 1/max_period and 1
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=device)
            / (half - 1)
        )  # (half,)
        # Normalise EV to [0, 1000]  ← FIXED: was raw ev * 1000
        ev_norm = (ev.float() - self.EV_MIN) / (self.EV_MAX - self.EV_MIN) * 1000.0
        args = ev_norm[:, None] * freqs[None, :]   # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, embed_dim)

    def forward(self, ev: torch.Tensor) -> torch.Tensor:
        """ev: (B,) → (B, embed_dim)"""
        sinus = self._sinusoidal(ev)   # (B, embed_dim)
        return self.mlp(sinus)         # (B, embed_dim)

    def null_cond(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Return the null (unconditional) embedding for a batch.
        Used for CFG inference: run UNet once with null and once with EV,
        then extrapolate: pred = null + scale * (ev_pred - null).

        Returns: (B, 1, embed_dim) — ready for cross-attention.
        """
        return self.null_embedding.unsqueeze(0).unsqueeze(0).expand(
            batch_size, 1, -1
        ).to(device)


# ──────────────────────────────────────────────────────────────────────────────
# Exponential Moving Average of model weights
# ──────────────────────────────────────────────────────────────────────────────

class EMA:
    """
    Maintains an exponential moving average of a model's parameters.

    Standard practice in diffusion model training (DDPM, Stable Diffusion, etc.).
    EMA weights are smoother than the latest checkpoint, which translates
    directly to less noise and sharper images at inference time.

    Usage:
        ema = EMA(unet, decay=0.9999)
        # After each optimiser step:
        ema.step(unet)
        # At inference time, temporarily apply the shadow weights:
        with ema.apply(unet):
            output = unet(...)
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        # Shadow parameters (EMA values) stored on CPU to save VRAM
        self.shadow: dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.float().cpu().clone()

    @torch.no_grad()
    def step(self, model: nn.Module, step: int = 0):
        """
        Update shadow weights with the current model parameters.

        Uses an adaptive decay that ramps up from ~0.1 at step 0 to the
        target decay asymptotically.  Formula: min(target, (1+step)/(10+step)).

        Without this, decay=0.9999 means the shadow weights are still
        74% random initialisation at step 3000 (0.9999^3000 = 0.74).
        Sampling with those weights produces pure noise even when the live
        model has converged.  The adaptive formula gives:
          step    0  →  decay ≈ 0.09  (fast tracking of live weights)
          step 1000  →  decay ≈ 0.990
          step 5000  →  decay ≈ 0.998
          step 10000 →  decay ≈ 0.999  (approaching target 0.9999)
        """
        # Adaptive decay: fast early, slow (stable) late
        effective_decay = min(self.decay, (1.0 + step) / (10.0 + step))
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(effective_decay).add_(
                    param.data.float().cpu(), alpha=1.0 - effective_decay
                )

    def apply_shadow(self, model: nn.Module):
        """Copy shadow weights into the model (in-place). Call restore() after use."""
        self._backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].to(param.device).to(param.dtype))

    def restore(self, model: nn.Module):
        """Restore original weights after apply_shadow."""
        for name, param in model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = {}

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict):
        self.decay  = state["decay"]
        self.shadow = state["shadow"]


# ──────────────────────────────────────────────────────────────────────────────
# Frozen VAE wrapper
# ──────────────────────────────────────────────────────────────────────────────

class VAEWrapper(nn.Module):
    """
    Thin wrapper around the SD VAE (AutoencoderKL).

    The VAE weights are *frozen*: we use it purely for encode/decode.
    Encoding is done in smaller micro-batches to avoid OOM on 4 GB GPUs.
    """

    LATENT_SCALE = 0.18215   # SD VAE normalisation constant

    def __init__(self, model_id: str, device: str, encode_batch: int = 1):
        super().__init__()
        self.device       = device
        self.encode_batch = encode_batch

        print(f"[VAE] Loading {model_id} …")
        self.vae = AutoencoderKL.from_pretrained(
            model_id,
            torch_dtype=torch.float32,   # VAE stays in fp32 for stability
        ).to(device)

        # Freeze all parameters
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae.eval()
        print(f"[VAE] Loaded and frozen. encode_batch={encode_batch}")

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) in [-1, 1], on self.device
        Returns: (B, 4, H/8, W/8) latents, normalised by LATENT_SCALE
        """
        x = x.to(self.device)
        latents = []
        for i in range(0, x.shape[0], self.encode_batch):
            chunk = x[i : i + self.encode_batch]
            dist  = self.vae.encode(chunk).latent_dist
            latents.append(dist.sample() * self.LATENT_SCALE)
        return torch.cat(latents, dim=0)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, 4, H/8, W/8) normalised latents
        Returns: (B, 3, H, W) in [-1, 1]
        """
        z = z.to(self.device) / self.LATENT_SCALE
        return self.vae.decode(z).sample


# ──────────────────────────────────────────────────────────────────────────────
# UNet builder
# ──────────────────────────────────────────────────────────────────────────────

def build_unet(
    image_size: int,
    unet_channels: tuple,
    unet_layers_per_block: int,
    ev_embed_dim: int,
    latent_channels: int = 4,
    gradient_checkpointing: bool = False,
) -> UNet2DConditionModel:
    """
    Build a UNet2DConditionModel configured for latent-space I2I.

    Conditioning strategy (dual-path):
      1. Spatial (Normal latent): concatenated channel-wise → in_channels = 2*latent_channels
      2. EV semantic: projected to cross_attention_dim and injected via cross-attention

    The extra in_channels (from spatial concat) are zero-initialised so the
    model starts close to an unconditional prior and learns to use the spatial
    conditioning signal gradually.
    """
    in_channels = latent_channels * 2  # noisy latent (4) + normal latent (4)

    unet = UNet2DConditionModel(
        sample_size          = image_size // 8,   # latent spatial size
        in_channels          = in_channels,
        out_channels         = latent_channels,
        layers_per_block     = unet_layers_per_block,
        block_out_channels   = unet_channels,
        down_block_types     = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types       = (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        cross_attention_dim  = ev_embed_dim,
        attention_head_dim   = 8,
        norm_num_groups      = 32,
        resnet_time_scale_shift = "default",
    )

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Zero-init the conditioning channels (normal latent) in conv_in.
    # This makes the model start as an unconditional denoiser and gradually
    # learn to use the spatial conditioning signal — improves training stability.
    with torch.no_grad():
        unet.conv_in.weight[:, latent_channels:, ...].zero_()

    return unet


# ──────────────────────────────────────────────────────────────────────────────
# Min-SNR loss weighting
# ──────────────────────────────────────────────────────────────────────────────

def compute_snr_weights(
    scheduler,
    timesteps: torch.Tensor,
    gamma: float = 5.0,
) -> torch.Tensor:
    """
    Min-SNR-γ weighting from "Analyzing and Improving the Training Dynamics
    of Diffusion Models" (Hang et al., 2023).

    Without weighting, high-noise timesteps dominate the MSE loss (they have
    large gradients but predict almost all-noise, giving little useful signal).
    Min-SNR clips the per-sample weight at γ/SNR, balancing the contribution
    of low-noise and high-noise timesteps.

    Returns a (B,) weight tensor with values in (0, 1].  Multiply element-wise
    with per-pixel MSE before calling .mean().
    """
    alphas_cumprod = scheduler.alphas_cumprod.to(timesteps.device)
    sqrt_alphas    = alphas_cumprod[timesteps] ** 0.5
    sqrt_one_minus = (1.0 - alphas_cumprod[timesteps]) ** 0.5
    snr = (sqrt_alphas / (sqrt_one_minus + 1e-8)) ** 2   # (B,)
    # clip SNR at gamma: weight = min(SNR, γ) / SNR
    weights = torch.minimum(snr, torch.full_like(snr, gamma)) / snr
    return weights


# ──────────────────────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────────────────────

class LPIPSLoss(nn.Module):
    """
    VGG-based perceptual (LPIPS-style) loss.

    Compatible with torchvision 0.12+ (no lpips package required).
    Uses VGG16 features at layers relu1_2, relu2_2, relu3_3, relu4_3.

    Supports a `luminance_only` mode (default: True for this pipeline):
      Convert both images to YCbCr and feed only the Y (luminance) channel —
      replicated to 3 channels — into VGG.  This measures structural/texture
      similarity WITHOUT penalising brightness differences.

    Why luminance-only matters here:
      - We compare the predicted image to the NORMAL input (same scene content,
        correct colors).
      - In RGB mode, LPIPS would penalise any brightness shift, fighting the
        exposure change the model is supposed to make.
      - In luminance-only mode, LPIPS measures fine-grained structural fidelity
        (edges, tissues, instruments) while ignoring how bright they are.
      - A separate ChrominanceLoss handles hue/saturation preservation.
    """

    _LAYERS = [3, 8, 15, 22]   # relu1_2, relu2_2, relu3_3, relu4_3

    # BT.601 luminance weights for RGB → Y conversion
    _LUMA_W = torch.tensor([0.299, 0.587, 0.114])

    def __init__(self, device: str, luminance_only: bool = True):
        super().__init__()
        self.device         = device
        self.luminance_only = luminance_only

        vgg = tv_models.vgg16(pretrained=True).features.to(device)
        for p in vgg.parameters():
            p.requires_grad_(False)
        vgg.eval()

        self.slices = nn.ModuleList()
        prev = 0
        for layer_idx in self._LAYERS:
            self.slices.append(vgg[prev : layer_idx + 1])
            prev = layer_idx + 1

        # ImageNet stats for the 3-channel (or replicated-Y) input
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)
        luma_w = self._LUMA_W.to(device).view(1, 3, 1, 1)
        self.register_buffer("luma_w", luma_w)

    def _to_luma3(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, 3, H, W) [-1,1] → (B, 3, H, W) where all 3 channels = Y (luminance).
        Output range: [0, 1].
        Replicating Y to 3 channels lets us pass it into VGG unchanged.
        """
        x01 = (x + 1.0) / 2.0                         # [0, 1]
        Y   = (x01 * self.luma_w).sum(dim=1, keepdim=True)   # (B,1,H,W)
        return Y.expand_as(x)                          # (B,3,H,W)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Optionally project to luminance, then ImageNet-normalise."""
        if self.luminance_only:
            x = self._to_luma3(x)   # already [0,1]
            return (x - self.mean) / self.std
        else:
            x = (x + 1.0) / 2.0    # [-1,1] → [0,1]
            return (x - self.mean) / self.std

    def forward(
        self, pred: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        """
        pred, reference: (B, 3, H, W) in [-1, 1], on self.device

        Compare pred to reference (should be the NORMAL image when
        luminance_only=True).  Gradients flow through pred; reference detached.
        """
        pred      = pred.to(self.device)
        reference = reference.to(self.device)

        x_p = self._preprocess(pred)
        with torch.no_grad():
            x_t = self._preprocess(reference)

        loss = torch.tensor(0.0, device=self.device)
        for s in self.slices:
            x_p = s(x_p)
            with torch.no_grad():
                x_t = s(x_t)
            norm = lambda f: F.normalize(f, p=2, dim=1)
            loss = loss + F.mse_loss(norm(x_p), norm(x_t))

        return loss / len(self.slices)


class SSIMLoss(nn.Module):
    """
    Differentiable SSIM loss applied only on the *luminance* channel (Y in
    YCbCr).  This allows the chrominance to change freely under exposure
    shifts while penalising structural degradation.

    Returns 1 - SSIM so it can be minimised.

    In this pipeline:
      - Compare PREDICTED image to the NORMAL input to encourage structure
        preservation.  The structural content (endoscope, tissue geometry)
        should not change between normal and exposure-shifted versions.
    """

    def __init__(self, window_size: int = 11, device: str = "cpu"):
        super().__init__()
        self.window_size = window_size
        self.device      = device

        kernel_1d = self._gaussian_kernel(window_size, sigma=1.5)
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)   # (1,1,W,W)
        self.register_buffer("kernel", kernel_2d.to(device))

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def _rgb_to_luminance(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) [-1,1] → (B,1,H,W) [0,1] luminance via BT.601"""
        x01 = (x + 1.0) / 2.0
        Y = (0.299 * x01[:, 0:1]
             + 0.587 * x01[:, 1:2]
             + 0.114 * x01[:, 2:3])
        return Y

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x, y: (B,1,H,W) in [0,1]"""
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        pad    = self.window_size // 2
        kernel = self.kernel.to(x.device)

        mu_x  = F.conv2d(x, kernel, padding=pad)
        mu_y  = F.conv2d(y, kernel, padding=pad)
        mu_x2 = mu_x ** 2
        mu_y2 = mu_y ** 2
        mu_xy = mu_x * mu_y

        sig_x  = F.conv2d(x * x, kernel, padding=pad) - mu_x2
        sig_y  = F.conv2d(y * y, kernel, padding=pad) - mu_y2
        sig_xy = F.conv2d(x * y, kernel, padding=pad) - mu_xy

        numer = (2 * mu_xy + C1) * (2 * sig_xy + C2)
        denom = (mu_x2 + mu_y2 + C1) * (sig_x + sig_y + C2)
        return (numer / (denom + 1e-8)).mean()

    def forward(
        self, pred: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes 1 - SSIM(luma(pred), luma(reference)).
        pred, reference: (B, 3, H, W) in [-1, 1]

        Mean-shift pred's luminance to match reference before computing SSIM.
        This removes the SSIM luminance term (2μₓμᵧ+C1)/(μₓ²+μᵧ²+C1) from
        the loss, because that term penalises global brightness differences —
        which is exactly the change we want the model to make for exposure.
        After mean-shifting, SSIM measures only texture/edge structure fidelity.
        """
        pred      = pred.to(self.device)
        reference = reference.to(self.device)
        luma_p = self._rgb_to_luminance(pred)       # (B,1,H,W)
        luma_r = self._rgb_to_luminance(reference)  # (B,1,H,W)

        # Shift pred's mean to match reference so SSIM ignores brightness delta
        mean_p = luma_p.mean(dim=[2, 3], keepdim=True)
        mean_r = luma_r.mean(dim=[2, 3], keepdim=True)
        luma_p = luma_p - mean_p + mean_r   # same mean, different texture

        return 1.0 - self._ssim(luma_p, luma_r)


class HistogramLoss(nn.Module):
    """
    Soft Wasserstein-1 histogram matching loss.

    Encourages the generated image's intensity distribution to match the
    target domain's distribution.  Only useful when paired data are available.

    Uses differentiable soft histogram via Gaussian KDE.
    """

    def __init__(self, bins: int = 64, sigma: float = 0.02, device: str = "cpu"):
        super().__init__()
        self.bins   = bins
        self.sigma  = sigma
        self.device = device
        centers = torch.linspace(0.0, 1.0, bins, device=device)
        self.register_buffer("centers", centers)

    def _soft_histogram(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) in [-1, 1] → (B, bins) normalised soft histogram"""
        x01     = ((x + 1.0) / 2.0).view(x.shape[0], -1)   # (B, 3*H*W)
        centers = self.centers.to(x.device)
        diff    = x01[:, :, None] - centers[None, None, :]   # (B, N, bins)
        weights = torch.exp(-0.5 * (diff / self.sigma) ** 2)
        hist    = weights.sum(dim=1)                          # (B, bins)
        hist    = hist / (hist.sum(dim=1, keepdim=True) + 1e-7)
        return hist

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        pred, target: (B, 3, H, W) in [-1, 1]
        Returns Wasserstein-1 distance between their soft histograms.
        """
        pred   = pred.to(self.device)
        target = target.to(self.device)
        hist_p = self._soft_histogram(pred)
        hist_t = self._soft_histogram(target.detach())
        # Wasserstein-1 = L1 distance between CDFs
        cdf_p = hist_p.cumsum(dim=1)
        cdf_t = hist_t.cumsum(dim=1)
        return F.l1_loss(cdf_p, cdf_t)


class ChrominanceConsistencyLoss(nn.Module):
    """
    Penalise hue and saturation shifts between the generated image and the
    normal (correctly-exposed) input image.

    ROOT CAUSE FIX for color shifts in generated images:

    Real camera exposure changes affect luminance (brightness), NOT hue.
    A correctly exposed red tissue should remain red when overexposed —
    only brighter and potentially desaturated at the extreme clip boundary.
    Without this loss, the model is entirely free to shift hue and saturation
    because:
      (a) SSIM operates on the Y (luminance) channel only
      (b) there was no other loss term touching Cb or Cr

    Implementation:
      Convert both pred and normal to YCbCr (BT.601).
      Compute L1 between their Cb and Cr channels.
      Scale Cb/Cr to [-1, 1] for numerical stability.

    This loss is intentionally lightweight — hue should be almost perfectly
    preserved, so even a small weight (0.5) provides strong signal when
    color shifts occur.
    """

    # BT.601 RGB → YCbCr conversion matrix (for [0,1] input, offsets Cb/Cr by 0.5)
    # Y  =  0.299*R + 0.587*G + 0.114*B
    # Cb = -0.169*R - 0.331*G + 0.500*B + 0.5
    # Cr =  0.500*R - 0.419*G - 0.081*B + 0.5
    _RGB2YCbCr = torch.tensor([
        [ 0.299,  0.587,  0.114],
        [-0.169, -0.331,  0.500],
        [ 0.500, -0.419, -0.081],
    ])  # (3, 3)

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        mat = self._RGB2YCbCr.to(device)         # (3, 3)
        self.register_buffer("mat", mat)
        offset = torch.tensor([0.0, 0.5, 0.5], device=device).view(1, 3, 1, 1)
        self.register_buffer("offset", offset)

    def _to_ycbcr(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) in [-1, 1]
        Returns: (B, 3, H, W)  channels = [Y, Cb, Cr]
                 Y ∈ [0, 1],  Cb/Cr ∈ [0, 1]  (offset by 0.5)
        """
        x01 = (x + 1.0) / 2.0                                # [0,1]
        # einsum: for each spatial position, multiply (3,) RGB by (3,3) matrix
        y = torch.einsum("bchw,oc->bohw", x01, self.mat)     # (B,3,H,W)
        return y + self.offset

    def forward(
        self, pred: torch.Tensor, normal: torch.Tensor
    ) -> torch.Tensor:
        """
        pred, normal: (B, 3, H, W) in [-1, 1]
        Returns mean L1 distance between their Cb and Cr channels.
        """
        pred   = pred.to(self.device)
        normal = normal.to(self.device)

        pred_ycbcr   = self._to_ycbcr(pred)
        normal_ycbcr = self._to_ycbcr(normal.detach())

        # Compare only Cb (index 1) and Cr (index 2) — ignore luminance Y
        cb_loss = F.l1_loss(pred_ycbcr[:, 1], normal_ycbcr[:, 1])
        cr_loss = F.l1_loss(pred_ycbcr[:, 2], normal_ycbcr[:, 2])
        return (cb_loss + cr_loss) / 2.0


class ExposureBrightnessLoss(nn.Module):
    """
    Exposure-aware brightness constraint.

    Penalises the model if the mean luminance of the generated image is not
    consistent with the target EV direction:
      - Overexposed (EV > 0): mean luminance should be higher than normal
      - Underexposed (EV < 0): mean luminance should be lower than normal

    This acts as a simple, differentiable sanity check that the model is
    actually shifting exposure in the right direction.

    Returns a non-negative scalar loss (hinge form: only penalises violations).
    """

    def forward(
        self,
        pred: torch.Tensor,       # (B, 3, H, W) in [-1, 1]
        normal: torch.Tensor,     # (B, 3, H, W) in [-1, 1]
        ev: torch.Tensor,         # (B,) signed EV values
    ) -> torch.Tensor:
        # BT.601 luminance, [0, 1] range
        def luma(x):
            x01 = (x + 1.0) / 2.0
            return (0.299 * x01[:, 0] + 0.587 * x01[:, 1] + 0.114 * x01[:, 2]).mean(dim=[1, 2])

        pred   = pred.to(ev.device)
        normal = normal.to(ev.device)
        luma_pred   = luma(pred)    # (B,)
        luma_normal = luma(normal)  # (B,)
        delta = luma_pred - luma_normal  # positive = brighter

        # For overexposure (EV > 0), delta should be positive.
        # Hinge: penalise if delta goes in the wrong direction.
        # sign(EV) * delta should be > 0.  We penalise max(0, -sign(EV) * delta).
        ev_sign  = ev.sign()
        violation = F.relu(-ev_sign * delta)   # (B,)  non-zero only when wrong direction
        return violation.mean()
