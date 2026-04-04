"""
model.py — Model components for the Illumination Diffusion pipeline.

Contains:
  • EVEmbedding         — sinusoidal embedding for the continuous EV scalar
  • IlluminationUNet    — UNet2DConditionModel configured for LDM I2I
  • VAEWrapper          — frozen SD VAE for encode/decode (pixel ↔ latent)
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

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
)


# ──────────────────────────────────────────────────────────────────────────────
# EV sinusoidal embedding
# ──────────────────────────────────────────────────────────────────────────────

class EVEmbedding(nn.Module):
    """
    Maps a scalar EV value (float, in stops) to a dense embedding vector
    using sinusoidal positional encoding followed by a two-layer MLP.

    This is strictly superior to a 2-class learned embedding because:
      1. Continuous control: any EV in [-4, +4] can be requested at inference.
      2. Monotone inductive bias: neighbouring EV values produce similar
         embeddings, so the model generalises between seen training EVs.

    Output shape: (B, embed_dim)
    """

    def __init__(self, embed_dim: int = 256, max_period: float = 10_000.0):
        super().__init__()
        self.embed_dim  = embed_dim
        self.max_period = max_period

        # MLP projects sinusoidal features → final embedding
        half = embed_dim // 2
        self.mlp = nn.Sequential(
            nn.Linear(half * 2, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def _sinusoidal(self, ev: torch.Tensor) -> torch.Tensor:
        """
        ev: (B,)  →  (B, embed_dim)

        Scale EV to a range that the sinusoidal frequencies can distinguish.
        We multiply by 1000 so the encoding has fine-grained resolution across
        the [-4, +4] stop range (similar to how timestep embeddings work in
        diffusion models where t ∈ [0, 1000]).
        """
        half   = self.embed_dim // 2
        device = ev.device
        # Frequency bands: log-spaced between 1/max_period and 1
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=device)
            / (half - 1)
        )  # (half,)
        # Scale EV into [0, 1000]-like range for numerical compatibility
        args = ev[:, None].float() * 1000.0 * freqs[None, :]   # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, embed_dim)

    def forward(self, ev: torch.Tensor) -> torch.Tensor:
        """ev: (B,) → (B, embed_dim)"""
        sinus = self._sinusoidal(ev)   # (B, embed_dim)
        return self.mlp(sinus)         # (B, embed_dim)


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
    model starts close to an unconditional prior.
    """
    # in_channels = noisy latent (4) + normal latent (4) = 8
    in_channels = latent_channels * 2

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

    # Zero-init the conditioning half of conv_in (channels latent_channels..2*latent_channels-1).
    #
    # UNet2DConditionModel is already built with in_channels = 2*latent_channels (8),
    # so conv.weight.shape = (out_ch, 8, kH, kW).  We want:
    #   channels [0 .. latent_channels-1]  — noisy latent  → keep random init
    #   channels [latent_channels .. 2*latent_channels-1]  — Normal cond → zero
    # This makes the model behave like an unconditional denoiser at the start of
    # training and gradually learn to use the spatial conditioning signal.
    with torch.no_grad():
        unet.conv_in.weight[:, latent_channels:, ...].zero_()

    return unet


# ──────────────────────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────────────────────

class LPIPSLoss(nn.Module):
    """
    VGG-based perceptual (LPIPS-style) loss.

    Compatible with torchvision 0.12 (no lpips package required).
    Uses VGG16 features at layers relu1_2, relu2_2, relu3_3, relu4_3.

    Inputs expected in [-1, 1]; internally normalised to ImageNet stats.
    """

    # Layers to extract features from (VGG16 feature indices)
    _LAYERS = [3, 8, 15, 22]   # relu1_2, relu2_2, relu3_3, relu4_3

    def __init__(self, device: str):
        super().__init__()
        self.device = device

        vgg = tv_models.vgg16(pretrained=True).features.to(device)
        for p in vgg.parameters():
            p.requires_grad_(False)
        vgg.eval()

        # Build sequential feature extractors stopping at each selected layer
        self.slices = nn.ModuleList()
        prev = 0
        for layer_idx in self._LAYERS:
            self.slices.append(vgg[prev : layer_idx + 1])
            prev = layer_idx + 1

        # ImageNet normalisation constants
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """[-1,1] → ImageNet-normalised."""
        x = (x + 1.0) / 2.0                    # [0, 1]
        return (x - self.mean) / self.std

    @torch.no_grad()
    def _get_features(self, x: torch.Tensor):
        x = self._preprocess(x)
        feats = []
        for s in self.slices:
            x = s(x)
            feats.append(x)
        return feats

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        pred, target: (B, 3, H, W) in [-1, 1], on self.device
        Returns scalar loss.
        """
        pred   = pred.to(self.device)
        target = target.to(self.device)
        # Features from generated image need gradients; target does not
        pred_x = self._preprocess(pred)
        feats_pred   = []
        feats_target = []
        with torch.no_grad():
            x_t = self._preprocess(target)
        x_p = pred_x

        loss = torch.tensor(0.0, device=self.device)
        for s in self.slices:
            x_p = s(x_p)
            with torch.no_grad():
                x_t = s(x_t.detach())
            # Normalise channels to unit norm (LPIPS convention)
            norm = lambda f: F.normalize(f, p=2, dim=1)
            loss = loss + F.mse_loss(norm(x_p), norm(x_t.detach()))

        return loss / len(self.slices)


class SSIMLoss(nn.Module):
    """
    Differentiable SSIM loss applied only on the *luminance* channel (Y in
    YCbCr).  This allows the chrominance to change freely under exposure
    shifts while penalising structural degradation.

    Returns 1 - SSIM so it can be minimised.
    """

    def __init__(self, window_size: int = 11, device: str = "cpu"):
        super().__init__()
        self.window_size = window_size
        self.device      = device

        # Gaussian kernel
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
        """(B,3,H,W) [-1,1] → (B,1,H,W) [0,1] luminance"""
        x01 = (x + 1.0) / 2.0
        # BT.601 coefficients
        Y = (0.299 * x01[:, 0:1]
             + 0.587 * x01[:, 1:2]
             + 0.114 * x01[:, 2:3])
        return Y

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x, y: (B,1,H,W) in [0,1]"""
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        pad = self.window_size // 2
        kernel = self.kernel.to(x.device)

        mu_x  = F.conv2d(x, kernel, padding=pad, groups=1)
        mu_y  = F.conv2d(y, kernel, padding=pad, groups=1)
        mu_x2 = mu_x ** 2
        mu_y2 = mu_y ** 2
        mu_xy = mu_x * mu_y

        sig_x  = F.conv2d(x * x, kernel, padding=pad, groups=1) - mu_x2
        sig_y  = F.conv2d(y * y, kernel, padding=pad, groups=1) - mu_y2
        sig_xy = F.conv2d(x * y, kernel, padding=pad, groups=1) - mu_xy

        numer = (2 * mu_xy + C1) * (2 * sig_xy + C2)
        denom = (mu_x2 + mu_y2 + C1) * (sig_x + sig_y + C2)
        ssim_map = numer / (denom + 1e-8)
        return ssim_map.mean()

    def forward(
        self, pred: torch.Tensor, normal: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes 1 - SSIM(luma(pred), luma(normal)).
        pred, normal: (B, 3, H, W) in [-1, 1]
        """
        pred   = pred.to(self.device)
        normal = normal.to(self.device)
        luma_p = self._rgb_to_luminance(pred)
        luma_n = self._rgb_to_luminance(normal)
        return 1.0 - self._ssim(luma_p, luma_n)


class HistogramLoss(nn.Module):
    """
    Soft Wasserstein-1 histogram matching loss.

    Encourages the generated image's intensity distribution to match the
    target domain's distribution.  Only useful when pseudo-pairs or paired
    data are available.

    Uses a differentiable soft histogram via Gaussian kernel density
    estimation.
    """

    def __init__(self, bins: int = 64, sigma: float = 0.02, device: str = "cpu"):
        super().__init__()
        self.bins   = bins
        self.sigma  = sigma
        self.device = device
        centers = torch.linspace(0.0, 1.0, bins, device=device)
        self.register_buffer("centers", centers)

    def _soft_histogram(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) in [-1, 1]
        Returns: (B, bins) — normalised soft histogram over all pixels/channels
        """
        x01 = ((x + 1.0) / 2.0).view(x.shape[0], -1)   # (B, 3*H*W)
        centers = self.centers.to(x.device)              # ensure same device
        # KDE: Gaussian kernel around each bin centre
        diff = x01[:, :, None] - centers[None, None, :]  # (B, N, bins)
        weights = torch.exp(-0.5 * (diff / self.sigma) ** 2)
        hist = weights.sum(dim=1)                        # (B, bins)
        hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-7)
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