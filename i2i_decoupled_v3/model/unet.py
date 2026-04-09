"""
model/unet.py  (v3 — conditioning-collapse fixes)
---------------------------------------------------
Root cause of spatial conditioning collapse:
  The model receives [noisy_L, cond_L] concatenated as 2 channels.
  During DDIM denoising at high-t (t≈980), noise amplitude ≈ 33× signal.
  The model can trivially minimise diffusion loss by predicting noise
  based only on the noisy channel, completely ignoring cond_L.

Fixes:
  1. DEDICATED CONDITIONING ENCODER: cond_L is passed through a separate
     shallow encoder whose features are added to EVERY encoder level via
     Feature-wise Linear Modulation (FiLM-style). This forces the model
     to use the conditioning at every scale, not just at the input.
     
  2. STRONGER INPUT SEPARATION: The input conv now processes noisy_L (1ch)
     and cond_L (1ch) through SEPARATE convolutions before summing.
     This prevents the model from treating them as interchangeable channels.

  3. ZERO-INIT CONDITIONING PATH: The conditioning encoder's output layers
     are zero-initialised so training starts from the unconditioned baseline
     and gradually learns to use the structure cue (curriculum-style).

  4. AdaGN uses _valid_groups to always find a valid group divisor.
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
def _valid_groups(num_channels: int, preferred: int = 32) -> int:
    for g in range(min(preferred, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half  = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


# ---------------------------------------------------------------------------
class TimeClassEmbedding(nn.Module):
    def __init__(self, model_channels: int, num_classes: int):
        super().__init__()
        self.time_proj = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels * 4),
        )
        # +1 for null (CFG) class
        self.class_emb = nn.Embedding(num_classes + 1, model_channels * 4)

    def forward(self, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        t_emb = timestep_embedding(t, self.time_proj[0].in_features)
        return self.time_proj(t_emb) + self.class_emb(c)


class AdaGN(nn.Module):
    def __init__(self, num_groups: int, in_channels: int, emb_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(_valid_groups(in_channels, num_groups), in_channels, affine=False)
        self.proj = nn.Linear(emb_channels, in_channels * 2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(emb).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_ch: int, dropout: float = 0.0):
        super().__init__()
        self.norm1   = AdaGN(32, in_ch,  emb_ch)
        self.conv1   = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.norm2   = AdaGN(32, out_ch, emb_ch)
        self.drop    = nn.Dropout2d(dropout)
        self.conv2   = zero_module(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.skip    = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act     = nn.SiLU()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x, emb))
        h = self.conv1(h)
        h = self.act(self.norm2(h, emb))
        h = self.drop(h)
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        while channels % num_heads != 0:
            num_heads //= 2
        self.norm  = nn.GroupNorm(_valid_groups(channels, 32), channels)
        self.qkv   = nn.Conv1d(channels, channels * 3, 1)
        self.proj  = zero_module(nn.Conv1d(channels, channels, 1))
        self.heads = num_heads
        self.scale = (channels // num_heads) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h   = self.norm(x).view(B, C, -1)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(B, self.heads, C // self.heads, -1)
        k = k.view(B, self.heads, C // self.heads, -1)
        v = v.view(B, self.heads, C // self.heads, -1)
        attn = (torch.einsum("bhci,bhcj->bhij", q, k) * self.scale).softmax(dim=-1)
        out  = torch.einsum("bhij,bhcj->bhci", attn, v).reshape(B, C, -1)
        return (self.proj(out) + h).view(B, C, H, W)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x): return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x): return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


# ---------------------------------------------------------------------------
# Conditioning Encoder
# ---------------------------------------------------------------------------

class ConditioningEncoder(nn.Module):
    """
    Shallow encoder for the clean cond_L.

    Produces multi-scale feature maps that are added to the main UNet's
    encoder features at each level. This forces structure conditioning
    at every scale and prevents the model from ignoring cond_L.

    All output convolutions are zero-initialised so the model starts
    from the unconditioned baseline and gradually learns to use cond.
    """
    def __init__(self, channel_mult: Tuple, base_channels: int):
        super().__init__()
        self.input_conv = nn.Conv2d(1, base_channels, 3, padding=1)
        self.act        = nn.SiLU()

        self.levels = nn.ModuleList()
        self.projs  = nn.ModuleList()   # zero-init projections

        ch = base_channels
        for mult in channel_mult:
            out_ch = base_channels * mult
            self.levels.append(nn.Sequential(
                nn.Conv2d(ch, out_ch, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.SiLU(),
            ))
            self.projs.append(zero_module(nn.Conv2d(out_ch, out_ch, 1)))
            ch = out_ch

        self.downsamples = nn.ModuleList([
            nn.AvgPool2d(2) for _ in range(len(channel_mult) - 1)
        ] + [nn.Identity()])

    def forward(self, cond: torch.Tensor) -> List[torch.Tensor]:
        """Returns list of feature maps, one per encoder level."""
        h      = self.act(self.input_conv(cond))
        feats  = []
        for level_net, proj, down in zip(self.levels, self.projs, self.downsamples):
            h = level_net(h)
            feats.append(proj(h))
            h = down(h)
        return feats   # [feat_level0, feat_level1, ...]


# ---------------------------------------------------------------------------
# Main UNet  (v3)
# ---------------------------------------------------------------------------

class IlluminationUNetV2(nn.Module):
    """
    Conditional UNet v3.

    Input  : noisy_L [B,1,H,W]  +  cond_L [B,1,H,W]  (passed separately)
    Output : noise_pred [B,1,H,W]

    The cond_L is processed by ConditioningEncoder whose multi-scale features
    are ADDED to each encoder level's output, ensuring the model never
    loses track of the input structure even at high noise levels.
    """

    def __init__(
        self,
        image_size:            int         = 256,
        base_channels:         int         = 128,
        channel_mult:          Tuple       = (1, 2, 3, 4),
        attention_resolutions: Tuple       = (16, 8),
        num_res_blocks:        int         = 2,
        dropout:               float       = 0.1,
        num_classes:           int         = 2,
    ):
        super().__init__()
        self.image_size    = image_size
        self.channel_mult  = channel_mult
        emb_ch             = base_channels * 4

        self.time_class_emb = TimeClassEmbedding(base_channels, num_classes)

        # ── Separate input paths ──────────────────────────────────────
        # noisy_L and cond_L processed separately, then summed.
        # Prevents the model treating them as interchangeable channels.
        self.noisy_in = nn.Conv2d(1, base_channels, 3, padding=1)
        self.cond_in  = zero_module(nn.Conv2d(1, base_channels, 3, padding=1))

        # ── Conditioning encoder (multi-scale structure cue) ──────────
        self.cond_encoder = ConditioningEncoder(channel_mult, base_channels)

        # ── Encoder ──────────────────────────────────────────────────
        self.enc_blocks:  nn.ModuleList = nn.ModuleList()
        self.enc_attns:   nn.ModuleList = nn.ModuleList()
        self.downsamples: nn.ModuleList = nn.ModuleList()
        enc_ch_list: List[int] = []

        ch  = base_channels
        res = image_size
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            level_rbs = nn.ModuleList()
            level_attn = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_rbs.append(ResBlock(ch, out_ch, emb_ch, dropout))
                level_attn.append(
                    SelfAttention(out_ch) if res in attention_resolutions else nn.Identity()
                )
                ch = out_ch
            self.enc_blocks.append(level_rbs)
            self.enc_attns.append(level_attn)
            enc_ch_list.append(ch)
            if level != len(channel_mult) - 1:
                self.downsamples.append(Downsample(ch)); res //= 2
            else:
                self.downsamples.append(nn.Identity())

        # ── Bottleneck ────────────────────────────────────────────────
        self.mid1     = ResBlock(ch, ch, emb_ch, dropout)
        self.mid_attn = SelfAttention(ch)
        self.mid2     = ResBlock(ch, ch, emb_ch, dropout)

        # ── Decoder ───────────────────────────────────────────────────
        self.dec_blocks:  nn.ModuleList = nn.ModuleList()
        self.dec_attns:   nn.ModuleList = nn.ModuleList()
        self.upsamples:   nn.ModuleList = nn.ModuleList()

        res = image_size // (2 ** (len(channel_mult) - 1))
        for level in reversed(range(len(channel_mult))):
            mult   = channel_mult[level]
            out_ch = base_channels * mult
            skip_ch = enc_ch_list[level]
            level_rbs = nn.ModuleList()
            level_attn = nn.ModuleList()
            for rb_idx in range(num_res_blocks + 1):
                in_c = (ch + skip_ch) if rb_idx == 0 else out_ch
                level_rbs.append(ResBlock(in_c, out_ch, emb_ch, dropout))
                level_attn.append(
                    SelfAttention(out_ch) if res in attention_resolutions else nn.Identity()
                )
                ch = out_ch
            self.dec_blocks.append(level_rbs)
            self.dec_attns.append(level_attn)
            if level != 0:
                self.upsamples.append(Upsample(ch)); res *= 2
            else:
                self.upsamples.append(nn.Identity())

        # ── Output ────────────────────────────────────────────────────
        self.out_norm = nn.GroupNorm(_valid_groups(ch, 32), ch)
        self.out_act  = nn.SiLU()
        self.out_conv = zero_module(nn.Conv2d(ch, 1, 3, padding=1))

    def forward(
        self,
        x: torch.Tensor,   # [B, 2, H, W]  — channel 0: noisy_L, channel 1: cond_L
        t: torch.Tensor,   # [B]
        c: torch.Tensor,   # [B]
    ) -> torch.Tensor:
        noisy_L = x[:, 0:1]   # [B,1,H,W]
        cond_L  = x[:, 1:2]   # [B,1,H,W]

        emb = self.time_class_emb(t, c)

        # Separate input encodings, summed
        h = self.noisy_in(noisy_L) + self.cond_in(cond_L)

        # Multi-scale conditioning features
        cond_feats = self.cond_encoder(cond_L)   # list[tensor] per level

        # ── Encoder ──────────────────────────────────────────────────
        enc_skips = []
        for level, (rbs, attns, down) in enumerate(
            zip(self.enc_blocks, self.enc_attns, self.downsamples)
        ):
            for rb, attn in zip(rbs, attns):
                h = rb(h, emb)
                if not isinstance(attn, nn.Identity):
                    h = attn(h)
            # Add conditioning feature at this level
            h = h + cond_feats[level]
            enc_skips.append(h)
            h = down(h)

        # ── Bottleneck ────────────────────────────────────────────────
        h = self.mid1(h, emb)
        h = self.mid_attn(h)
        h = self.mid2(h, emb)

        # ── Decoder ───────────────────────────────────────────────────
        for rbs, attns, up in zip(self.dec_blocks, self.dec_attns, self.upsamples):
            skip = enc_skips.pop()
            h    = torch.cat([h, skip], dim=1)
            for rb, attn in zip(rbs, attns):
                h = rb(h, emb)
                if not isinstance(attn, nn.Identity):
                    h = attn(h)
            h = up(h)

        return self.out_conv(self.out_act(self.out_norm(h)))