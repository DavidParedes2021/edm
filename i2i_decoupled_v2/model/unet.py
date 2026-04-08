"""
model/unet.py
--------------
Class-conditioned UNet for luminance-channel diffusion.

Architecture decisions
----------------------
1. Input channels = 2  (noisy L_target  +  L_normal as spatial conditioning).
   Concatenating the clean normal-L to the noisy input gives the model a
   spatial anchor – the structural content it must preserve while shifting
   exposure.  This is the ControlNet-lite strategy: cheap and very effective.

2. Class embedding (0=over, 1=under) is injected at every residual block via
   AdaGN (Adaptive Group Normalization), which modulates both scale and shift
   of the feature maps.  This gives the model a *strong, global* signal about
   which exposure direction to target.

3. Self-attention at 32×32 and 16×16 (and 8×8 on DGX) captures long-range
   luminance relationships, critical for uniform global brightness shifts.

4. All convolutions use GroupNorm (not BatchNorm) – stable with small batches.

5. The output is 1 channel: the predicted noise in the L channel only.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility blocks
# ---------------------------------------------------------------------------

def zero_module(module: nn.Module) -> nn.Module:
    """Zero-initialise weights – standard trick for residual paths."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embedding (Vaswani et al.)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


# ---------------------------------------------------------------------------
# Time + class embedding MLP
# ---------------------------------------------------------------------------

class TimeClassEmbedding(nn.Module):
    """
    Projects (timestep sinusoidal embedding + class embedding) into a shared
    latent for AdaGN conditioning.
    """
    def __init__(self, model_channels: int, num_classes: int):
        super().__init__()
        self.time_proj = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels * 4),
        )
        self.class_emb = nn.Embedding(num_classes + 1, model_channels * 4)  # +1 for null class (CFG)

    def forward(self, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        t_emb = timestep_embedding(t, self.time_proj[0].in_features).to(t.device)
        return self.time_proj(t_emb) + self.class_emb(c)


# ---------------------------------------------------------------------------
# Adaptive Group Norm
# ---------------------------------------------------------------------------

def _valid_groups(num_channels: int, preferred: int = 32) -> int:
    """Return the largest divisor of num_channels that is ≤ preferred."""
    for g in range(min(preferred, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


class AdaGN(nn.Module):
    """Adaptive Group Normalization: scale/shift from embedding vector."""
    def __init__(self, num_groups: int, in_channels: int, emb_channels: int):
        super().__init__()
        real_groups = _valid_groups(in_channels, num_groups)
        self.norm = nn.GroupNorm(real_groups, in_channels, affine=False)
        self.proj = nn.Linear(emb_channels, in_channels * 2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(emb).chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        return self.norm(x) * (1.0 + scale) + shift


# ---------------------------------------------------------------------------
# Residual block with AdaGN conditioning
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        dropout: float = 0.0,
        num_groups: int = 32,
    ):
        super().__init__()
        self.norm1 = AdaGN(min(num_groups, in_channels), in_channels, emb_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm2 = AdaGN(min(num_groups, out_channels), out_channels, emb_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1))

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x, emb))
        h = self.conv1(h)
        h = self.act(self.norm2(h, emb))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Self-attention block (efficient: computes over spatial positions)
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        # Clamp heads so channels is divisible
        while channels % num_heads != 0:
            num_heads //= 2
        self.norm  = nn.GroupNorm(_valid_groups(channels, 32), channels)
        self.qkv   = nn.Conv1d(channels, channels * 3, 1)
        self.proj  = zero_module(nn.Conv1d(channels, channels, 1))
        self.heads = num_heads
        self.scale = (channels // num_heads) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, -1)        # [B, C, HW]
        qkv = self.qkv(h)                       # [B, 3C, HW]
        q, k, v = qkv.chunk(3, dim=1)          # each [B, C, HW]

        # Split heads
        q = q.view(B, self.heads, C // self.heads, -1)
        k = k.view(B, self.heads, C // self.heads, -1)
        v = v.view(B, self.heads, C // self.heads, -1)

        attn = torch.einsum("bhci,bhcj->bhij", q, k) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum("bhij,bhcj->bhci", attn, v)
        out = out.reshape(B, C, -1)
        out = self.proj(out)
        return (out + h).view(B, C, H, W)


# ---------------------------------------------------------------------------
# Down / Up sampling
# ---------------------------------------------------------------------------

class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# Full UNet
# ---------------------------------------------------------------------------

class IlluminationUNet(nn.Module):
    """
    Conditional UNet for illumination diffusion.

    Input  : [B, 2, H, W]  (noisy L_target  ||  clean L_normal)
    Output : [B, 1, H, W]  (predicted noise in L_target channel)
    """

    def __init__(
        self,
        image_size: int        = 256,
        in_channels: int       = 2,    # noisy_L + cond_L
        out_channels: int      = 1,
        base_channels: int     = 128,
        channel_mult: List[int] = (1, 2, 3, 4),
        attention_resolutions: List[int] = (16, 8),
        num_res_blocks: int    = 2,
        dropout: float         = 0.1,
        num_classes: int       = 2,
        num_groups: int        = 32,
    ):
        super().__init__()
        self.image_size = image_size

        emb_ch = base_channels * 4
        self.time_class_emb = TimeClassEmbedding(base_channels, num_classes)

        # ---- Input projection ----
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # ---- Encoder ----
        ch = base_channels
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        skip_channels: List[int] = [ch]

        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            res = image_size // (2 ** level)
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(ch, out_ch, emb_ch, dropout, num_groups))
                skip_channels.append(out_ch)
                if res in attention_resolutions:
                    self.down_blocks.append(SelfAttention(out_ch))
                    skip_channels.append(out_ch)  # attention doesn't change ch but we still push skip
                ch = out_ch
            if level != len(channel_mult) - 1:
                self.down_samples.append(Downsample(ch))
                skip_channels.append(ch)
            else:
                self.down_samples.append(nn.Identity())  # no downsample at last level

        # ---- Bottleneck ----
        self.mid_block1  = ResBlock(ch, ch, emb_ch, dropout, num_groups)
        self.mid_attn    = SelfAttention(ch)
        self.mid_block2  = ResBlock(ch, ch, emb_ch, dropout, num_groups)

        # ---- Decoder ----
        self.up_blocks   = nn.ModuleList()
        self.up_samples  = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            res = image_size // (2 ** level)
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                in_ch   = ch + skip_ch
                self.up_blocks.append(ResBlock(in_ch, out_ch, emb_ch, dropout, num_groups))
                if res in attention_resolutions:
                    self.up_blocks.append(SelfAttention(out_ch))
                ch = out_ch
            if level != 0:
                self.up_samples.append(Upsample(ch))
            else:
                self.up_samples.append(nn.Identity())

        # ---- Output ----
        self.out_norm = nn.GroupNorm(min(num_groups, ch), ch)
        self.out_act  = nn.SiLU()
        self.out_conv = zero_module(nn.Conv2d(ch, out_channels, 3, padding=1))

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,           # [B, 2, H, W]  noisy_L || cond_L
        t: torch.Tensor,           # [B]  timesteps
        c: torch.Tensor,           # [B]  class labels (0=over,1=under; num_classes=null)
    ) -> torch.Tensor:
        emb = self.time_class_emb(t, c)  # [B, emb_ch]

        h = self.input_conv(x)
        skips = [h]

        # ---- Encoder pass ----
        down_idx  = 0
        samp_idx  = 0
        for block in self.down_blocks:
            if isinstance(block, ResBlock):
                h = block(h, emb)
            else:  # SelfAttention
                h = block(h)
            skips.append(h)

        # Apply downsamples at correct points (between levels)
        # Rebuild with explicit level tracking
        h, skips = self._encoder_forward(x, emb)

        # Bottleneck
        h = self.mid_block1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, emb)

        # Decoder
        h = self._decoder_forward(h, skips, emb)

        return self.out_conv(self.out_act(self.out_norm(h)))

    def _encoder_forward(self, x: torch.Tensor, emb: torch.Tensor):
        """Structured encoder traversal returning (last_feat, skip_list)."""
        h = self.input_conv(x)
        skips = [h]

        block_iter  = iter(self.down_blocks)
        sample_iter = iter(self.down_samples)

        cfg = list(zip(
            [hasattr(b, 'conv1') for b in self.down_blocks],  # is ResBlock
        ))

        # We iterate manually to interleave ResBlocks, optional Attentions, and Downsamples
        remaining = list(self.down_blocks)
        level_boundaries = self._level_boundaries()

        pos = 0
        for level_blocks, downsample in zip(level_boundaries, self.down_samples):
            for block in level_blocks:
                if isinstance(block, ResBlock):
                    h = block(h, emb)
                else:
                    h = block(h)
                skips.append(h)
            h = downsample(h)
            if not isinstance(downsample, nn.Identity):
                skips.append(h)

        return h, skips

    def _decoder_forward(self, h: torch.Tensor, skips: list, emb: torch.Tensor) -> torch.Tensor:
        block_iter  = iter(self.up_blocks)
        sample_iter = iter(self.up_samples)

        level_boundaries_up = self._level_boundaries_up()

        for level_blocks, upsample in zip(level_boundaries_up, self.up_samples):
            for block in level_blocks:
                if isinstance(block, ResBlock):
                    skip = skips.pop()
                    h    = torch.cat([h, skip], dim=1)
                    h    = block(h, emb)
                else:
                    h    = block(h)
            h = upsample(h)

        return h

    def _level_boundaries(self):
        """Group down_blocks by encoder level."""
        from model.unet import ResBlock, SelfAttention  # local import to avoid circular
        groups = []
        current = []
        res_count = 0
        for block in self.down_blocks:
            current.append(block)
            if isinstance(block, ResBlock):
                res_count += 1
        # Simple: return all as one group and rely on downsamples list for structure
        # Rebuild properly below
        return self._group_blocks(self.down_blocks)

    def _level_boundaries_up(self):
        return self._group_blocks(self.up_blocks)

    @staticmethod
    def _group_blocks(blocks):
        """
        Group blocks into levels by tracking ResBlock counts.
        Each level ends after num_res_blocks ResBlocks (+optional attentions).
        This is a heuristic; the UNet is built so levels naturally alternate.
        """
        # Since we have a flat list, group by encountering Downsample markers
        # Instead, we partition by the structure embedded during init.
        # We'll use a simple approach: collect until we've seen a Downsample-worthy boundary.
        # Because UNet construction appended level-by-level, we can just return all blocks
        # in one group – the downsamples list handles level transitions.
        return [blocks]  # single group; downsamples handles level boundaries


# ---------------------------------------------------------------------------
# Simplified, robust UNet (avoids the complex grouping above)
# ---------------------------------------------------------------------------

class IlluminationUNetV2(nn.Module):
    """
    Cleaner, fully explicit UNet with no dynamic grouping.
    This is the version actually used in training.
    """

    def __init__(
        self,
        image_size: int         = 256,
        base_channels: int      = 128,
        channel_mult: Tuple     = (1, 2, 3, 4),
        attention_resolutions: Tuple = (16, 8),
        num_res_blocks: int     = 2,
        dropout: float          = 0.1,
        num_classes: int        = 2,
    ):
        super().__init__()
        self.image_size = image_size
        num_groups = 32
        emb_ch = base_channels * 4

        self.time_class_emb = TimeClassEmbedding(base_channels, num_classes)

        # Input: noisy_L (1ch) + cond_L_normal (1ch) = 2 channels
        self.input_conv = nn.Conv2d(2, base_channels, 3, padding=1)

        # ── Build encoder ──────────────────────────────────────────────
        self.enc_blocks: nn.ModuleList  = nn.ModuleList()
        self.enc_attns:  nn.ModuleList  = nn.ModuleList()
        self.downsamples: nn.ModuleList = nn.ModuleList()
        enc_ch_list: List[int]          = []

        ch = base_channels
        res = image_size
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            level_res_blocks = nn.ModuleList()
            level_attn_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_res_blocks.append(
                    ResBlock(ch, out_ch, emb_ch, dropout, min(num_groups, out_ch))
                )
                if res in attention_resolutions:
                    level_attn_blocks.append(SelfAttention(out_ch))
                else:
                    level_attn_blocks.append(nn.Identity())
                ch = out_ch

            self.enc_blocks.append(level_res_blocks)
            self.enc_attns.append(level_attn_blocks)
            enc_ch_list.append(ch)

            if level != len(channel_mult) - 1:
                self.downsamples.append(Downsample(ch))
                res = res // 2
            else:
                self.downsamples.append(nn.Identity())

        # ── Bottleneck ──────────────────────────────────────────────────
        self.mid1 = ResBlock(ch, ch, emb_ch, dropout, min(num_groups, ch))
        self.mid_attn = SelfAttention(ch)
        self.mid2 = ResBlock(ch, ch, emb_ch, dropout, min(num_groups, ch))

        # ── Build decoder ───────────────────────────────────────────────
        self.dec_blocks:  nn.ModuleList = nn.ModuleList()
        self.dec_attns:   nn.ModuleList = nn.ModuleList()
        self.upsamples:   nn.ModuleList = nn.ModuleList()

        res = image_size // (2 ** (len(channel_mult) - 1))
        for level in reversed(range(len(channel_mult))):
            mult   = channel_mult[level]
            out_ch = base_channels * mult
            skip_ch = enc_ch_list[level]
            in_ch_first = ch + skip_ch

            level_res_blocks = nn.ModuleList()
            level_attn_blocks = nn.ModuleList()
            for rb_idx in range(num_res_blocks + 1):
                in_c = in_ch_first if rb_idx == 0 else out_ch
                level_res_blocks.append(
                    ResBlock(in_c, out_ch, emb_ch, dropout, min(num_groups, out_ch))
                )
                if res in attention_resolutions:
                    level_attn_blocks.append(SelfAttention(out_ch))
                else:
                    level_attn_blocks.append(nn.Identity())
                ch = out_ch

            self.dec_blocks.append(level_res_blocks)
            self.dec_attns.append(level_attn_blocks)

            if level != 0:
                self.upsamples.append(Upsample(ch))
                res = res * 2
            else:
                self.upsamples.append(nn.Identity())

        # ── Output ──────────────────────────────────────────────────────
        self.out_norm = nn.GroupNorm(_valid_groups(ch, num_groups), ch)
        self.out_act  = nn.SiLU()
        self.out_conv = zero_module(nn.Conv2d(ch, 1, 3, padding=1))

    def forward(
        self,
        x: torch.Tensor,    # [B, 2, H, W]
        t: torch.Tensor,    # [B]
        c: torch.Tensor,    # [B]
    ) -> torch.Tensor:
        emb = self.time_class_emb(t, c)

        h = self.input_conv(x)
        enc_skips: List[torch.Tensor] = []

        # Encoder
        for res_blocks, attn_blocks, down in zip(
            self.enc_blocks, self.enc_attns, self.downsamples
        ):
            for res_blk, attn_blk in zip(res_blocks, attn_blocks):
                h = res_blk(h, emb)
                if not isinstance(attn_blk, nn.Identity):
                    h = attn_blk(h)
            enc_skips.append(h)
            h = down(h)

        # Bottleneck
        h = self.mid1(h, emb)
        h = self.mid_attn(h)
        h = self.mid2(h, emb)

        # Decoder
        for res_blocks, attn_blocks, up in zip(
            self.dec_blocks, self.dec_attns, self.upsamples
        ):
            skip = enc_skips.pop()
            h    = torch.cat([h, skip], dim=1)
            for rb_idx, (res_blk, attn_blk) in enumerate(zip(res_blocks, attn_blocks)):
                h = res_blk(h, emb)
                if not isinstance(attn_blk, nn.Identity):
                    h = attn_blk(h)

            h = up(h)

        return self.out_conv(self.out_act(self.out_norm(h)))
