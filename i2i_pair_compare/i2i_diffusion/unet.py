"""Compact conditional UNet for single-channel L-residual diffusion.

Inputs:
    x:       (B, in_ch, H, W)   noisy_target_L concat [cond_L, depth]
    t:       (B,)               diffusion timesteps
    y:       (B,)               class labels (0=over, 1=under). num_classes
                                slot is reserved for the unconditional token.

Output:
    eps_pred (B, out_ch, H, W)

Design notes:
- GroupNorm + SiLU + 3x3 convs, FiLM time/class injection.
- Channel mults define a 4-level UNet; attention only at low-res levels
  to keep VRAM in check on a 16GB GPU.
- Optional class-free guidance: passing y == num_classes uses a learned
  null embedding so the same network does conditional + unconditional.
"""
from __future__ import annotations

import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    groups = max_groups
    while groups > 1 and channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels, eps=1e-6)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("dim must be even")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.pos = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pos(t))


class ResBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        emb_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = _gn(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.norm2 = _gn(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        if channels % num_heads != 0:
            num_heads = max(1, channels // 32)
            while channels % num_heads != 0:
                num_heads -= 1
        self.num_heads = num_heads
        self.norm = _gn(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)  # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)
        # reshape to (B, heads, head_dim, N)
        head_dim = C // self.num_heads
        q = q.reshape(B, self.num_heads, head_dim, H * W)
        k = k.reshape(B, self.num_heads, head_dim, H * W)
        v = v.reshape(B, self.num_heads, head_dim, H * W)
        # scaled dot product on last dim
        scale = head_dim ** -0.5
        attn = torch.einsum("bhcn,bhcm->bhnm", q, k) * scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhnm,bhcm->bhcn", attn, v)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,         # noisy_target + cond_L + depth
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mult: Sequence[int] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Sequence[int] = (16, 8),
        dropout: float = 0.1,
        num_classes: int = 2,
        time_emb_dim: int = 256,
        input_resolution: int = 256,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        emb_dim = time_emb_dim
        self.time_embed = TimeEmbedding(base_channels, emb_dim)
        # +1 slot for the null token used by classifier-free guidance
        self.class_embed = nn.Embedding(num_classes + 1, emb_dim)

        # input projection
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # ── Encoder ───────────────────────────────────────────────────────
        self.down_blocks = nn.ModuleList()
        skip_channels: List[int] = [base_channels]
        ch = base_channels
        cur_res = input_resolution
        attn_set = set(int(r) for r in attn_resolutions)
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                block = nn.ModuleList([ResBlock(ch, out_ch, emb_dim, dropout)])
                ch = out_ch
                if cur_res in attn_set:
                    block.append(SelfAttention(ch))
                self.down_blocks.append(block)
                skip_channels.append(ch)
            if level != len(channel_mult) - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(ch)]))
                skip_channels.append(ch)
                cur_res //= 2

        # ── Bottleneck ────────────────────────────────────────────────────
        self.mid_block1 = ResBlock(ch, ch, emb_dim, dropout)
        self.mid_attn = SelfAttention(ch)
        self.mid_block2 = ResBlock(ch, ch, emb_dim, dropout)

        # ── Decoder ───────────────────────────────────────────────────────
        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            for i in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                block = nn.ModuleList(
                    [ResBlock(ch + skip_ch, out_ch, emb_dim, dropout)]
                )
                ch = out_ch
                if cur_res in attn_set:
                    block.append(SelfAttention(ch))
                self.up_blocks.append(block)
            if level != 0:
                self.up_blocks.append(nn.ModuleList([Upsample(ch)]))
                cur_res *= 2

        # ── Output ────────────────────────────────────────────────────────
        self.out_norm = _gn(ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self.time_embed(t)
        if y is None:
            # treat as null token
            y = torch.full((x.shape[0],), self.num_classes, device=x.device, dtype=torch.long)
        emb = emb + self.class_embed(y)

        h = self.in_conv(x)
        skips: List[torch.Tensor] = [h]

        for block in self.down_blocks:
            modules = list(block)
            if isinstance(modules[0], (Downsample,)):
                h = modules[0](h)
            else:
                h = modules[0](h, emb)
                for m in modules[1:]:
                    h = m(h)
            skips.append(h)

        h = self.mid_block1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, emb)

        for block in self.up_blocks:
            modules = list(block)
            if isinstance(modules[0], Upsample):
                h = modules[0](h)
            else:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = modules[0](h, emb)
                for m in modules[1:]:
                    h = m(h)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)
