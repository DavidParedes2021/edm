"""
Conditional UNet operating on the Y channel.

Inputs:
    y_t      : (B, 1, H, W) -- noisy target Y at timestep t
    y_cond   : (B, 1, H, W) -- structural condition (the source normal Y)
    t        : (B,)          -- diffusion timestep
    cls      : (B,)          -- class label, 0=normal, 1=under, 2=null (CFG)

Output:
    eps      : (B, 1, H, W) -- predicted noise

Conditioning:
    Time and class are mapped to a shared embedding vector that modulates
    every ResBlock through Adaptive GroupNorm (FiLM-style scale + shift).
    The class embedding has an extra "null" slot used for CFG dropout.

Param budget: ~31M @ base=64, mults=[1,2,3,4]. Activation memory at
256x256, batch=8, fp16 fits comfortably under 16 GB.
"""
from __future__ import annotations
import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- helpers ----------------------------------------------------------------

def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Standard DDPM timestep embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def _largest_divisor_le(n: int, target: int) -> int:
    """Largest integer <= target that divides n. Used to choose GroupNorm groups."""
    g = min(target, n)
    while n % g != 0:
        g -= 1
    return max(g, 1)


# ---- building blocks --------------------------------------------------------

class AdaGN(nn.Module):
    """GroupNorm without affine, then FiLM modulation from the (t, cls) embedding."""

    def __init__(self, num_channels: int, emb_dim: int, num_groups: int = 32):
        super().__init__()
        groups = _largest_divisor_le(num_channels, num_groups)
        self.norm = nn.GroupNorm(groups, num_channels, affine=False)
        self.proj = nn.Linear(emb_dim, num_channels * 2)
        # init scale modulation to 0 -> AdaGN starts as plain GroupNorm
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        scale, shift = self.proj(F.silu(emb)).chunk(2, dim=-1)
        return h * (1.0 + scale[..., None, None]) + shift[..., None, None]


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = AdaGN(in_ch, emb_dim)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = AdaGN(out_ch, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        # zero-init the second conv -> identity init for the residual branch
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x, emb)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h, emb))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        if channels % num_heads != 0:
            num_heads = _largest_divisor_le(channels, num_heads)
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        groups = _largest_divisor_le(channels, 32)
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(dim=1)                               # (B, h, d, N)
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v).reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    def forward(self, x): return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode='nearest'))


# ---- the UNet ---------------------------------------------------------------

class ConditionalYUNet(nn.Module):
    """
    Conditional UNet for 1-channel diffusion targets with 1-channel image
    conditioning concatenated channel-wise.
    """
    def __init__(
        self,
        in_channels: int = 2,           # y_t + y_cond
        out_channels: int = 1,          # eps
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 3, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Sequence[int] = (32,),  # feature-map sizes
        num_real_classes: int = 2,
        image_size: int = 256,
        dropout: float = 0.1,
        use_grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_real_classes = num_real_classes
        self.null_cls = num_real_classes  # last index reserved for CFG dropout
        self.use_grad_checkpointing = use_grad_checkpointing
        self.num_res_blocks = num_res_blocks

        # ---- embeddings ----
        self.t_emb_dim_in = base_channels
        emb_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(self.t_emb_dim_in, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.class_emb = nn.Embedding(num_real_classes + 1, emb_dim)
        nn.init.normal_(self.class_emb.weight, std=0.02)

        # ---- input projection ----
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # ---- down path ----
        cur_res = image_size
        ch = base_channels
        skip_channels: List[int] = [ch]  # one skip per "thing pushed"

        self.downs = nn.ModuleList()
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                blk = nn.ModuleList([ResBlock(ch, out_ch, emb_dim, dropout)])
                ch = out_ch
                if cur_res in attention_resolutions:
                    blk.append(SelfAttention(ch))
                self.downs.append(blk)
                skip_channels.append(ch)
            if i != len(channel_mults) - 1:
                self.downs.append(nn.ModuleList([Downsample(ch)]))
                skip_channels.append(ch)
                cur_res //= 2

        # ---- bottleneck ----
        self.mid = nn.ModuleList([
            ResBlock(ch, ch, emb_dim, dropout),
            SelfAttention(ch),
            ResBlock(ch, ch, emb_dim, dropout),
        ])

        # ---- up path ----
        # We pop skips one-per-"thing pushed". Number of pops = number of pushes.
        # Per-level pops: num_res_blocks + 1 (consumes the level's RB skips
        # plus the previous-level Downsample skip).
        self.ups = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                blk = nn.ModuleList([ResBlock(ch + skip_ch, out_ch, emb_dim, dropout)])
                ch = out_ch
                if cur_res in attention_resolutions:
                    blk.append(SelfAttention(ch))
                self.ups.append(blk)
            if i != len(channel_mults) - 1:
                self.ups.append(nn.ModuleList([Upsample(ch)]))
                cur_res *= 2

        assert not skip_channels, f"unconsumed skips: {skip_channels}"

        # ---- output ----
        groups = _largest_divisor_le(ch, 32)
        self.out_norm = nn.GroupNorm(groups, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    # -------- helpers --------
    def _is_resblock(self, m: nn.Module) -> bool:
        return isinstance(m, ResBlock)

    def _run_block(self, layers: nn.ModuleList, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        for layer in layers:
            if self._is_resblock(layer):
                if self.use_grad_checkpointing and self.training:
                    h = torch.utils.checkpoint.checkpoint(layer, h, emb, use_reentrant=False)
                else:
                    h = layer(h, emb)
            else:
                h = layer(h)
        return h

    # -------- forward --------
    def forward(self, y_t: torch.Tensor, y_cond: torch.Tensor,
                t: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(sinusoidal_timestep_embedding(t, self.t_emb_dim_in))
        emb = emb + self.class_emb(cls)

        h = self.in_conv(torch.cat([y_t, y_cond], dim=1))
        skips: List[torch.Tensor] = [h]

        # Down
        for layers in self.downs:
            h = self._run_block(layers, h, emb)
            skips.append(h)

        # Mid
        h = self._run_block(self.mid, h, emb)

        # Up
        for layers in self.ups:
            first = layers[0]
            if isinstance(first, ResBlock):
                h = torch.cat([h, skips.pop()], dim=1)
            h = self._run_block(layers, h, emb)

        return self.out_conv(F.silu(self.out_norm(h)))


# ---- self-test --------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    net = ConditionalYUNet(
        base_channels=32,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks=1,
        attention_resolutions=(16,),
        image_size=64,
    )
    n_params = sum(p.numel() for p in net.parameters())
    print(f"params: {n_params/1e6:.2f} M")

    y_t = torch.randn(2, 1, 64, 64)
    y_c = torch.randn(2, 1, 64, 64)
    t = torch.randint(0, 1000, (2,))
    c = torch.tensor([0, 1])
    out = net(y_t, y_c, t, c)
    print(f"out: {tuple(out.shape)}")
    assert out.shape == y_t.shape
    print("model.py OK")
