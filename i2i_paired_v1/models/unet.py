"""
models/unet.py
Palette-style conditional U-Net for illumination artifact generation.

Key design choices vs. vanilla DDPM U-Net:
  1. Input channels = 6  (3 noisy target + 3 concatenated normal frame)
  2. Exposure MLP embeds a scalar exposure label → same dim as timestep embedding.
     Both are *added* so AdaGN receives combined time+exposure context.
  3. v-prediction output (resolves blurry generations).
  4. Attention at multiple scales for global context.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Sinusoidal time embedding  (standard DDPM)
# ──────────────────────────────────────────────────────────────────────────────

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] long or float → [B, dim]"""
        device    = t.device
        half_dim  = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None].float() * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


# ──────────────────────────────────────────────────────────────────────────────
# Exposure MLP  — maps scalar float ∈ [-1, +1] → same embedding space as time
# ──────────────────────────────────────────────────────────────────────────────

class ExposureMLP(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, exposure: torch.Tensor) -> torch.Tensor:
        """exposure: [B] float → [B, out_dim]"""
        return self.net(exposure.unsqueeze(-1))  # [B,1] → [B, out_dim]


# ──────────────────────────────────────────────────────────────────────────────
# Basic building blocks
# ──────────────────────────────────────────────────────────────────────────────

def _norm(channels: int) -> nn.GroupNorm:
    num_groups = min(32, channels)
    # Make sure channels is divisible by num_groups
    while channels % num_groups != 0 and num_groups > 1:
        num_groups //= 2
    return nn.GroupNorm(num_groups, channels)


class ResBlock(nn.Module):
    """
    Residual block with AdaGN conditioning on (time + exposure) embedding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = _norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # Linear projection of embedding → scale + shift for AdaGN
        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels * 2),  # *2 for scale & shift
        )

        self.norm2   = _norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2   = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        x   : [B, C, H, W]
        emb : [B, emb_dim]
        """
        h = self.conv1(F.silu(self.norm1(x)))

        # AdaGN: compute scale & shift from embedding
        scale_shift = self.emb_proj(emb)           # [B, out_ch*2]
        scale_shift = scale_shift[:, :, None, None]  # [B, out_ch*2, 1, 1]
        scale, shift = scale_shift.chunk(2, dim=1)

        h = self.norm2(h) * (1 + scale) + shift
        h = self.dropout(F.silu(h))
        h = self.conv2(h)

        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention block (single head, efficient)."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = _norm(channels)
        self.qkv  = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h   = self.norm(x)
        qkv = self.qkv(h)                              # [B, 3C, H, W]
        q, k, v = qkv.chunk(3, dim=1)                  # each [B, C, H, W]

        # Flatten spatial
        q = q.reshape(B, C, H * W).permute(0, 2, 1)   # [B, HW, C]
        k = k.reshape(B, C, H * W).permute(0, 2, 1)
        v = v.reshape(B, C, H * W).permute(0, 2, 1)

        scale = C ** -0.5
        attn  = torch.bmm(q, k.permute(0, 2, 1)) * scale  # [B, HW, HW]
        attn  = attn.softmax(dim=-1)
        out   = torch.bmm(attn, v)                         # [B, HW, C]

        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(out)


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
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ──────────────────────────────────────────────────────────────────────────────
# Main U-Net
# ──────────────────────────────────────────────────────────────────────────────

class IlluminationUNet(nn.Module):
    """
    Palette-style U-Net for conditional illumination I2I diffusion.

    Input:  [noisy_target (3ch) | normal_cond (3ch)] = 6 channels
    Output: [3ch] predicted velocity (v) or noise (ε)

    Args:
        image_size            : spatial resolution (H = W)
        base_channels         : channel width at first level
        channel_multipliers   : per-level multiplier list, e.g. [1,2,4,8]
        num_res_blocks        : ResBlocks per level
        attention_resolutions : spatial resolutions at which to add attention
        dropout               : dropout rate in ResBlocks
        timesteps             : total diffusion steps (for positional embed init)
    """

    def __init__(
        self,
        image_size: int            = 256,
        base_channels: int         = 128,
        channel_multipliers: List[int] = None,
        num_res_blocks: int        = 2,
        attention_resolutions: List[int] = None,
        dropout: float             = 0.1,
        timesteps: int             = 1000,
        in_channels: int           = 6,   # 3 noisy + 3 condition
        out_channels: int          = 3,
    ):
        super().__init__()

        if channel_multipliers is None:
            channel_multipliers = [1, 2, 4, 8]
        if attention_resolutions is None:
            attention_resolutions = [32, 16, 8]

        self.image_size           = image_size
        self.base_channels        = base_channels
        self.channel_multipliers  = channel_multipliers
        self.num_res_blocks       = num_res_blocks
        self.attention_resolutions = set(attention_resolutions)

        # ── Embedding pipeline ────────────────────────────────────────
        emb_dim = base_channels * 4

        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.exposure_embed = nn.Sequential(
            ExposureMLP(emb_dim),
        )
        # Both embeddings will be summed → single combined embedding
        # No extra params needed; already in emb_dim space.

        # ── Encoder ───────────────────────────────────────────────────
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down_blocks: nn.ModuleList = nn.ModuleList()
        self.down_samples: nn.ModuleList = nn.ModuleList()

        ch        = base_channels
        skip_chs  = [ch]       # track channels for skip connections
        curr_res  = image_size

        for level, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            level_blocks = nn.ModuleList()

            for _ in range(num_res_blocks):
                level_blocks.append(ResBlock(ch, out_ch, emb_dim, dropout))
                ch = out_ch
                if curr_res in self.attention_resolutions:
                    level_blocks.append(AttentionBlock(ch))
                skip_chs.append(ch)

            self.down_blocks.append(level_blocks)

            if level < len(channel_multipliers) - 1:
                self.down_samples.append(Downsample(ch))
                curr_res = curr_res // 2
            else:
                self.down_samples.append(nn.Identity())

        # ── Bottleneck ────────────────────────────────────────────────
        self.mid_block1   = ResBlock(ch, ch, emb_dim, dropout)
        self.mid_attention = AttentionBlock(ch)
        self.mid_block2   = ResBlock(ch, ch, emb_dim, dropout)

        # ── Decoder ───────────────────────────────────────────────────
        self.up_blocks: nn.ModuleList  = nn.ModuleList()
        self.up_samples: nn.ModuleList = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = base_channels * mult
            level_blocks = nn.ModuleList()

            for i in range(num_res_blocks):
                skip_ch = skip_chs.pop()
                level_blocks.append(ResBlock(ch + skip_ch, out_ch, emb_dim, dropout))
                ch = out_ch
                if curr_res in self.attention_resolutions:
                    level_blocks.append(AttentionBlock(ch))

            self.up_blocks.append(level_blocks)

            if level > 0:
                self.up_samples.append(Upsample(ch))
                curr_res = curr_res * 2
            else:
                self.up_samples.append(nn.Identity())

        # ── Output head ───────────────────────────────────────────────
        self.out_norm = _norm(ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    # ── Forward ───────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,            # [B, 3, H, W] noisy target
        t: torch.Tensor,            # [B] timestep indices
        cond: torch.Tensor,         # [B, 3, H, W] normal frame (condition)
        exposure: torch.Tensor,     # [B] float in [-1, +1]
    ) -> torch.Tensor:

        # ── Build combined embedding ──────────────────────────────────
        t_emb   = self.time_embed(t)                   # [B, emb_dim]
        exp_emb = self.exposure_embed(exposure)        # [B, emb_dim]
        emb     = t_emb + exp_emb                      # [B, emb_dim]

        # ── Concat input with condition ───────────────────────────────
        h = torch.cat([x, cond], dim=1)                # [B, 6, H, W]
        h = self.input_conv(h)                          # [B, base_ch, H, W]

        # ── Encoder forward ───────────────────────────────────────────
        skips = []
        for level_blocks, down_op in zip(self.down_blocks, self.down_samples):
            for block in level_blocks:
                if isinstance(block, ResBlock):
                    h = block(h, emb)
                    skips.append(h)
                else:  # AttentionBlock
                    h = block(h)
            h = down_op(h)

        # ── Bottleneck ────────────────────────────────────────────────
        h = self.mid_block1(h, emb)
        h = self.mid_attention(h)
        h = self.mid_block2(h, emb)

        # ── Decoder forward ───────────────────────────────────────────
        for level_blocks, up_op in zip(self.up_blocks, self.up_samples):
            for block in level_blocks:
                if isinstance(block, ResBlock):
                    skip = skips.pop()
                    h    = torch.cat([h, skip], dim=1)
                    h    = block(h, emb)
                else:  # AttentionBlock
                    h = block(h)
            h = up_op(h)

        # ── Output ────────────────────────────────────────────────────
        h = self.out_conv(F.silu(self.out_norm(h)))
        return h


# ──────────────────────────────────────────────────────────────────────────────
# EMA helper
# ──────────────────────────────────────────────────────────────────────────────

class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model  = model
        self.decay  = decay
        self.shadow: dict = {}
        self.backup: dict = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data
                )

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data        = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state: dict):
        self.shadow = state["shadow"]
        self.decay  = state["decay"]


# ──────────────────────────────────────────────────────────────────────────────
# Quick instantiation helper
# ──────────────────────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> IlluminationUNet:
    model_cfg = cfg["model"]
    return IlluminationUNet(
        image_size           = cfg["data"]["image_size"],
        base_channels        = model_cfg["base_channels"],
        channel_multipliers  = model_cfg["channel_multipliers"],
        num_res_blocks       = model_cfg["num_res_blocks"],
        attention_resolutions= model_cfg["attention_resolutions"],
        dropout              = model_cfg["dropout"],
        timesteps            = model_cfg["timesteps"],
    )


if __name__ == "__main__":
    # Smoke test
    B, H = 2, 64
    net  = IlluminationUNet(image_size=H, base_channels=32,
                            channel_multipliers=[1,2,2],
                            num_res_blocks=1,
                            attention_resolutions=[16],
                            dropout=0.0, timesteps=100)
    x    = torch.randn(B, 3, H, H)
    cond = torch.randn(B, 3, H, H)
    t    = torch.randint(0, 100, (B,))
    exp  = torch.tensor([-1.0, 0.5])
    out  = net(x, t, cond, exp)
    print(f"U-Net output: {out.shape}")   # should be [2, 3, 64, 64]

    params = sum(p.numel() for p in net.parameters())
    print(f"Parameters: {params:,}")
