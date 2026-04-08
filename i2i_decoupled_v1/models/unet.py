# models/unet.py
"""
Conditional UNet for Luminance Diffusion.

Input:  noisy luminance x_t  (B, 1, H, W)   ← single channel (Y)
        conditioning Y_normal (B, 1, H, W)   ← normal frame luminance
        timestep t            (B,)
        exposure_label        (B,)            ← 0=over, 1=under, 2=null

Output: predicted noise ε    (B, 1, H, W)   ← same shape as input

Architecture highlights:
  - Input is concat(x_t, Y_cond) → 2-channel input to first conv
  - ResNet blocks with Group Normalization (stable for small batch sizes)
  - Self-attention at specified resolutions (captures global exposure patterns)
  - Conditioning via scale-shift (AdaGN): each ResBlock modulates its GN
    with the conditioning vector — this is the primary control mechanism
  - Skip connections for sharpness preservation
  - Gradient checkpointing support (saves ~30% VRAM at cost of ~20% speed)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.embeddings import ExposureConditionEmbedding


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class GroupNorm32(nn.GroupNorm):
    """GroupNorm with 32 groups (standard for diffusion models)."""
    def __init__(self, channels: int):
        # Use min(32, channels) groups to handle small channel counts
        num_groups = min(32, channels)
        # Ensure divisibility
        while channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        super().__init__(num_groups, channels)


def zero_module(module: nn.Module) -> nn.Module:
    """Zero-initialize module weights (final layer of each ResBlock)."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class ResBlock(nn.Module):
    """
    ResNet block with AdaGN (Adaptive Group Normalization) conditioning.
    AdaGN: y = scale(cond) * GroupNorm(x) + shift(cond)
    This is the control mechanism — the exposure embedding modulates every
    residual block's normalization, injecting the exposure type at all scales.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        dropout: float = 0.1,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.in_channels  = in_channels
        self.out_channels = out_channels

        self.norm1 = GroupNorm32(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # AdaGN projection: cond → (scale, shift) for norm2
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_channels * 2),  # *2 for scale + shift
        )

        self.norm2 = GroupNorm32(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1))

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def _forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        # AdaGN: conditioning → scale, shift
        cond_out = self.cond_proj(cond)                      # (B, out*2)
        scale, shift = cond_out.chunk(2, dim=1)             # each (B, out)
        scale = scale[:, :, None, None]                      # (B, out, 1, 1)
        shift = shift[:, :, None, None]

        h = self.norm2(h) * (1.0 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, cond, use_reentrant=True)
        return self._forward(x, cond)


class SelfAttentionBlock(nn.Module):
    """
    Multi-head self-attention at spatial positions.
    Allows the model to reason about global exposure patterns
    (e.g., blown-out highlights or crushed shadows across the frame).
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = GroupNorm32(channels)
        self.qkv  = nn.Conv1d(channels, channels * 3, 1)
        self.proj = zero_module(nn.Conv1d(channels, channels, 1))
        self.scale = (channels // num_heads) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, -1)            # (B, C, HW)

        qkv = self.qkv(h)                           # (B, 3C, HW)
        q, k, v = qkv.chunk(3, dim=1)              # each (B, C, HW)

        # Reshape for multi-head attention
        head_dim = C // self.num_heads
        q = q.view(B, self.num_heads, head_dim, -1).permute(0,1,3,2)
        k = k.view(B, self.num_heads, head_dim, -1).permute(0,1,3,2)
        v = v.view(B, self.num_heads, head_dim, -1).permute(0,1,3,2)

        attn = torch.softmax(q @ k.transpose(-2,-1) * self.scale, dim=-1)
        out  = (attn @ v).permute(0,1,3,2).reshape(B, C, -1)

        out = self.proj(out).view(B, C, H, W)
        return x + out


# ---------------------------------------------------------------------------
# Encoder / Decoder Levels
# ---------------------------------------------------------------------------

class DownBlock(nn.Module):
    """Encoder level: N ResBlocks [+ optional attention] + downsample."""
    def __init__(
        self,
        in_ch: int, out_ch: int, cond_dim: int,
        num_res_blocks: int, use_attention: bool,
        dropout: float, use_checkpoint: bool,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            self.res_blocks.append(
                ResBlock(
                    in_ch if i == 0 else out_ch,
                    out_ch, cond_dim, dropout, use_checkpoint,
                )
            )
        self.attn = SelfAttentionBlock(out_ch) if use_attention else None
        self.downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, cond):
        for block in self.res_blocks:
            x = block(x, cond)
        if self.attn is not None:
            x = self.attn(x)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """Decoder level: upsample + skip concat + N ResBlocks [+ optional attention]."""
    def __init__(
        self,
        in_ch: int, skip_ch: int, out_ch: int, cond_dim: int,
        num_res_blocks: int, use_attention: bool,
        dropout: float, use_checkpoint: bool,
    ):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            block_in = (in_ch + skip_ch) if i == 0 else out_ch
            self.res_blocks.append(
                ResBlock(block_in, out_ch, cond_dim, dropout, use_checkpoint)
            )
        self.attn = SelfAttentionBlock(out_ch) if use_attention else None

    def forward(self, x, skip, cond):
        x = self.upsample(x)
        # Handle potential size mismatch from odd dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
        x = torch.cat([x, skip], dim=1)
        for block in self.res_blocks:
            x = block(x, cond)
        if self.attn is not None:
            x = self.attn(x)
        return x


# ---------------------------------------------------------------------------
# Main UNet
# ---------------------------------------------------------------------------

class IlluminationUNet(nn.Module):
    """
    Conditional UNet for Luminance-domain Illumination Diffusion.

    This model takes:
      - x_t:        noisy luminance at timestep t     (B, 1, H, W)
      - condition_y: clean luminance of normal frame  (B, 1, H, W)
      - t:           diffusion timestep               (B,)
      - label:       exposure type (0=over, 1=under)  (B,)

    Input to first conv: cat(x_t, condition_y) → (B, 2, H, W)
    """

    def __init__(
        self,
        base_channels: int = 64,
        channel_multipliers: tuple = (1, 2, 4, 8),
        attention_resolutions: tuple = (32, 16, 8),
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        exposure_embed_dim: int = 256,
        image_size: int = 256,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.image_size = image_size

        # time_embed_dim = 4 * base_channels (standard)
        time_embed_dim = base_channels * 4

        # Conditioning: timestep + exposure class → single vector
        self.condition_embed = ExposureConditionEmbedding(
            embed_dim=exposure_embed_dim,
            time_embed_dim=time_embed_dim,
        )

        # Determine which levels use attention
        # attention_resolutions: spatial sizes (e.g. 32, 16, 8 pixels)
        # We'll track current spatial size through the encoder
        levels = len(channel_multipliers)
        current_res = image_size
        use_attn_at = []
        for _ in range(levels):
            use_attn_at.append(current_res in attention_resolutions)
            current_res //= 2

        # Input projection: 2-channel (x_t + condition_y) → base_channels
        self.input_conv = nn.Conv2d(2, base_channels, 3, padding=1)

        # Build channel schedule
        ch_schedule = [base_channels * m for m in channel_multipliers]

        # ---- Encoder ----
        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        for i, out_ch in enumerate(ch_schedule):
            self.down_blocks.append(DownBlock(
                in_ch=in_ch, out_ch=out_ch, cond_dim=time_embed_dim,
                num_res_blocks=num_res_blocks, use_attention=use_attn_at[i],
                dropout=dropout, use_checkpoint=use_checkpoint,
            ))
            in_ch = out_ch

        # ---- Bottleneck ----
        self.mid_block1 = ResBlock(in_ch, in_ch, time_embed_dim, dropout, use_checkpoint)
        self.mid_attn   = SelfAttentionBlock(in_ch)
        self.mid_block2 = ResBlock(in_ch, in_ch, time_embed_dim, dropout, use_checkpoint)

        # ---- Decoder ----
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(levels)):
            skip_ch = ch_schedule[i]
            out_ch  = base_channels if i == 0 else ch_schedule[i - 1]
            self.up_blocks.append(UpBlock(
                in_ch=in_ch, skip_ch=skip_ch, out_ch=out_ch,
                cond_dim=time_embed_dim,
                num_res_blocks=num_res_blocks + 1,  # +1 extra in decoder
                use_attention=use_attn_at[i],
                dropout=dropout, use_checkpoint=use_checkpoint,
            ))
            in_ch = out_ch

        # Output projection: base_channels → 1 (luminance)
        self.out_norm = GroupNorm32(in_ch)
        self.out_conv = zero_module(nn.Conv2d(in_ch, 1, 3, padding=1))

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        exposure_label: torch.Tensor,
        condition_y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:              (B, 1, H, W) noisy luminance
            timesteps:      (B,) long
            exposure_label: (B,) long — 0, 1, or 2
            condition_y:    (B, 1, H, W) normal frame luminance
        Returns:
            pred_noise:     (B, 1, H, W)
        """
        # --- Conditioning vector ---
        cond = self.condition_embed(timesteps, exposure_label)  # (B, time_embed_dim)

        # --- Input: concatenate noisy + condition ---
        x_in = torch.cat([x, condition_y], dim=1)  # (B, 2, H, W)
        h = self.input_conv(x_in)                  # (B, base_ch, H, W)

        # --- Encoder ---
        skips = []
        for block in self.down_blocks:
            h, skip = block(h, cond)
            skips.append(skip)

        # --- Bottleneck ---
        h = self.mid_block1(h, cond)
        h = self.mid_attn(h)
        h = self.mid_block2(h, cond)

        # --- Decoder ---
        for block in self.up_blocks:
            skip = skips.pop()
            h = block(h, skip, cond)

        # --- Output ---
        h = F.silu(self.out_norm(h))
        return self.out_conv(h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)