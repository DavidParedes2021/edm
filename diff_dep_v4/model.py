"""UNet builder.

A 1-channel class-conditional UNet for diffusing the Y channel.
Class layout: 0=normal, 1=overexposed, 2=underexposed, 3=null (CFG).

We use diffusers 0.14.0's ``UNet2DModel`` with ``resnet_time_scale_shift="scale_shift"``
to enable AdaGN-style modulation: the (timestep + class) embedding produces a per-block
scale/shift on GroupNorm features. This injects the exposure-class signal at every
ResNet block, which is what gives strong, controllable conditioning.
"""
from __future__ import annotations

from typing import Any

from diffusers import UNet2DModel


# Order of (down/up) block types in this 5-level UNet:
#   resolution per level: 256 -> 128 -> 64 -> 32 -> 16 -> mid(8)
#
# Self-attention is *only* in the mid block (8x8 = 64 tokens, ~free) -- it is
# added automatically by UNet2DModel at the bottleneck. We deliberately do
# NOT use AttnDown/AttnUpBlock2D anywhere on the sides: at 32x32 the attention
# has 1024 tokens, and diffusers casts the softmax to fp32, so for batch 8
# with 32 heads the scores tensor alone is 8 * 32 * 1024 * 1024 * 4 B = 1 GiB.
# That's exactly the cumulative footprint that OOMs a 16 GB GPU during the
# up-path under AMP. For Y-only luminance diffusion the bottleneck attention
# is enough; nothing on the sides needs global mixing.
_DOWN_BLOCK_TYPES = (
    "DownBlock2D",
    "DownBlock2D",
    "DownBlock2D",
    "DownBlock2D",
    "DownBlock2D",
)
_UP_BLOCK_TYPES = (
    "UpBlock2D",
    "UpBlock2D",
    "UpBlock2D",
    "UpBlock2D",
    "UpBlock2D",
)


def build_unet(
    image_size: int,
    block_out_channels: tuple[int, ...] = (128, 128, 256, 256, 512),
    layers_per_block: int = 2,
    attention_head_dim: int = 8,
    norm_num_groups: int = 32,
    resnet_time_scale_shift: str = "scale_shift",
    num_class_embeds: int = 4,            # 3 real classes + 1 null for CFG
    **overrides: Any,
) -> UNet2DModel:
    config = dict(
        sample_size=image_size,
        in_channels=1,
        out_channels=1,
        center_input_sample=False,
        time_embedding_type="positional",
        freq_shift=0,
        flip_sin_to_cos=True,
        down_block_types=_DOWN_BLOCK_TYPES,
        up_block_types=_UP_BLOCK_TYPES,
        block_out_channels=tuple(block_out_channels),
        layers_per_block=layers_per_block,
        attention_head_dim=attention_head_dim,
        norm_num_groups=norm_num_groups,
        norm_eps=1e-5,
        resnet_time_scale_shift=resnet_time_scale_shift,
        num_class_embeds=num_class_embeds,
    )
    config.update(overrides or {})
    return UNet2DModel(**config)