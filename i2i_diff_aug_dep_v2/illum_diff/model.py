"""UNet builder over the diffusers UNet2DModel.

We do NOT fine-tune Stable Diffusion or any large RGB latent diffusion model.
Instead we train a small, dedicated 1-channel UNet over the L* channel only.

Why a small custom UNet beats SD-style fine-tuning here:
    * SD's VAE is the dominant source of "blur" in I2I pipelines, especially
      when the change is small and high-frequency (specular saturation, dark
      lumen edges). Operating directly in pixel space on a single channel
      removes the VAE entirely.
    * The luminance manifold is far simpler than full natural images, so a
      small UNet trains well on the available data (281/817 frames).
    * 1-channel I/O comfortably fits the 16 GB DGX target (and starts on a
      4 GB RTX 3050) with FP16 + EMA.
"""

from diffusers import UNet2DModel


def build_unet(image_size: int, in_channels: int = 4, out_channels: int = 1,
               base_channels: int = 64, channel_mults=(1, 2, 2, 4, 4),
               num_attn_blocks_from_bottom: int = 2,
               layers_per_block: int = 2,
               norm_num_groups: int = 32) -> UNet2DModel:
    block_out = tuple(int(base_channels) * int(m) for m in channel_mults)
    n = len(block_out)
    if num_attn_blocks_from_bottom > n:
        num_attn_blocks_from_bottom = n

    down_block_types = []
    for i in range(n):
        if i >= n - num_attn_blocks_from_bottom:
            down_block_types.append("AttnDownBlock2D")
        else:
            down_block_types.append("DownBlock2D")

    up_block_types = []
    for i in range(n):
        # i=0 is the deepest up block (mirror of the deepest down block).
        if i < num_attn_blocks_from_bottom:
            up_block_types.append("AttnUpBlock2D")
        else:
            up_block_types.append("UpBlock2D")

    return UNet2DModel(
        sample_size=int(image_size),
        in_channels=int(in_channels),
        out_channels=int(out_channels),
        layers_per_block=int(layers_per_block),
        block_out_channels=block_out,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        norm_num_groups=int(norm_num_groups),
    )
