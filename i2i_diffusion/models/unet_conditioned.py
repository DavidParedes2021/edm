"""
models/unet_conditioned.py
--------------------------
Class-conditioned U-Net wrapping diffusers 0.14 UNet2DConditionModel.

Design
------
1. Source Normal frame concatenated channel-wise (Palette-style hint).
   in_channels = 4  (3 RGB noisy + 1 luminance slice of source)
2. Class embedding (over=0, under=1, null=2) injected via cross-attention.
3. Unconditional null class for classifier-free guidance (CFG).
4. ControlNet residuals injected externally via forward hooks
   (see controlnet_lite.ControlNetHookContext).

diffusers 0.14 UNet2DConditionModel forward signature
------------------------------------------------------
    unet(sample, timestep, encoder_hidden_states) -> UNet2DConditionOutput
    .sample  ->  (B, out_channels, H, W)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel


class ClassConditionedUNet(nn.Module):
    """
    Parameters
    ----------
    num_classes : int
        Number of target exposure classes. Default 2 (over / under).
    class_embed_dim : int
        Dimension of the class embedding fed into cross-attention.
        Must equal cross_attention_dim of the inner U-Net.
    in_channels : int
        Input channels of the noisy tensor fed to the U-Net.
        4 = 3-channel noisy target + 1 source luminance hint (training).
    image_size : int
        Spatial resolution (H = W) of the training images.
    block_out_channels : tuple[int, ...]
        Output channels for each U-Net encoder stage.
        Length determines the number of down/up sampling stages.
        Must be consistent with what ControlNetLite uses.
    layers_per_block : int
        Number of ResNet layers inside each encoder/decoder block.
    dropout : float
        Dropout probability applied inside the U-Net.
    """

    def __init__(
        self,
        num_classes:        int   = 2,
        class_embed_dim:    int   = 512,
        in_channels:        int   = 4,
        image_size:         int   = 256,
        block_out_channels: tuple = (128, 256, 512, 512),
        layers_per_block:   int   = 2,
    ) -> None:
        super().__init__()

        self.num_classes     = num_classes
        self.class_embed_dim = class_embed_dim
        self.null_class_idx  = num_classes   # reserved index for CFG null token

        # +1 embedding slot for the null / unconditional class
        self.class_emb = nn.Embedding(num_classes + 1, class_embed_dim)

        n_stages = len(block_out_channels)

        # attention_head_dim must divide every stage channel count that uses
        # attention. Pick the largest power-of-2 <= 8 that divides the
        # smallest stage channel count.
        min_ch = min(block_out_channels)
        attention_head_dim = next(
            h for h in [8, 4, 2, 1] if min_ch % h == 0
        )

        # First stage: plain DownBlock (no attention — too spatially large).
        # All subsequent stages use attention.
        down_block_types = (
            ("DownBlock2D",)
            + ("AttnDownBlock2D",) * (n_stages - 1)
        )
        up_block_types = (
            ("AttnUpBlock2D",) * (n_stages - 1)
            + ("UpBlock2D",)
        )

        # dropout is NOT a top-level kwarg in diffusers 0.14's
        # UNet2DConditionModel — it is set internally per-block.
        # Passing it raises TypeError, so we drop it here.
        self.unet = UNet2DConditionModel(
            sample_size         = image_size,
            in_channels         = in_channels,
            out_channels        = 3,
            layers_per_block    = layers_per_block,
            block_out_channels  = block_out_channels,
            down_block_types    = down_block_types,
            up_block_types      = up_block_types,
            cross_attention_dim = class_embed_dim,
            attention_head_dim  = attention_head_dim,
        )

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        noisy_sample:         torch.Tensor,           # (B, 3, H, W)
        source_image:         torch.Tensor,           # (B, 3|1, H, W)
        timestep:             torch.Tensor,           # (B,)
        class_labels:         Optional[torch.Tensor] = None,  # (B,) long
        controlnet_residuals: Optional[list]         = None,  # consumed by hooks
    ) -> torch.Tensor:
        """
        Returns predicted noise of shape (B, 3, H, W).

        class_labels=None uses the null embedding for the unconditioned
        branch of classifier-free guidance.
        """
        B = noisy_sample.shape[0]

        # 1. Concatenate source luminance hint as 4th input channel
        if source_image.shape[1] > 1:
            hint = source_image[:, :1, :, :]   # luminance slice  (B, 1, H, W)
        else:
            hint = source_image
        x = torch.cat([noisy_sample, hint], dim=1)   # (B, 4, H, W)

        # 2. Class embedding -> cross-attention context  (B, 1, class_embed_dim)
        if class_labels is None:
            idx = torch.full(
                (B,), self.null_class_idx,
                dtype=torch.long, device=noisy_sample.device,
            )
        else:
            idx = class_labels
        ctx = self.class_emb(idx).unsqueeze(1)   # (B, 1, class_embed_dim)

        # 3. ControlNet residuals are injected via external forward hooks
        #    (ControlNetHookContext) -- nothing to do here explicitly.
        _ = controlnet_residuals

        out = self.unet(
            sample                = x,
            timestep              = timestep,
            encoder_hidden_states = ctx,
        )
        return out.sample   # (B, 3, H, W)

    # ── classifier-free guidance helper ───────────────────────────────────────

    @torch.no_grad()
    def cfg_forward(
        self,
        noisy_sample:   torch.Tensor,
        source_image:   torch.Tensor,
        timestep:       torch.Tensor,
        class_labels:   torch.Tensor,
        guidance_scale: float = 5.0,
    ) -> torch.Tensor:
        """
        Single denoising step with CFG:
            eps_hat = eps_uncond + gamma * (eps_cond - eps_uncond)
        """
        eps_cond   = self.forward(noisy_sample, source_image, timestep, class_labels)
        eps_uncond = self.forward(noisy_sample, source_image, timestep, None)
        return eps_uncond + guidance_scale * (eps_cond - eps_uncond)