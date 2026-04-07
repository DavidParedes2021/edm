# models/embeddings.py
"""
Exposure Conditioning Embeddings.

Maps discrete exposure type (OVER=0, UNDER=1, NULL=2) to a dense vector
that conditions the UNet at every resolution via cross-attention or addition.

Design choices:
  - Sinusoidal timestep embedding (standard DDPM)
  - Learned exposure class embedding (3 classes: over, under, null)
  - Combined via MLP → single conditioning vector
"""
import math
import torch
import torch.nn as nn


class SinusoidalTimestepEmbedding(nn.Module):
    """
    Standard sinusoidal position embedding for diffusion timesteps.
    Projects scalar t → R^{embed_dim}.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        self.embed_dim = embed_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (B,) long tensor
        Returns:
            emb: (B, embed_dim)
        """
        device = timesteps.device
        half_dim = self.embed_dim // 2
        freq = math.log(10000) / (half_dim - 1)
        freq = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32) * -freq
        )
        args = timesteps.float().unsqueeze(1) * freq.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, embed_dim)
        return emb


class ExposureConditionEmbedding(nn.Module):
    """
    Learned embedding for exposure type conditioning.

    Classes:
      0 → OVEREXPOSED
      1 → UNDEREXPOSED
      2 → NULL (unconditional, used for CFG dropout)

    The null class is initialized close to zero to act as a neutral baseline
    during Classifier-Free Guidance unconditional forward passes.
    """
    NUM_CLASSES = 3  # over, under, null

    LABEL_OVER  = 0
    LABEL_UNDER = 1
    LABEL_NULL  = 2

    def __init__(self, embed_dim: int, time_embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

        # Sinusoidal time embedding
        self.time_embed = SinusoidalTimestepEmbedding(embed_dim)
        self.time_proj  = nn.Sequential(
            nn.Linear(embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Learned class embedding
        self.class_embed = nn.Embedding(self.NUM_CLASSES, embed_dim)

        # Initialize null token near zero
        with torch.no_grad():
            self.class_embed.weight[self.LABEL_NULL].fill_(0.0)

        # Fusion MLP: time + class → conditioning vector
        self.fusion = nn.Sequential(
            nn.Linear(time_embed_dim + embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(
        self,
        timesteps: torch.Tensor,
        exposure_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            timesteps:       (B,) long — diffusion step index
            exposure_labels: (B,) long — 0, 1, or 2

        Returns:
            cond: (B, time_embed_dim) — single conditioning vector
        """
        t_emb = self.time_proj(self.time_embed(timesteps))   # (B, time_embed_dim)
        c_emb = self.class_embed(exposure_labels)             # (B, embed_dim)
        combined = torch.cat([t_emb, c_emb], dim=-1)         # (B, time+embed)
        return self.fusion(combined)                           # (B, time_embed_dim)
