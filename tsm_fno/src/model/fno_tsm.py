"""Dual-head FNO for Tissue Strain Mapping.

Inputs:
    X : (B, in_channels, H, W) — wave field channels + Lamé prior + distance
        We append two normalised grid-coordinate channels (x, y), so the
        first 1×1 lift sees `in_channels + 2` features.

Outputs:
    G   : (B, H, W) shear modulus [Pa], in [G_min, G_max]
    eps : (B, H, W) latent strain  [-], in [0, eps_max]
    A   : (B,)      acoustoelastic constant [-], constrained < -2 (we keep
          the negative-A convention for a global learned scalar; the
          training pipeline supplies positive ground-truth A and the loss
          flips signs when needed — see losses.py)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fourier_block import FourierBlock


def _grid_coords(B: int, H: int, W: int, device) -> torch.Tensor:
    """Return (B, 2, H, W) tensor with normalised x/y coords in [0, 1]."""
    ys = torch.linspace(0.0, 1.0, H, device=device)
    xs = torch.linspace(0.0, 1.0, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
    return grid


class FNO_TSM(nn.Module):
    """Dual-head FNO predicting G(x,y), ε_latent(x,y) and a global A scalar."""

    def __init__(
        self,
        in_channels: int = 6,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 48,
        n_layers: int = 4,
        G_min: float = 200.0,
        G_max: float = 80000.0,
        eps_max: float = 3.0,
    ):
        super().__init__()
        self.G_min = float(G_min)
        self.G_max = float(G_max)
        self.eps_max = float(eps_max)

        self.lift = nn.Conv2d(in_channels + 2, width, 1)
        self.blocks = nn.ModuleList([
            FourierBlock(width, modes1, modes2, activation=(i < n_layers - 1))
            for i in range(n_layers)
        ])

        # Stiffness head
        self.G_proj = nn.Sequential(
            nn.Conv2d(width, 128, 1), nn.GELU(),
            nn.Conv2d(128, 1, 1),
        )

        # Latent strain head
        self.eps_proj = nn.Sequential(
            nn.Conv2d(width, 64, 1), nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )

        # Acoustoelastic constant head — global average pool -> scalar
        # Output is forced into A < -2.0 via -softplus - 2.0
        self.A_proj = nn.Linear(width, 1)

    def _backbone(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared encoder: returns (spatial_feat (B,W,H,W), global_feat (B,W))."""
        B, C, H, W = x.shape
        grid = _grid_coords(B, H, W, x.device)
        x = torch.cat([x, grid], dim=1)
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        return x, x.mean(dim=(-2, -1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spatial, global_feat = self._backbone(x)

        G_raw = self.G_proj(spatial).squeeze(1)
        G = self.G_min + (self.G_max - self.G_min) * torch.sigmoid(G_raw)

        eps_raw = self.eps_proj(spatial).squeeze(1)
        eps = self.eps_max * torch.sigmoid(eps_raw)

        A_raw = self.A_proj(global_feat).squeeze(-1)
        A = -F.softplus(A_raw) - 2.0

        return G, eps, A

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning predictions + latent representations.

        Returns
        -------
        G, eps, A  — same as forward()
        spatial    — (B, width, H, W) per-pixel feature map after all Fourier blocks
        global_vec — (B, width) global-average-pooled embedding
        """
        spatial, global_feat = self._backbone(x)

        G_raw = self.G_proj(spatial).squeeze(1)
        G = self.G_min + (self.G_max - self.G_min) * torch.sigmoid(G_raw)

        eps_raw = self.eps_proj(spatial).squeeze(1)
        eps = self.eps_max * torch.sigmoid(eps_raw)

        A_raw = self.A_proj(global_feat).squeeze(-1)
        A = -F.softplus(A_raw) - 2.0

        return G, eps, A, spatial, global_feat
