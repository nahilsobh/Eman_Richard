"""Fourier block: spectral conv + 1×1 bypass + InstanceNorm + GeLU (Phase 0 reuse)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectral_conv import SpectralConv2d


class FourierBlock(nn.Module):
    def __init__(self, width: int, modes1: int = 12, modes2: int = 12,
                 activation: bool = True):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.conv = nn.Conv2d(width, width, 1)
        self.norm = nn.InstanceNorm2d(width, affine=True)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.spectral(x) + self.conv(x))
        if self.activation:
            x = F.gelu(x)
        return x
