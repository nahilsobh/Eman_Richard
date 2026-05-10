"""2D spectral convolution layer (Phase 0 reuse).

Multiplies the low-frequency Fourier modes of an input tensor by a learnable
complex weight, mirroring `mre_pipeline.src.fno_model.SpectralConv2d`.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 modes1: int = 12, modes2: int = 12):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2,
                                dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2,
                                dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            B, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1,
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2,
        )
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
