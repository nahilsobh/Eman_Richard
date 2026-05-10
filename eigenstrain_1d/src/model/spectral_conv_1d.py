"""1D spectral convolution layer for FNO.

MKL FFT backward is broken on this system (Intel oneMKL DFTI ERROR).
torch.bmm backward also segfaults for large (m, B, C) batches on this system.

Workaround: precomputed DFT basis (avoids FFT), matmul+reshape for DFT steps
(4× faster than einsum), and einsum for the weight multiply (stable backward).

Forward:
    y_r = x.reshape(B*C, N) @ F_r.T  → reshape (B, C, m)   real DFT coeffs
    y_i = x.reshape(B*C, N) @ F_i.T  → reshape (B, C, m)   imag DFT coeffs
    complex multiply: einsum("bcm,com->bom", y_r, Wr) etc.
    IDFT: out_r.reshape(B*O, m) @ F_r  → reshape (B, O, N)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


class SpectralConv1d(nn.Module):
    """1D Fourier layer via precomputed DFT matrices (avoids MKL FFT autograd)."""

    def __init__(self, in_channels: int, out_channels: int, modes: int,
                 N: int = 256):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.N = N

        scale = 1.0 / (in_channels * out_channels)
        self.weights_r = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes))
        self.weights_i = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes))

        # Precompute DFT basis: F[k,n] = exp(-2πi*k*n/N)
        k = np.arange(modes, dtype=np.float64)[:, None]
        n = np.arange(N,     dtype=np.float64)[None, :]
        phase = -2.0 * np.pi * k * n / N
        F_r = torch.tensor(np.cos(phase), dtype=torch.float32)  # (modes, N)
        F_i = torch.tensor(np.sin(phase), dtype=torch.float32)
        self.register_buffer("F_r", F_r)
        self.register_buffer("F_i", F_i)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N)  real-valued
        B, C, N = x.shape
        O = self.out_channels
        m = min(self.modes, N // 2)
        F_r = self.F_r[:m]   # (m, N)
        F_i = self.F_i[:m]

        # DFT forward: matmul reshape is 4× faster than einsum on this system
        # (B*C, N) @ (N, m) → (B, C, m)
        xf = x.reshape(B * C, N)
        y_r = (xf @ F_r.T).reshape(B, C, m)
        y_i = (xf @ F_i.T).reshape(B, C, m)

        # Complex multiply with learnable weights
        # Use einsum — bmm backward segfaults on this system (MKL issue)
        Wr = self.weights_r[:, :, :m]   # (C, O, m)
        Wi = self.weights_i[:, :, :m]
        out_r = (torch.einsum("bcm,com->bom", y_r, Wr)
                 - torch.einsum("bcm,com->bom", y_i, Wi))
        out_i = (torch.einsum("bcm,com->bom", y_r, Wi)
                 + torch.einsum("bcm,com->bom", y_i, Wr))

        # IDFT: matmul reshape for inverse
        # (B*O, m) @ (m, N) → (B, O, N)
        orf = out_r.reshape(B * O, m)
        oif = out_i.reshape(B * O, m)
        out = ((orf @ F_r) + (oif @ (-F_i))).reshape(B, O, N) / N
        return out  # (B, out_channels, N)
