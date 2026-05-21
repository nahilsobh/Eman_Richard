"""2D FNO and SIREN operators for brain MRE G inversion.

A pair of 2D inverse operators matching the 1D architectures (Fig 5):
spectral-conv FNO and 1D-conv-with-sin SIREN, both adapted to 2D.
Input is (2, 80, 80) -- real and imaginary parts of one component of the
shear-wave displacement field, normalised so ||u||_inf = 1.  Output is
(2, 80, 80) -- the complex shear modulus G / G_scale.

The intention is to drop these into the BBIR pretrain-then-finetune
recipe:

  1. Pretrain on synthetic 2D phantom + real-data brain slices.
  2. Fine-tune with PDE residual (interior nodes only -- BC-agnostic).
  3. Evaluate on held-out subjects.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 2D spectral conv (FNO) ────────────────────────────────────────────────────

class SpectralConv2d(nn.Module):
    """2D Fourier neural operator layer (Li et al. 2020) with truncation."""
    def __init__(self, c_in: int, c_out: int, modes1: int, modes2: int):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (c_in * c_out)
        self.W1 = nn.Parameter(scale * torch.randn(c_in, c_out, modes1, modes2,
                                                    dtype=torch.cfloat))
        self.W2 = nn.Parameter(scale * torch.randn(c_in, c_out, modes1, modes2,
                                                    dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # (B, C, H, W)
        B, C, H, W = x.shape
        Xf = torch.fft.rfft2(x, dim=(-2, -1))                   # (B, C, H, W//2+1)
        Yf = torch.zeros(B, self.W1.shape[1], H, W // 2 + 1,
                          dtype=torch.cfloat, device=x.device)
        m1 = min(self.modes1, H)
        m2 = min(self.modes2, W // 2 + 1)
        Yf[..., :m1, :m2] = torch.einsum(
            "bcij,cdij->bdij", Xf[..., :m1, :m2], self.W1[..., :m1, :m2])
        Yf[..., -m1:, :m2] = torch.einsum(
            "bcij,cdij->bdij", Xf[..., -m1:, :m2], self.W2[..., :m1, :m2])
        return torch.fft.irfft2(Yf, s=(H, W), dim=(-2, -1))


class FNO2dBlock(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spec = SpectralConv2d(width, width, modes1, modes2)
        self.lin  = nn.Conv2d(width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spec(x) + self.lin(x))


class FNO2dBrain(nn.Module):
    """Drop-in 2D FNO for the BBIR brain G-inversion task.

    Default ~1M parameters at width=48, modes=16, n_blocks=4 -- roughly
    10x the 23k 1D model and comparable to small published 2D FNO MRE
    inverters.
    """
    def __init__(self, in_ch: int = 2, out_ch: int = 2,
                 width: int = 48, modes1: int = 16, modes2: int = 16,
                 n_blocks: int = 4):
        super().__init__()
        self.lift   = nn.Conv2d(in_ch + 2, width, 1)            # +2 for x/y coords
        self.blocks = nn.ModuleList([FNO2dBlock(width, modes1, modes2)
                                      for _ in range(n_blocks)])
        self.proj   = nn.Sequential(
            nn.Conv2d(width, width, 1), nn.GELU(),
            nn.Conv2d(width, out_ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # (B, C, H, W)
        B, C, H, W = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, H, device=x.device),
            torch.linspace(0, 1, W, device=x.device),
            indexing="ij",
        )
        coord = torch.stack([xx, yy]).unsqueeze(0).expand(B, 2, H, W)
        x = torch.cat([x, coord], dim=1)
        x = self.lift(x)
        for b in self.blocks:
            x = b(x)
        return self.proj(x)


# ── 2D SIREN (sin-activated CNN) ──────────────────────────────────────────────

class SineLayer2d(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int = 7,
                 is_first: bool = False, omega_0: float = 30.0):
        super().__init__()
        self.omega_0  = omega_0
        self.is_first = is_first
        pad = kernel_size // 2
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, padding=pad)
        with torch.no_grad():
            fan_in = c_in * kernel_size * kernel_size
            if is_first:
                self.conv.weight.uniform_(-1.0 / fan_in, 1.0 / fan_in)
            else:
                b = math.sqrt(6.0 / fan_in) / omega_0
                self.conv.weight.uniform_(-b, b)
            self.conv.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.conv(x))


class SIREN2dBrain(nn.Module):
    """2D analogue of the 1D SIREN used in Fig 5/9. ~600k params at
    width=48, kernel=7, blocks=4."""
    def __init__(self, in_ch: int = 2, out_ch: int = 2,
                 width: int = 48, kernel_size: int = 7, n_blocks: int = 4,
                 first_omega_0: float = 30.0, hidden_omega_0: float = 30.0):
        super().__init__()
        self.first = SineLayer2d(in_ch + 2, width, kernel_size=1,
                                  is_first=True, omega_0=first_omega_0)
        self.hidden = nn.ModuleList([
            SineLayer2d(width, width, kernel_size=kernel_size,
                         is_first=False, omega_0=hidden_omega_0)
            for _ in range(n_blocks)
        ])
        self.last = nn.Conv2d(width, out_ch, 1)
        with torch.no_grad():
            b = math.sqrt(6.0 / width) / hidden_omega_0
            self.last.weight.uniform_(-b, b)
            self.last.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, H, device=x.device),
            torch.linspace(0, 1, W, device=x.device),
            indexing="ij",
        )
        coord = torch.stack([xx, yy]).unsqueeze(0).expand(B, 2, H, W)
        x = torch.cat([x, coord], dim=1)
        x = self.first(x)
        for h in self.hidden:
            x = h(x)
        return self.last(x)


def helmholtz_residual_2d(Y_pred: torch.Tensor, X: torch.Tensor,
                          mask: torch.Tensor, dx: float, freq: float,
                          rho: float, G_scale: float
                          ) -> torch.Tensor:
    """Mean relative |G * lap(u) + rho omega^2 u| / |rho omega^2 u| on
    interior brain voxels.

    Y_pred : (B, 2, H, W)   complex G / G_scale (Re, Im)
    X      : (B, 2, H, W)   complex displacement (Re, Im)
    mask   : (B, 1, H, W)   1 inside brain
    """
    omega = 2.0 * math.pi * freq
    G = (Y_pred[:, 0] + 1j * Y_pred[:, 1]) * G_scale          # (B, H, W)
    u =  X[:, 0]      + 1j * X[:, 1]                           # (B, H, W)

    # 5-point Laplacian on interior pixels
    lap_re = (X[:, 0, 2:, 1:-1] - 2 * X[:, 0, 1:-1, 1:-1] + X[:, 0, :-2, 1:-1]
              + X[:, 0, 1:-1, 2:] - 2 * X[:, 0, 1:-1, 1:-1] + X[:, 0, 1:-1, :-2]
              ) / (dx ** 2)
    lap_im = (X[:, 1, 2:, 1:-1] - 2 * X[:, 1, 1:-1, 1:-1] + X[:, 1, :-2, 1:-1]
              + X[:, 1, 1:-1, 2:] - 2 * X[:, 1, 1:-1, 1:-1] + X[:, 1, 1:-1, :-2]
              ) / (dx ** 2)
    lap_u = lap_re + 1j * lap_im

    G_int = G[:, 1:-1, 1:-1]
    u_int = u[:, 1:-1, 1:-1]
    m_int = mask[:, 0, 1:-1, 1:-1]

    res = G_int * lap_u + rho * omega ** 2 * u_int
    denom = (rho * omega ** 2) * u_int.abs().clamp_min(1e-10)
    rel  = (res.abs() / denom) * m_int
    if m_int.sum() < 1.0:
        return torch.tensor(0.0, device=X.device)
    return rel.sum() / m_int.sum()
