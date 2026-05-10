"""1D FNO: u(x) → ε*(x) eigenstrain inversion."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectral_conv_1d import SpectralConv1d


class FNO1d(nn.Module):
    """1D Fourier Neural Operator for eigenstrain inversion.

    Input:  (B, 5, N)  — Re(u60), Im(u60), Re(u120), Im(u120), Lamé prior
    Output: (B, N)     — ε*(x) ∈ [0, eps_max]
    """

    def __init__(self, modes: int = 32, width: int = 64, n_layers: int = 4,
                 in_channels: int = 5, eps_max: float = 0.12):
        super().__init__()
        self.eps_max = eps_max

        # Lift: (in_channels + 1 grid coord) → width
        self.fc_in = nn.Linear(in_channels + 1, width)

        # Fourier blocks
        self.spectral_convs = nn.ModuleList([
            SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.bypass_convs = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size=1) for _ in range(n_layers)])
        self.norms = nn.ModuleList([
            nn.InstanceNorm1d(width) for _ in range(n_layers)])

        # Project: width → 128 → 1
        self.fc_out1 = nn.Linear(width, 128)
        self.fc_out2 = nn.Linear(128, 1)

        # Binary classification head: expanding vs control
        self.classifier = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (eps_pred (B,N), expand_logit (B,))."""
        B, C, N = x.shape

        # Grid coordinate channel
        grid = torch.linspace(0, 1, N, device=x.device)
        grid = grid.unsqueeze(0).unsqueeze(0).expand(B, 1, N)
        x = torch.cat([x, grid], dim=1)              # (B, 6, N)

        # Lift
        x = x.permute(0, 2, 1)                       # (B, N, 6)
        x = self.fc_in(x)                             # (B, N, width)
        x = x.permute(0, 2, 1)                       # (B, width, N)

        # Fourier blocks
        for spec, bypass, norm in zip(self.spectral_convs,
                                       self.bypass_convs,
                                       self.norms):
            residual = x
            x = norm(F.gelu(spec(x) + bypass(x))) + residual

        # Global average for classification
        g = x.mean(dim=-1)                            # (B, width)
        expand_logit = self.classifier(g).squeeze(-1) # (B,)

        # Project to eigenstrain
        x = x.permute(0, 2, 1)                       # (B, N, width)
        x = F.gelu(self.fc_out1(x))                  # (B, N, 128)
        x = self.fc_out2(x).squeeze(-1)              # (B, N)

        eps_pred = torch.sigmoid(x) * self.eps_max   # (B, N) ∈ [0, eps_max]
        return eps_pred, expand_logit
