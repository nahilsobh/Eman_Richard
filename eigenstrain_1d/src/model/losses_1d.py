"""Loss functions for 1D eigenstrain inversion.

Dual comparison: FNO vs ground truth AND vs analytical formula.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def relative_l2(pred: torch.Tensor, true: torch.Tensor,
                eps_floor: float = 1e-6) -> torch.Tensor:
    """Mean relative L² error over batch. pred, true: (B, N).

    Uses a batch-adaptive floor (1% of batch mean norm) to avoid blow-up
    on zero-target control cases while preserving accuracy for expanding cases.
    """
    norm_diff = torch.norm(pred - true, dim=-1)
    norm_true = torch.norm(true, dim=-1)
    # adaptive floor: 1% of batch mean, floored at eps_floor
    floor = 0.01 * norm_true.mean().detach() + eps_floor
    return (norm_diff / (norm_true + floor)).mean()


def lesion_weighted_l2(pred: torch.Tensor, true: torch.Tensor,
                        lesion_mask: torch.Tensor,
                        w_lesion: float = 0.7, w_bg: float = 0.3
                        ) -> torch.Tensor:
    """Weighted RL² with more emphasis on lesion region. mask: (B, N) bool."""
    m = lesion_mask.float()
    bg = (1.0 - m)
    l_lesion = relative_l2(pred * m, true * m)
    l_bg     = relative_l2(pred * bg, true * bg)
    return w_lesion * l_lesion + w_bg * l_bg


def inversion_consistency_loss(
    eps_pred: torch.Tensor,
    u_input:  torch.Tensor,
    E_bg:     torch.Tensor,
    rho:      float,
    freq1:    float,
    freq2:    float,
    dx:       float,
) -> torch.Tensor:
    """Physics-informed loss: eps_pred should match analytical inversion formula.

    eps_pred : (B, N)
    u_input  : (B, 5, N)  — first 4 channels are Re/Im at 60/120 Hz
    E_bg     : (B,)       — background modulus for each sample
    """
    B, N = eps_pred.shape
    device = eps_pred.device

    k_np = np.fft.fftfreq(N, d=dx) * 2.0 * np.pi
    k    = torch.tensor(k_np, dtype=torch.float32, device=device)   # (N,)

    def invert_torch(u_re, u_im, E, freq):
        """Recover ε*(x) from complex u field using the k-space formula."""
        omega = 2.0 * np.pi * freq
        u_complex = torch.complex(u_re, u_im)           # (B, N)
        U = torch.fft.fft(u_complex, norm="backward")
        eps_hat = 1j * k.unsqueeze(0) * U               # (B, N)
        k2 = k.pow(2)
        k2_safe = k2.clone(); k2_safe[0] = 1.0
        E_col = E.unsqueeze(-1)                          # (B, 1)
        correction = 1.0 - rho * omega**2 / (k2_safe.unsqueeze(0) * E_col)
        correction[:, 0] = 0.0
        eps_rec = torch.fft.ifft(eps_hat * correction, norm="backward").real
        return eps_rec

    eps_60  = invert_torch(u_input[:, 0], u_input[:, 1], E_bg, freq1)
    eps_120 = invert_torch(u_input[:, 2], u_input[:, 3], E_bg, freq2)
    eps_analytic = (eps_60 + eps_120) / 2.0

    return relative_l2(eps_pred, eps_analytic)


class TSMLoss1D(nn.Module):
    """Combined loss for 1D eigenstrain inversion."""

    def __init__(self, lambda_true: float = 1.0, lambda_analytic: float = 0.5,
                 lambda_pde: float = 0.1, lambda_expand: float = 0.2,
                 dx: float = 2*0.10/256, rho: float = 1000.0,
                 freq1: float = 60.0, freq2: float = 120.0):
        super().__init__()
        self.lambda_true     = lambda_true
        self.lambda_analytic = lambda_analytic
        self.lambda_pde      = lambda_pde
        self.lambda_expand   = lambda_expand
        self.dx   = dx
        self.rho  = rho
        self.freq1 = freq1
        self.freq2 = freq2

    def forward(self, eps_pred: torch.Tensor, expand_logit: torch.Tensor,
                eps_true: torch.Tensor, eps_analytic: torch.Tensor,
                is_expanding: torch.Tensor,
                u_input: torch.Tensor, E_bg: torch.Tensor) -> dict:
        """Compute total loss and all components."""
        l_true     = relative_l2(eps_pred, eps_true)
        l_analytic = relative_l2(eps_pred, eps_analytic)
        l_pde      = inversion_consistency_loss(
            eps_pred, u_input, E_bg, self.rho, self.freq1, self.freq2, self.dx)
        l_expand   = F.binary_cross_entropy_with_logits(
            expand_logit, is_expanding)

        total = (self.lambda_true     * l_true
               + self.lambda_analytic * l_analytic
               + self.lambda_pde      * l_pde
               + self.lambda_expand   * l_expand)

        return {
            "loss":       total,
            "l_true":     l_true,
            "l_analytic": l_analytic,
            "l_pde":      l_pde,
            "l_expand":   l_expand,
        }
