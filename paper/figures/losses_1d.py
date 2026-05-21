"""Loss functions for the 1D brain-scale architecture+loss ablation (Fig 5).

All loss callables share the signature

    loss(Y_pred, Y_true, X, dx, freq, rho, G_scale) -> scalar tensor

where
  Y_pred, Y_true : (B, 2, N) real -- (Re G, Im G) / G_scale
  X              : (B, 2, N) real -- (Re u, Im u) normalised to ||u||_inf=1
  dx, freq, rho  : floats, problem physical parameters
  G_scale        : float -- normalisation applied to G during training

This shared signature lets the training loop swap loss functions
without changing any other code. Losses that do not need the unused
arguments simply ignore them.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def _relative_l2(Y_pred: torch.Tensor, Y_true: torch.Tensor) -> torch.Tensor:
    num = torch.linalg.vector_norm(Y_pred - Y_true, dim=(-2, -1))
    den = torch.linalg.vector_norm(Y_true,           dim=(-2, -1)).clamp_min(1e-6)
    return (num / den).mean()


def loss_rl2(Y_pred: torch.Tensor, Y_true: torch.Tensor,
             X: torch.Tensor, dx: float, freq: float, rho: float,
             G_scale: float) -> torch.Tensor:
    """Per-sample relative L^2 on the normalised G(x)."""
    return _relative_l2(Y_pred, Y_true)


def loss_mse(Y_pred: torch.Tensor, Y_true: torch.Tensor,
             X: torch.Tensor, dx: float, freq: float, rho: float,
             G_scale: float) -> torch.Tensor:
    """Mean-squared error on the normalised G(x)."""
    return F.mse_loss(Y_pred, Y_true)


def helmholtz_residual_1d(Y_pred: torch.Tensor, X: torch.Tensor,
                          dx: float, freq: float, rho: float,
                          G_scale: float) -> torch.Tensor:
    """Mean relative |G*(d^2 u/dx^2) + rho*omega^2*u| / |rho*omega^2*u|.

    Uses a 3-point FD Laplacian on interior nodes.  Y_pred is in
    normalised (G/G_scale) units, so we rescale by G_scale before
    forming the residual.  The complex displacement is X[:, 0] + i X[:, 1].
    """
    omega = 2.0 * math.pi * freq
    # Recover the complex G and u
    G  = (Y_pred[:, 0] + 1j * Y_pred[:, 1]) * G_scale     # (B, N), complex
    u  =  X[:, 0]      + 1j * X[:, 1]                      # (B, N), complex

    # Second derivative via central differences on interior nodes
    lap_u = (u[:, 2:] - 2.0 * u[:, 1:-1] + u[:, :-2]) / (dx ** 2)
    G_int = G[:, 1:-1]
    u_int = u[:, 1:-1]

    residual = G_int * lap_u + rho * omega ** 2 * u_int           # complex
    denom    = (rho * omega ** 2) * u_int.abs().clamp_min(1e-10)
    return (residual.abs() / denom).mean()


def loss_pde(Y_pred: torch.Tensor, Y_true: torch.Tensor,
             X: torch.Tensor, dx: float, freq: float, rho: float,
             G_scale: float) -> torch.Tensor:
    """Pure physics-informed loss (Helmholtz residual on predicted G)."""
    return helmholtz_residual_1d(Y_pred, X, dx, freq, rho, G_scale)


def loss_composite(Y_pred: torch.Tensor, Y_true: torch.Tensor,
                   X: torch.Tensor, dx: float, freq: float, rho: float,
                   G_scale: float,
                   lambda_pde: float = 0.05) -> torch.Tensor:
    """RL^2 + lambda_pde * Helmholtz residual."""
    L_data = _relative_l2(Y_pred, Y_true)
    L_pde  = helmholtz_residual_1d(Y_pred, X, dx, freq, rho, G_scale)
    return L_data + lambda_pde * L_pde


LOSS_FNS = {
    "rl2":       loss_rl2,
    "mse":       loss_mse,
    "pde":       loss_pde,
    "composite": loss_composite,
}
