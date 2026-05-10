"""Composite loss for the dual-head FNO-TSM.

L_total = L_G
        + λ_eps      · L_eps           (overall ε relative-L²)
        + λ_ring     · L_eps_ring      (ring-only relative-L²)
        + λ_acoustic · L_acoustic      (G_eff − G_bg ≈ |A| · ε · G_bg in ring)
        + λ_pde      · L_pde           (Helmholtz residual on (u, G))
        + λ_expand   · L_expand        (BCE: is the lesion expanding?)
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ── Building-block losses ────────────────────────────────

def relative_l2(pred: torch.Tensor, target: torch.Tensor,
                eps: float = 1e-6, floor: float = 0.0) -> torch.Tensor:
    """Mean over batch of ‖pred-target‖₂ / (‖target‖₂ + floor).

    For nearly-zero targets (e.g. ε_latent for control samples), pass
    `floor > 0` so the loss reduces to MSE-like behaviour instead of
    exploding. Set `floor=0` for the standard relative-L² (with eps clamp).
    """
    num = torch.norm(pred - target, dim=(-2, -1))
    den = torch.norm(target, dim=(-2, -1))
    if floor > 0:
        return (num / (den + floor)).mean()
    return (num / den.clamp(min=eps)).mean()


def masked_relative_l2(pred: torch.Tensor, target: torch.Tensor,
                        mask: torch.Tensor,
                        eps_floor: float = 1e-3) -> torch.Tensor:
    """Per-sample relative-L² evaluated only on the True pixels of `mask`.

    Falls back to MSE when target is identically zero in the masked region
    (control samples) — the eps_floor in the denominator ensures stability.
    """
    diff = (pred - target) * mask
    tgt  = target * mask
    num = torch.norm(diff.flatten(1), dim=-1)
    den = torch.norm(tgt.flatten(1),  dim=-1).clamp(min=eps_floor)
    return (num / den).mean()


def ssim_metric(pred: torch.Tensor, target: torch.Tensor,
                C1: float = 1e-4, C2: float = 9e-4) -> torch.Tensor:
    """Per-sample SSIM averaged across the batch."""
    mu_p = pred.mean(dim=(-2, -1), keepdim=True)
    mu_t = target.mean(dim=(-2, -1), keepdim=True)
    sig_p = pred.var(dim=(-2, -1), keepdim=True)
    sig_t = target.var(dim=(-2, -1), keepdim=True)
    sig_pt = ((pred - mu_p) * (target - mu_t)).mean(dim=(-2, -1), keepdim=True)
    num = (2 * mu_p * mu_t + C1) * (2 * sig_pt + C2)
    den = (mu_p ** 2 + mu_t ** 2 + C1) * (sig_p + sig_t + C2)
    return (num / den).mean()


# ── Helmholtz PDE residual ───────────────────────────────

def helmholtz_residual_loss(
    u_re: torch.Tensor, u_im: torch.Tensor, G_pred: torch.Tensor,
    dx: float = 0.002, freq: float = 60.0,
    rho: float = 1000.0, damping: float = 0.05,
) -> torch.Tensor:
    """Mean relative |Δu·G + ρω²u| / |ρω²u| on interior pixels."""
    omega = 2.0 * math.pi * freq
    lap_k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                          dtype=torch.float32, device=G_pred.device
                          ).view(1, 1, 3, 3) / dx ** 2

    def lap(u):
        return F.conv2d(u.unsqueeze(1), lap_k, padding=1).squeeze(1)

    Gr = G_pred
    Gi = G_pred * damping
    lap_re = lap(u_re)
    lap_im = lap(u_im)

    res_re = Gr * lap_re - Gi * lap_im + rho * omega ** 2 * u_re
    res_im = Gr * lap_im + Gi * lap_re + rho * omega ** 2 * u_im

    norm = (rho * omega ** 2 *
            (u_re[:, 1:-1, 1:-1] ** 2 + u_im[:, 1:-1, 1:-1] ** 2)
            .sqrt().clamp(min=1e-10))
    res = (res_re[:, 1:-1, 1:-1] ** 2 + res_im[:, 1:-1, 1:-1] ** 2).sqrt()
    return (res / norm).mean()


# ── Acoustoelastic consistency (in the ring) ─────────────

def acoustic_consistency_loss(
    G_pred: torch.Tensor, eps_pred: torch.Tensor,
    A_pred: torch.Tensor, G_bg_mean: torch.Tensor,
    ring_mask: torch.Tensor,
) -> torch.Tensor:
    """In ring: G_pred − G_bg_mean ≈ |A| · ε_pred · G_bg_mean.

    A_pred is the model's per-sample scalar (constrained < -2). We use
    its absolute value when comparing to ground truth (which is positive
    in our convention).
    """
    if ring_mask.sum() == 0:
        return torch.tensor(0.0, device=G_pred.device)

    # Broadcast A and G_bg_mean across spatial dims
    A_abs = A_pred.abs().view(-1, 1, 1)
    Gbg = G_bg_mean.view(-1, 1, 1)

    G_recon = Gbg + A_abs * eps_pred * Gbg
    diff = (G_recon - G_pred) * ring_mask
    return (diff ** 2).mean()


# ── Expansion classifier ─────────────────────────────────

def expansion_loss(
    eps_pred: torch.Tensor, ring_mask: torch.Tensor,
    is_expanding: torch.Tensor,
    eps_threshold: float = 0.01,
) -> torch.Tensor:
    """BCE on max(eps in ring) > threshold being a positive prediction."""
    masked = eps_pred * ring_mask
    score = masked.flatten(1).amax(dim=-1)            # (B,)
    logit = (score - eps_threshold) / (eps_threshold + 1e-8)
    target = is_expanding.float()
    return F.binary_cross_entropy_with_logits(logit, target)


# ── Composite total loss ─────────────────────────────────

def total_loss(
    G_pred: torch.Tensor, eps_pred: torch.Tensor, A_pred: torch.Tensor,
    G_true: torch.Tensor, eps_true: torch.Tensor,
    ring_mask: torch.Tensor,
    G_bg_mean: torch.Tensor,
    is_expanding: torch.Tensor,
    u_re: torch.Tensor, u_im: torch.Tensor,
    *,
    lambda_eps: float = 1.0,
    lambda_ring: float = 0.5,
    lambda_acoustic: float = 0.1,
    lambda_pde: float = 0.05,
    lambda_expand: float = 0.2,
) -> tuple[torch.Tensor, dict]:
    L_G       = relative_l2(G_pred, G_true)
    # ε values are bounded in [0, eps_max] — MSE is numerically stable
    # whether the sample is expanding or a zero-target control.
    L_eps     = F.mse_loss(eps_pred, eps_true)
    L_ring    = masked_relative_l2(eps_pred, eps_true, ring_mask, eps_floor=1.0)
    L_acoust  = acoustic_consistency_loss(G_pred, eps_pred, A_pred,
                                           G_bg_mean, ring_mask)
    L_pde     = helmholtz_residual_loss(u_re, u_im, G_pred)
    L_expand  = expansion_loss(eps_pred, ring_mask, is_expanding)

    L = (L_G
         + lambda_eps      * L_eps
         + lambda_ring     * L_ring
         + lambda_acoustic * L_acoust
         + lambda_pde      * L_pde
         + lambda_expand   * L_expand)

    return L, {
        "L_total":   L.detach(),
        "L_G":       L_G.detach(),
        "L_eps":     L_eps.detach(),
        "L_ring":    L_ring.detach(),
        "L_acoust":  L_acoust.detach(),
        "L_pde":     L_pde.detach(),
        "L_expand":  L_expand.detach(),
    }
