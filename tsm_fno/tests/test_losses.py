import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.losses import (relative_l2, masked_relative_l2,
                                acoustic_consistency_loss, expansion_loss,
                                helmholtz_residual_loss, total_loss)


def test_relative_l2_perfect():
    pred = torch.randn(4, 8, 8)
    assert relative_l2(pred, pred).item() < 1e-6


def test_masked_relative_l2_zero_target():
    """When target is zero in the mask, loss should be MSE-style — finite."""
    pred = torch.zeros(2, 8, 8)
    target = torch.zeros(2, 8, 8)
    mask = torch.ones(2, 8, 8)
    L = masked_relative_l2(pred, target, mask)
    assert torch.isfinite(L) and L.item() == 0.0


def test_acoustic_consistency_zero_when_consistent():
    """G = G_bg + |A| · ε · G_bg should give zero loss in the ring."""
    B, N = 2, 16
    G_bg = torch.tensor([1500.0, 2000.0])
    A_pred = torch.tensor([-5.0, -4.0])   # |A| = [5, 4]
    eps = torch.zeros(B, N, N)
    eps[:, 4:8, 4:8] = 0.05
    G = G_bg.view(B, 1, 1) + A_pred.abs().view(B, 1, 1) * eps * G_bg.view(B, 1, 1)
    ring_mask = torch.zeros(B, N, N)
    ring_mask[:, 4:8, 4:8] = 1.0
    L = acoustic_consistency_loss(G, eps, A_pred, G_bg, ring_mask)
    assert L.item() < 1e-6


def test_expansion_loss_low_for_correct_classification():
    """Low score for control + low ε; high score for expanding + high ε ⇒ low BCE."""
    eps_pred = torch.zeros(4, 16, 16)
    eps_pred[2:, 7, 7] = 0.05    # samples 2,3 expanding
    ring_mask = torch.ones(4, 16, 16)
    is_exp = torch.tensor([0, 0, 1, 1])
    L = expansion_loss(eps_pred, ring_mask, is_exp, eps_threshold=0.01)
    assert L.item() < 0.7  # well below random (≈ ln 2 = 0.693)


def test_total_loss_finite():
    B, N = 2, 16
    G_pred = torch.full((B, N, N), 1500.0).requires_grad_()
    eps_pred = torch.full((B, N, N), 0.01).requires_grad_()
    A_pred = torch.tensor([-3.0, -3.0]).requires_grad_()
    G_true = torch.full((B, N, N), 1500.0)
    eps_true = torch.zeros(B, N, N)
    ring_mask = torch.zeros(B, N, N); ring_mask[:, 4:8, 4:8] = 1.0
    G_bg_mean = torch.tensor([1500.0, 1500.0])
    is_exp = torch.tensor([0, 1])
    u_re = torch.randn(B, N, N) * 0.1
    u_im = torch.randn(B, N, N) * 0.1
    L, parts = total_loss(G_pred, eps_pred, A_pred,
                          G_true, eps_true, ring_mask, G_bg_mean,
                          is_exp, u_re, u_im)
    assert torch.isfinite(L)
    L.backward()
    for tensor_name, t in [("G_pred", G_pred), ("eps_pred", eps_pred),
                             ("A_pred", A_pred)]:
        assert torch.isfinite(t.grad).all(), f"NaN grad in {tensor_name}"
