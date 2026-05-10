"""Tests for 1D FNO model."""
import torch
import pytest
import numpy as np

from src.model.fno_1d import FNO1d

B, N = 4, 256
EPS_MAX = 0.12


@pytest.fixture
def model():
    return FNO1d(modes=32, width=64, n_layers=4, eps_max=EPS_MAX).eval()


def test_output_shape(model):
    x = torch.randn(B, 5, N)
    eps, logit = model(x)
    assert eps.shape   == (B, N)
    assert logit.shape == (B,)


def test_output_range(model):
    x = torch.randn(B, 5, N)
    with torch.no_grad():
        eps, _ = model(x)
    assert float(eps.min()) >= 0.0
    assert float(eps.max()) <= EPS_MAX + 1e-5


def test_gradient_flows(model):
    model.train()
    x = torch.randn(B, 5, N, requires_grad=False)
    eps, logit = model(x)
    loss = eps.mean() + logit.mean()
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No grad for {name}"
            assert torch.isfinite(p.grad).all(), f"NaN/Inf grad for {name}"


def test_zero_input(model):
    """Zero displacement input → output must be finite and in valid range."""
    x = torch.zeros(1, 5, N)
    with torch.no_grad():
        eps, logit = model(x)
    assert torch.isfinite(eps).all()
    assert torch.isfinite(logit).all()
    assert float(eps.min()) >= 0.0
    assert float(eps.max()) <= EPS_MAX + 1e-5
