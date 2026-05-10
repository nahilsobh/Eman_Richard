import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.fno_tsm import FNO_TSM


def test_output_shapes():
    model = FNO_TSM(in_channels=6, width=32, n_layers=2)
    x = torch.randn(2, 6, 64, 64)
    G, eps, A = model(x)
    assert G.shape == (2, 64, 64)
    assert eps.shape == (2, 64, 64)
    assert A.shape == (2,)


def test_G_range():
    model = FNO_TSM(in_channels=6, width=24, n_layers=2,
                     G_min=200.0, G_max=80000.0)
    x = torch.randn(4, 6, 32, 32)
    G, _, _ = model(x)
    assert torch.all(G >= 200.0 - 1e-3)
    assert torch.all(G <= 80000.0 + 1e-3)


def test_eps_nonneg():
    model = FNO_TSM(in_channels=6, width=24, n_layers=2)
    x = torch.randn(4, 6, 32, 32)
    _, eps, _ = model(x)
    assert torch.all(eps >= 0.0)
    assert torch.all(eps <= model.eps_max + 1e-6)


def test_A_negative():
    model = FNO_TSM(in_channels=6, width=24, n_layers=2)
    x = torch.randn(4, 6, 32, 32)
    _, _, A = model(x)
    assert torch.all(A < -2.0 + 1e-6)


def test_gradient_flows():
    model = FNO_TSM(in_channels=6, width=24, n_layers=2)
    x = torch.randn(2, 6, 32, 32)
    G, eps, A = model(x)
    loss = G.mean() + eps.mean() + A.mean()
    loss.backward()
    # All parameters should have a gradient
    for p in model.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
