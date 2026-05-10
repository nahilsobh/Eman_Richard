import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.phantom.geometry import LesionGeometry, perilesional_shell
from src.phantom.acoustoelastic import (make_effective_G,
                                          make_latent_strain,
                                          G_MIN_PA, G_MAX_PA, EPS_MAX)


def _setup(p=4000.0):
    g = LesionGeometry(center=(32, 32), semi_axes=(10, 8),
                        angle=0.3, pressure=p)
    return g


def test_ring_present_for_expanding():
    g = _setup(p=4000.0)
    G_bg = 1500.0
    G_lesion = 8000.0
    A = 5.0
    G = make_effective_G(64, dx=0.002, geometry=g,
                          G_bg=G_bg, G_lesion=G_lesion, A_coeff=A)
    shell = perilesional_shell(g.mask(64), shell_mm=4.0, dx=0.002)
    assert G[shell].mean() > G_bg, "Expected stiffening ring (mean above G_bg)"


def test_no_ring_for_control():
    g = _setup(p=0.0)
    G_bg = 1500.0
    G_lesion = 8000.0
    A = 5.0
    G = make_effective_G(64, dx=0.002, geometry=g,
                          G_bg=G_bg, G_lesion=G_lesion, A_coeff=A)
    shell = perilesional_shell(g.mask(64), shell_mm=4.0, dx=0.002)
    assert np.allclose(G[shell], G_bg, atol=1e-9)


def test_epsilon_bounds():
    g = _setup(p=6000.0)
    eps = make_latent_strain(64, dx=0.002, geometry=g, G_bg=1200.0)
    assert eps.min() >= 0.0
    assert eps.max() <= EPS_MAX + 1e-12


def test_epsilon_zero_for_control():
    g = _setup(p=0.0)
    eps = make_latent_strain(64, dx=0.002, geometry=g, G_bg=1500.0)
    assert np.all(eps == 0.0)


def test_acoustic_coupling():
    """At any voxel where ε is unclipped: G_eff − G_bg ≈ A · ε · G_bg.

    Use a low-pressure case so most of the field stays below EPS_MAX.
    """
    g = _setup(p=200.0)
    G_bg, G_lesion, A = 1500.0, 8000.0, 5.0
    G = make_effective_G(64, dx=0.002, geometry=g,
                          G_bg=G_bg, G_lesion=G_lesion, A_coeff=A)
    eps = make_latent_strain(64, dx=0.002, geometry=g, G_bg=G_bg)
    outside = ~g.mask(64)
    unclipped = outside & (eps < EPS_MAX - 1e-6) & (eps > 1e-8)
    assert unclipped.sum() > 100, f"Need unclipped pixels, got {unclipped.sum()}"
    lhs = G[unclipped] - G_bg
    rhs = A * eps[unclipped] * G_bg
    err = np.abs(lhs - rhs) / (np.abs(rhs).mean() + 1e-9)
    assert err.mean() < 0.05, f"Mean rel err {err.mean():.3f} > 5%"


def test_G_clipping():
    g = _setup(p=8000.0)
    G_bg, G_lesion, A = 800.0, 14000.0, 8.0
    G = make_effective_G(64, dx=0.002, geometry=g,
                          G_bg=G_bg, G_lesion=G_lesion, A_coeff=A)
    assert G.min() >= G_MIN_PA
    assert G.max() <= G_MAX_PA
