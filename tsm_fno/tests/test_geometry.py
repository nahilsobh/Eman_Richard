import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.phantom.geometry import (LesionGeometry, compute_lame_field,
                                    perilesional_shell)


def test_mask_shape():
    g = LesionGeometry(center=(32, 32), semi_axes=(10, 7),
                        angle=0.5, pressure=2000.0)
    m = g.mask(64)
    assert m.shape == (64, 64)
    assert m.dtype == np.bool_
    assert m[32, 32]
    assert not m[0, 0]


def test_normals_zero_inside():
    g = LesionGeometry(center=(32, 32), semi_axes=(10, 8),
                        angle=0.0, pressure=1000.0)
    n = g.boundary_normals(64)
    m = g.mask(64)
    assert n.shape == (64, 64, 2)
    assert np.allclose(n[m], 0.0)
    norms = np.linalg.norm(n[~m], axis=-1)
    # outside normals should be (close to) unit length
    assert (norms > 0.9).mean() > 0.9


def test_lame_decay():
    """Outside the lesion, Δσ ∝ 1/r²: ratio at 2r vs r should be ~0.25."""
    g = LesionGeometry(center=(32, 32), semi_axes=(6, 6),
                        angle=0.0, pressure=4000.0)
    f = compute_lame_field(g, 64, dx=1.0, G_background=1.0)
    # Sample at radii outside the lesion, both along +x from centre
    r1 = 10   # > a_eq = 6
    r2 = 20
    val_r1 = f[32, 32 + r1]
    val_r2 = f[32, 32 + r2]
    ratio = val_r2 / val_r1
    assert 0.20 < ratio < 0.30, f"Expected ~0.25, got {ratio:.3f}"


def test_lame_zero_pressure():
    g = LesionGeometry(center=(32, 32), semi_axes=(8, 6),
                        angle=0.0, pressure=0.0)
    f = compute_lame_field(g, 64, dx=1.0)
    assert np.all(f == 0.0)


def test_perilesional_shell():
    g = LesionGeometry(center=(32, 32), semi_axes=(10, 8),
                        angle=0.0, pressure=1000.0)
    m = g.mask(64)
    shell = perilesional_shell(m, shell_mm=5.0, dx=0.002)
    assert shell.shape == (64, 64)
    assert shell.dtype == np.bool_
    # Shell must be entirely outside the lesion
    assert not (shell & m).any()
    # Shell must contain at least some voxels
    assert shell.sum() > 0
