"""Tests for the 3D Helmholtz solver and 3D phantom helpers."""
from __future__ import annotations

import numpy as np
import pytest

from src.solver.helmholtz_fd_3d import (
    bottom_plate_driver_sources_3d,
    direct_inversion_3d,
    helmholtz_solve_3d,
    lfe_inversion_3d,
)
from src.phantom.geometry_3d import (
    SphericalBalloon,
    make_effective_G_3d,
    perilesional_shell_3d,
)


N_SMALL = 16   # keep tests fast; 16³ = 4096 DOFs solves in ~0.3 s


def _uniform_G(N: int = N_SMALL, val: float = 2500.0) -> np.ndarray:
    return np.full((N, N, N), val, dtype=float)


def test_bottom_plate_driver_disk():
    N = 32
    src = bottom_plate_driver_sources_3d(N, radius_frac=0.5)
    assert all(i == N - 1 for (i, _, _, _) in src), "all sources must be on bottom face"
    # For radius_frac=0.5 in a 32³ cube, disk radius = 8 → ~π·8² ≈ 201 nodes
    n = len(src)
    assert 180 < n < 220, f"expected ~π·r² disk nodes, got {n}"
    # Coherent phase.
    amps = {a for (_, _, _, a) in src}
    assert amps == {1.0 + 0.0j}


def test_default_bc_clamps_top_face():
    G = _uniform_G()
    src = bottom_plate_driver_sources_3d(N_SMALL, radius_frac=0.5)
    u = helmholtz_solve_3d(G, freq=60.0, dx=0.005, sources=src, top_free=False)
    top_max = np.max(np.abs(u[0, :, :]))
    assert top_max < 1e-12, f"clamped top must be pinned to 0, got max={top_max:.3g}"


def test_top_free_soft_reflection_vs_clamped_3d():
    G = _uniform_G()
    src = bottom_plate_driver_sources_3d(N_SMALL, radius_frac=0.5)
    u_clamp = helmholtz_solve_3d(G, freq=60.0, dx=0.005, sources=src, top_free=False)
    u_free  = helmholtz_solve_3d(G, freq=60.0, dx=0.005, sources=src, top_free=True)

    # Ignore edges of the top face (still Dirichlet in the free-top case).
    top_clamp = np.sqrt(np.mean(np.abs(u_clamp[0, 2:-2, 2:-2]) ** 2))
    top_free  = np.sqrt(np.mean(np.abs(u_free[0,  2:-2, 2:-2]) ** 2))
    assert top_free > 10 * top_clamp, \
        f"free-top face amplitude should dominate clamped: {top_free:.3g} vs {top_clamp:.3g}"


def test_top_slab_stencil_residual():
    # Row-0 stencil residual under free-top must be ~0 in a uniform medium.
    G_val = 2500.0
    dx    = 0.005
    freq  = 60.0
    rho   = 1000.0
    xi    = 0.05
    G = _uniform_G(val=G_val)
    src = bottom_plate_driver_sources_3d(N_SMALL, radius_frac=0.5)
    u = helmholtz_solve_3d(G, freq=freq, dx=dx, damping=xi, sources=src, top_free=True)

    omega = 2 * np.pi * freq
    Gc = G_val * (1 + 1j * xi)
    inv_dx2 = 1.0 / dx ** 2
    # Interior of the top face (avoid the Dirichlet edges).
    j_range = np.arange(2, N_SMALL - 2)
    k_range = np.arange(2, N_SMALL - 2)
    J, K = np.meshgrid(j_range, k_range, indexing="ij")
    # Stencil at i=0 (ghost mirror): −6 Gc u[0] + Gc(u[0,j±1] + u[0,,k±1]) + 2 Gc u[1] + ρω² u[0] = 0
    u0 = u[0, J, K]
    resid = (
        -6.0 * Gc * inv_dx2 * u0
        + Gc * inv_dx2 * (u[0, J + 1, K] + u[0, J - 1, K])
        + Gc * inv_dx2 * (u[0, J, K + 1] + u[0, J, K - 1])
        + 2.0 * Gc * inv_dx2 * u[1, J, K]
        + rho * omega ** 2 * u0
    )
    scale = np.max(np.abs(u)) * (6.0 * abs(Gc) * inv_dx2 + rho * omega ** 2)
    rel   = np.max(np.abs(resid)) / (scale + 1e-30)
    assert rel < 1e-10, f"top-slab stencil residual too large: {rel:.3g}"


def test_spherical_balloon_geometry():
    N = 32
    balloon = SphericalBalloon(center=(16, 16, 16), radius_vx=6.0, pressure=1000.0)
    mask = balloon.mask(N)
    # Volume should be close to (4/3)π r³ = 904 voxels (discretisation error < 10%).
    expected = (4 / 3) * np.pi * balloon.radius_vx ** 3
    assert 0.9 * expected < mask.sum() < 1.1 * expected


def test_perilesional_shell_3d_nonempty():
    N = 32
    balloon = SphericalBalloon(center=(16, 16, 16), radius_vx=6.0, pressure=1000.0)
    shell = perilesional_shell_3d(balloon.mask(N), shell_mm=8.0, dx=0.003)
    assert shell.sum() > 0
    assert not shell[balloon.mask(N)].any(), "shell must not overlap lesion"


def test_effective_G_stiffens_around_balloon():
    N = 32
    balloon_off = SphericalBalloon((16, 16, 16), 6.0, pressure=0.0)
    balloon_on  = SphericalBalloon((16, 16, 16), 6.0, pressure=4000.0)
    G_off = make_effective_G_3d(N, balloon_off, G_bg=2500, G_lesion=2000, A_coeff=5.0)
    G_on  = make_effective_G_3d(N, balloon_on,  G_bg=2500, G_lesion=2000, A_coeff=5.0)
    shell = perilesional_shell_3d(balloon_on.mask(N), shell_mm=8.0, dx=0.003)
    assert G_on[shell].mean() > G_off[shell].mean() + 500.0, \
        "pressurised balloon must produce a stiffer shell"


def test_stiffening_exponent_default_matches_linear():
    # m = 1.0 must reproduce the original linear form exactly.
    N = 24
    balloon = SphericalBalloon((12, 12, 12), 5.0, pressure=3000.0)
    G_default = make_effective_G_3d(N, balloon, G_bg=2500, G_lesion=2000, A_coeff=5.0)
    G_m1      = make_effective_G_3d(N, balloon, G_bg=2500, G_lesion=2000, A_coeff=5.0,
                                     stiffening_exponent=1.0)
    assert np.allclose(G_default, G_m1), "m=1 must equal default (linear) form"


def test_stiffening_exponent_superlinear():
    # m = 2 must give a strictly stiffer shell than m = 1 wherever Δσ > 0.
    N = 32
    balloon = SphericalBalloon((16, 16, 16), 6.0, pressure=3000.0)
    G_lin = make_effective_G_3d(N, balloon, 2500, 2000, 5.0, stiffening_exponent=1.0)
    G_hyp = make_effective_G_3d(N, balloon, 2500, 2000, 5.0, stiffening_exponent=2.0)
    shell = perilesional_shell_3d(balloon.mask(N), shell_mm=8.0, dx=0.003)
    # In the ring, Δσ > 0 → hyperelastic is strictly greater (up to the clip).
    ring_lin = G_lin[shell]
    ring_hyp = G_hyp[shell]
    # Every voxel in the ring should satisfy G_hyp >= G_lin.
    assert np.all(ring_hyp >= ring_lin - 1e-6), \
        "hyperelastic G must dominate linear G in the pre-stressed ring"
    # And the shell mean should be strictly greater by a wide margin.
    assert ring_hyp.mean() > 1.5 * ring_lin.mean(), \
        f"expected >1.5x ring mean, got {ring_hyp.mean()/ring_lin.mean():.2f}"


def test_powerlaw_default_backward_compat():
    # Default constitutive must match the old power-law behavior byte-for-byte.
    N = 24
    balloon = SphericalBalloon((12, 12, 12), 5.0, pressure=3000.0)
    G_default = make_effective_G_3d(N, balloon, 2500, 2000, 0.5, stiffening_exponent=1.5)
    G_explicit = make_effective_G_3d(N, balloon, 2500, 2000, 0.5,
                                      stiffening_exponent=1.5, constitutive="powerlaw")
    assert np.array_equal(G_default, G_explicit)


def test_ogden_at_zero_pressure_matches_base():
    # At Δσ = 0 the Ogden form must give exactly G_base everywhere.
    N = 24
    balloon = SphericalBalloon((12, 12, 12), 5.0, pressure=0.0)
    G = make_effective_G_3d(N, balloon, 2500, 2000, 5.0,
                             stiffening_exponent=3.0, constitutive="ogden")
    outside = ~balloon.mask(N)
    assert np.allclose(G[outside], 2500.0)
    assert np.allclose(G[balloon.mask(N)], 2000.0)


def test_ogden_softer_than_powerlaw_at_small_strain():
    # Ogden's small-strain rise is quadratic; power-law's is linear.
    # For small A·Δσ/G, Ogden should give a smaller G_eff-vs-baseline lift
    # than the power-law with the same exponent.
    N = 24
    balloon = SphericalBalloon((12, 12, 12), 5.0, pressure=1000.0)
    G_pow = make_effective_G_3d(N, balloon, 2500, 2000, 0.10,
                                 stiffening_exponent=2.0, constitutive="powerlaw")
    G_ogd = make_effective_G_3d(N, balloon, 2500, 2000, 0.10,
                                 stiffening_exponent=2.0, constitutive="ogden")
    from src.phantom.geometry_3d import perilesional_shell_3d
    shell = perilesional_shell_3d(balloon.mask(N), shell_mm=8.0, dx=0.003)
    ring_pow = G_pow[shell].mean() - 2500
    ring_ogd = G_ogd[shell].mean() - 2500
    assert ring_ogd < ring_pow, \
        f"Ogden should be softer at low strain: pow={ring_pow:.0f}, ogd={ring_ogd:.0f}"
    assert ring_ogd > 0, "Ogden should still stiffen under pressure"


def test_ogden_stiffens_ring_at_high_pressure():
    # High Δσ → λ ≫ 1 → Ogden term (1/2)·λ^α dominates → strong stiffening.
    N = 24
    balloon = SphericalBalloon((12, 12, 12), 5.0, pressure=5000.0)
    G_off = make_effective_G_3d(N, SphericalBalloon((12,12,12), 5.0, 0.0),
                                 2500, 2000, 0.5, 3.0, constitutive="ogden")
    G_on  = make_effective_G_3d(N, balloon, 2500, 2000, 0.5, 3.0, constitutive="ogden")
    from src.phantom.geometry_3d import perilesional_shell_3d
    shell = perilesional_shell_3d(balloon.mask(N), shell_mm=8.0, dx=0.003)
    assert G_on[shell].mean() > G_off[shell].mean() + 200.0, \
        "Ogden shell must stiffen at high pressure"


def test_unknown_constitutive_raises():
    N = 24
    balloon = SphericalBalloon((12, 12, 12), 5.0, pressure=1000.0)
    import pytest
    with pytest.raises(ValueError, match="unknown constitutive"):
        make_effective_G_3d(N, balloon, 2500, 2000, 0.5, 1.5, constitutive="bogus")


def test_stiffening_exponent_pressure_zero_invariant():
    # With p = 0, Δσ ≡ 0 → G_eff is independent of the exponent.
    N = 24
    balloon = SphericalBalloon((12, 12, 12), 5.0, pressure=0.0)
    for m in (1.0, 1.5, 2.5, 3.0):
        G = make_effective_G_3d(N, balloon, 2500, 2000, 5.0, stiffening_exponent=m)
        # Background voxels stay at G_bg regardless of m.
        outside = ~balloon.mask(N)
        assert np.allclose(G[outside], 2500.0), \
            f"m={m}: background changed at p=0 ({G[outside].mean():.1f} vs 2500)"


def test_median_filter_none_is_no_op():
    # median_filter_size=None must give the same output as the default.
    N = 16
    G = _uniform_G(val=2500.0)
    src = bottom_plate_driver_sources_3d(N, radius_frac=0.5)
    u = helmholtz_solve_3d(G, freq=60, dx=0.005, sources=src)
    di_default = direct_inversion_3d(u, freq=60, dx=0.005)
    di_none    = direct_inversion_3d(u, freq=60, dx=0.005, median_filter_size=None)
    di_one     = direct_inversion_3d(u, freq=60, dx=0.005, median_filter_size=1)
    assert np.array_equal(di_default, di_none, equal_nan=True)
    assert np.array_equal(di_default, di_one,  equal_nan=True)


def test_median_filter_reduces_outliers():
    # Median should suppress isolated large spikes. Fabricate a G map with
    # spikes by using a very steep artificial pressure and check that the
    # filtered result has a smaller max/median ratio.
    N = 20
    balloon = SphericalBalloon((10, 10, 10), 4.0, pressure=7000.0)
    G_true = make_effective_G_3d(N, balloon, 2500, 2000, 0.5,
                                  stiffening_exponent=1.5, G_max_pa=500000.0)
    src = bottom_plate_driver_sources_3d(N, radius_frac=0.5)
    u = helmholtz_solve_3d(G_true, freq=60, dx=0.005, sources=src, top_free=True)
    di_raw = direct_inversion_3d(u, freq=60, dx=0.005)
    di_med = direct_inversion_3d(u, freq=60, dx=0.005, median_filter_size=3)
    # Both should have similar median (bulk stiffness preserved) but
    # median filter should reduce the extreme upper tail.
    med_raw = np.nanmedian(di_raw)
    med_med = np.nanmedian(di_med)
    max_raw = np.nanpercentile(di_raw, 99)
    max_med = np.nanpercentile(di_med, 99)
    # Medians roughly match (bulk not shifted).
    assert abs(med_raw - med_med) / max(abs(med_raw), 1.0) < 0.5, \
        f"medians drift too much: raw {med_raw:.0f}, filtered {med_med:.0f}"
    # 99th percentile should shrink significantly.
    assert max_med < max_raw, \
        f"median filter should suppress outliers: raw 99pct {max_raw:.0f}, filt {max_med:.0f}"


def test_lfe_returns_positive_finite_G():
    # Gradient-based LFE (|k|² = |∇u|²/|u|²) has a known standing-wave
    # bias in a bounded Dirichlet domain — the ratio isn't a plane-wave
    # k in general. Test only that it returns something physical.
    N = 16
    G_true = 2500.0
    G = _uniform_G(val=G_true)
    src = bottom_plate_driver_sources_3d(N, radius_frac=0.5)
    u = helmholtz_solve_3d(G, freq=60, dx=0.005, sources=src, top_free=True)
    G_lfe = lfe_inversion_3d(u, freq=60, dx=0.005)
    core = G_lfe[3:-3, 4:-4, 4:-4]
    core = core[np.isfinite(core)]
    assert core.size > 0
    mean = np.median(core)
    # Standing-wave bias can be up to ~50% on uniform G. Just check
    # order-of-magnitude sanity.
    assert 500.0 < mean < 20000.0, \
        f"LFE gave unphysical G: mean={mean:.0f}"


def test_lfe_scales_with_true_G():
    # LFE should scale monotonically with true G, even if biased.
    # Compare a stiff phantom to a soft one — LFE median should be larger
    # for the stiffer material.
    N = 16
    src = bottom_plate_driver_sources_3d(N, radius_frac=0.5)
    u_soft = helmholtz_solve_3d(_uniform_G(val=1000.0), freq=60, dx=0.005,
                                 sources=src, top_free=True)
    u_stiff = helmholtz_solve_3d(_uniform_G(val=5000.0), freq=60, dx=0.005,
                                 sources=src, top_free=True)
    lfe_soft  = np.nanmedian(lfe_inversion_3d(u_soft,  freq=60, dx=0.005)[3:-3, 4:-4, 4:-4])
    lfe_stiff = np.nanmedian(lfe_inversion_3d(u_stiff, freq=60, dx=0.005)[3:-3, 4:-4, 4:-4])
    assert lfe_stiff > lfe_soft, \
        f"LFE should scale with G: soft={lfe_soft:.0f}, stiff={lfe_stiff:.0f}"


def test_direct_inversion_recovers_uniform_G():
    N = 16
    G_true = 2500.0
    G = _uniform_G(val=G_true)
    src = bottom_plate_driver_sources_3d(N, radius_frac=0.5)
    u = helmholtz_solve_3d(G, freq=60.0, dx=0.005, sources=src, top_free=True)
    G_di = direct_inversion_3d(u, freq=60.0, dx=0.005)
    # Interior mean should be within ~20% of truth (DI is noisy near sources).
    core = G_di[3:-3, 4:-4, 4:-4]
    core_valid = core[np.isfinite(core)]
    assert core_valid.size > 0
    mean_di = np.median(core_valid)
    rel = abs(mean_di - G_true) / G_true
    assert rel < 0.25, f"DI far from true G: mean={mean_di:.0f} (target {G_true}), rel {rel:.2%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
