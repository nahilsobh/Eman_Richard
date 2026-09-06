"""Tests for anisotropic pre-stress + direction-dependent G + KV viscosity."""
from __future__ import annotations

import numpy as np
import pytest

from src.phantom.geometry_3d import (
    SphericalBalloon,
    effective_G_for_direction,
    lame_field_sphere,
    stress_tensor_sphere,
)
from src.solver.helmholtz_fd_3d import (
    bottom_plate_driver_sources_3d,
    helmholtz_solve_3d,
)


def _balloon(N=24, r=4.0, p=3000.0):
    return SphericalBalloon(center=(N//2, N//2, N//2), radius_vx=r, pressure=p)


# ── Stress-tensor tests ──────────────────────────────────────────────────

def test_stress_tensor_symmetric():
    sig = stress_tensor_sphere(_balloon(), 24)
    assert np.allclose(sig, np.swapaxes(sig, -1, -2)), "σ must be symmetric"


def test_stress_tensor_zero_at_zero_pressure():
    sig = stress_tensor_sphere(_balloon(p=0.0), 24)
    assert np.all(sig == 0.0)


def test_stress_tensor_deviator_matches_lame_scalar():
    # The deviatoric magnitude σ_θθ − σ_rr should equal (3/2) p (a/r)^3
    # outside the balloon — the same field lame_field_sphere returns.
    N = 32
    balloon = _balloon(N=N, r=6.0, p=2000.0)
    sig  = stress_tensor_sphere(balloon, N)
    lame = lame_field_sphere(balloon, N)
    # Take the deviator on the equatorial ring (j=k=N/2, i sweeping).
    j = k = N // 2
    outside = ~balloon.mask(N)
    for i in range(N):
        if not outside[i, j, k]:
            continue
        # r̂ = (di, 0, 0)/|di| points along i (since j=k=cy).
        # So σ_rr = σ_ii, σ_θθ = σ_jj = σ_kk.
        sig_rr = sig[i, j, k, 0, 0]
        sig_tt = sig[i, j, k, 1, 1]
        deviator = sig_tt - sig_rr
        assert np.isclose(deviator, lame[i, j, k], rtol=1e-6), \
            f"mismatch at i={i}: σ_θθ-σ_rr={deviator}, Δσ={lame[i, j, k]}"


# ── Direction-dependent G_eff tests ──────────────────────────────────────

def test_G_eff_tangential_greater_than_radial():
    # At an equatorial shell voxel, radial propagation should soften, tangential
    # should stiffen. Use a large A_coeff to make the effect obvious.
    N = 32
    balloon = _balloon(N=N, r=6.0, p=3000.0)
    sig = stress_tensor_sphere(balloon, N)
    G_base = np.full((N, N, N), 2500.0)
    G_base[balloon.mask(N)] = 2000.0

    G_radial = effective_G_for_direction(sig, np.array([1., 0, 0]), G_base, A_coeff=5.0)
    G_tang   = effective_G_for_direction(sig, np.array([0., 1, 0]), G_base, A_coeff=5.0)

    # Sample points along the +i axis outside the balloon (radial from center).
    # At those points, radial direction = +i, tangential = +j.
    j = k = N // 2
    outside_i = [i for i in range(N // 2 + 1, N)
                 if not balloon.mask(N)[i, j, k]]
    for i in outside_i[:5]:
        assert G_radial[i, j, k] < G_base[i, j, k], \
            f"radial G should soften at i={i}, got {G_radial[i,j,k]:.1f} vs base {G_base[i,j,k]:.1f}"
        assert G_tang[i, j, k]   > G_base[i, j, k], \
            f"tangential G should stiffen at i={i}, got {G_tang[i,j,k]:.1f}"


def test_G_eff_at_zero_pressure_equals_base():
    N = 24
    balloon = _balloon(N=N, p=0.0)
    sig = stress_tensor_sphere(balloon, N)
    G_base = np.full((N, N, N), 2500.0)
    for k in (np.array([1., 0, 0]), np.array([0, 1., 0]), np.array([0, 0, 1.])):
        G_eff = effective_G_for_direction(sig, k, G_base, A_coeff=5.0)
        assert np.allclose(G_eff, G_base), "no pre-stress → G_eff must equal G_base"


# ── Kelvin-Voigt viscosity tests ─────────────────────────────────────────

def test_kv_viscosity_no_op_when_none():
    # viscosity=None must give byte-identical output to unspecified.
    N = 16
    G = np.full((N, N, N), 2500.0)
    src = bottom_plate_driver_sources_3d(N, radius_frac=0.5)
    u1 = helmholtz_solve_3d(G, freq=60.0, dx=0.005, sources=src)
    u2 = helmholtz_solve_3d(G, freq=60.0, dx=0.005, sources=src, viscosity=None)
    assert np.array_equal(u1, u2)


def test_kv_viscosity_adds_damping():
    # A non-zero viscosity must attenuate the wave amplitude compared to
    # the same setup without viscosity (all else equal).
    N = 24
    G = np.full((N, N, N), 2500.0)
    src = bottom_plate_driver_sources_3d(N, radius_frac=0.5)
    u_dry = helmholtz_solve_3d(G, freq=60.0, dx=0.005, sources=src,
                                damping=0.0, viscosity=None)
    u_wet = helmholtz_solve_3d(G, freq=60.0, dx=0.005, sources=src,
                                damping=0.0, viscosity=1.0)   # 1 Pa·s
    # Amplitude far from the driver — top face, interior.
    dry_amp = np.max(np.abs(u_dry[2, 4:-4, 4:-4]))
    wet_amp = np.max(np.abs(u_wet[2, 4:-4, 4:-4]))
    assert wet_amp < 0.9 * dry_amp, \
        f"expected wet-gel attenuation, got dry={dry_amp:.3g}, wet={wet_amp:.3g}"


def test_kv_viscosity_frequency_scaling():
    # For pure KV (no hysteretic damping), doubling ω should roughly double
    # the loss term's effect on penetration depth. Not exact due to standing
    # waves, but higher frequency must attenuate strictly more.
    N = 24
    G = np.full((N, N, N), 2500.0)
    src = bottom_plate_driver_sources_3d(N, radius_frac=0.5)
    u_60  = helmholtz_solve_3d(G, freq=60.0,  dx=0.005, sources=src,
                                damping=0.0, viscosity=2.0)
    u_120 = helmholtz_solve_3d(G, freq=120.0, dx=0.005, sources=src,
                                damping=0.0, viscosity=2.0)
    # Compare far-from-driver amplitude — 120 Hz should be more damped.
    amp_60  = np.max(np.abs(u_60[2, 4:-4, 4:-4]))
    amp_120 = np.max(np.abs(u_120[2, 4:-4, 4:-4]))
    assert amp_120 < amp_60, \
        f"expected 120Hz more damped than 60Hz: 60→{amp_60:.3g}, 120→{amp_120:.3g}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
