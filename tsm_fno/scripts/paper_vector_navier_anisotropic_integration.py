#!/usr/bin/env python3
"""Anisotropic-tensor vector Navier integration (6th TSM pipeline stage).

Uses the newly-added rank-2 μ_ij(x) vector Navier solver on the full
Ogden anisotropic G tensor field (not the isotropic scalar mean). This
is the closest step in-session toward Murnaghan-tensor elasticity: it
preserves the radial-vs-tangential shear anisotropy of the pre-stress
field while solving proper vector Navier physics.

Constitutive law:
    σ_ij = λ · δ_ij · tr(ε)  +  (μ_ik ε_kj + μ_jk ε_ki)
with μ_ij(x) = G_r(x)·r̂_i r̂_j + G_θ(x)·(δ_ij − r̂_i r̂_j)  (Ogden principal).

Prediction: μ_conv should drop toward Yin's flat value (2.7 kPa) because
the anisotropic shear response cancels partially under direction-averaged
inversion. μ_TSM may or may not improve — depends on interference.

Grid: N=28, dx=3 mm, peak state only. ~10-15 min per phantom.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phantom.geometry_3d import (
    ogden_G_tensor_field,
    perilesional_shell_3d,
    SphericalBalloon,
)
from src.solver.helmholtz_fd_3d import (
    direct_inversion_3d,
    directional_filter_3d,
)
from src.solver.vector_elasticity_3d import (
    curl_of_displacement_3d,
    navier_solve_3d_tensor_mu,
)


# N=28 with λ=100·μ_max caused SuperLU factorisation to hang for >2 hours
# (ill-conditioned direct solve on a denser 3D FEM system). We downscale
# to N=24 with λ=10·μ_max and use the 200-mL state (r=12.1 vx fits within
# the 72-mm FOV with a small buffer) — a proof-of-concept run that shows
# the anisotropic-tensor vector Navier produces physical results.
N        = 24
DX       = 0.003
FREQ     = 60.0
RHO      = 1000.0
DAMPING  = 0.05
G_LESION = 2000.0
DRIVER_R = 0.5
CENTER   = (N // 2, N // 2, N // 2)
SHELL_MM        = 5.0
SHELL_OFFSET_MM = 6.0    # 2 vx at dx=3mm; scaled down from 9mm because
                          # the shell would otherwise fall past the grid.
MEDIAN_SIZE     = 3
NDIRS           = 20
WEDGE_WIDTH     = 0.35
AMP_THRESHOLD   = 0.15
LAM_MULTIPLIER  = 10.0    # was 100 (ill-conditioned)


def _r_vx(vol):
    return ((3 * vol * 1e-6 / (4 * math.pi)) ** (1/3)) / DX

A0_VX  = _r_vx(50)
# Use 200 mL (r ≈ 12.1 vx = 36.3 mm) so balloon diameter (72.6 mm)
# fits in the 72 mm FOV of the N=24 grid.
PEAK_A = _r_vx(200)

PHANTOMS = [
    dict(name="Phantom 1 (gelatin)",
         mu=[1100.0, 1400.0], alpha=[7.0, 1.0]),
    dict(name="Phantom 2 (cellulose)",
         mu=[2000.0,  500.0], alpha=[2.0, 10.0]),
]


def _fibonacci_dirs(n):
    phi = (1 + np.sqrt(5)) / 2
    out = []
    for i in range(n):
        z = 1 - (2*i + 1) / n
        theta = 2 * np.pi * i / phi
        r = np.sqrt(max(0.0, 1 - z*z))
        khat = np.array([r*np.cos(theta), r*np.sin(theta), z])
        out.append(khat / (np.linalg.norm(khat) + 1e-30))
    return out


def _sources_bottom_plate_vector(N_grid, driver_frac, comp=1, amp=1.0+0.0j):
    cy = (N_grid - 1) / 2.0
    r_max = (N_grid / 2.0) * driver_frac
    src = []
    for a in range(N_grid):
        for b in range(N_grid):
            if (a - cy) ** 2 + (b - cy) ** 2 > r_max ** 2:
                continue
            src.append((N_grid - 1, a, b, comp, complex(amp)))
    return src


def run_phantom_peak(phantom):
    balloon = SphericalBalloon(center=CENTER, radius_vx=PEAK_A, pressure=0.0)
    G_tensor = ogden_G_tensor_field(N=N, dx=DX, center=CENTER,
                                     a0_vx=A0_VX, a_vx=PEAK_A,
                                     mu_list=phantom["mu"],
                                     alpha_list=phantom["alpha"],
                                     G_lesion=G_LESION,
                                     balloon_mask=balloon.mask(N))
    # Use REAL tensor field (not the trace/3 scalar).
    mu_tensor_real = np.real(G_tensor)
    lam = LAM_MULTIPLIER * float(mu_tensor_real.max())
    tr_mean = np.mean(np.einsum("xyzii->xyz", mu_tensor_real) / 3.0)
    print(f"  μ_ij tensor: min={mu_tensor_real.min():.0f}, "
          f"max={mu_tensor_real.max():.0f}, tr/3 mean={tr_mean:.0f} Pa; λ={lam:.0f}")

    src = _sources_bottom_plate_vector(N, driver_frac=DRIVER_R, comp=1)
    t0 = time.time()
    u_vec = navier_solve_3d_tensor_mu(mu_tensor_real, lam=lam, freq=FREQ,
                                        rho=RHO, dx=DX, damping=DAMPING,
                                        sources=src)
    print(f"  Solve done in {time.time()-t0:.1f} s. Extracting shear …")

    curl = curl_of_displacement_3d(u_vec, dx=DX)
    u_scalar = curl[..., 2]

    directions = _fibonacci_dirs(NDIRS)
    di_maps, amp_maps = [], []
    for khat in directions:
        u_k = directional_filter_3d(u_scalar, khat=khat, angular_width=WEDGE_WIDTH)
        G_DI = direct_inversion_3d(u_k, freq=FREQ, rho=RHO, dx=DX,
                                    median_filter_size=MEDIAN_SIZE)
        di_maps.append(G_DI)
        amp_maps.append(np.abs(u_k))
    di_stack  = np.stack(di_maps,  axis=0)
    amp_stack = np.stack(amp_maps, axis=0)

    peaks = amp_stack.reshape(NDIRS, -1).max(axis=1)
    thresh = peaks[:, None, None, None] * AMP_THRESHOLD
    di_stack = np.where(amp_stack >= thresh, di_stack, np.nan)

    w = amp_stack ** 2
    with np.errstate(invalid="ignore"):
        num = np.nansum(np.where(np.isnan(di_stack), 0.0, w * di_stack), axis=0)
        den = np.nansum(np.where(np.isnan(di_stack), 0.0, w),            axis=0)
        mu_conv = num / (den + 1e-30)
        mu_conv[den == 0] = np.nan
    mu_tsm = np.nanmax(di_stack, axis=0)

    shell = perilesional_shell_3d(balloon.mask(N), shell_mm=SHELL_MM, dx=DX,
                                   inner_offset_mm=SHELL_OFFSET_MM)

    def _ring(f):
        v = f[shell]; v = v[np.isfinite(v)]
        if v.size == 0: return float("nan")
        lo, hi = np.percentile(v, [10, 90])
        tr = v[(v >= lo) & (v <= hi)]
        return float(tr.mean()) if tr.size else float("nan")

    return dict(ring_conv=_ring(mu_conv), ring_tsm=_ring(mu_tsm))


def main():
    out_dir = ROOT / "results" / "paper_vector_navier_anisotropic"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nAnisotropic-tensor vector Navier — peak state only")
    print(f"N={N}, DX={DX*1000:.0f} mm\n")

    results = []
    for phantom in PHANTOMS:
        print(f"[{phantom['name']}]")
        print(f"  Ogden μ={phantom['mu']}  α={phantom['alpha']}")
        r = run_phantom_peak(phantom)
        r["name"] = phantom["name"]
        results.append(r)
        print(f"  μ_conv = {r['ring_conv']/1000:.2f} kPa   "
              f"μ_TSM = {r['ring_tsm']/1000:.2f} kPa\n")

    lines = [
        "Anisotropic-tensor vector Navier — peak state only",
        "=" * 78,
        f"Grid: N={N}³, dx={DX*1000:.0f} mm",
        "Method: Ogden G_ij(x) tensor field → vector Navier with quasi-anisotropic",
        "        constitutive (σ = λ·tr(ε)·δ + μ_ik ε_kj + μ_jk ε_ki) → curl_z(u)",
        f"        → {NDIRS}-direction filter → DI (median={MEDIAN_SIZE}, "
        f"offset={SHELL_OFFSET_MM}mm)",
        "",
        "Yin reference @ peak: P1 TSM=4.40, P2 TSM=5.15, conv~2.8/3.3 (flat)",
        "",
    ]
    for r in results:
        lines.append(f"{r['name']}:")
        lines.append(f"  μ_conv = {r['ring_conv']/1000:.3f} kPa")
        lines.append(f"  μ_TSM  = {r['ring_tsm']/1000:.3f} kPa")
        lines.append("")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
