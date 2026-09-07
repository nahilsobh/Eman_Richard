#!/usr/bin/env python3
"""Vector Navier integration into the TSM pipeline (5th data lineage).

Uses the newly-added vector elasticity solver on the Ogden-derived μ(x)
field, extracts the shear component via curl(u), then feeds through the
existing directional-filter + DI machinery. Produces μ_TSM and μ_conv at
peak inflation for both phantoms — added to the pipeline summary as a
5th comparison point.

Grid: N=28 (dx=3 mm, FOV 84 mm) — smaller than the other pipelines' N=32
because the vector solver has 3× the DOF and O(N³) Python assembly.
Peak state only (250 mL, r=13 vx = 39 mm diameter fits in 84 mm FOV
with ~3 mm buffer per side).

Runtime ~4 min per phantom, ~8 min total.
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
    multi_face_broadband_sources,
)
from src.solver.vector_elasticity_3d import (
    curl_of_displacement_3d,
    navier_solve_3d_isotropic,
)


N        = 28
DX       = 0.003
FREQ     = 60.0
RHO      = 1000.0
DAMPING  = 0.05
G_LESION = 2000.0
DRIVER_R = 0.5
CENTER   = (N // 2, N // 2, N // 2)
SHELL_MM        = 5.0
SHELL_OFFSET_MM = 9.0
MEDIAN_SIZE     = 3
NDIRS           = 20
WEDGE_WIDTH     = 0.35
AMP_THRESHOLD   = 0.15
# Nearly-incompressible penalty via λ. Real gel is essentially
# incompressible (K → ∞); we use λ = 100·μ_max as a soft penalty.
LAM_MULTIPLIER = 100.0


def _r_vx(vol):
    return ((3 * vol * 1e-6 / (4 * math.pi)) ** (1/3)) / DX


A0_VX  = _r_vx(50)
PEAK_A = _r_vx(250)


PHANTOMS = [
    dict(name="Phantom 1 (gelatin)",
         mu=[1100.0, 1400.0], alpha=[7.0, 1.0]),   # best-fit Ogden
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
    """Coherent piston-plate source on bottom face (i = N-1). Vector-typed:
    drives a specific component (default y, i.e. comp=1)."""
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
    """Solve the Ogden-derived μ field for one phantom at peak inflation
    using the vector Navier solver, then run the standard TSM pipeline on
    the shear component (curl of u)."""
    balloon = SphericalBalloon(center=CENTER, radius_vx=PEAK_A, pressure=0.0)
    G_tensor = ogden_G_tensor_field(N=N, dx=DX, center=CENTER,
                                     a0_vx=A0_VX, a_vx=PEAK_A,
                                     mu_list=phantom["mu"],
                                     alpha_list=phantom["alpha"],
                                     G_lesion=G_LESION,
                                     balloon_mask=balloon.mask(N))
    # Scalar μ field for the vector solver = isotropic mean of the tensor.
    mu_field = np.einsum("xyzii->xyz", G_tensor) / 3.0
    lam = LAM_MULTIPLIER * float(mu_field.max())

    print(f"  Solving vector Navier: N={N}, DOF={3*N**3}, μ∈[{mu_field.min():.0f},"
          f" {mu_field.max():.0f}] Pa, λ={lam:.0f} Pa (penalty)")
    src = _sources_bottom_plate_vector(N, driver_frac=DRIVER_R, comp=1)
    t0 = time.time()
    u_vec = navier_solve_3d_isotropic(mu_field, lam=lam, freq=FREQ,
                                        rho=RHO, dx=DX, damping=DAMPING,
                                        sources=src)
    print(f"  Solve done in {time.time()-t0:.1f} s. Extracting shear …")

    curl = curl_of_displacement_3d(u_vec, dx=DX)          # (N,N,N,3)
    u_scalar = curl[..., 2]  # z-component of curl — the dominant shear mode
                              # for a y-polarised driver on the bottom face.

    # Apply the standard directional-filter + DI pipeline to the scalar field.
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

    return dict(ring_conv=_ring(mu_conv), ring_tsm=_ring(mu_tsm),
                mu_field_max=float(mu_field.max()))


def main():
    out_dir = ROOT / "results" / "paper_vector_navier_integration"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nVector-Navier TSM integration at peak inflation")
    print(f"N={N}, DX={DX*1000:.0f} mm, FOV={N*DX*1000:.0f} mm")
    print(f"Balloon peak radius: {PEAK_A:.2f} vx = {PEAK_A*DX*1000:.0f} mm "
          f"(fits with buffer {(N/2 - PEAK_A)*DX*1000:.0f} mm/side)\n")

    results = []
    for phantom in PHANTOMS:
        print(f"[{phantom['name']}]")
        print(f"  Ogden mu={phantom['mu']}  alpha={phantom['alpha']}")
        r = run_phantom_peak(phantom)
        r["name"] = phantom["name"]
        results.append(r)
        print(f"  μ_conv = {r['ring_conv']/1000:.2f} kPa   "
              f"μ_TSM = {r['ring_tsm']/1000:.2f} kPa\n")

    lines = [
        "Vector-Navier TSM integration (peak state only)",
        "=" * 78,
        f"Grid: N={N}³, dx={DX*1000:.0f} mm  (vs N=32 in other pipelines)",
        f"Method: Ogden G(x) → vector Navier → curl_z(u) → 20-dir filter → DI",
        f"        median={MEDIAN_SIZE}, shell offset={SHELL_OFFSET_MM} mm",
        "",
        "Peak (250 mL) reference values:",
        "  Yin measured        P1 TSM = 4.40 kPa   P2 TSM = 5.15 kPa",
        "  Yin measured        P1 conv= 2.80 kPa   P2 conv= 3.30 kPa",
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
