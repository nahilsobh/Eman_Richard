#!/usr/bin/env python3
"""Vector Navier + tensor µ_ij + top-free + MULTI-FACE broadband + curl→MIP.

This is the "physics-correct + rich-direction-content" variant of the
vector pipeline:

  Forward:  vector Navier with rank-2 tensor µ_ij(x)  (rotated Cartesian)
            top-free ghost mirror on the +z face,
            5 rigid walls elsewhere,
            multi-face broadband source (bottom + 4 sides, top left free),
            each face driven on its face-normal displacement component.

  Inversion: q = ∇ × u  →  q_z as scalar wave field  →  20-direction
            k-space wedge filter  →  3D direct inversion with 3×3×3
            median filter per direction  →  µ_TSM (max) and µ_conv
            (amplitude-weighted mean).

Grid: 32³ hex at dx = 6 mm (fits vector Navier's SuperLU footprint).
Frequency: 80 Hz (Yin's phantom frequency).
"""
from __future__ import annotations

import gc, math, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.solver.helmholtz_fd_3d import (
    directional_filter_3d,
    direct_inversion_3d,
)
from src.solver.vector_elasticity_3d import (
    curl_of_displacement_3d,
    navier_solve_3d_tensor_mu,
)


# Yin's container: 15 cm (x) × 15 cm (y) × 18 cm (z).  Grid at dx = 3 mm to
# match the 3-mm acquisition voxel resolution exactly.
# Axis convention: (i, j, k) = (x, y, z), z vertical increasing upward.
# Bottom (driver) at k=0, top (traction-free) at k=NZ-1.
NX, NY, NZ = 50, 50, 60
DX  = 0.003
FREQ_HZ = 80.0
RHO = 1000.0
DAMPING = 0.05
# Bulk-penalty first Lamé (spatially variable):
#   gel  →  LAM_GEL  = 100 kPa  (soft-incompressibility penalty, matches paper)
#   ball →  LAM_BALL = 10 MPa   (water incompressibility, K much larger than gel)
# Combined into a lam_field of shape (Nx, Ny, Nz) at run time.
LAM_GEL  = 1.0e5
LAM_BALL = 1.0e7
DRIVER_AMP = 1.0e-6
DRIVER_R_FRAC = 0.5

A_REF_M = 0.02285
V_INJECTION_ML = 250
A_INFL_M = ((3 * V_INJECTION_ML * 1e-6) / (4 * math.pi)) ** (1/3)

# Fitted Sobh-Ehman material model (see perilesional_model/sce_models.py).
# mu0 is pinned to the digitised Fig. 6 baseline at lam_theta = 1; P2 carries
# a power-law fibre recruitment identified from the P1-normalised ratio.
P1_MU0 = 3530.0   # Pa   — neo-Hookean matrix baseline
P2_MU0 = 3340.0   # Pa   — matrix + fibre baseline
P2_C   = 0.2728   # power-law fibre recruitment coefficient (dimensionless)
P2_M   = 0.5791   # power-law fibre recruitment exponent    (dimensionless)
# Balloon + water treated as a single incompressible fluid inclusion:
#   µ = 0 (a fluid cannot support shear)
#   K = large (water is incompressible, K ~ 2.2 GPa)
# The high bulk stiffness is captured by the global LAM_C above (10 MPa is
# sufficient for MRE at 80 Hz — P-wavelength >> domain — while keeping the
# sparse-system conditioning well within PARDISO's safe range).
# G_BALL is set slightly above zero for numerical stability (avoids exactly
# singular shear operator inside the ball).  See problem_formulation.tex
# §Balloon-model paragraph for physics discussion.
G_BALL = 1.0

N_DIRECTIONS = 20
WEDGE_WIDTH = 0.35
MEDIAN_FILTER = 3


def _W1_P1(lam):
    """Neo-Hookean matrix: W1 = mu0 / 2 (constant)."""
    return np.full_like(lam, P1_MU0 / 2.0)


def _W1_P2(lam):
    """Matrix + power-law fibre: W1 = (mu0 / 2) * (1 + c*g^m), g = I1 - 3.

    Incompressible spherical kinematics: I1 = lam^-4 + 2*lam^2.
    """
    I1 = lam ** (-4) + 2.0 * lam ** 2
    g  = np.maximum(I1 - 3.0, 0.0)
    f  = 1.0 + P2_C * g ** P2_M
    return (P2_MU0 / 2.0) * f


def build_tensor_field(W1_fn):
    # (i, j, k) = (x, y, z).  Origin at grid centre.
    ii, jj, kk = np.indices((NX, NY, NZ))
    cx = (NX - 1) / 2.0; cy = (NY - 1) / 2.0; cz = (NZ - 1) / 2.0
    dx_c = (ii - cx) * DX; dy_c = (jj - cy) * DX; dz_c = (kk - cz) * DX
    r = np.sqrt(dx_c ** 2 + dy_c ** 2 + dz_c ** 2)
    in_balloon = r < A_INFL_M
    r_safe = np.where(r < 1e-12, 1e-12, r)

    R_ref = np.cbrt(r_safe ** 3 - A_INFL_M ** 3 + A_REF_M ** 3)
    lam_theta = np.where(~in_balloon, r_safe / R_ref, 1.0)

    W1 = W1_fn(lam_theta)
    mu_tt = 2.0 * W1 * lam_theta ** 2
    mu_rr = 2.0 * W1 * lam_theta ** (-4)

    mu_tt = np.where(in_balloon, G_BALL, mu_tt)
    mu_rr = np.where(in_balloon, G_BALL, mu_rr)

    # Variable Lamé λ: gel gets the soft-incompressibility penalty,
    # ball gets water-scale incompressibility.
    lam_field = np.where(in_balloon, LAM_BALL, LAM_GEL)

    # Cartesian components of r̂ in (x, y, z) ordering (matches solver axes).
    rhat_x = dx_c / r_safe
    rhat_y = dy_c / r_safe
    rhat_z = dz_c / r_safe
    rhat = np.stack([rhat_x, rhat_y, rhat_z], axis=-1)
    RR = rhat[..., :, None] * rhat[..., None, :]
    dij = np.eye(3)[None, None, None, :, :]
    mu_tensor = (mu_rr[..., None, None] * RR
                    + mu_tt[..., None, None] * (dij - RR))

    return mu_tensor, in_balloon, mu_tt, lam_field


def multi_face_vector_sources(radius_frac=DRIVER_R_FRAC, amp=DRIVER_AMP):
    """5-face broadband disk sources (bottom + 4 sides, top left free).

    Axis convention (i, j, k) = (x, y, z), z vertical.  Top face at k=NZ-1
    is left free; bottom face at k=0 carries the driver.
    Each face is driven on its own face-normal displacement component so
    the source behaves like a piston pushing perpendicular to that wall.
    Disk radius = radius_frac × half-min-in-plane-dim on each face.
    """
    src = []
    cx = (NX - 1) / 2.0; cy = (NY - 1) / 2.0; cz = (NZ - 1) / 2.0
    ampC = complex(amp)

    # Bottom face k = 0 — driven on k-component (z, comp 2), disk in (i, j)
    r_max_bot = (min(NX, NY) / 2.0) * radius_frac
    for i in range(NX):
        for j in range(NY):
            if (i - cx) ** 2 + (j - cy) ** 2 <= r_max_bot ** 2:
                src.append((i, j, 0, 2, ampC))

    # ±x sides (i = NX-1 and i = 0) — driven on i-component (x, comp 0), disk in (j, k)
    r_max_i = (min(NY, NZ) / 2.0) * radius_frac
    for j in range(NY):
        for k in range(NZ):
            if (j - cy) ** 2 + (k - cz) ** 2 <= r_max_i ** 2:
                src.append((NX - 1, j, k, 0, ampC))
                src.append((0,      j, k, 0, ampC))

    # ±y sides (j = NY-1 and j = 0) — driven on j-component (y, comp 1), disk in (i, k)
    r_max_j = (min(NX, NZ) / 2.0) * radius_frac
    for i in range(NX):
        for k in range(NZ):
            if (i - cx) ** 2 + (k - cz) ** 2 <= r_max_j ** 2:
                src.append((i, NY - 1, k, 1, ampC))
                src.append((i, 0,      k, 1, ampC))

    return src


def fibonacci_sphere(n):
    i = np.arange(n)
    phi = (1 + 5 ** 0.5) / 2
    theta = 2 * np.pi * i / phi
    z = 1 - 2 * (i + 0.5) / n
    rxy = np.sqrt(np.maximum(0.0, 1 - z * z))
    return np.stack([rxy * np.cos(theta), rxy * np.sin(theta), z], axis=1)


def ring_mask(balloon):
    # (i, j, k) = (x, y, z).  12 mm shell around the cavity edge.
    ii, jj, kk = np.indices((NX, NY, NZ))
    cx = (NX - 1) / 2.0; cy = (NY - 1) / 2.0; cz = (NZ - 1) / 2.0
    r = np.sqrt(((ii - cx) * DX) ** 2 + ((jj - cy) * DX) ** 2 + ((kk - cz) * DX) ** 2)
    return (~balloon) & (r >= A_INFL_M) & (r <= A_INFL_M + 0.012)


def ring_mean(G, ring):
    v = G[ring]; v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def curl_mip(u_vec, ring_msk, label):
    q = curl_of_displacement_3d(u_vec, DX)
    q_z = q[..., 2]         # (i,j,k) = (x,y,z), so component 2 is the z-curl
    del q
    directions = fibonacci_sphere(N_DIRECTIONS)
    G_stack = np.zeros((N_DIRECTIONS, NX, NY, NZ), dtype=np.float32)
    amp_stack = np.zeros((N_DIRECTIONS, NX, NY, NZ), dtype=np.float32)
    for d, khat in enumerate(directions):
        q_k = directional_filter_3d(q_z, khat=khat, angular_width=WEDGE_WIDTH)
        G_k = direct_inversion_3d(q_k, freq=FREQ_HZ, rho=RHO, dx=DX,
                                       median_filter_size=MEDIAN_FILTER)
        G_stack[d] = np.where(np.isfinite(G_k), G_k, 0.0).astype(np.float32)
        amp_stack[d] = np.abs(q_k).astype(np.float32)
        del q_k, G_k
        if (d + 1) % 5 == 0:
            print(f"    {label} direction {d+1}/{N_DIRECTIONS}", flush=True)

    pos = np.where(G_stack > 0, G_stack, 0.0)
    G_tsm = pos.max(axis=0).astype(float)
    wsum = amp_stack.sum(axis=0)
    wsafe = np.where(wsum > 1e-30, wsum, 1e-30)
    G_conv = ((pos * amp_stack).sum(axis=0) / wsafe).astype(float)
    del G_stack, amp_stack, pos, wsum, wsafe, q_z
    gc.collect()
    return G_conv, G_tsm


def run_phantom(W1_fn, label, sources, out_dir):
    print(f"\n[{label}] tensor stiffness…")
    mu_tensor, balloon, mu_tt, lam_field = build_tensor_field(W1_fn)
    print(f"  µ_θθ range {mu_tt.min()/1000:.2f} – {mu_tt.max()/1000:.2f} kPa")
    print(f"  λ field:  gel {LAM_GEL/1000:.0f} kPa, ball {LAM_BALL/1000:.0f} kPa")

    print(f"[{label}] vector Navier at {FREQ_HZ} Hz, top_free=True…", flush=True)
    t0 = time.time()
    u_vec = navier_solve_3d_tensor_mu(mu_tensor, lam=lam_field, freq=FREQ_HZ,
                                          rho=RHO, dx=DX, damping=DAMPING,
                                          sources=sources, top_free=True)
    print(f"  Navier: {time.time()-t0:.1f} s   |u|_max = "
          f"{np.abs(u_vec).max():.2e} m", flush=True)

    ring_msk = ring_mask(balloon)
    G_conv, G_tsm = curl_mip(u_vec, ring_msk, label)

    G_conv = np.where(balloon, np.nan, G_conv)
    G_tsm  = np.where(balloon, np.nan, G_tsm)

    ring_gt   = ring_mean(mu_tt, ring_msk)
    ring_conv = ring_mean(G_conv, ring_msk)
    ring_tsm  = ring_mean(G_tsm,  ring_msk)
    print(f"  GT µ_θθ ring = {ring_gt/1000:.3f} kPa")
    print(f"  µ_conv  ring = {ring_conv/1000:.3f} kPa")
    print(f"  µ_TSM   ring = {ring_tsm/1000:.3f} kPa")
    print(f"  TSM/conv     = {ring_tsm/max(ring_conv,1):.3f}")

    np.savez_compressed(out_dir / f"vector_{label}.npz",
                          mu_tt=mu_tt, balloon=balloon, u_vec=u_vec,
                          G_conv=G_conv, G_tsm=G_tsm)
    return {"gt": ring_gt, "conv": ring_conv, "tsm": ring_tsm}


def main():
    out_dir = ROOT / "results" / "paper_wave_sim_sobh_vector_multiface_topfree_yindim_sce_rigidball"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Grid: (NX,NY,NZ) = ({NX},{NY},{NZ}) at dx = {DX*1000:.1f} mm "
          f"({NX*DX*100:.1f} x {NY*DX*100:.1f} x {NZ*DX*100:.1f} cm, "
          f"matches Yin's 15 x 15 x 18 cm)")
    print(f"Frequency: {FREQ_HZ} Hz (Yin's)")
    print(f"BCs: 5 walls fixed, top σ·n=0 (ghost-node at k=Nz-1)")
    sources = multi_face_vector_sources()
    print(f"Multi-face broadband source: {len(sources)} nodes across "
          f"bottom + 4 sides, driven on face-normal component")

    r1 = run_phantom(_W1_P1, "P1", sources, out_dir)
    gc.collect()
    r2 = run_phantom(_W1_P2, "P2", sources, out_dir)

    lines = [
        "Vector Navier + tensor µ_ij + multi-face broadband + top-free + curl→MIP",
        f"at Yin's {FREQ_HZ} Hz on the Sobh-Ehman phantoms at 250 mL",
        "=" * 76,
        f"Grid: (NX,NY,NZ) = ({NX},{NY},{NZ}) at dx = {DX*1000:.1f} mm "
        f"({NX*DX*100:.1f} x {NY*DX*100:.1f} x {NZ*DX*100:.1f} cm, matches Yin)",
        f"Driver: multi-face broadband at bottom + 4 sides, face-normal component",
        f"BCs: 5 walls Dirichlet u=0, top σ·n=0 (ghost-node)",
        "",
        "12 mm perilesional ring means:",
        f"{'':16s}{'GT µ_θθ':>12s}{'µ_conv':>12s}{'µ_TSM':>12s}{'TSM/conv':>12s}",
        f"  P1              {r1['gt']/1000:>12.3f}{r1['conv']/1000:>12.3f}"
        f"{r1['tsm']/1000:>12.3f}{r1['tsm']/max(r1['conv'],1):>12.3f}",
        f"  P2              {r2['gt']/1000:>12.3f}{r2['conv']/1000:>12.3f}"
        f"{r2['tsm']/1000:>12.3f}{r2['tsm']/max(r2['conv'],1):>12.3f}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
