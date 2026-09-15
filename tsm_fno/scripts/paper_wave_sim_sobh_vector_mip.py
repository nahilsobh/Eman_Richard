#!/usr/bin/env python3
"""Vector Navier + tensor µ_ij(x) + curl→filter→DI→MIP at 80 Hz.

Closes the gap between our previous "scalar Helmholtz + isotropic G(x)"
pipeline and Yin's actual TSM protocol:

  1. Build spatially-varying rank-2 tensor stiffness µ_ij(x) in
     Cartesian coordinates by rotating the Sobh-Ehman spherical-frame
     acoustoelastic moduli:
         µ_ij(x) = µ_rr(r) · r̂_i r̂_j + µ_θθ(r) · (δ_ij − r̂_i r̂_j)

  2. Solve the time-harmonic vector Navier equation with that tensor
     stiffness (ρω² u_i = ∂_j σ_ij, σ from the quasi-anisotropic
     constitutive law in `vector_elasticity_3d.py`):
         all 6 walls fixed, bottom disk driver in the +i direction,
         80 Hz (Yin's phantom frequency),
         water inside the balloon (isotropic soft µ_ij = G_water δ_ij).

  3. Extract the vector displacement u(x) and compute its curl
     q(x) = ∇ × u(x). Represent the shear-wave content by the
     z-component q_z (tangential-along-z rotation) as the scalar
     wave field for direct inversion.

  4. Apply Yin's 20-direction k-space wedge filter to q_z, direct-
     invert each filtered field with a 3×3×3 median filter, then
     combine as µ_TSM (voxelwise MAX) and µ_conv (amp-weighted mean).

  5. Report ring-mean stiffness inside the 12 mm perilesional shell
     alongside the ground-truth µ_θθ and the earlier scalar-Helmholtz
     numbers.

Grid: 32³ hex at dx = 6 mm → 19.2 cm cube (matches the earlier
paper_wave_sim_sobh_vector run so vector-vs-scalar tensor-vs-isotropic
runs can be directly compared).
"""
from __future__ import annotations

import gc
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


N   = 32
DX  = 0.006
CUBE_L_M = N * DX
FREQ_HZ = 80.0
RHO = 1000.0
DAMPING = 0.05
LAM_C = 1.0e5
DRIVER_AMP = 1.0e-6
DRIVER_R_FRAC = 0.5

A_REF_M = 0.02285
V_INJECTION_ML = 250
A_INFL_M = ((3 * V_INJECTION_ML * 1e-6) / (4 * math.pi)) ** (1/3)

MU_MEAN = 2750.0
K1_MEAN = 3250.0
K2_MEAN = 1.25
G_WATER = 1.0

N_DIRECTIONS = 20
WEDGE_WIDTH = 0.35
MEDIAN_FILTER = 3


def _W1_P1(lam):
    return np.full_like(lam, MU_MEAN / 2.0)


def _W1_P2(lam):
    eps = lam - 1.0
    lam6_m1 = lam ** 6 - 1.0
    safe = np.where(np.abs(lam6_m1) < 1e-12, 1e-12, lam6_m1)
    fiber_W1 = (K1_MEAN * eps * np.exp(K2_MEAN * eps ** 2) * lam ** 5
                  / (4.0 * safe))
    fiber_W1 = np.where(np.abs(lam - 1.0) < 1e-8, K1_MEAN / 24.0, fiber_W1)
    return MU_MEAN / 2.0 + fiber_W1


def build_tensor_field(W1_fn, name):
    print(f"  building tensor field for {name}...")
    ii, jj, kk = np.indices((N, N, N))
    c = (N - 1) / 2.0
    dz = (ii - c) * DX
    dy = (jj - c) * DX
    dxv = (kk - c) * DX
    r = np.sqrt(dxv ** 2 + dy ** 2 + dz ** 2)
    in_balloon = r < A_INFL_M
    in_gel = ~in_balloon
    r_safe = np.where(r < 1e-12, 1e-12, r)

    R_ref = np.cbrt(r_safe ** 3 - A_INFL_M ** 3 + A_REF_M ** 3)
    lam_theta = np.where(in_gel, r_safe / R_ref, 1.0)

    W1 = W1_fn(lam_theta)
    mu_tt = 2.0 * W1 * lam_theta ** 2
    mu_rr = 2.0 * W1 * lam_theta ** (-4)

    mu_tt = np.where(in_balloon, G_WATER, mu_tt)
    mu_rr = np.where(in_balloon, G_WATER, mu_rr)

    rhat_i = dz  / r_safe
    rhat_j = dy  / r_safe
    rhat_k = dxv / r_safe
    rhat = np.stack([rhat_i, rhat_j, rhat_k], axis=-1)

    RR = rhat[..., :, None] * rhat[..., None, :]
    dij = np.eye(3)[None, None, None, :, :]
    mu_tensor = (mu_rr[..., None, None] * RR
                    + mu_tt[..., None, None] * (dij - RR))

    return mu_tensor, in_balloon, mu_tt, mu_rr


def bottom_disk_driver(comp=0, amp=DRIVER_AMP, radius_frac=DRIVER_R_FRAC):
    cy = (N - 1) / 2.0
    cz = (N - 1) / 2.0
    r_max = (N / 2.0) * radius_frac
    src = []
    for j in range(N):
        for k in range(N):
            if (j - cy) ** 2 + (k - cz) ** 2 <= r_max ** 2:
                src.append((N - 1, j, k, comp, complex(amp)))
    return src


def fibonacci_sphere(n):
    i = np.arange(n)
    phi = (1 + 5 ** 0.5) / 2
    theta = 2 * np.pi * i / phi
    z = 1 - 2 * (i + 0.5) / n
    rxy = np.sqrt(np.maximum(0.0, 1 - z * z))
    return np.stack([rxy * np.cos(theta), rxy * np.sin(theta), z], axis=1)


def ring_mask(balloon):
    ii, jj, kk = np.indices((N, N, N))
    c = (N - 1) / 2.0
    r = np.sqrt(((ii - c) * DX) ** 2 + ((jj - c) * DX) ** 2 + ((kk - c) * DX) ** 2)
    return (~balloon) & (r >= A_INFL_M) & (r <= A_INFL_M + 0.012)


def ring_mean(G, ring):
    v = G[ring]; v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def curl_mip_pipeline(u_vec, ring_msk, label):
    """Yin's TSM pipeline applied to the vector displacement u_vec.

    1. curl q = ∇ × u  (complex 3-vector field)
    2. use q_z (component about the vertical axis) as the scalar wave field
       for DI — a tangential-plane rotation, natural for the shell ring
    3. for each of 20 Fibonacci directions: k-space wedge filter → DI → collect
    4. combine: µ_TSM (max), µ_conv (amplitude-weighted mean)
    """
    print(f"  [{label}] computing curl ∇ × u ...", flush=True)
    q = curl_of_displacement_3d(u_vec, DX)   # (N,N,N,3) complex
    q_z = q[..., 0]                            # use i-axis component (vertical)
    del q

    directions = fibonacci_sphere(N_DIRECTIONS)
    G_stack   = np.zeros((N_DIRECTIONS, N, N, N), dtype=np.float32)
    amp_stack = np.zeros((N_DIRECTIONS, N, N, N), dtype=np.float32)
    for d, khat in enumerate(directions):
        q_k = directional_filter_3d(q_z, khat=khat, angular_width=WEDGE_WIDTH)
        G_k = direct_inversion_3d(q_k, freq=FREQ_HZ, rho=RHO, dx=DX,
                                       median_filter_size=MEDIAN_FILTER)
        G_stack[d] = np.where(np.isfinite(G_k), G_k, 0.0).astype(np.float32)
        amp_stack[d] = np.abs(q_k).astype(np.float32)
        del q_k, G_k
        if (d + 1) % 5 == 0:
            print(f"    direction {d+1}/{N_DIRECTIONS}", flush=True)

    pos = np.where(G_stack > 0, G_stack, 0.0)
    G_tsm = pos.max(axis=0).astype(float)
    wsum = amp_stack.sum(axis=0)
    wsafe = np.where(wsum > 1e-30, wsum, 1e-30)
    G_conv = ((pos * amp_stack).sum(axis=0) / wsafe).astype(float)

    del G_stack, amp_stack, pos, wsum, wsafe, q_z
    gc.collect()

    return G_conv, G_tsm


def run_phantom(W1_fn, label, out_dir):
    print(f"\n[{label}] building tensor stiffness…")
    mu_tensor, balloon, mu_tt, mu_rr = build_tensor_field(W1_fn, label)
    print(f"  µ_θθ range {mu_tt.min()/1000:.2f} – {mu_tt.max()/1000:.2f} kPa")
    print(f"  µ_rr range {mu_rr.min()/1000:.4f} – {mu_rr.max()/1000:.4f} kPa")

    sources = bottom_disk_driver(comp=0)   # driver in +i (vertical) direction
    print(f"  {len(sources)} bottom-face driver nodes, amp = "
          f"{DRIVER_AMP*1e6:.1f} µm on component 0 (vertical)")

    print(f"\n[{label}] solving vector Navier with tensor µ at {FREQ_HZ} Hz…",
          flush=True)
    t0 = time.time()
    u_vec = navier_solve_3d_tensor_mu(mu_tensor, lam=LAM_C, freq=FREQ_HZ,
                                          rho=RHO, dx=DX, damping=DAMPING,
                                          sources=sources)
    print(f"  Navier solve: {time.time()-t0:.1f} s   |u| range "
          f"{np.abs(u_vec).min():.2e} – {np.abs(u_vec).max():.2e} m",
          flush=True)

    ring_msk = ring_mask(balloon)
    ring_gt = ring_mean(mu_tt, ring_msk)
    print(f"  GT ring µ_θθ = {ring_gt/1000:.3f} kPa")

    G_conv, G_tsm = curl_mip_pipeline(u_vec, ring_msk, label)

    G_conv_masked = np.where(balloon, np.nan, G_conv)
    G_tsm_masked  = np.where(balloon, np.nan, G_tsm)

    ring_conv = ring_mean(G_conv_masked, ring_msk)
    ring_tsm  = ring_mean(G_tsm_masked, ring_msk)
    print(f"  ring µ_conv = {ring_conv/1000:.3f} kPa")
    print(f"  ring µ_TSM  = {ring_tsm/1000:.3f} kPa")
    print(f"  TSM/conv    = {ring_tsm/max(ring_conv,1):.3f}")

    # Save phantom result immediately (defense against OOM)
    np.savez_compressed(out_dir / f"vector_{label}.npz",
                          mu_tt=mu_tt, mu_rr=mu_rr, balloon=balloon,
                          u_vec=u_vec,
                          G_conv=G_conv_masked, G_tsm=G_tsm_masked)
    print(f"  saved vector_{label}.npz", flush=True)
    return {"ring_gt": ring_gt, "ring_conv": ring_conv, "ring_tsm": ring_tsm,
            "mu_tt": mu_tt, "mu_rr": mu_rr, "balloon": balloon,
            "u_vec": u_vec, "G_conv": G_conv_masked, "G_tsm": G_tsm_masked}


def main():
    out_dir = ROOT / "results" / "paper_wave_sim_sobh_vector_mip"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Vector-Navier + tensor µ_ij(x) + curl→MIP pipeline")
    print(f"Grid: {N}³ at dx = {DX*1000:.1f} mm → {CUBE_L_M*100:.1f}³ cm cube")
    print(f"Frequency: {FREQ_HZ} Hz (Yin's phantom frequency)")
    print(f"Balloon at 250 mL (a = {A_INFL_M*100:.3f} cm)")
    print(f"BCs: all 6 walls fixed (solver limitation), bottom disk driver")
    print()

    r_P1 = run_phantom(_W1_P1, "P1", out_dir)
    gc.collect()
    r_P2 = run_phantom(_W1_P2, "P2", out_dir)

    lines = [
        f"Vector Navier + rotated Cartesian tensor µ_ij(x) + curl→MIP",
        f"at Yin's 80 Hz on the Sobh-Ehman phantoms at 250 mL",
        "=" * 76,
        f"Grid: {N}³ at dx = {DX*1000:.1f} mm → {CUBE_L_M*100:.1f}³ cm cube",
        f"BCs: all 6 walls fixed (solver limitation); bottom-plate driver",
        "      at 1 µm amplitude on the +i (vertical) component.",
        "Curl representative: q_z (vertical component of ∇ × u).",
        "",
        "12 mm perilesional ring means:",
        f"{'':16s}{'GT µ_θθ':>12s}{'µ_conv':>12s}{'µ_TSM':>12s}"
        f"{'TSM/conv':>12s}",
        f"  P1              {r_P1['ring_gt']/1000:>12.3f}"
        f"{r_P1['ring_conv']/1000:>12.3f}"
        f"{r_P1['ring_tsm']/1000:>12.3f}"
        f"{r_P1['ring_tsm']/max(r_P1['ring_conv'],1):>12.3f}",
        f"  P2              {r_P2['ring_gt']/1000:>12.3f}"
        f"{r_P2['ring_conv']/1000:>12.3f}"
        f"{r_P2['ring_tsm']/1000:>12.3f}"
        f"{r_P2['ring_tsm']/max(r_P2['ring_conv'],1):>12.3f}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
