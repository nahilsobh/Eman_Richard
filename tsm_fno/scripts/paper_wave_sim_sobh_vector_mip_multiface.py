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


N   = 48
DX  = 0.18 / N   # 3.75 mm — near Yin's 3 mm voxel resolution
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
    fiber_W1 = K1_MEAN * eps * np.exp(K2_MEAN * eps ** 2) * lam ** 5 / (4.0 * safe)
    fiber_W1 = np.where(np.abs(lam - 1.0) < 1e-8, K1_MEAN / 24.0, fiber_W1)
    return MU_MEAN / 2.0 + fiber_W1


def build_tensor_field(W1_fn):
    ii, jj, kk = np.indices((N, N, N))
    c = (N - 1) / 2.0
    dz = (ii - c) * DX; dy = (jj - c) * DX; dxv = (kk - c) * DX
    r = np.sqrt(dxv ** 2 + dy ** 2 + dz ** 2)
    in_balloon = r < A_INFL_M
    r_safe = np.where(r < 1e-12, 1e-12, r)

    R_ref = np.cbrt(r_safe ** 3 - A_INFL_M ** 3 + A_REF_M ** 3)
    lam_theta = np.where(~in_balloon, r_safe / R_ref, 1.0)

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

    return mu_tensor, in_balloon, mu_tt


def multi_face_vector_sources(radius_frac=DRIVER_R_FRAC, amp=DRIVER_AMP):
    """4-face broadband disk sources (bottom + 4 sides, top left free).

    Each face is driven on its own face-normal displacement component so
    the source behaves like a piston pushing perpendicular to that wall.
    """
    src = []
    cy = cz = (N - 1) / 2.0
    r_max = (N / 2.0) * radius_frac
    ampC = complex(amp)

    # Bottom face i = N-1 — driven on i-component (comp 0)
    for j in range(N):
        for k in range(N):
            if (j - cy) ** 2 + (k - cz) ** 2 <= r_max ** 2:
                src.append((N - 1, j, k, 0, ampC))

    # j = N-1 side — driven on j-component (comp 1)
    for i in range(N):
        for k in range(N):
            if (i - cy) ** 2 + (k - cz) ** 2 <= r_max ** 2:
                src.append((i, N - 1, k, 1, ampC))
    # j = 0 side
    for i in range(N):
        for k in range(N):
            if (i - cy) ** 2 + (k - cz) ** 2 <= r_max ** 2:
                src.append((i, 0, k, 1, ampC))

    # k = N-1 side — driven on k-component (comp 2)
    for i in range(N):
        for j in range(N):
            if (i - cy) ** 2 + (j - cz) ** 2 <= r_max ** 2:
                src.append((i, j, N - 1, 2, ampC))
    # k = 0 side
    for i in range(N):
        for j in range(N):
            if (i - cy) ** 2 + (j - cz) ** 2 <= r_max ** 2:
                src.append((i, j, 0, 2, ampC))

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


def curl_mip(u_vec, ring_msk, label):
    q = curl_of_displacement_3d(u_vec, DX)
    q_z = q[..., 0]
    del q
    directions = fibonacci_sphere(N_DIRECTIONS)
    G_stack = np.zeros((N_DIRECTIONS, N, N, N), dtype=np.float32)
    amp_stack = np.zeros((N_DIRECTIONS, N, N, N), dtype=np.float32)
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
    mu_tensor, balloon, mu_tt = build_tensor_field(W1_fn)
    print(f"  µ_θθ range {mu_tt.min()/1000:.2f} – {mu_tt.max()/1000:.2f} kPa")

    print(f"[{label}] vector Navier at {FREQ_HZ} Hz, top_free=True…", flush=True)
    t0 = time.time()
    u_vec = navier_solve_3d_tensor_mu(mu_tensor, lam=LAM_C, freq=FREQ_HZ,
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
    out_dir = ROOT / "results" / "paper_wave_sim_sobh_vector_multiface_topfree_N48"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Grid: {N}³ at dx = {DX*1000:.1f} mm ({N*DX*100:.1f} cm cube)")
    print(f"Frequency: {FREQ_HZ} Hz (Yin's)")
    print(f"BCs: 5 walls fixed, top ∂u/∂z=0 (mirror)")
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
        f"Grid: {N}³ at dx = {DX*1000:.1f} mm ({N*DX*100:.1f} cm cube)",
        f"Driver: multi-face broadband, face-normal component per face",
        f"BCs: 5 walls Dirichlet u=0, top ∂u/∂z=0 (mirror)",
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
