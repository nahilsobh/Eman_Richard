#!/usr/bin/env python3
"""80 Hz standalone MIP sim on the Sobh-Ehman phantoms (Yin's frequency).

Runs the same multi-face broadband + 20-direction filter + DI + MIP
pipeline as `paper_wave_sim_sobh_mip.py` but at 80 Hz (Yin's phantom
frequency).  Kept as a standalone so the process only holds one
phantom's state at a time — the combined 60+80 script OOM'd on the
login node when it accumulated both.

Output goes to results/paper_wave_sim_sobh_80hz/; use with the earlier
60 Hz results in results/paper_wave_sim_sobh_mip/ for a 60 vs 80
comparison.
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
    direct_inversion_3d,
    directional_filter_3d,
    helmholtz_solve_3d,
    multi_face_broadband_sources,
)


N = 60
DX_M = 0.003
FREQ_HZ = 80.0
RHO = 1000.0
DAMPING = 0.05

A_REF_M = 0.02285
V_INJECTION_ML = 250
A_INFL_M = ((3 * V_INJECTION_ML * 1e-6) / (4 * math.pi)) ** (1/3)

MU_MEAN = 2750.0
K1_MEAN = 3250.0
K2_MEAN = 1.25
G_WATER = 1.0

DRIVER_AMP = 1.0e-6
DRIVER_R_FRAC = 0.5
N_DIRECTIONS = 20
WEDGE_WIDTH = 0.35
MEDIAN_FILTER = 3


def _mu_P1(lam):
    return MU_MEAN * lam ** 2


def _mu_P2(lam):
    eps = lam - 1.0
    lam6_m1 = lam ** 6 - 1.0
    safe = np.where(np.abs(lam6_m1) < 1e-12, 1e-12, lam6_m1)
    fiber = K1_MEAN * lam ** 7 * eps * np.exp(K2_MEAN * eps ** 2) / (2.0 * safe)
    fiber = np.where(np.abs(lam - 1.0) < 1e-8, K1_MEAN / 12.0, fiber)
    return MU_MEAN * lam ** 2 + fiber


def build_stiffness():
    i_idx, j_idx, k_idx = np.indices((N, N, N))
    c = (N - 1) / 2.0
    r = np.sqrt(((i_idx - c) * DX_M) ** 2 + ((j_idx - c) * DX_M) ** 2 +
                  ((k_idx - c) * DX_M) ** 2)
    in_balloon = r < A_INFL_M
    r_safe = np.where(r < 1e-12, 1e-12, r)
    R_ref = np.cbrt(r_safe ** 3 - A_INFL_M ** 3 + A_REF_M ** 3)
    lam = np.where(~in_balloon, r_safe / R_ref, 1.0)
    G_P1 = np.where(~in_balloon, _mu_P1(lam), G_WATER).astype(float)
    G_P2 = np.where(~in_balloon, _mu_P2(lam), G_WATER).astype(float)
    return G_P1, G_P2, in_balloon


def fibonacci_sphere(n):
    i = np.arange(n)
    phi = (1 + 5 ** 0.5) / 2
    theta = 2 * np.pi * i / phi
    z = 1 - 2 * (i + 0.5) / n
    rxy = np.sqrt(np.maximum(0.0, 1 - z * z))
    return np.stack([rxy * np.cos(theta), rxy * np.sin(theta), z], axis=1)


def ring_stats(G, balloon, shell_inner_mm=0.0, shell_outer_mm=12.0):
    i_idx, j_idx, k_idx = np.indices(G.shape)
    c = (N - 1) / 2.0
    r = np.sqrt(((i_idx - c) * DX_M) ** 2 +
                  ((j_idx - c) * DX_M) ** 2 +
                  ((k_idx - c) * DX_M) ** 2)
    ring = (~balloon) & (r >= A_INFL_M + shell_inner_mm * 1e-3) \
                        & (r <= A_INFL_M + shell_outer_mm * 1e-3)
    v = G[ring]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(v.mean())


def run_mip(G, balloon, label, out_dir):
    print(f"\n[{label}] running MIP at 80 Hz…", flush=True)
    directions = fibonacci_sphere(N_DIRECTIONS)
    sources = multi_face_broadband_sources(N, radius_frac=DRIVER_R_FRAC,
                                                faces=("iN", "jN", "j0", "kN", "k0"))
    sources = [(i, j, k, DRIVER_AMP + 0.0j) for (i, j, k, _) in sources]

    t0 = time.time()
    u = helmholtz_solve_3d(G, freq=FREQ_HZ, rho=RHO, dx=DX_M, damping=DAMPING,
                              sources=sources, top_free=True)
    print(f"  Helmholtz: {time.time() - t0:.1f} s  "
          f"|u| range {np.abs(u).min():.2e} – {np.abs(u).max():.2e} m",
          flush=True)

    G_est_stack = np.zeros((N_DIRECTIONS, N, N, N), dtype=np.float32)
    amp_stack   = np.zeros((N_DIRECTIONS, N, N, N), dtype=np.float32)
    for d, khat in enumerate(directions):
        u_k = directional_filter_3d(u, khat=khat, angular_width=WEDGE_WIDTH)
        G_k = direct_inversion_3d(u_k, freq=FREQ_HZ, rho=RHO, dx=DX_M,
                                      median_filter_size=MEDIAN_FILTER)
        G_est_stack[d] = np.where(np.isfinite(G_k), G_k, 0.0).astype(np.float32)
        amp_stack[d]   = np.abs(u_k).astype(np.float32)
        del u_k, G_k
        if (d + 1) % 5 == 0:
            print(f"    direction {d+1}/{N_DIRECTIONS}", flush=True)

    pos = np.where(G_est_stack > 0, G_est_stack, 0.0)
    G_tsm = pos.max(axis=0).astype(float)
    wsum = amp_stack.sum(axis=0)
    wsafe = np.where(wsum > 1e-30, wsum, 1e-30)
    G_conv = ((pos * amp_stack).sum(axis=0) / wsafe).astype(float)

    G_tsm  = np.where(balloon, np.nan, G_tsm)
    G_conv = np.where(balloon, np.nan, G_conv)

    # Save immediately so an OOM later doesn't lose this phantom
    np.savez_compressed(out_dir / f"tsm_{label}.npz",
                          u=u, G_conv=G_conv, G_tsm=G_tsm)
    print(f"  saved tsm_{label}.npz", flush=True)

    # Free heavy locals aggressively
    del G_est_stack, amp_stack, pos, wsum, wsafe, u
    gc.collect()

    return G_conv, G_tsm


def main():
    out_dir = ROOT / "results" / "paper_wave_sim_sobh_80hz"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Grid: {N}³ hex at dx = {DX_M*1000:.1f} mm → 18 cm cube")
    print(f"Balloon 250 mL (a = {A_INFL_M*100:.3f} cm)")
    print(f"Frequency: {FREQ_HZ} Hz (Yin's phantom frequency)")
    cs = math.sqrt(MU_MEAN / RHO)
    print(f"  shear λ = c_s/f = {cs/FREQ_HZ*1000:.1f} mm "
          f"→ λ/dx = {cs/FREQ_HZ/DX_M:.1f}")
    print()

    G_P1, G_P2, balloon = build_stiffness()
    ring_gt_P1 = ring_stats(G_P1, balloon)
    ring_gt_P2 = ring_stats(G_P2, balloon)
    print(f"GT ring: P1 = {ring_gt_P1/1000:.3f} kPa, "
          f"P2 = {ring_gt_P2/1000:.3f} kPa")

    conv_P1, tsm_P1 = run_mip(G_P1, balloon, "P1", out_dir)
    conv_P2, tsm_P2 = run_mip(G_P2, balloon, "P2", out_dir)

    conv_P1_ring = ring_stats(conv_P1, balloon)
    tsm_P1_ring  = ring_stats(tsm_P1,  balloon)
    conv_P2_ring = ring_stats(conv_P2, balloon)
    tsm_P2_ring  = ring_stats(tsm_P2,  balloon)

    lines = [
        f"80 Hz MIP recovery on Sobh-Ehman phantoms at 250 mL "
        f"(Yin's phantom frequency)",
        "=" * 78,
        f"Grid: {N}³ at dx = {DX_M*1000:.1f} mm (18 cm cube)",
        f"Shear wavelength at 80 Hz in unstretched gel: "
        f"{cs/FREQ_HZ*1000:.1f} mm ({cs/FREQ_HZ/DX_M:.1f} voxels)",
        "",
        "Perilesional ring (r ∈ [a, a+12 mm]) mean stiffness at 80 Hz:",
        f"{'':16s}{'GT':>10s}{'µ_conv':>12s}{'µ_TSM':>12s}"
        f"{'TSM/conv':>12s}",
        f"  P1              {ring_gt_P1/1000:>10.3f}"
        f"{conv_P1_ring/1000:>12.3f}{tsm_P1_ring/1000:>12.3f}"
        f"{tsm_P1_ring/max(conv_P1_ring,1):>12.3f}",
        f"  P2              {ring_gt_P2/1000:>10.3f}"
        f"{conv_P2_ring/1000:>12.3f}{tsm_P2_ring/1000:>12.3f}"
        f"{tsm_P2_ring/max(conv_P2_ring,1):>12.3f}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
