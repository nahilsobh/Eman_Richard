#!/usr/bin/env python3
"""Iterate Phantom 1 with shell offset = 3 mm and alpha_1 = 7.

Runs the full P1 inflation + deflation cycle with:
    mu = (1100, 1400) Pa      alpha = (7.0, 1.0)      G0 = 2500 Pa
    shell inner offset = 3 mm (vs 9 mm in the previous final run)

Goal: close the residual −16 % peak gap for Phantom 1 (previous run
gave 3.70 kPa vs Yin's 4.4).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    helmholtz_solve_3d_anisotropic,
    multi_face_broadband_sources,
)


N        = 32
DX       = 0.003
FREQ     = 60.0
RHO      = 1000.0
DAMPING  = 0.05
G_LESION = 2000.0
DRIVER_R = 0.5
CENTER   = (N // 2, N // 2, N // 2)
SHELL_MM        = 5.0
SHELL_OFFSET_MM = 3.0    # <-- iterated (was 9.0)
MEDIAN_SIZE     = 3
NDIRS           = 20
WEDGE_WIDTH     = 0.35
AMP_THRESHOLD   = 0.15

BALLOON_VOLUMES_ML = [50, 100, 150, 200, 250]
def _r_vx(vol):
    return ((3 * vol * 1e-6 / (4 * math.pi)) ** (1/3)) / DX
BALLOON_RADII_VX = [_r_vx(v) for v in BALLOON_VOLUMES_ML]
A0_VX = BALLOON_RADII_VX[0]

MU    = [1100.0, 1400.0]
ALPHA = [7.0,    1.0]        # <-- iterated (was 5.0)

YIN_P1_TSM  = [3.5, 3.8, 3.9, 4.2, 4.4, 4.2, 3.7, 3.6, 3.6]
YIN_P1_CONV = [2.7, 2.7, 2.8, 2.8, 2.8, 2.8, 2.8, 2.7, 2.8]


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


def solve_state(a_vx):
    balloon = SphericalBalloon(center=CENTER, radius_vx=a_vx, pressure=0.0)
    G_tensor = ogden_G_tensor_field(N=N, dx=DX, center=CENTER,
                                     a0_vx=A0_VX, a_vx=a_vx,
                                     mu_list=MU, alpha_list=ALPHA,
                                     G_lesion=G_LESION,
                                     balloon_mask=balloon.mask(N))
    src = multi_face_broadband_sources(N, radius_frac=DRIVER_R,
                                        faces=("iN", "jN", "j0", "kN", "k0"))
    u_full = helmholtz_solve_3d_anisotropic(G_tensor, freq=FREQ, rho=RHO,
                                             dx=DX, damping=DAMPING, sources=src)

    directions = _fibonacci_dirs(NDIRS)
    di_maps, amp_maps = [], []
    for khat in directions:
        u_k = directional_filter_3d(u_full, khat=khat, angular_width=WEDGE_WIDTH)
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

    return _ring(mu_conv), _ring(mu_tsm)


def main():
    out_dir = ROOT / "results" / "paper_ogden_p1_iter_a7_shell3"
    out_dir.mkdir(parents=True, exist_ok=True)

    schedule = [(BALLOON_RADII_VX[i], BALLOON_VOLUMES_ML[i], "inflation")
                for i in range(len(BALLOON_VOLUMES_ML))]
    for i in range(len(BALLOON_VOLUMES_ML) - 2, -1, -1):
        schedule.append((BALLOON_RADII_VX[i], BALLOON_VOLUMES_ML[i], "deflation"))

    print(f"Phantom 1 iteration:  mu={MU}  alpha={ALPHA}  shell offset={SHELL_OFFSET_MM} mm")
    print(f"G0 = {sum(MU)} Pa (Abaqus small-strain identity)\n")

    results = []
    for step, (a_vx, vol, branch) in enumerate(schedule):
        c, t = solve_state(a_vx)
        results.append(dict(vol=vol, a_vx=a_vx, branch=branch,
                             ring_conv=c, ring_tsm=t))
        print(f"  {vol:3d} mL  r={a_vx:5.2f} vx  [{branch:9s}]  "
              f"conv={c/1000:.2f} kPa   TSM={t/1000:.2f} kPa")

    # ── Compare directly to Yin P1 curve ──
    x = list(range(len(results)))
    ours_tsm  = [r["ring_tsm"]  / 1000 for r in results]
    ours_conv = [r["ring_conv"] / 1000 for r in results]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x, YIN_P1_TSM,  "*-", color="black",     ms=12, lw=1.5,
            label="Yin P1 μ_TSM (measured)", alpha=0.8)
    ax.plot(x, YIN_P1_CONV, "*-", color="tab:orange", ms=12, lw=1.5,
            label="Yin P1 μ_conv (measured, flat)", alpha=0.6)
    ax.plot(x, ours_tsm,  "o-", color="tab:red",   ms=8, lw=2.2,
            label=f"Ours P1 μ_TSM  (α₁=7, shell 3 mm)")
    ax.plot(x, ours_conv, "^--", color="tab:red",  ms=7, lw=1.4, alpha=0.6,
            label="Ours P1 μ_conv (α₁=7, shell 3 mm)")

    n_infl = len(BALLOON_VOLUMES_ML)
    ax.axvspan(-0.5, n_infl - 0.5, alpha=0.05, color="tab:blue")
    ax.axvspan(n_infl - 0.5, 2 * n_infl - 1.5, alpha=0.05, color="tab:orange")
    labels = BALLOON_VOLUMES_ML + BALLOON_VOLUMES_ML[-2::-1]
    ax.set_xticks(x); ax.set_xticklabels([f"{v}" for v in labels], fontsize=10)
    ax.set_xlabel("Balloon water volume (mL)     [inflation ← | → deflation]")
    ax.set_ylabel("Perilesional G_ring [kPa]")
    peak_ours = max(ours_tsm)
    ax.set_title(
        f"Phantom 1 iteration — Ogden N=2 with α₁=7, shell offset 3 mm\n"
        f"Peak μ_TSM = {peak_ours:.2f} kPa  (Yin: 4.4,  Δ = {(peak_ours-4.4)/4.4*100:+.0f}%)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    out_fig = out_dir / "p1_iter_a7_shell3.png"
    fig.savefig(out_fig, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_fig}")

    lines = [
        f"P1 iteration — Ogden N=2, mu = {MU}, alpha = {ALPHA}, shell = {SHELL_OFFSET_MM} mm",
        "=" * 78,
        "",
        f"Yin P1 μ_TSM :  {'  '.join(f'{v:5.2f}' for v in YIN_P1_TSM)}",
        f"Ours μ_TSM   :  {'  '.join(f'{v:5.2f}' for v in ours_tsm)}",
        f"Yin P1 μ_conv:  {'  '.join(f'{v:5.2f}' for v in YIN_P1_CONV)}",
        f"Ours μ_conv  :  {'  '.join(f'{v:5.2f}' for v in ours_conv)}",
        "",
        f"Peak μ_TSM at 250 mL:  Ours = {peak_ours:.2f} kPa   Yin = 4.40 kPa   "
        f"Δ = {(peak_ours - 4.4)/4.4*100:+.1f}%",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
