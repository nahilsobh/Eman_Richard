#!/usr/bin/env python3
"""Final Yin Fig 6 reproduction with best-fit Ogden N=2 parameters.

Runs the full inflation + deflation cycle for all three phantoms using
the parameter set that came out of the 4-config sweep:

  Phantom 1 (gelatin):   mu = (1100, 1400) Pa,  alpha = (5.0, 1.0)
  Phantom 2 (cellulose): mu = (2000,  500) Pa,  alpha = (2.0, 10.0)
  Phantom 3 (control):   fixed at 250 mL, never inflated (no stretch)

Both P1 and P2 preserve G0 = 2500 Pa exactly (Abaqus small-strain sum).

Pipeline per state:
  analytical Lamé stretch field → Ogden anisotropic G_ij(x)
     → 3D anisotropic Helmholtz + broadband multi-face source
     → 20-direction Fibonacci k-space filter
     → per-direction DI (3x3x3 median, 9 mm shell offset)
     → μ_conv (amp-weighted mean) + μ_TSM (MIP)

Produces:
  results/paper_ogden_yin_final/summary.txt
  results/paper_ogden_yin_final/yin_fig6_final.png
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


# ── Config ─────────────────────────────────────────────────────────────
N        = 32
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

# Balloon inflation states (same as Yin: 50, 100, 150, 200, 250 mL).
BALLOON_VOLUMES_ML = [50, 100, 150, 200, 250]
def _r_vx(vol):
    return ((3 * vol * 1e-6 / (4 * math.pi)) ** (1/3)) / DX
BALLOON_RADII_VX = [_r_vx(v) for v in BALLOON_VOLUMES_ML]
A0_VX = BALLOON_RADII_VX[0]        # 50 mL reference (undeformed matrix)
A_CTRL_VX = BALLOON_RADII_VX[-1]   # 250 mL for Phantom 3 control (never inflated)

# Best-fit Ogden N=2 parameters (from the 4-config sweep).
PHANTOMS = [
    dict(name="Phantom 1 (gelatin)",
         mu=[1100.0, 1400.0], alpha=[5.0, 1.0],
         a0=A0_VX, control=False,
         style=dict(color="black",      marker="o", ms=8)),
    dict(name="Phantom 2 (cellulose)",
         mu=[2000.0,  500.0], alpha=[2.0, 10.0],
         a0=A0_VX, control=False,
         style=dict(color="tab:cyan",   marker="s", ms=8)),
    dict(name="Phantom 3 (control)",
         mu=[1100.0, 1400.0], alpha=[5.0, 1.0],   # same as P1 chemistry
         a0=A_CTRL_VX, control=True,
         style=dict(color="tab:gray",   marker="D", ms=7)),
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


def solve_state(a_vx, a0_vx, mu_list, alpha_list):
    balloon = SphericalBalloon(center=CENTER, radius_vx=a_vx, pressure=0.0)
    G_tensor = ogden_G_tensor_field(N=N, dx=DX, center=CENTER,
                                     a0_vx=a0_vx, a_vx=a_vx,
                                     mu_list=mu_list, alpha_list=alpha_list,
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

    peaks  = amp_stack.reshape(NDIRS, -1).max(axis=1)
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


def run_phantom(phantom):
    """Run one full inflate/deflate cycle for one phantom."""
    # Schedule: inflation 50→250, then deflation 200→50 (skipping peak).
    schedule = [(BALLOON_RADII_VX[i], BALLOON_VOLUMES_ML[i], "inflation")
                for i in range(len(BALLOON_VOLUMES_ML))]
    for i in range(len(BALLOON_VOLUMES_ML) - 2, -1, -1):
        schedule.append((BALLOON_RADII_VX[i], BALLOON_VOLUMES_ML[i], "deflation"))

    if phantom["control"]:
        # P3: always at fixed size (never inflated). All states use same
        # geometry → same solve → same G_ring. Only run once for speed.
        a = phantom["a0"]
        c, t = solve_state(a, phantom["a0"], phantom["mu"], phantom["alpha"])
        return [dict(vol=vol, a_vx=a, branch=br, ring_conv=c, ring_tsm=t)
                for (_, vol, br) in schedule]

    results = []
    for step, (a_vx, vol, branch) in enumerate(schedule):
        c, t = solve_state(a_vx, phantom["a0"], phantom["mu"], phantom["alpha"])
        results.append(dict(vol=vol, a_vx=a_vx, branch=branch, step=step,
                             ring_conv=c, ring_tsm=t))
        print(f"    {vol:3d} mL  r={a_vx:5.2f} vx  [{branch:9s}]  "
              f"conv={c/1000:.2f} kPa   TSM={t/1000:.2f} kPa")
    return results


def make_figure(all_results, out_path):
    """Yin Fig 6 lookalike — full cycle, both TSM and conv, all 3 phantoms."""
    n_infl = len(BALLOON_VOLUMES_ML)
    x_labels = BALLOON_VOLUMES_ML + BALLOON_VOLUMES_ML[-2::-1]

    fig, ax = plt.subplots(figsize=(11, 6.2))
    # Background shading — inflation blue, deflation orange (Yin style).
    ax.axvspan(-0.5, n_infl - 0.5, alpha=0.05, color="tab:blue")
    ax.axvspan(n_infl - 0.5, 2 * n_infl - 1.5, alpha=0.05, color="tab:orange")
    ax.text(1.5, 5.65, "Balloon inflation", ha="center", fontsize=10,
             color="tab:blue", fontweight="bold", alpha=0.7)
    ax.text(6.5, 5.65, "Balloon deflation", ha="center", fontsize=10,
             color="tab:orange", fontweight="bold", alpha=0.7)

    # Yin's actual measured curves (from Fig 6 of PMC13010385).
    yin_p1_tsm  = [3.5, 3.8, 3.9, 4.2, 4.4, 4.2, 3.7, 3.6, 3.6]
    yin_p2_tsm  = [3.5, 3.9, 4.2, 4.9, 5.15, 4.6, 3.9, 3.6, 3.6]
    yin_p1_conv = [2.7, 2.7, 2.8, 2.8, 2.8, 2.8, 2.8, 2.7, 2.8]
    yin_p2_conv = [3.0, 3.0, 3.0, 3.1, 3.3, 3.0, 3.0, 3.2, 3.2]
    yin_p3_tsm  = [3.35] * len(x_labels)
    yin_p3_conv = [2.8]  * len(x_labels)
    x_full = list(range(len(x_labels)))

    # Yin measured curves — plotted as thin lines with star markers.
    ax.plot(x_full, yin_p1_tsm,  "*-", color="black",    ms=10, lw=1.2, alpha=0.7,
            label="Yin P1 μ_TSM (measured)")
    ax.plot(x_full, yin_p2_tsm,  "*-", color="tab:cyan", ms=10, lw=1.2, alpha=0.7,
            label="Yin P2 μ_TSM (measured)")
    ax.plot(x_full, yin_p1_conv, "^-", color="tab:orange", ms=6, lw=1.0, alpha=0.5,
            label="Yin conv (measured, ~flat)")
    ax.plot(x_full, yin_p3_tsm,   ":", color="gray", lw=1.0,
            label="Yin P3 TSM (control ~3.35)")

    # Our simulation curves — solid, larger markers.
    for phantom, results in zip(PHANTOMS, all_results):
        if phantom["control"]:
            ring_tsm = results[0]["ring_tsm"] / 1000
            ring_conv = results[0]["ring_conv"] / 1000
            ax.axhline(ring_tsm, ls="--", color=phantom["style"]["color"], lw=1.2,
                        label=f"Ours P3 TSM = {ring_tsm:.2f} (control)")
            continue

        tsm = [r["ring_tsm"] / 1000 for r in results]
        cv  = [r["ring_conv"] / 1000 for r in results]
        st = phantom["style"]
        ax.plot(x_full, tsm, "-",  color=st["color"], marker=st["marker"],
                 ms=st["ms"], lw=2.2, label=f"Ours {phantom['name']}: μ_TSM")
        ax.plot(x_full, cv,  "--", color=st["color"], marker=st["marker"],
                 ms=st["ms"] - 2, lw=1.4, alpha=0.55,
                 label=f"Ours {phantom['name']}: μ_conv")

    ax.set_xticks(x_full)
    ax.set_xticklabels([f"{v}" for v in x_labels], fontsize=10)
    ax.set_xlabel("Balloon water volume (mL)     [inflation ← | → deflation]",
                   fontsize=11)
    ax.set_ylabel("Perilesional G_ring [kPa]  (DI, 3×3×3 median, 9 mm shell offset)",
                   fontsize=11)
    ax.set_title(
        "Yin Fig 6 reproduction — Ogden N=2 acoustoelastic simulation\n"
        "P1: μ=(1100,1400) α=(5,1)    P2: μ=(2000,500) α=(2,10)    G₀=2.5 kPa both",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1.8, 6.0)
    ax.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_path}")


def main():
    out_dir = ROOT / "results" / "paper_ogden_yin_final"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for phantom in PHANTOMS:
        print(f"\n=== {phantom['name']} ===")
        print(f"  mu = {phantom['mu']}    alpha = {phantom['alpha']}"
              f"    G0 = {sum(phantom['mu'])} Pa")
        print(f"  a0 = {phantom['a0']:.2f} vx    control = {phantom['control']}")
        r = run_phantom(phantom)
        all_results.append(r)

    make_figure(all_results, out_dir / "yin_fig6_final.png")

    # Text summary.
    x_labels = BALLOON_VOLUMES_ML + BALLOON_VOLUMES_ML[-2::-1]
    lines = [
        "Final Yin Fig 6 reproduction — Ogden N=2 acoustoelastic simulation",
        "=" * 78,
        f"Grid: {N}³, dx={DX*1000:.0f} mm    freq={FREQ:.0f} Hz    "
        f"shell offset={SHELL_OFFSET_MM} mm    median={MEDIAN_SIZE}",
        f"Directions: {NDIRS} Fibonacci, wedge σ = {WEDGE_WIDTH} rad",
        "",
        "Yin reference (measured):",
        "  P1 μ_TSM :  3.5  3.8  3.9  4.2  4.4  4.2  3.7  3.6  3.6",
        "  P2 μ_TSM :  3.5  3.9  4.2  4.9  5.15 4.6  3.9  3.6  3.6",
        "  P1 μ_conv:  2.7  2.7  2.8  2.8  2.8  2.8  2.8  2.7  2.8  (flat)",
        "  P2 μ_conv:  3.0  3.0  3.0  3.1  3.3  3.0  3.0  3.2  3.2",
        "  P3 (control): TSM ~3.35 flat, conv ~2.8 flat",
        "",
        f"State label:  {'  '.join(f'{v:4d}' for v in x_labels)}",
        f"                  {' '*4}  <-- inflation -->  {' '*4}  <-- deflation -->",
        "",
    ]
    for phantom, results in zip(PHANTOMS, all_results):
        lines.append(f"{phantom['name']}:")
        if phantom["control"]:
            t = results[0]["ring_tsm"] / 1000
            c = results[0]["ring_conv"] / 1000
            lines.append(f"  (fixed at {BALLOON_VOLUMES_ML[-1]} mL — no stretch)")
            lines.append(f"  μ_TSM :  {t:.3f} kPa  (Yin ~3.35)")
            lines.append(f"  μ_conv:  {c:.3f} kPa  (Yin ~2.8)")
        else:
            tsm = "  ".join(f"{r['ring_tsm']/1000:4.2f}" for r in results)
            cv  = "  ".join(f"{r['ring_conv']/1000:4.2f}" for r in results)
            lines.append(f"  μ_TSM :  {tsm}")
            lines.append(f"  μ_conv:  {cv}")
        lines.append("")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
