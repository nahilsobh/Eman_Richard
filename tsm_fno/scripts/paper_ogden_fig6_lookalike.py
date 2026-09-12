#!/usr/bin/env python3
"""Yin Fig 6 lookalike using ONLY the Ogden constitutive law.

The physically-cleanest reproduction of Yin's Figure 6 available here:
takes the tuned Ogden N=2 material for each phantom, applies it to the
analytical Lamé stretch field for a balloon inflating in an incompressible
matrix, and reads off the perilesional ring stiffness directly from the
resulting tensor field G_ij(x). No wave solve, no direct inversion, no
numerical noise — just Ogden + Lamé + shell average.

Two ring quantities per phantom (both from the same tensor):
    μ_TSM  = G_θ = Σ_p μ_p · λ_θ^(α_p − 2)     (tangential principal)
    μ_conv = trace(G_ij) / 3                   (isotropic mean)

Because Yin's TSM is a MIP over 20 directions preferentially picking the
tangentially-propagating direction, μ_TSM here is the pure-tangential
Ogden principal — the theoretical ceiling of what MIP could recover.
μ_conv is the direction-averaged mean of the tensor — what conventional
inversion recovers when it averages over all directions.

Phantoms (best-fit Ogden parameters, all G₀ = 2500 Pa):
    P1 gelatin:   μ = (1100, 1400) Pa,  α = (7.0, 1.0)
    P2 cellulose: μ = (2000,  500) Pa,  α = (2.0, 10.0)
    P3 control:   fixed at 250 mL water, never inflated (no stretch)

Full 9-state cycle: 5 inflation (50→250 mL) + 4 deflation (200→50 mL).
Because Ogden is memoryless, deflation retraces inflation exactly.

Runtime ~1 s (analytical, no solve).
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
    ogden_lame_stretch_field,
    perilesional_shell_3d,
    SphericalBalloon,
)


# ── Config ──────────────────────────────────────────────────────────────
N        = 32
DX       = 0.003
G_LESION = 2000.0
CENTER   = (N // 2, N // 2, N // 2)
SHELL_MM        = 5.0
SHELL_OFFSET_MM = 3.0    # closer to the stretched region for cleaner G_θ

BALLOON_VOLUMES_ML = [50, 100, 150, 200, 250]
def _r_vx(vol):
    return ((3 * vol * 1e-6 / (4 * math.pi)) ** (1/3)) / DX
BALLOON_RADII_VX = [_r_vx(v) for v in BALLOON_VOLUMES_ML]
A0_VX  = BALLOON_RADII_VX[0]
A_CTRL = BALLOON_RADII_VX[-1]

# Full inflation + deflation schedule (9 states — memoryless so peak
# appears once, deflation retraces from 200 back to 50).
SCHEDULE = list(zip(BALLOON_RADII_VX, BALLOON_VOLUMES_ML, ["infl"] * 5))
for i in range(3, -1, -1):
    SCHEDULE.append((BALLOON_RADII_VX[i], BALLOON_VOLUMES_ML[i], "defl"))

PHANTOMS = [
    dict(name="P1 gelatin",       mu=[1100.0, 1400.0], alpha=[7.0, 1.0],  a0=A0_VX,
         color="black",   marker="o", control=False),
    dict(name="P2 cellulose",     mu=[2000.0,  500.0], alpha=[2.0, 10.0], a0=A0_VX,
         color="tab:cyan", marker="s", control=False),
    dict(name="P3 control (p=0)", mu=[1100.0, 1400.0], alpha=[7.0, 1.0],  a0=A_CTRL,
         color="tab:gray", marker="D", control=True),
]


def analytical_ring(a_vx, a0_vx, mu, alpha):
    """Return (G_theta, G_iso) ring means from the analytical Ogden tensor."""
    balloon = SphericalBalloon(center=CENTER, radius_vx=a_vx, pressure=0.0)
    G_tensor = ogden_G_tensor_field(N=N, dx=DX, center=CENTER,
                                     a0_vx=a0_vx, a_vx=a_vx,
                                     mu_list=mu, alpha_list=alpha,
                                     G_lesion=G_LESION,
                                     balloon_mask=balloon.mask(N))
    G_iso = np.einsum("xyzii->xyz", G_tensor) / 3.0

    # G_theta from principal decomposition of the stress-affected shear:
    _lam_r, lam_theta, _rhat = ogden_lame_stretch_field(N, DX, CENTER, a0_vx, a_vx)
    G_theta = np.zeros_like(lam_theta)
    for m, al in zip(mu, alpha):
        G_theta = G_theta + float(m) * np.power(lam_theta, al - 2.0)

    shell = perilesional_shell_3d(balloon.mask(N), shell_mm=SHELL_MM, dx=DX,
                                   inner_offset_mm=SHELL_OFFSET_MM)

    def _ring(f):
        v = f[shell]; v = v[np.isfinite(v)]
        if v.size == 0: return float("nan")
        lo, hi = np.percentile(v, [10, 90])
        tr = v[(v >= lo) & (v <= hi)]
        return float(tr.mean()) if tr.size else float("nan")

    return _ring(G_theta), _ring(G_iso)


def main():
    out_dir = ROOT / "results" / "paper_ogden_fig6_lookalike"
    out_dir.mkdir(parents=True, exist_ok=True)

    curves = {p["name"]: dict(TSM=[], conv=[]) for p in PHANTOMS}
    for a_vx, vol, branch in SCHEDULE:
        for p in PHANTOMS:
            if p["control"]:
                # Fixed at a_ctrl (250 mL undeformed) — no stretch, no
                # acoustoelastic contribution. G_θ = G_iso = G_bg.
                g_th, g_iso = analytical_ring(a_vx=p["a0"], a0_vx=p["a0"],
                                                mu=p["mu"], alpha=p["alpha"])
            else:
                g_th, g_iso = analytical_ring(a_vx=a_vx, a0_vx=p["a0"],
                                                mu=p["mu"], alpha=p["alpha"])
            curves[p["name"]]["TSM"].append(g_th / 1000)
            curves[p["name"]]["conv"].append(g_iso / 1000)

    # Yin Fig 6 measured (from paper).
    yin = dict(
        P1_TSM  = [3.5, 3.8, 3.9, 4.2, 4.4, 4.2, 3.7, 3.6, 3.6],
        P2_TSM  = [3.5, 3.9, 4.2, 4.9, 5.15, 4.6, 3.9, 3.6, 3.6],
        P1_conv = [2.7, 2.7, 2.8, 2.8, 2.8, 2.8, 2.8, 2.7, 2.8],
        P2_conv = [3.0, 3.0, 3.0, 3.1, 3.3, 3.0, 3.0, 3.2, 3.2],
        P3_TSM  = [3.35] * 9,
        P3_conv = [2.8]  * 9,
    )

    n = len(SCHEDULE)
    x = list(range(n))
    x_labels = [s[1] for s in SCHEDULE]

    fig, ax = plt.subplots(figsize=(11, 6.3))
    # Yin-style inflation/deflation background shading.
    ax.axvspan(-0.5, 4.5, alpha=0.06, color="tab:blue")
    ax.axvspan(4.5,  n - 0.5, alpha=0.06, color="tab:orange")
    ax.text(2, 5.9, "Balloon inflation", ha="center", fontsize=10,
            color="tab:blue",    fontweight="bold", alpha=0.7)
    ax.text(6.5, 5.9, "Balloon deflation", ha="center", fontsize=10,
            color="tab:orange",  fontweight="bold", alpha=0.7)

    # Yin measured curves — thin star markers.
    ax.plot(x, yin["P1_TSM"],  "*-", color="black",       ms=13, lw=1.0, alpha=0.55,
            label="Yin P1 TSM (measured)")
    ax.plot(x, yin["P2_TSM"],  "*-", color="tab:cyan",    ms=13, lw=1.0, alpha=0.55,
            label="Yin P2 TSM (measured)")
    ax.plot(x, yin["P1_conv"], "^-", color="tab:orange",  ms=7,  lw=0.8, alpha=0.45,
            label="Yin conv (measured, flat)")
    ax.plot(x, yin["P3_TSM"],  ":",  color="dimgray",     lw=1.0,
            label=f"Yin P3 TSM control = 3.35 kPa")

    # Our Ogden analytical curves — solid, thick.
    for p in PHANTOMS:
        c = p["color"]
        style_marker = p["marker"]
        ms = 8
        # μ_TSM (G_θ tangential) — solid line
        ax.plot(x, curves[p["name"]]["TSM"],
                 "-", color=c, marker=style_marker, ms=ms, lw=2.4,
                 label=f"Ours {p['name']}: G_θ (μ_TSM analog)")
        # μ_conv (trace/3 isotropic) — dashed line
        ax.plot(x, curves[p["name"]]["conv"],
                 "--", color=c, marker=style_marker, ms=ms-2, lw=1.6, alpha=0.65,
                 label=f"Ours {p['name']}: tr(G)/3 (μ_conv analog)")

    ax.set_xticks(x); ax.set_xticklabels([f"{v}" for v in x_labels], fontsize=10)
    ax.set_xlabel("Balloon water volume (mL)     [inflation ← | → deflation]",
                   fontsize=11)
    ax.set_ylabel("Perilesional ring stiffness [kPa]", fontsize=11)
    ax.set_title(
        "Ogden N=2 analytical reproduction of Yin Fig 6\n"
        "P1: μ=(1100,1400) α=(7,1)    P2: μ=(2000,500) α=(2,10)    G₀ = 2.5 kPa (both)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1.5, 6.2)
    ax.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.92)
    plt.tight_layout()
    out_fig = out_dir / "ogden_fig6_lookalike.png"
    fig.savefig(out_fig, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_fig}")

    # ── Summary ─────────────────────────────────────────────────────────
    def _fmt(vec): return "  ".join(f"{v:5.2f}" for v in vec)
    lines = [
        "Ogden N=2 analytical reproduction of Yin Fig 6",
        "=" * 78,
        "Method: analytical Lamé stretch → Ogden G_ij(x) → shell mean.",
        "        No wave solve, no inversion — pure constitutive law.",
        f"Grid: N={N}³, dx={DX*1000:.0f} mm    shell offset={SHELL_OFFSET_MM} mm.",
        "",
        f"State (mL): {_fmt(x_labels)}",
        f"Branch:      {' '.join(f'{s[2]:>4s}' for s in SCHEDULE):>{5*n-1}}",
        "",
        f"Yin P1 TSM:  {_fmt(yin['P1_TSM'])}",
        f"Ours P1 G_θ: {_fmt(curves['P1 gelatin']['TSM'])}",
        f"Ours P1 tr/3:{_fmt(curves['P1 gelatin']['conv'])}",
        f"Yin P1 conv: {_fmt(yin['P1_conv'])}",
        "",
        f"Yin P2 TSM:  {_fmt(yin['P2_TSM'])}",
        f"Ours P2 G_θ: {_fmt(curves['P2 cellulose']['TSM'])}",
        f"Ours P2 tr/3:{_fmt(curves['P2 cellulose']['conv'])}",
        f"Yin P2 conv: {_fmt(yin['P2_conv'])}",
        "",
        f"Ours P3 G_θ (control): {curves['P3 control (p=0)']['TSM'][0]:.3f} kPa "
        f"(Yin ~ 3.35)",
        f"Ours P3 tr/3(control): {curves['P3 control (p=0)']['conv'][0]:.3f} kPa "
        f"(Yin ~ 2.8)",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
