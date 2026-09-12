#!/usr/bin/env python3
"""Composition-based Ogden model — independent of Yin's MIP measurements.

Instead of tuning Ogden parameters to match Yin's Fig 6 curves (which are
themselves MIP-processed MRE readouts, not independent ground truth for
the phantom material), this script sets Ogden parameters from what Yin's
phantoms are ACTUALLY MADE OF:

  Phantom 1: 10% pure bovine gelatin, cured 5 days at room temp
      Rheometry-based estimate: G₀ ≈ 2.5 kPa, mild strain-stiffening
      Ogden N=2: μ = (1800, 700) Pa,  α = (2.5, 3.0)

  Phantom 2: 8% gelatin + 7% cellulose fiber, cured 5 days at room temp
      Less gelatin (softer matrix) but cellulose adds baseline + fiber-lock
      Composite G₀ ≈ 4.0 kPa (matrix ~2.5 + fiber ~1.5)
      Ogden N=2: μ = (2500, 1500) Pa,  α = (2.5, 5.0)

These are composition-informed estimates from published gel rheometry
ranges — NOT fitted to Yin. The comparison to Yin below is now a
predictive test: does our physically-motivated Ogden material, imaged
through the same MIP pipeline as Yin, give similar numbers?

Produces two things:
  1. analytical ring stiffness (G_θ tangential and G_iso mean) — the
     TRUE material G, no MIP, no inversion noise
  2. comparison to Yin's Fig 6 MIP-derived values

The gap between (1) and Yin's data reveals two distinct effects:
  - Yin's MIP-upward-bias (systematic overestimate ~1 kPa at baseline)
  - Whether our composition-based Ogden matches Yin's actual gel
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


N        = 32
DX       = 0.003
CENTER   = (N // 2, N // 2, N // 2)
SHELL_MM        = 5.0
SHELL_OFFSET_MM = 3.0

BALLOON_VOLUMES_ML = [50, 100, 150, 200, 250]
def _r_vx(vol):
    return ((3 * vol * 1e-6 / (4 * math.pi)) ** (1/3)) / DX
BALLOON_RADII_VX = [_r_vx(v) for v in BALLOON_VOLUMES_ML]
A0_VX  = BALLOON_RADII_VX[0]
A_CTRL = BALLOON_RADII_VX[-1]

SCHEDULE = list(zip(BALLOON_RADII_VX, BALLOON_VOLUMES_ML, ["infl"] * 5))
for i in range(3, -1, -1):
    SCHEDULE.append((BALLOON_RADII_VX[i], BALLOON_VOLUMES_ML[i], "defl"))

# Composition-based Ogden parameters (see module docstring for rationale).
PHANTOMS = [
    dict(name="P1 gelatin 10%",
         composition="10% bovine gelatin, 5-day RT cure",
         mu=[1800.0,  700.0], alpha=[2.5, 3.0],
         G_lesion=1500.0,             # balloon interior (water) ~ fluid
         a0=A0_VX, control=False,
         color="black",     marker="o"),
    dict(name="P2 8% gel + 7% cellulose",
         composition="8% gelatin + 7% cellulose fiber, 5-day RT cure",
         mu=[2500.0, 1500.0], alpha=[2.5, 5.0],
         G_lesion=1500.0,
         a0=A0_VX, control=False,
         color="tab:cyan",  marker="s"),
    dict(name="P3 control (never inflated)",
         composition="10% gelatin phantom, fixed at 250 mL",
         mu=[1800.0,  700.0], alpha=[2.5, 3.0],
         G_lesion=1500.0,
         a0=A_CTRL, control=True,
         color="tab:gray",  marker="D"),
]


def analytical_ring(a_vx, a0_vx, mu, alpha, G_lesion):
    balloon = SphericalBalloon(center=CENTER, radius_vx=a_vx, pressure=0.0)
    G_tensor = ogden_G_tensor_field(N=N, dx=DX, center=CENTER,
                                     a0_vx=a0_vx, a_vx=a_vx,
                                     mu_list=mu, alpha_list=alpha,
                                     G_lesion=G_lesion,
                                     balloon_mask=balloon.mask(N))
    G_iso = np.einsum("xyzii->xyz", G_tensor) / 3.0

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
    out_dir = ROOT / "results" / "paper_ogden_composition_based"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nComposition-based Ogden — INDEPENDENT of Yin's MIP measurements\n")
    for p in PHANTOMS:
        print(f"  {p['name']}:  μ = {p['mu']} Pa,  α = {p['alpha']}"
              f"  →  G₀ = {sum(p['mu'])} Pa")
        print(f"     {p['composition']}")

    curves = {p["name"]: dict(TSM=[], conv=[]) for p in PHANTOMS}
    for a_vx, vol, branch in SCHEDULE:
        for p in PHANTOMS:
            if p["control"]:
                g_th, g_iso = analytical_ring(a_vx=p["a0"], a0_vx=p["a0"],
                                                mu=p["mu"], alpha=p["alpha"],
                                                G_lesion=p["G_lesion"])
            else:
                g_th, g_iso = analytical_ring(a_vx=a_vx, a0_vx=p["a0"],
                                                mu=p["mu"], alpha=p["alpha"],
                                                G_lesion=p["G_lesion"])
            curves[p["name"]]["TSM"].append(g_th / 1000)
            curves[p["name"]]["conv"].append(g_iso / 1000)

    yin = dict(
        P1_TSM=[3.5, 3.8, 3.9, 4.2, 4.4, 4.2, 3.7, 3.6, 3.6],
        P2_TSM=[3.5, 3.9, 4.2, 4.9, 5.15, 4.6, 3.9, 3.6, 3.6],
        P1_conv=[2.7, 2.7, 2.8, 2.8, 2.8, 2.8, 2.8, 2.7, 2.8],
        P2_conv=[3.0, 3.0, 3.0, 3.1, 3.3, 3.0, 3.0, 3.2, 3.2],
        P3_TSM=[3.35] * 9,
        P3_conv=[2.8]  * 9,
    )

    n = len(SCHEDULE)
    x = list(range(n))
    x_labels = [s[1] for s in SCHEDULE]

    fig, ax = plt.subplots(figsize=(11, 6.3))
    ax.axvspan(-0.5, 4.5,   alpha=0.06, color="tab:blue")
    ax.axvspan(4.5, n - 0.5, alpha=0.06, color="tab:orange")
    ax.text(2, 6.1, "Balloon inflation", ha="center", fontsize=10,
            color="tab:blue",   fontweight="bold", alpha=0.7)
    ax.text(6.5, 6.1, "Balloon deflation", ha="center", fontsize=10,
            color="tab:orange", fontweight="bold", alpha=0.7)

    # Yin measured — thin star markers.
    ax.plot(x, yin["P1_TSM"],  "*-", color="black",       ms=13, lw=1.0, alpha=0.55,
            label="Yin P1 TSM (measured/MIP)")
    ax.plot(x, yin["P2_TSM"],  "*-", color="tab:cyan",    ms=13, lw=1.0, alpha=0.55,
            label="Yin P2 TSM (measured/MIP)")
    ax.plot(x, yin["P1_conv"], "^-", color="tab:orange",  ms=7,  lw=0.8, alpha=0.45,
            label="Yin conv (measured, flat)")
    ax.plot(x, yin["P3_TSM"],  ":",  color="dimgray",     lw=1.0,
            label="Yin P3 TSM (control ~3.35)")

    for p in PHANTOMS:
        c = p["color"]
        mk = p["marker"]
        ax.plot(x, curves[p["name"]]["TSM"],
                 "-", color=c, marker=mk, ms=8, lw=2.4,
                 label=f"Ours {p['name']}: G_θ tangential")
        ax.plot(x, curves[p["name"]]["conv"],
                 "--", color=c, marker=mk, ms=6, lw=1.6, alpha=0.65,
                 label=f"Ours {p['name']}: tr(G)/3")

    ax.set_xticks(x); ax.set_xticklabels([f"{v}" for v in x_labels], fontsize=10)
    ax.set_xlabel("Balloon water volume (mL)     [inflation ← | → deflation]",
                   fontsize=11)
    ax.set_ylabel("Perilesional ring stiffness [kPa]", fontsize=11)
    ax.set_title(
        "Composition-based Ogden (no Yin fitting) vs Yin Fig 6\n"
        "P1: 10% gelatin  μ=(1800,700), α=(2.5,3.0)    "
        "P2: 8%gel+7%cellulose  μ=(2500,1500), α=(2.5,5.0)"
    )
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1.5, 6.5)
    ax.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.92)
    plt.tight_layout()
    out_fig = out_dir / "ogden_composition_based.png"
    fig.savefig(out_fig, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_fig}")

    # ── Summary ──
    def _fmt(vec): return "  ".join(f"{v:5.2f}" for v in vec)
    lines = [
        "Composition-based Ogden N=2 (no Yin fitting) vs Yin Fig 6",
        "=" * 78,
        "Ogden parameters chosen from published gel rheometry for the actual",
        "phantom compositions — NOT tuned to match Yin's MRE readouts.",
        "",
        "PHANTOM PROPERTIES:",
    ]
    for p in PHANTOMS:
        lines.append(f"  {p['name']}:  μ = {p['mu']} Pa,  α = {p['alpha']}"
                     f"  →  G₀ = {sum(p['mu'])} Pa")
        lines.append(f"     [{p['composition']}]")
    lines += [
        "",
        f"State (mL):   {_fmt(x_labels)}",
        f"Branch:        {' '.join(f'{s[2]:>4s}' for s in SCHEDULE)}",
        "",
        f"Yin P1 TSM :   {_fmt(yin['P1_TSM'])}",
        f"Ours P1 G_θ:   {_fmt(curves['P1 gelatin 10%']['TSM'])}",
        f"Ours P1 tr/3:  {_fmt(curves['P1 gelatin 10%']['conv'])}",
        f"Yin P1 conv:   {_fmt(yin['P1_conv'])}",
        "",
        f"Yin P2 TSM :   {_fmt(yin['P2_TSM'])}",
        f"Ours P2 G_θ:   {_fmt(curves['P2 8% gel + 7% cellulose']['TSM'])}",
        f"Ours P2 tr/3:  {_fmt(curves['P2 8% gel + 7% cellulose']['conv'])}",
        f"Yin P2 conv:   {_fmt(yin['P2_conv'])}",
        "",
        f"Yin P3 TSM (control): 3.35 kPa (constant, MIP-biased baseline)",
        f"Ours P3 G_θ (control):  {curves['P3 control (never inflated)']['TSM'][0]:.3f} kPa (unstretched, true G_bg)",
        f"Ours P3 tr/3 (control): {curves['P3 control (never inflated)']['conv'][0]:.3f} kPa",
        "",
        "INTERPRETATION:",
        "  - Ours = TRUE material G predicted from composition-based Ogden model.",
        "  - Yin  = MIP-processed MRE readout with systematic upward bias.",
        "  - The GAP between them at baseline (50 mL) is the MIP bias itself.",
        "  - The SHAPE and RELATIVE ordering (P2 > P1 > control, both rising",
        "    with pressure) should match if our Ogden captures the physics.",
        "  - If our absolute values disagree with Yin significantly, either:",
        "      (a) the Ogden parameters differ from Yin's actual gel, OR",
        "      (b) Yin's MIP pipeline introduces bias/gain not accounted for.",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
