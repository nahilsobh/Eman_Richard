#!/usr/bin/env python3
"""Ogden N=2 acoustoelastic-Lamé simulation of Yin et al. 2025 phantoms.

Equivalent to running the user-supplied Abaqus Ogden models through a
balloon inflation/deflation cycle, extracting the deformation field, and
computing what MRE would measure. Skips the actual Abaqus solve by using
the analytical Lamé solution for an incompressible spherical inclusion
in an infinite matrix — the physics is exact for the spherical geometry.

Constitutive laws (Abaqus convention, initial G0 = 2500 Pa each):
  Phantom 1 — gelatin:
    (μ_1, α_1) = (1100 Pa, 3.0),   (μ_2, α_2) = (1400 Pa, 1.0)
  Phantom 2 — cellulose fiber:
    (μ_1, α_1) = (2400 Pa, 2.0),   (μ_2, α_2) = ( 100 Pa, 10.0)

Balloon inflation: 50 mL (r = 7.6 vx) → 250 mL (r = 13.0 vx), peak
tangential stretch λ_θ = (250/50)^(1/3) = 1.71 at the balloon surface.

Pipeline per state:
  1. Compute Lamé stretch field (λ_r, λ_θ) at every voxel outside the
     balloon (analytical, no FE solve).
  2. Build anisotropic tangent-shear tensor G_ij(x) via Ogden principal
     moduli G_r = Σμ_p·λ_r^(α_p-2), G_θ = Σμ_p·λ_θ^(α_p-2).
  3. Solve anisotropic scalar Helmholtz with broadband multi-face source.
  4. Directional filter → per-direction G_DI via median-filtered DI at
     9 mm shell offset (Yin's protocol).
  5. μ_conv (amp-weighted mean) and μ_TSM (MIP over directions).

Produces:
  results/paper_ogden_fea/summary.txt
  results/paper_ogden_fea/ogden_fig6_reproduction.png
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


# ── Config ──────────────────────────────────────────────────────────────
N        = 32
DX       = 0.003         # 3 mm/voxel
FREQ     = 60.0
RHO      = 1000.0
DAMPING  = 0.05
G_LESION = 2000.0
DRIVER_R = 0.5
CENTER   = (N // 2, N // 2, N // 2)
SHELL_MM        = 5.0
SHELL_OFFSET_MM = 9.0    # Yin: 3 px away from balloon edge
MEDIAN_SIZE     = 3      # Yin: 3x3x3 median filter
NDIRS           = 20
WEDGE_WIDTH     = 0.35
AMP_THRESHOLD   = 0.15

BALLOON_VOLUMES_ML = [50, 100, 150, 200, 250]
def _r_vx(vol):
    return ((3 * vol * 1e-6 / (4 * math.pi)) ** (1/3)) / DX
BALLOON_RADII_VX = [_r_vx(v) for v in BALLOON_VOLUMES_ML]
A0_VX = BALLOON_RADII_VX[0]   # 50 mL is the undeformed reference

# User's corrected Ogden N=2 parameters.
PHANTOMS = [
    dict(name="Phantom 1 (gelatin)",
         mu=[1100.0, 1400.0], alpha=[3.0, 1.0]),
    dict(name="Phantom 2 (cellulose)",
         mu=[2400.0,  100.0], alpha=[2.0, 10.0]),
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


def solve_state(a_vx, mu_list, alpha_list):
    """Run one state: Ogden G_ij → anisotropic Helmholtz → filter → DI."""
    balloon = SphericalBalloon(center=CENTER, radius_vx=a_vx, pressure=0.0)
    G_tensor = ogden_G_tensor_field(N=N, dx=DX, center=CENTER,
                                     a0_vx=A0_VX, a_vx=a_vx,
                                     mu_list=mu_list, alpha_list=alpha_list,
                                     G_lesion=G_LESION,
                                     balloon_mask=balloon.mask(N))
    # Also compute G_true for reference — trace/3 gives isotropic-avg.
    G_true = np.einsum("xyzii->xyz", G_tensor) / 3.0

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

    return dict(ring_true=_ring(G_true), ring_conv=_ring(mu_conv),
                ring_tsm=_ring(mu_tsm))


def main():
    out_dir = ROOT / "results" / "paper_ogden_fea"
    out_dir.mkdir(parents=True, exist_ok=True)

    schedule = [(BALLOON_RADII_VX[i], BALLOON_VOLUMES_ML[i], "inflation")
                for i in range(len(BALLOON_VOLUMES_ML))]
    # Deflation: retrace peak-1 down to 50 mL (skip the peak).
    for i in range(len(BALLOON_VOLUMES_ML) - 2, -1, -1):
        schedule.append((BALLOON_RADII_VX[i], BALLOON_VOLUMES_ML[i], "deflation"))

    all_curves = {}
    for phantom in PHANTOMS:
        print(f"\n[{phantom['name']}]")
        mu_str = ", ".join(f"({m},{a})" for m, a in zip(phantom['mu'], phantom['alpha']))
        print(f"  Ogden N=2  (μ,α): {mu_str}    G0 = {sum(phantom['mu'])} Pa")
        results = []
        for step, (a_vx, vol, branch) in enumerate(schedule):
            print(f"  [{step+1:2d}/{len(schedule)}] {vol} mL  r={a_vx:.2f} vx  "
                  f"λ_θ_surf = {(a_vx/A0_VX):.3f}  [{branch}]")
            s = solve_state(a_vx, phantom["mu"], phantom["alpha"])
            s.update(vol=vol, a_vx=a_vx, branch=branch, step=step)
            results.append(s)
            print(f"        ring_true={s['ring_true']/1000:.2f} kPa   "
                  f"conv={s['ring_conv']/1000:.2f}   TSM={s['ring_tsm']/1000:.2f}")
        all_curves[phantom["name"]] = results

    # ── Figure ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.8))
    n_infl = len(BALLOON_VOLUMES_ML)
    ax.axvspan(-0.5, n_infl - 0.5, alpha=0.05, color="tab:blue")
    ax.axvspan(n_infl - 0.5, 2 * n_infl - 1.5, alpha=0.05, color="tab:orange")
    # Yin reference bands.
    ax.axhspan(3.4, 3.6, alpha=0.10, color="black",
               label="Yin baseline TSM band")
    ax.axhspan(2.7, 2.8, alpha=0.10, color="tab:orange",
               label="Yin conv band (flat)")

    styles = {
        "Phantom 1 (gelatin)":   dict(color="black",    marker="o"),
        "Phantom 2 (cellulose)": dict(color="tab:cyan", marker="s"),
    }
    for name, results in all_curves.items():
        infl = [r for r in results if r["branch"] == "inflation"]
        defl = [r for r in results if r["branch"] == "deflation"]
        infl_x = list(range(len(infl)))
        defl_x = list(range(len(infl) - 1, len(infl) - 1 + len(defl)))
        infl_tsm  = [r["ring_tsm"]/1000  for r in infl]
        defl_tsm  = [r["ring_tsm"]/1000  for r in defl]
        infl_conv = [r["ring_conv"]/1000 for r in infl]
        defl_conv = [r["ring_conv"]/1000 for r in defl]
        st = styles[name]
        ax.plot(infl_x + defl_x, infl_tsm + defl_tsm,
                 color=st["color"], marker=st["marker"], ms=8, lw=1.8,
                 label=f"{name}: μ_TSM")
        ax.plot(infl_x + defl_x, infl_conv + defl_conv,
                 color=st["color"], marker=st["marker"], ms=8, lw=1.8,
                 linestyle="--", alpha=0.6,
                 label=f"{name}: μ_conv")

    labels = BALLOON_VOLUMES_ML + BALLOON_VOLUMES_ML[-2::-1]
    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels([f"{v}" for v in labels], fontsize=9)
    ax.set_xlabel("Balloon water volume (mL)     [inflation ← | → deflation]")
    ax.set_ylabel("Perilesional G_ring [kPa]  (DI, 3×3×3 median, 9 mm shell offset)")
    ax.set_title(
        "Ogden N=2 balloon inflation/deflation — Yin Fig 6 reproduction\n"
        "Analytical-Lamé stretch field  →  anisotropic Helmholtz  →  20-direction TSM"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    plt.tight_layout()
    out_fig = out_dir / "ogden_fig6_reproduction.png"
    fig.savefig(out_fig, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_fig}")

    # ── Summary text ────────────────────────────────────────────────────
    lines = [
        "Ogden N=2 balloon inflation/deflation — Yin Fig 6 reproduction",
        "=" * 78,
        f"Grid: {N}³, dx={DX*1000:.0f} mm    freq={FREQ:.0f} Hz",
        f"Undeformed balloon a_0 = {A0_VX:.2f} vx ({A0_VX*DX*1000:.0f} mm, 50 mL)",
        f"Peak balloon      a    = {BALLOON_RADII_VX[-1]:.2f} vx ({BALLOON_RADII_VX[-1]*DX*1000:.0f} mm, 250 mL)",
        f"Peak surface stretch λ_θ = {BALLOON_RADII_VX[-1]/A0_VX:.3f}",
        "",
        f"Method: analytical Lamé stretch → Ogden tangent G_ij(x) → "
        f"anisotropic Helmholtz → {NDIRS}-direction filter → DI with "
        f"median={MEDIAN_SIZE}, shell offset={SHELL_OFFSET_MM} mm",
        "",
        "Yin Fig 6 reference:",
        "  Phantom 1 TSM  (kPa):  3.5   3.8   3.9   4.2   4.4",
        "  Phantom 2 TSM  (kPa):  3.5   3.9   4.2   4.9   5.15",
        "  Conv (all P):          2.7   2.7   2.8   2.8   2.8   (essentially flat)",
        "",
    ]
    for name, results in all_curves.items():
        infl = [r for r in results if r["branch"] == "inflation"]
        tsm  = [r["ring_tsm"] / 1000 for r in infl]
        conv = [r["ring_conv"] / 1000 for r in infl]
        tru  = [r["ring_true"] / 1000 for r in infl]
        lines.append(f"{name}:")
        lines.append(f"  G_true (mean 1/3 tr(G))     : " + "  ".join(f"{v:5.2f}" for v in tru))
        lines.append(f"  μ_conv (amp-weighted mean)   : " + "  ".join(f"{v:5.2f}" for v in conv))
        lines.append(f"  μ_TSM  (MIP over 20 dirs)    : " + "  ".join(f"{v:5.2f}" for v in tsm))
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
