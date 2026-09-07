#!/usr/bin/env python3
"""Sweep four Ogden N=2 configurations against Yin Fig 6 phantoms.

Each configuration inflates+deflates both phantoms through 5 volume
states (50, 100, 150, 200, 250 mL) and reports the perilesional
G_ring for μ_TSM and μ_conv. Uses the analytical-Lamé + anisotropic
Helmholtz + 20-direction filter + median DI pipeline (same as
paper_ogden_fea_reproduction.py).

Four configs — each isolates or combines the three fixes we
identified after the baseline undershot Yin's TSM peak:

  A. baseline           — shell 9 mm, user's original Ogden params
  B. shell 3 mm         — sample voxels closer to balloon (Fix 1)
  C. P2 mu2 rebalanced  — μ_2 = 500 (α_2 = 10 fiber-lock weight up),
                          μ_1 = 2000 (down to keep G0 = 2500)  (Fix 2)
  D. combined           — shell 3 mm + P1 α_1 raised 3 → 5 + P2 rebalanced

Runtime ~20 min at N=32.
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


# ── Common config ─────────────────────────────────────────────────────
N        = 32
DX       = 0.003
FREQ     = 60.0
RHO      = 1000.0
DAMPING  = 0.05
G_LESION = 2000.0
DRIVER_R = 0.5
CENTER   = (N // 2, N // 2, N // 2)
SHELL_MM        = 5.0
MEDIAN_SIZE     = 3
NDIRS           = 20
WEDGE_WIDTH     = 0.35
AMP_THRESHOLD   = 0.15

BALLOON_VOLUMES_ML = [50, 100, 150, 200, 250]
def _r_vx(vol):
    return ((3 * vol * 1e-6 / (4 * math.pi)) ** (1/3)) / DX
BALLOON_RADII_VX = [_r_vx(v) for v in BALLOON_VOLUMES_ML]
A0_VX = BALLOON_RADII_VX[0]

# ── Four experimental configurations ──────────────────────────────────
CONFIGS = [
    dict(
        label="A. baseline (shell 9 mm)",
        shell_offset_mm=9.0,
        p1_mu=[1100.0, 1400.0], p1_alpha=[3.0, 1.0],
        p2_mu=[2400.0,  100.0], p2_alpha=[2.0, 10.0],
    ),
    dict(
        label="B. shell 3 mm (Fix 1)",
        shell_offset_mm=3.0,
        p1_mu=[1100.0, 1400.0], p1_alpha=[3.0, 1.0],
        p2_mu=[2400.0,  100.0], p2_alpha=[2.0, 10.0],
    ),
    dict(
        label="C. P2 rebalanced (Fix 2)",
        shell_offset_mm=9.0,
        p1_mu=[1100.0, 1400.0], p1_alpha=[3.0, 1.0],
        p2_mu=[2000.0,  500.0], p2_alpha=[2.0, 10.0],
    ),
    dict(
        label="D. all fixes combined",
        shell_offset_mm=3.0,
        p1_mu=[1100.0, 1400.0], p1_alpha=[5.0, 1.0],   # Fix 3: raise α_1
        p2_mu=[2000.0,  500.0], p2_alpha=[2.0, 10.0],  # Fix 2
    ),
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


def _solve_and_ring(a_vx, mu_list, alpha_list, shell_offset_mm):
    """Solve one state and return the μ_conv, μ_TSM ring stats."""
    balloon = SphericalBalloon(center=CENTER, radius_vx=a_vx, pressure=0.0)
    G_tensor = ogden_G_tensor_field(N=N, dx=DX, center=CENTER,
                                     a0_vx=A0_VX, a_vx=a_vx,
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
                                   inner_offset_mm=shell_offset_mm)

    def _ring(f):
        v = f[shell]; v = v[np.isfinite(v)]
        if v.size == 0: return float("nan")
        lo, hi = np.percentile(v, [10, 90])
        tr = v[(v >= lo) & (v <= hi)]
        return float(tr.mean()) if tr.size else float("nan")

    return _ring(mu_conv), _ring(mu_tsm)


def run_config(config):
    """Run inflation-only for a config on both phantoms (deflation would be
    identical for our memoryless model — skip for speed)."""
    out = {"config": config["label"]}
    for pkey, mu_list, alpha_list in [
        ("P1", config["p1_mu"], config["p1_alpha"]),
        ("P2", config["p2_mu"], config["p2_alpha"]),
    ]:
        conv_curve, tsm_curve = [], []
        for a_vx, vol in zip(BALLOON_RADII_VX, BALLOON_VOLUMES_ML):
            c, t = _solve_and_ring(a_vx, mu_list, alpha_list,
                                    shell_offset_mm=config["shell_offset_mm"])
            conv_curve.append(c / 1000)
            tsm_curve.append(t / 1000)
            print(f"    {pkey}  {vol} mL  conv={c/1000:.2f}  TSM={t/1000:.2f}")
        out[f"{pkey}_conv"] = conv_curve
        out[f"{pkey}_tsm"]  = tsm_curve
    return out


def main():
    out_dir = ROOT / "results" / "paper_ogden_fea_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning {len(CONFIGS)} configs × 2 phantoms × 5 states "
          f"(~5 min per config × 4 = ~20 min total)\n")
    all_results = []
    for config in CONFIGS:
        print(f"\n=== {config['label']} ===")
        print(f"  shell_offset = {config['shell_offset_mm']} mm")
        print(f"  P1: mu={config['p1_mu']}, alpha={config['p1_alpha']}"
              f"   G0={sum(config['p1_mu'])} Pa")
        print(f"  P2: mu={config['p2_mu']}, alpha={config['p2_alpha']}"
              f"   G0={sum(config['p2_mu'])} Pa")
        r = run_config(config)
        all_results.append(r)

    # ── Comparison plot: 2 axes side-by-side, one per phantom ─────────
    yin_p1 = [3.5, 3.8, 3.9, 4.2, 4.4]
    yin_p2 = [3.5, 3.9, 4.2, 4.9, 5.15]
    yin_conv = [2.7, 2.7, 2.8, 2.8, 2.8]
    colors = ["tab:gray", "tab:blue", "tab:green", "tab:red"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, phantom, yin_curve in zip(axes, ("P1", "P2"), (yin_p1, yin_p2)):
        # Yin reference: TSM (solid black), conv (dashed orange).
        ax.plot(BALLOON_VOLUMES_ML, yin_curve,   "k*-", ms=12, lw=2,
                label="Yin μ_TSM (measured)")
        ax.plot(BALLOON_VOLUMES_ML, yin_conv,    "*--", color="tab:orange",
                ms=12, lw=1.5, label="Yin μ_conv (measured, ~flat)")
        for r, c in zip(all_results, colors):
            tsm = r[f"{phantom}_tsm"]
            cv  = r[f"{phantom}_conv"]
            ax.plot(BALLOON_VOLUMES_ML, tsm, "o-",  color=c, ms=6, lw=1.4,
                    label=f"{r['config']} — TSM")
            ax.plot(BALLOON_VOLUMES_ML, cv,  "^--", color=c, ms=5, lw=1.0, alpha=0.55,
                    label=f"{r['config']} — conv")
        ax.set_title(f"Phantom {phantom[1]} — {'gelatin' if phantom == 'P1' else 'cellulose'}",
                     fontsize=12)
        ax.set_xlabel("Balloon water volume (mL)")
        ax.grid(True, alpha=0.3)
        if phantom == "P1":
            ax.set_ylabel("Perilesional G_ring [kPa]")
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        "Ogden N=2 sweep — closing the peak-amplitude gap to Yin Fig 6",
        fontsize=13,
    )
    plt.tight_layout()
    out_fig = out_dir / "ogden_sweep.png"
    fig.savefig(out_fig, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_fig}")

    # ── Text summary ────────────────────────────────────────────────────
    lines = [
        "Ogden N=2 sweep — closing the gap to Yin Fig 6",
        "=" * 78,
        "",
        "Yin Phantom 1 TSM (kPa):  3.5   3.8   3.9   4.2   4.4",
        "Yin Phantom 2 TSM (kPa):  3.5   3.9   4.2   4.9   5.15",
        "Yin conv (flat):          2.7   2.7   2.8   2.8   2.8",
        "",
    ]
    for r in all_results:
        lines.append(f"{r['config']}:")
        for phantom in ("P1", "P2"):
            tsm  = "  ".join(f"{v:5.2f}" for v in r[f"{phantom}_tsm"])
            conv = "  ".join(f"{v:5.2f}" for v in r[f"{phantom}_conv"])
            lines.append(f"  {phantom} μ_TSM :  {tsm}")
            lines.append(f"  {phantom} μ_conv:  {conv}")
        lines.append("")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
