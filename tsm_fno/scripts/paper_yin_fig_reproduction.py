#!/usr/bin/env python3
"""Reproduce Yin et al. (2025) phantom figures — 3, 5, 6.

Runs the best-fit configuration identified in this session:
    - method:  filter (broadband multi-face + k-space wedge)
    - dirs:    20 (Fibonacci-lattice DF set matching Yin's protocol)
    - DI post-processing: 3x3x3 spatial median + 9 mm shell edge-exclusion
    - constitutive: powerlaw m=1 for Phantom 1 (gelatin),
                    powerlaw m=1.5 for Phantom 2 (cellulose-reinforced),
                    p=0 for Phantom 3 (non-pressurising control).

Produces three PNGs matching the layout of Yin's Figures 3, 5, 6:
    fig3_phantom_geometry.png   — mid-slice T2W-like maps, 3 phantoms x 6 states
    fig5_stiffness_maps.png     — G_true / μ_conv / μ_TSM, 3 phantoms x 3 states,
                                  with annular perilesional ROI overlay
    fig6_ring_curves.png        — G_ring vs balloon volume, all 3 phantoms +
                                  both inversions + inflation & deflation

Runtime ~5-8 min at N=32 on a single core.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phantom.geometry_3d import (
    SphericalBalloon,
    make_effective_G_3d,
    perilesional_shell_3d,
)
from src.solver.helmholtz_fd_3d import (
    direct_inversion_3d,
    directional_filter_3d,
    helmholtz_solve_3d,
    multi_face_broadband_sources,
)


# ── Config ──────────────────────────────────────────────────────────────
N        = 32
DX       = 0.003
FREQ     = 60.0
RHO      = 1000.0
DAMPING  = 0.05
G_BG     = 2500.0
G_LESION = 2000.0
A_COEFF  = 0.20
DRIVER_R = 0.5
CENTER   = (N // 2, N // 2, N // 2)
SHELL_MM        = 5.0
SHELL_OFFSET_MM = 9.0    # Yin: 3 px away from balloon edge
MEDIAN_SIZE     = 3      # Yin: 3x3x3 median filter
NDIRS           = 20
WEDGE_WIDTH     = 0.35
AMP_THRESHOLD   = 0.15   # per-direction amplitude gate

BALLOON_VOLUMES_ML = [0, 50, 100, 150, 200, 250]
PRESSURE_STATES    = [0, 1000, 2000, 3000, 5000, 7000]
def _r_vx(vol):
    if vol <= 0: return 4.0
    return ((3 * vol * 1e-6 / (4 * math.pi)) ** (1/3)) / DX
BALLOON_RADII_VX = [_r_vx(v) for v in BALLOON_VOLUMES_ML]

# Phantom definitions matching Yin:
#   1: pure gelatin — inflated 0→250 mL, m=1 (linear acoustoelastic)
#   2: cellulose-reinforced — same states, m=1.5 (super-linear)
#   3: static control — fixed at 250 mL, never pressurised (p=0 forced)
PHANTOMS = [
    dict(name="Phantom 1",  label="P1 (gelatin)",       m=1.0, control=False),
    dict(name="Phantom 2",  label="P2 (cellulose)",     m=1.5, control=False),
    dict(name="Phantom 3",  label="P3 (control p=0)",   m=1.0, control=True),
]


def _fibonacci_dirs(n):
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    out = []
    for i in range(n):
        z = 1.0 - (2.0 * i + 1.0) / n
        theta = 2.0 * np.pi * i / phi
        r = np.sqrt(max(0.0, 1.0 - z * z))
        khat = np.array([r * np.cos(theta), r * np.sin(theta), z])
        out.append(khat / (np.linalg.norm(khat) + 1e-30))
    return out


def solve_state(radius_vx, pressure, stiffening_exponent):
    """Returns (G_true, mu_conv, mu_TSM) mid-slice arrays + ring stats."""
    balloon = SphericalBalloon(center=CENTER, radius_vx=radius_vx, pressure=pressure)
    G_iso = make_effective_G_3d(N, balloon, G_BG, G_LESION, A_COEFF,
                                 stiffening_exponent=stiffening_exponent,
                                 G_max_pa=500000.0)
    src = multi_face_broadband_sources(N, radius_frac=DRIVER_R,
                                        faces=("iN", "jN", "j0", "kN", "k0"))
    u_full = helmholtz_solve_3d(G_iso, freq=FREQ, rho=RHO, dx=DX,
                                 damping=DAMPING, sources=src, top_free=False)

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

    # Amplitude gate.
    peaks = amp_stack.reshape(NDIRS, -1).max(axis=1)
    thresh = peaks[:, None, None, None] * AMP_THRESHOLD
    di_stack = np.where(amp_stack >= thresh, di_stack, np.nan)

    # μ_conv (amplitude-weighted mean).
    w = amp_stack ** 2
    with np.errstate(invalid="ignore"):
        num = np.nansum(np.where(np.isnan(di_stack), 0.0, w * di_stack), axis=0)
        den = np.nansum(np.where(np.isnan(di_stack), 0.0, w),            axis=0)
        mu_conv = num / (den + 1e-30)
        mu_conv[den == 0] = np.nan
    # μ_TSM (MIP).
    mu_tsm = np.nanmax(di_stack, axis=0)

    shell = perilesional_shell_3d(balloon.mask(N), shell_mm=SHELL_MM, dx=DX,
                                   inner_offset_mm=SHELL_OFFSET_MM)

    def _ring(field):
        v = field[shell]; v = v[np.isfinite(v)]
        if v.size == 0: return float("nan")
        lo, hi = np.percentile(v, [10, 90])
        tr = v[(v >= lo) & (v <= hi)]
        return float(tr.mean()) if tr.size else float("nan")

    return dict(
        G_true=G_iso, mu_conv=mu_conv, mu_tsm=mu_tsm,
        balloon_mask=balloon.mask(N),
        shell=shell,
        ring_true=_ring(G_iso), ring_conv=_ring(mu_conv), ring_tsm=_ring(mu_tsm),
    )


def build_phantom_cycle(phantom):
    """Returns list of dict per state — 11 states (6 inflation + 5 deflation)."""
    states = []
    schedule = [(i, "inflation") for i in range(len(PRESSURE_STATES))]
    schedule += [(i, "deflation") for i in range(len(PRESSURE_STATES) - 2, -1, -1)]

    for step, (idx, branch) in enumerate(schedule):
        p = 0.0 if phantom["control"] else float(PRESSURE_STATES[idx])
        r_vx = BALLOON_RADII_VX[idx] if not phantom["control"] else BALLOON_RADII_VX[-1]
        vol  = BALLOON_VOLUMES_ML[idx] if not phantom["control"] else 250
        print(f"  [{step+1:2d}/{len(schedule)}] {phantom['name']}: "
              f"{vol} mL, p={p:.0f} Pa, r={r_vx:.1f}vx [{branch}]")
        s = solve_state(r_vx, p, phantom["m"])
        s.update(step=step, vol_ml=vol, pressure=p, radius_vx=r_vx, branch=branch)
        states.append(s)
    return states


# ── Figures ─────────────────────────────────────────────────────────────
MID = N // 2

def fig3_phantom_geometry(all_states, out_path):
    """T2W-like phantom cross-section grid: 3 rows × 6 states (inflation only)."""
    inflation_only = [[s for s in ph if s["branch"] == "inflation"] for ph in all_states]
    n_states = len(BALLOON_VOLUMES_ML)
    fig, axes = plt.subplots(3, n_states, figsize=(2.2 * n_states, 6.4))
    for row, (ph, states) in enumerate(zip(PHANTOMS, inflation_only)):
        for col, s in enumerate(states):
            ax = axes[row, col]
            # T2W-like display: bright balloon (water) on dark gel background.
            mask = s["balloon_mask"][MID]
            gray = np.where(mask, 1.0, 0.15)   # white balloon on grey gel
            ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"{BALLOON_VOLUMES_ML[col]} mL", fontsize=10)
            if col == 0:
                ax.text(-0.05, 0.5, ph["label"], transform=ax.transAxes,
                        rotation=90, va="center", ha="right", fontsize=10,
                        fontweight="bold")
    fig.suptitle("Fig 3 analogue — phantom cross-sections at 6 inflation states",
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def fig5_stiffness_maps(all_states, out_path):
    """G_true vs μ_conv vs μ_TSM at 3 states: baseline (50 mL), peak (250 mL),
    deflation-back-to-baseline (50 mL). Only Phantoms 1 & 2 (Yin's Fig 5)."""
    # Pull the exact states we want.
    def _find(states, vol, branch):
        for s in states:
            if s["vol_ml"] == vol and s["branch"] == branch:
                return s
        return None

    p1, p2 = all_states[0], all_states[1]
    columns = [
        ("Baseline (50 mL)",   _find(p1, 50, "inflation"), _find(p2, 50, "inflation")),
        ("Peak (250 mL)",      _find(p1, 250, "inflation"), _find(p2, 250, "inflation")),
        ("Deflated (50 mL)",   _find(p1, 50, "deflation"), _find(p2, 50, "deflation")),
    ]

    # Common colour scales across the panel.
    all_G  = np.concatenate([[s["G_true"][MID] for s in (col[1], col[2])]
                              for col in columns], axis=None)
    G_vmin = np.nanpercentile(all_G, 5)
    G_vmax = np.nanpercentile(all_G, 99)

    fig, axes = plt.subplots(6, 3, figsize=(3.4 * 3, 3.0 * 6))
    row_labels = [
        ("P1  G_true",  0, "G_true"),
        ("P1  μ_conv",  0, "mu_conv"),
        ("P1  μ_TSM",   0, "mu_tsm"),
        ("P2  G_true",  1, "G_true"),
        ("P2  μ_conv",  1, "mu_conv"),
        ("P2  μ_TSM",   1, "mu_tsm"),
    ]
    for row_i, (rlabel, phi, field) in enumerate(row_labels):
        for col_i, (ctitle, s1, s2) in enumerate(columns):
            s = (s1, s2)[phi]
            ax = axes[row_i, col_i]
            img = s[field][MID]
            im = ax.imshow(img, cmap="hot", vmin=G_vmin, vmax=G_vmax)
            # Balloon outline + annular ROI overlay.
            mask_slice = s["balloon_mask"][MID]
            ax.contour(mask_slice.astype(float), levels=[0.5],
                        colors=["white"], linewidths=1.0)
            shell_slice = s["shell"][MID]
            ax.contour(shell_slice.astype(float), levels=[0.5],
                        colors=["cyan"], linewidths=0.6, linestyles="dashed")
            ax.axis("off")
            if row_i == 0:
                ax.set_title(ctitle, fontsize=10, fontweight="bold")
            if col_i == 0:
                ax.text(-0.06, 0.5, rlabel, transform=ax.transAxes,
                        rotation=90, va="center", ha="right", fontsize=9,
                        fontweight="bold")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                          label="G [Pa]" if col_i == 2 else None)
            # Annotation: ring value for the map.
            ring_val = s.get({
                "G_true": "ring_true", "mu_conv": "ring_conv", "mu_tsm": "ring_tsm"
            }[field], float("nan"))
            ax.text(0.02, 0.98,
                    f"ring={ring_val/1000:.2f} kPa" if np.isfinite(ring_val) else "ring=—",
                    transform=ax.transAxes, va="top", color="white", fontsize=8,
                    bbox=dict(facecolor="black", alpha=0.5, pad=1))

    fig.suptitle(
        "Fig 5 analogue — mid-slice stiffness maps (G_true, μ_conv, μ_TSM)\n"
        "white outline = balloon boundary,  cyan dashed = perilesional shell ROI",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def fig6_ring_curves(all_states, out_path):
    """G_ring vs volume for all 3 phantoms, both TSM and conv, inflation +
    deflation. Layout mirrors Yin's Figure 6: shaded inflation zone on left,
    deflation on right, control as a horizontal dotted line."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    # Shaded background: inflation left half, deflation right half.
    n = len(BALLOON_VOLUMES_ML)
    ax.axvspan(-0.5, n - 1 + 0.5, alpha=0.05, color="tab:blue")   # inflation
    ax.axvspan(n - 1 + 0.5, 2 * (n - 1) + 0.5, alpha=0.05, color="tab:orange")

    colors_tsm  = {"Phantom 1": "black",      "Phantom 2": "tab:cyan",   "Phantom 3": "gray"}
    colors_conv = {"Phantom 1": "tab:orange", "Phantom 2": "tab:red",    "Phantom 3": "gray"}
    markers     = {"Phantom 1": "o",          "Phantom 2": "s",          "Phantom 3": "D"}

    for ph, states in zip(PHANTOMS, all_states):
        infl = [s for s in states if s["branch"] == "inflation"]
        defl = [s for s in states if s["branch"] == "deflation"]
        infl_x = list(range(len(infl)))
        defl_x = list(range(len(infl) - 1, len(infl) - 1 + len(defl)))
        infl_tsm  = [s["ring_tsm"]/1000  for s in infl]
        defl_tsm  = [s["ring_tsm"]/1000  for s in defl]
        infl_conv = [s["ring_conv"]/1000 for s in infl]
        defl_conv = [s["ring_conv"]/1000 for s in defl]

        style = dict(marker=markers[ph["name"]], ms=8, lw=1.8)
        if ph["control"]:
            # Dashed control line: single horizontal (constant p=0).
            y = infl_tsm[0]
            ax.axhline(y, ls=":", color=colors_tsm[ph["name"]],
                        label=f"{ph['label']}: TSM (control)")
            y = infl_conv[0]
            ax.axhline(y, ls=":", color=colors_conv[ph["name"]], alpha=0.6,
                        label=f"{ph['label']}: conv (control)")
        else:
            ax.plot(infl_x + defl_x, infl_tsm + defl_tsm,
                     color=colors_tsm[ph["name"]],
                     label=f"{ph['label']}: μ_TSM",
                     **style)
            ax.plot(infl_x + defl_x, infl_conv + defl_conv,
                     color=colors_conv[ph["name"]], linestyle="--",
                     label=f"{ph['label']}: μ_conv",
                     **style)

    labels = BALLOON_VOLUMES_ML + BALLOON_VOLUMES_ML[-2::-1]
    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels([f"{v}" for v in labels], fontsize=9)
    ax.set_xlabel("Balloon water volume (mL)     [inflation ← | → deflation]")
    ax.set_ylabel("Perilesional G_ring [kPa]  (DI trimmed mean)")
    ax.set_title(
        "Fig 6 analogue — G_ring vs balloon volume, full cycle\n"
        "P1 (gelatin, m=1) · P2 (cellulose, m=1.5) · P3 (static control, p=0)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    # Yin reference band overlay: shaded region ~ Yin's actual data.
    ax.axhspan(3.4, 3.6, alpha=0.10, color="black")
    ax.text(0.15, 3.5, "Yin baseline TSM ~3.5", fontsize=7, color="black", alpha=0.6)
    ax.axhspan(2.7, 2.8, alpha=0.10, color="tab:orange")
    ax.text(0.15, 2.75, "Yin conv ~2.7-2.8 (flat)", fontsize=7, color="tab:orange", alpha=0.6)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def summary_text(all_states, out_path):
    """Write a Yin-comparison table."""
    lines = [
        "Yin Fig 6 reproduction — quantitative summary",
        "=" * 78,
        f"Grid: {N}³, dx={DX*1000:.0f} mm    A_coeff = {A_COEFF}    FREQ = {FREQ:.0f} Hz",
        f"Method: filter + {NDIRS} directions, median={MEDIAN_SIZE}, shell offset={SHELL_OFFSET_MM} mm",
        "",
        "Yin Phantom 1 TSM (kPa):     3.5  3.5  3.8  3.9  4.2  4.4",
        "Yin Phantom 2 TSM (kPa):     3.5  3.5  3.9  4.2  4.9  5.15",
        "Yin conv    (kPa, all P):    2.7  2.7  2.7  2.8  2.8  2.8  (essentially flat)",
        "",
    ]
    for ph, states in zip(PHANTOMS, all_states):
        infl = [s for s in states if s["branch"] == "inflation"]
        tsm  = [s["ring_tsm"] / 1000 for s in infl]
        conv = [s["ring_conv"] / 1000 for s in infl]
        lines.append(f"{ph['label']}:")
        lines.append(f"  μ_TSM  (0/50/100/150/200/250 mL): "
                     + "  ".join(f"{v:5.2f}" for v in tsm))
        lines.append(f"  μ_conv (0/50/100/150/200/250 mL): "
                     + "  ".join(f"{v:5.2f}" for v in conv))
    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"  saved {out_path}")
    print("\n" + "\n".join(lines))


def main():
    out_dir = ROOT / "results" / "paper_yin_figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Running 3 phantoms, filter+20+median+edge ===\n")
    all_states = []
    for phantom in PHANTOMS:
        print(f"[{phantom['label']}]  m={phantom['m']}  control={phantom['control']}")
        states = build_phantom_cycle(phantom)
        all_states.append(states)

    print("\n=== Generating figures ===")
    fig3_phantom_geometry(all_states, out_dir / "fig3_phantom_geometry.png")
    fig5_stiffness_maps  (all_states, out_dir / "fig5_stiffness_maps.png")
    fig6_ring_curves     (all_states, out_dir / "fig6_ring_curves.png")
    summary_text         (all_states, out_dir / "summary.txt")


if __name__ == "__main__":
    main()
