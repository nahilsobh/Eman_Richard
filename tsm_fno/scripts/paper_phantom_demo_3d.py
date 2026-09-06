#!/usr/bin/env python3
"""3D balloon-phantom demo — the 3D companion to paper_phantom_demo.py.

Physical setup
--------------
- Cubic grid, N × N × N at dx m/voxel.
- Spherical pressurised balloon centred in the cube.
- Bottom face (i = N-1): coherent circular piston-plate driver (real MRE).
- Top face (i = 0):      traction-free (Neumann) — container open at top.
- Side faces + bottom outside the driver disk: clamped Dirichlet u = 0.

Because we do not (yet) have a trained 3D FNO_TSM, we use
**direct inversion** as the stiffness estimator:
    G_DI(x) ≈ -ρω² u(x) / ∇²u(x)
which is the standard MRE baseline and requires no learning.

Outputs
-------
results/paper_demo_3d/inflation_series.png   — mid-slice panel, 6 states
results/paper_demo_3d/summary.txt            — quantitative table
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phantom.geometry_3d import (
    SphericalBalloon,
    make_effective_G_3d,
    perilesional_shell_3d,
)
from src.phantom.viscoelastic import sls_effective_pressures
from src.solver.helmholtz_fd_3d import (
    bottom_plate_driver_sources_3d,
    direct_inversion_3d,
    helmholtz_solve_3d,
)


# ── Configuration ────────────────────────────────────────────────────────
N        = 32
DX       = 0.003          # 3 mm/voxel → 9.6 cm FOV
FREQ     = 60.0
RHO      = 1000.0
DAMPING  = 0.05

G_BG     = 2500.0         # Pa
G_LESION = 2000.0
# A_COEFF = 0.20 calibrated against Yin Fig 6 Phantom 1 (pure gelatin) —
# peak G_ring at 250 mL = 4.28 kPa vs Yin's 4.4 kPa. Values in the paper's
# random-training-distribution [2, 8] were chosen for phantom diversity,
# not gel realism; 0.20 is the physically-honest number for gelatin.
A_COEFF  = 0.20

CENTER   = (N // 2, N // 2, N // 2)
SHELL_MM = 5.0
DRIVER_R = 0.5            # driver disk radius as fraction of N/2

BALLOON_VOLUMES_ML = [0,   50,   100,   150,   200,   250]
PRESSURE_STATES    = [0, 1000,  2000,  3000,  5000,  7000]   # Pa


def _balloon_vx(vol_ml: float, dx: float = DX) -> float:
    if vol_ml <= 0:
        return 4.0   # baseline radius (12 mm at dx=3mm)
    r_m = (3 * vol_ml * 1e-6 / (4 * math.pi)) ** (1/3)
    return r_m / dx


BALLOON_RADII_VX = [_balloon_vx(v) for v in BALLOON_VOLUMES_ML]
STATE_LABELS     = [
    f"Baseline (0 mL)         r={BALLOON_RADII_VX[0]*DX*1000:.0f} mm",
    f"+50 mL   (p=1 kPa)      r={BALLOON_RADII_VX[1]*DX*1000:.0f} mm",
    f"+100 mL  (p=2 kPa)      r={BALLOON_RADII_VX[2]*DX*1000:.0f} mm",
    f"+150 mL  (p=3 kPa)      r={BALLOON_RADII_VX[3]*DX*1000:.0f} mm",
    f"+200 mL  (p=5 kPa)      r={BALLOON_RADII_VX[4]*DX*1000:.0f} mm",
    f"+250 mL  (p=7 kPa)      r={BALLOON_RADII_VX[5]*DX*1000:.0f} mm",
]


def solve_state(balloon: SphericalBalloon,
                stiffening_exponent: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (u, G_true, G_di) for one balloon inflation state."""
    # 500 kPa cap: enough headroom for hyperelastic runs at high pressure.
    # The DI baseline doesn't share the 2D FNO's training-distribution cap.
    G_true = make_effective_G_3d(N, balloon, G_BG, G_LESION, A_COEFF,
                                  stiffening_exponent=stiffening_exponent,
                                  G_max_pa=500000.0)
    src    = bottom_plate_driver_sources_3d(N, radius_frac=DRIVER_R)
    t0     = time.time()
    u      = helmholtz_solve_3d(G_true, freq=FREQ, rho=RHO, dx=DX,
                                damping=DAMPING, sources=src, top_free=True)
    print(f"  solve took {time.time()-t0:.1f}s   |u|max={np.max(np.abs(u)):.3f}")
    G_di   = direct_inversion_3d(u, freq=FREQ, rho=RHO, dx=DX)
    return u, G_true, G_di


def ring_stats(G_di: np.ndarray, balloon: SphericalBalloon) -> tuple[float, float]:
    """(mean, median) DI-recovered stiffness in the perilesional shell."""
    shell = perilesional_shell_3d(balloon.mask(N), shell_mm=SHELL_MM, dx=DX)
    vals  = G_di[shell]
    vals  = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    # Trim extreme DI outliers (divergences where ∇²u ≈ 0).
    lo, hi = np.percentile(vals, [10, 90])
    trimmed = vals[(vals >= lo) & (vals <= hi)]
    return float(np.mean(trimmed)), float(np.median(vals))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="paper_demo_3d",
                        help="Result subdir name under tsm_fno/results/")
    parser.add_argument("--deflation", action="store_true",
                        help="After inflating to peak, run the deflation branch "
                             "(peak → baseline). Doubles the run time. Because the "
                             "forward model is memoryless (linear or hyperelastic, "
                             "no viscoelasticity), deflation matches inflation "
                             "at each pressure — Yin Fig. 6 lookalike + identity "
                             "regression test.")
    parser.add_argument("--stiffening-exponent", "-m", type=float, default=1.0,
                        help="Hyperelastic power-law exponent for the "
                             "acoustoelastic effective stiffness "
                             "G_eff = G_base·(1 + A·Δσ/G_base)^m. Default 1.0 "
                             "= linear (Phantom 1 flavor). Try 2.0 for a "
                             "Phantom 2 (cellulose-reinforced) analogue.")
    parser.add_argument("--viscoelastic-tau", type=float, default=None,
                        help="If set, apply an SLS viscoelastic relaxation to "
                             "the applied-pressure schedule. Value is the "
                             "gel relaxation time τ [s]. Typical gelatin: "
                             "30–120 s. Requires --scan-pause. Effective "
                             "pre-stress at each measurement is "
                             "p_eff[k] = p_applied[k] + (p_eff[k-1] - p_applied[k])·exp(-Δt/τ). "
                             "Produces genuine inflation/deflation hysteresis.")
    parser.add_argument("--scan-pause", type=float, default=45.0,
                        help="Wait time between each injection step and the "
                             "MRE measurement [s]. Only used with --viscoelastic-tau. "
                             "Yin's protocol ~30–60 s.")
    args = parser.parse_args()

    out_name = args.out
    if out_name == "paper_demo_3d":
        # Auto-suffix so specialized runs don't clobber the linear baseline.
        parts = []
        if args.stiffening_exponent != 1.0:
            parts.append(f"hyper{args.stiffening_exponent:g}")
        if args.viscoelastic_tau is not None:
            parts.append(f"visc{args.viscoelastic_tau:g}")
        if parts:
            out_name = "paper_demo_3d_" + "_".join(parts)
    save_dir = ROOT / "results" / out_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build the state schedule. Inflation is always 0→peak; deflation retraces
    # peak−1 → 0 (skipping the peak duplicate).
    schedule = list(zip(STATE_LABELS, PRESSURE_STATES, BALLOON_RADII_VX,
                        ["inflation"] * len(PRESSURE_STATES)))
    if args.deflation:
        n = len(PRESSURE_STATES)
        for i in range(n - 2, -1, -1):
            schedule.append((STATE_LABELS[i] + "  [defl.]",
                             PRESSURE_STATES[i], BALLOON_RADII_VX[i], "deflation"))

    # Viscoelastic effective-pressure sequence.
    applied_seq = [float(s[1]) for s in schedule]
    if args.viscoelastic_tau is not None:
        pauses = [float(args.scan_pause)] * len(schedule)
        p_eff_seq = sls_effective_pressures(applied_seq, pauses,
                                             tau=float(args.viscoelastic_tau)).tolist()
        print(f"\nViscoelastic on: τ={args.viscoelastic_tau}s, pause={args.scan_pause}s")
        print("  step | branch     | p_applied | p_effective")
        for k, (label, p, _, br) in enumerate(schedule):
            print(f"  {k+1:>4} | {br:<10} | {applied_seq[k]:>9.0f} | {p_eff_seq[k]:>11.1f}")
    else:
        p_eff_seq = applied_seq  # memoryless fallback

    states = []
    for (label, p_applied, r_vx, branch), p_eff in zip(schedule, p_eff_seq):
        print(f"\n{label}  (r={r_vx:.1f} vx, p_applied={p_applied} Pa, "
              f"p_eff={p_eff:.1f} Pa)  [{branch}]")
        # Balloon geometry follows applied volume/pressure (fast latex).
        # But the Δσ field the gel actually sees uses p_eff (viscoelastic lag).
        balloon = SphericalBalloon(center=CENTER, radius_vx=r_vx, pressure=float(p_eff))
        u, G_true, G_di = solve_state(balloon, stiffening_exponent=args.stiffening_exponent)
        mean_r, med_r   = ring_stats(G_di, balloon)
        states.append(dict(
            label=label, p=p_applied, p_eff=p_eff, radius_vx=r_vx,
            balloon=balloon, branch=branch,
            u=u, G_true=G_true, G_di=G_di,
            ring_mean_di=mean_r, ring_median_di=med_r,
        ))
        print(f"  G_ring (DI trimmed mean) = {mean_r:>8.1f} Pa"
              f"   G_ring (DI median) = {med_r:>8.1f} Pa")

    # ── Figure: mid-slice panel per state (Re(u), G_true, G_di) ─────────────
    mid = N // 2
    n_states = len(states)
    fig, axes = plt.subplots(n_states, 3, figsize=(10, 2.6 * n_states))
    G_vmin = min(s["G_true"].min() for s in states)
    G_vmax = max(s["G_true"].max() for s in states)

    for row, s in enumerate(states):
        ax = axes[row]
        im0 = ax[0].imshow(s["u"][mid, :, :].real, cmap="RdBu_r")
        ax[0].axis("off")
        ax[0].set_ylabel(s["label"], fontsize=7, rotation=0, labelpad=140, va="center")
        if row == 0:
            ax[0].set_title("Re(u) [mid-slice]", fontsize=9, fontweight="bold")

        im1 = ax[1].imshow(s["G_true"][mid, :, :], cmap="hot",
                            vmin=G_vmin, vmax=G_vmax)
        ax[1].axis("off")
        if row == 0:
            ax[1].set_title("G_true [Pa]", fontsize=9, fontweight="bold")
        plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

        di_mid = s["G_di"][mid, :, :]
        di_finite = di_mid[np.isfinite(di_mid)]
        vmin_di = np.percentile(di_finite, 5) if di_finite.size else 0.0
        vmax_di = np.percentile(di_finite, 95) if di_finite.size else 1.0
        im2 = ax[2].imshow(di_mid, cmap="hot", vmin=vmin_di, vmax=vmax_di)
        ax[2].axis("off")
        ax[2].set_title(f"G_DI [Pa]  ring={s['ring_mean_di']:.0f}",
                        fontsize=8)
        plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

    plt.suptitle(
        f"3D balloon inflation (spherical, {N}³ grid, dx={DX*1000:.0f} mm, "
        f"FOV {N*DX*100:.1f} cm)\n"
        f"open top + bottom-plate driver ({DRIVER_R*100:.0f}% radius) — "
        f"{FREQ:.0f} Hz — direct inversion baseline",
        fontsize=10,
    )
    plt.tight_layout()
    out_fig = save_dir / "inflation_series.png"
    fig.savefig(out_fig, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_fig}")

    # ── Figure 2: G_ring vs step, inflation + deflation (Yin Fig 6 lookalike) ─
    if args.deflation:
        infl = [s for s in states if s["branch"] == "inflation"]
        defl = [s for s in states if s["branch"] == "deflation"]
        infl_x = list(range(len(infl)))
        defl_x = list(range(len(infl) - 1, len(infl) - 1 - len(defl), -1))
        infl_y = [s["ring_mean_di"] / 1000.0 for s in infl]     # kPa
        defl_y = [s["ring_mean_di"] / 1000.0 for s in defl]

        fig2, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(infl_x, infl_y, "o-",  color="tab:red",  label="Inflation",
                markersize=8, linewidth=2)
        ax.plot(defl_x, defl_y, "s--", color="tab:blue", label="Deflation",
                markersize=8, linewidth=2)
        ax.set_xticks(list(range(len(infl))))
        ax.set_xticklabels([f"{p} Pa" for p in PRESSURE_STATES], rotation=30, ha="right")
        ax.set_xlabel("Balloon inflation state (pressure)")
        ax.set_ylabel("Perilesional G_ring [kPa]  (DI trimmed mean)")
        subtitle = (f"SLS viscoelastic (τ={args.viscoelastic_tau:g}s, "
                    f"pause={args.scan_pause:g}s) → hysteresis loop"
                    if args.viscoelastic_tau is not None
                    else "Memoryless model → curves overlay by construction")
        ax.set_title(f"3D balloon inflation → deflation cycle ({FREQ:.0f} Hz)\n"
                     + subtitle)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        out_fig2 = save_dir / "hysteresis_curve.png"
        fig2.savefig(out_fig2, dpi=130, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved {out_fig2}")

        # Hysteresis measurement (or reversibility check if memoryless).
        max_abs_diff = max(abs(a - b) for a, b in zip(infl_y, defl_y[::-1]))
        label = ("Hysteresis" if args.viscoelastic_tau is not None
                 else "Reversibility check")
        print(f"\n{label}: max |ΔG_ring| between inflation & deflation "
              f"at matched applied p = {max_abs_diff*1000:.2f} Pa")

    # ── Summary table ───────────────────────────────────────────────────────
    lines = [
        "3D Balloon Phantom Demo — spherical inclusion, open top, bottom driver",
        "=" * 78,
        f"Grid: {N}³, dx={DX*1000:.0f} mm  →  FOV {N*DX*100:.1f}×{N*DX*100:.1f}×{N*DX*100:.1f} cm",
        f"G_bg={G_BG:.0f} Pa  G_lesion={G_LESION:.0f} Pa  A_coeff={A_COEFF}",
        f"Frequency: {FREQ:.0f} Hz    Damping ξ={DAMPING}",
        f"Driver: bottom-face disk, radius = {DRIVER_R*N/2:.1f} vx ({DRIVER_R*N/2*DX*1000:.0f} mm)",
        f"Constitutive law: G_eff = G_base · (1 + A·Δσ/G_base)^m,  m={args.stiffening_exponent}"
        + ("  [linear]" if args.stiffening_exponent == 1.0
           else "  [hyperelastic strain-stiffening]"),
        (f"Viscoelastic (SLS): τ={args.viscoelastic_tau:g}s, "
         f"scan pause={args.scan_pause:g}s  →  history-dependent p_eff"
         if args.viscoelastic_tau is not None
         else "Memoryless: p_eff = p_applied at every step"),
        "",
        f"{'State':<48} {'branch':<10} {'p_appl':>7}  {'p_eff':>8}  {'G_ring DI mean':>15}  {'G_ring DI median':>18}",
        "-" * 118,
    ]
    for s in states:
        lines.append(
            f"{s['label']:<48} {s['branch']:<10} {s['p']:>7}  "
            f"{s['p_eff']:>8.1f}  "
            f"{s['ring_mean_di']:>15.1f}  {s['ring_median_di']:>18.1f}"
        )
    lines += [
        "-" * 118,
        "",
        "Notes:",
        "  - Stiffness estimator: local direct inversion (DI). No FNO — a 3D",
        "    FNO_TSM would need its own training pipeline (see RESUME.md).",
        "  - DI is noisy near sources and at the free surface; trimmed mean",
        "    uses the 10–90 percentile band within the perilesional shell.",
        "  - Compare 2D vs 3D G_ring @ baseline (0 Pa) as the reference for",
        "    the sim-dimensionality gap.",
    ]
    if args.deflation:
        if args.viscoelastic_tau is not None:
            lines += [
                "  - Deflation branch is included, with SLS viscoelasticity.",
                "    Effective pressure lags applied pressure: inflation p_eff",
                "    stays below applied, deflation p_eff stays above applied,",
                "    so the two branches SEPARATE — that separation is the",
                "    hysteresis loop Yin reports for Phantom 1 & 2.",
                "  - This is a lumped-parameter model of gel creep; the wave-",
                "    scale damping ξ handles frequency-domain losses separately.",
            ]
        else:
            lines += [
                "  - Deflation branch is included. The constitutive law is",
                "    memoryless (linear or hyperelastic), so deflation numbers",
                "    equal inflation numbers at the same applied pressure —",
                "    cross-check with the reversibility diff above.",
                "  - Yin's real gel shows hysteresis from viscoelastic creep;",
                "    enable it with --viscoelastic-tau <seconds>.",
            ]
    summary = save_dir / "summary.txt"
    summary.write_text("\n".join(lines) + "\n")
    print(f"Saved {summary}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
