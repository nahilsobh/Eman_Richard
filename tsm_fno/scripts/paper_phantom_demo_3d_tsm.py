#!/usr/bin/env python3
"""3D TSM (tissue strain mapping) demo — the Yin-style multi-direction pipeline.

For each of ``N_dirs`` propagation directions k̂:
  1. Compute direction-dependent effective stiffness
       G_eff(k̂, x) = G_base(x) · (1 + A · k̂·σ·k̂ / G_base(x))^m
     from the anisotropic Lamé stress tensor. Radial propagation sees the
     softened (compressed) radial direction; tangential propagation sees
     the stiffened (stretched) tangential direction.
  2. Place a coherent piston-plate source on the face perpendicular to k̂
     (matching how Yin drives the phantom).
  3. Solve the 3D Helmholtz for the complex displacement field u_k̂.
  4. Run direct inversion (DI) to get a direction-specific stiffness map
     G_DI(k̂, x).

Then combine the direction-specific maps two ways:
  * μ_conv  = amplitude-weighted mean across directions (conventional MRE)
  * μ_TSM   = voxelwise maximum across directions (Yin's TSM MIP)

μ_TSM should show a perilesional stiffening ring; μ_conv should not.

This is a single-inflation-state demo (peak by default) because the
6-direction sweep is 6× slower than the memoryless demo. Runs in ~1 min
at N=32.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phantom.geometry_3d import (
    SphericalBalloon,
    effective_G_for_direction,
    make_effective_G_3d,
    perilesional_shell_3d,
    stress_tensor_sphere,
)
from src.solver.helmholtz_fd_3d import (
    bottom_plate_driver_sources_3d,
    direct_inversion_3d,
    directional_filter_3d,
    helmholtz_solve_3d,
    multi_face_broadband_sources,
)


# ── Configuration ────────────────────────────────────────────────────────
N        = 32
DX       = 0.003
FREQ     = 60.0
RHO      = 1000.0
DAMPING  = 0.05
G_BG     = 2500.0
G_LESION = 2000.0
# Calibrated against Yin Fig 6 Phantom 1 (see paper_phantom_demo_3d.py).
A_COEFF  = 0.20
DRIVER_R = 0.5
CENTER   = (N // 2, N // 2, N // 2)
SHELL_MM = 5.0

# Full-cycle state schedule — mirrors paper_phantom_demo_3d.py.
import math as _math
BALLOON_VOLUMES_ML = [0, 50, 100, 150, 200, 250]
PRESSURE_STATES    = [0, 1000, 2000, 3000, 5000, 7000]  # Pa
def _balloon_vx(vol_ml, dx=DX):
    if vol_ml <= 0: return 4.0
    return ((3 * vol_ml * 1e-6 / (4 * _math.pi)) ** (1/3)) / dx
BALLOON_RADII_VX = [_balloon_vx(v) for v in BALLOON_VOLUMES_ML]

# Six face-normal directions — matches the smallest useful DF set Yin
# discusses. The face for each direction is where the piston plate sits.
# Convention: face index is the axis that the driver is on; sign indicates
# +/- side of the cube.
SIX_DIRECTIONS = [
    (np.array([ 1., 0, 0]), "+i (bottom)",  ("i", N - 1)),
    (np.array([-1., 0, 0]), "-i (top)",     ("i", 0)),
    (np.array([0.,  1, 0]), "+j (right)",   ("j", N - 1)),
    (np.array([0., -1, 0]), "-j (left)",    ("j", 0)),
    (np.array([0., 0,  1]), "+k (back)",    ("k", N - 1)),
    (np.array([0., 0, -1]), "-k (front)",   ("k", 0)),
]


def _closest_face(khat: np.ndarray) -> tuple[str, int]:
    """Return the cube face (axis, index) whose outward normal is most
    opposite k̂ — the physically-sensible face to place a source on so the
    wave enters the domain propagating along k̂. Ties broken by axis order."""
    # Outward normals: (axis, index, normal-vector)
    faces = [
        ("i", 0,     np.array([-1., 0, 0])),
        ("i", N - 1, np.array([ 1., 0, 0])),
        ("j", 0,     np.array([ 0.,-1, 0])),
        ("j", N - 1, np.array([ 0., 1, 0])),
        ("k", 0,     np.array([ 0., 0,-1])),
        ("k", N - 1, np.array([ 0., 0, 1])),
    ]
    # Want the face whose outward normal is most anti-parallel to k̂,
    # i.e. minimises k̂ · n̂ (most negative dot product).
    best = min(faces, key=lambda f: float(np.dot(khat, f[2])))
    return best[0], best[1]


def fibonacci_directions(n: int) -> list[tuple[np.ndarray, str, tuple[str, int]]]:
    """n unit vectors uniformly on the sphere (Fibonacci lattice) with
    source face assigned to each via `_closest_face`. Used for the 20-
    direction Yin-style DF set (or any n)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    out = []
    for i in range(n):
        z     = 1.0 - (2.0 * i + 1.0) / n
        theta = 2.0 * np.pi * i / phi
        r     = np.sqrt(max(0.0, 1.0 - z * z))
        khat  = np.array([r * np.cos(theta), r * np.sin(theta), z])
        khat  = khat / (np.linalg.norm(khat) + 1e-30)
        face  = _closest_face(khat)
        out.append((khat, f"d{i:02d} ({khat[0]:+.2f},{khat[1]:+.2f},{khat[2]:+.2f})", face))
    return out


def plate_sources(face_axis: str, face_idx: int) -> list[tuple[int, int, int, complex]]:
    """Coherent disk source on a specified face of the cube."""
    cy = (N - 1) / 2.0
    r_max = (N / 2.0) * DRIVER_R
    src = []
    for a in range(N):
        for b in range(N):
            if (a - cy) ** 2 + (b - cy) ** 2 > r_max ** 2:
                continue
            if face_axis == "i":
                src.append((face_idx, a, b, 1.0 + 0.0j))
            elif face_axis == "j":
                src.append((a, face_idx, b, 1.0 + 0.0j))
            elif face_axis == "k":
                src.append((a, b, face_idx, 1.0 + 0.0j))
    return src


def solve_one_direction(khat: np.ndarray,
                         face: tuple[str, int],
                         balloon: SphericalBalloon,
                         sigma: np.ndarray,
                         G_base: np.ndarray,
                         stiffening_exponent: float,
                         viscosity: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Returns (|u|, G_DI) for one propagation direction."""
    G_eff = effective_G_for_direction(sigma, khat, G_base,
                                       A_coeff=A_COEFF,
                                       stiffening_exponent=stiffening_exponent)
    G_eff = np.clip(G_eff, 200.0, 500000.0)
    src   = plate_sources(*face)
    u     = helmholtz_solve_3d(G_eff, freq=FREQ, rho=RHO, dx=DX,
                                damping=DAMPING, sources=src,
                                top_free=False,
                                viscosity=viscosity)
    G_DI  = direct_inversion_3d(u, freq=FREQ, rho=RHO, dx=DX)
    return np.abs(u), G_DI


def _tsm_for_state(balloon: SphericalBalloon, sigma: np.ndarray, G_base: np.ndarray,
                    G_iso: np.ndarray, args, directions) -> dict:
    """Compute (mu_conv, mu_TSM, ring stats) for one balloon state.

    Returns a dict with keys: 'mu_conv', 'mu_tsm', 'ring_conv', 'ring_tsm',
    'ring_iso'. Extracted from main() so the full-cycle driver can loop.
    """
    di_maps: list[np.ndarray]  = []
    amp_maps: list[np.ndarray] = []
    if args.method == "solves":
        for khat, name, face in directions:
            amp, G_DI = solve_one_direction(khat, face, balloon, sigma, G_base,
                                             stiffening_exponent=args.stiffening_exponent,
                                             viscosity=args.viscosity)
            di_maps.append(G_DI); amp_maps.append(amp)
    else:
        src = multi_face_broadband_sources(N, radius_frac=DRIVER_R,
                                            faces=("iN", "jN", "j0", "kN", "k0"))
        u_full = helmholtz_solve_3d(G_iso, freq=FREQ, rho=RHO, dx=DX,
                                     damping=DAMPING, sources=src,
                                     top_free=False, viscosity=args.viscosity)
        for khat, _name, _face in directions:
            u_k  = directional_filter_3d(u_full, khat=khat, angular_width=args.wedge_width)
            G_DI = direct_inversion_3d(u_k, freq=FREQ, rho=RHO, dx=DX)
            di_maps.append(G_DI); amp_maps.append(np.abs(u_k))

    di_stack  = np.stack(di_maps, axis=0)
    amp_stack = np.stack(amp_maps, axis=0)
    if args.amp_threshold > 0.0:
        per_dir_peak = amp_stack.reshape(len(directions), -1).max(axis=1)
        thresh = per_dir_peak[:, None, None, None] * args.amp_threshold
        di_stack = np.where(amp_stack >= thresh, di_stack, np.nan)

    w = amp_stack ** 2
    with np.errstate(invalid="ignore"):
        num = np.nansum(np.where(np.isnan(di_stack), 0.0, w * di_stack), axis=0)
        den = np.nansum(np.where(np.isnan(di_stack), 0.0, w),            axis=0)
        mu_conv = num / (den + 1e-30)
        mu_conv[den == 0] = np.nan
    mu_tsm = np.nanmax(di_stack, axis=0)

    shell = perilesional_shell_3d(balloon.mask(N), shell_mm=SHELL_MM, dx=DX)
    def _mean_shell(field):
        vals = field[shell]; vals = vals[np.isfinite(vals)]
        if not vals.size: return float("nan")
        lo, hi = np.percentile(vals, [10, 90])
        trimmed = vals[(vals >= lo) & (vals <= hi)]
        return float(np.mean(trimmed)) if trimmed.size else float("nan")

    return dict(
        mu_conv=mu_conv, mu_tsm=mu_tsm,
        ring_iso=_mean_shell(G_iso),
        ring_conv=_mean_shell(mu_conv),
        ring_tsm=_mean_shell(mu_tsm),
    )


def _run_full_cycle(args, save_dir, directions):
    """Loop TSM over 6 inflation + 5 deflation states, produce Fig-6 plot."""
    schedule = list(zip(PRESSURE_STATES, BALLOON_RADII_VX,
                         ["inflation"] * len(PRESSURE_STATES),
                         [f"{v} mL" for v in BALLOON_VOLUMES_ML]))
    n = len(PRESSURE_STATES)
    for i in range(n - 2, -1, -1):
        schedule.append((PRESSURE_STATES[i], BALLOON_RADII_VX[i], "deflation",
                          f"{BALLOON_VOLUMES_ML[i]} mL"))

    results = []
    for step, (p, r_vx, branch, vol_label) in enumerate(schedule):
        print(f"[{step+1:2d}/{len(schedule)}]  {vol_label:<7s} p={p:>4} Pa "
              f"r={r_vx:>4.1f}vx  [{branch}] …")
        balloon = SphericalBalloon(center=CENTER, radius_vx=r_vx, pressure=float(p))
        sigma   = stress_tensor_sphere(balloon, N)
        G_base  = np.full((N, N, N), G_BG); G_base[balloon.mask(N)] = G_LESION
        G_iso   = make_effective_G_3d(N, balloon, G_BG, G_LESION, A_COEFF,
                                       stiffening_exponent=args.stiffening_exponent,
                                       G_max_pa=500000.0)
        stats   = _tsm_for_state(balloon, sigma, G_base, G_iso, args, directions)
        results.append(dict(p=p, r_vx=r_vx, branch=branch, vol_label=vol_label,
                            **{k: v for k, v in stats.items()
                               if k not in ("mu_conv", "mu_tsm")}))
        print(f"      ring_iso={stats['ring_iso']:>7.0f} Pa  "
              f"ring_conv={stats['ring_conv']:>7.0f} Pa  "
              f"ring_tsm={stats['ring_tsm']:>7.0f} Pa")

    # ── Figure: Yin Fig 6 lookalike, μ_conv & μ_TSM per branch ─────────
    infl = [r for r in results if r["branch"] == "inflation"]
    defl = [r for r in results if r["branch"] == "deflation"]
    # x-axis: step index, but labeled by volume
    fig, ax = plt.subplots(figsize=(9, 5))
    infl_x = list(range(len(infl)))
    # Deflation states are volumes 200,150,100,50,0 mL → x = 4,3,2,1,0
    # (not 5,4,3,2,1 — deflation starts one step below the peak).
    defl_x = list(range(len(infl) - 2, len(infl) - 2 - len(defl), -1))
    ax.plot(infl_x, [r["ring_tsm"] / 1000  for r in infl], "o-",  color="tab:red",
             label="μ_TSM inflation", ms=8, lw=2)
    ax.plot(defl_x, [r["ring_tsm"] / 1000  for r in defl], "s--", color="tab:red",
             label="μ_TSM deflation", ms=8, lw=2, alpha=0.6)
    ax.plot(infl_x, [r["ring_conv"] / 1000 for r in infl], "^-",  color="tab:blue",
             label="μ_conv inflation", ms=8, lw=2)
    ax.plot(defl_x, [r["ring_conv"] / 1000 for r in defl], "v--", color="tab:blue",
             label="μ_conv deflation", ms=8, lw=2, alpha=0.6)
    ax.set_xticks(list(range(len(infl))))
    ax.set_xticklabels([r["vol_label"] for r in infl], rotation=30, ha="right")
    ax.set_xlabel("Balloon inflation state (water volume)")
    ax.set_ylabel("Perilesional G_ring [kPa]  (DI trimmed mean)")
    ax.set_title(
        f"Yin-style TSM full cycle — {args.method} + {args.num_directions} directions\n"
        f"m={args.stiffening_exponent},  A={A_COEFF},  {FREQ:.0f} Hz"
    )
    ax.grid(True, alpha=0.3); ax.legend(loc="upper left")
    plt.tight_layout()
    out_fig = save_dir / "yin_fig6_reproduction.png"
    fig.savefig(out_fig, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_fig}")

    # ── Summary table ───────────────────────────────────────────────────
    lines = [
        "Yin Fig 6 reproduction — 3D balloon full inflation + deflation cycle",
        "=" * 78,
        f"Grid: {N}³, dx={DX*1000:.0f} mm    m = {args.stiffening_exponent}    "
        f"A = {A_COEFF}",
        f"Method: {args.method} + {len(directions)} directions "
        f"(wedge σ={args.wedge_width} rad) @ {FREQ:.0f} Hz",
        f"Amplitude gate: |u| >= {args.amp_threshold*100:.0f}% of per-direction peak",
        "",
        f"{'State':<28} {'branch':<10} {'p [Pa]':>7}  "
        f"{'ring_iso':>9}  {'ring_conv':>10}  {'ring_TSM':>9}  {'TSM/conv':>9}",
        "-" * 100,
    ]
    for r in results:
        ratio = r["ring_tsm"] / (r["ring_conv"] + 1e-9) if r["ring_conv"] > 0 else float("nan")
        lines.append(
            f"{r['vol_label']:<28} {r['branch']:<10} {r['p']:>7}  "
            f"{r['ring_iso']:>9.0f}  {r['ring_conv']:>10.0f}  {r['ring_tsm']:>9.0f}  "
            f"{ratio:>9.2f}"
        )
    lines += [
        "-" * 100,
        "",
        "Compare to Yin Fig 6 (Phantom 1 TSM, kPa): 3.5, 3.8, 3.9, 4.2, 4.4",
        "                              (Phantom 1 conv, kPa): 2.7, 2.7, 2.8, 2.8, 2.8",
    ]
    (save_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="paper_demo_3d_tsm",
                        help="Result subdir under tsm_fno/results/")
    parser.add_argument("--pressure", type=float, default=3000.0,
                        help="Balloon pressure [Pa] for this single-state demo.")
    parser.add_argument("--radius-vx", type=float, default=11.0,
                        help="Balloon radius [voxels] (matches ~150 mL inflation).")
    parser.add_argument("--stiffening-exponent", "-m", type=float, default=1.0)
    parser.add_argument("--constitutive", choices=("powerlaw", "ogden"),
                        default="powerlaw",
                        help="Acoustoelastic constitutive law. Passed through "
                             "to make_effective_G_3d.")
    parser.add_argument("--viscosity", type=float, default=None,
                        help="Kelvin-Voigt viscosity η [Pa·s]. If set, damping "
                             "grows linearly with ω (frequency-dependent).")
    parser.add_argument("--amp-threshold", type=float, default=0.15,
                        help="Amplitude cutoff (as fraction of peak |u| per "
                             "direction). Voxels below this are masked to NaN "
                             "before MIP / amp-weighted combining. Matches Yin's "
                             "semi-automatic amplitude gate. Default 0.15; "
                             "set 0.0 to disable.")
    parser.add_argument("--num-directions", type=int, default=6,
                        choices=(6, 20),
                        help="Number of propagation directions to sweep. 6 = "
                             "cube face normals (±i,±j,±k). 20 = Fibonacci-"
                             "sphere sampling matching Yin's 20-direction 3D "
                             "directional filter set. 20 gives finer angular "
                             "coverage → stronger tangential-stiffening TSM "
                             "signal but ~3–4× more compute per state.")
    parser.add_argument("--method", choices=("solves", "filter"), default="solves",
                        help="TSM combining method. 'solves' (default): N "
                             "independent solves, each with a direction-"
                             "dependent G_eff, then MIP. 'filter': ONE solve "
                             "with broadband multi-face sources on the "
                             "direction-averaged G_eff, then apply k-space "
                             "wedge filter to isolate each direction — Yin's "
                             "actual algorithm. Much faster (1 solve vs N).")
    parser.add_argument("--wedge-width", type=float, default=0.35,
                        help="Angular σ (radians in sin(θ) space) for the "
                             "directional filter wedge when --method filter. "
                             "0.35 ≈ 20° FWHM.")
    parser.add_argument("--full-cycle", action="store_true",
                        help="Run the full 11-state inflation + deflation "
                             "schedule (matches paper_phantom_demo_3d.py) "
                             "and produce a Yin Fig-6-style plot with μ_conv "
                             "and μ_TSM curves for both branches. Overrides "
                             "--pressure / --radius-vx.")
    args = parser.parse_args()

    # Auto-suffix so different runs don't clobber each other.
    out_name = args.out
    if out_name == "paper_demo_3d_tsm":
        parts = []
        if args.num_directions != 6: parts.append(f"dir{args.num_directions}")
        if args.method != "solves":  parts.append(args.method)
        if args.full_cycle:          parts.append("cycle")
        if parts: out_name = "paper_demo_3d_tsm_" + "_".join(parts)
    save_dir = ROOT / "results" / out_name
    save_dir.mkdir(parents=True, exist_ok=True)

    directions = (SIX_DIRECTIONS if args.num_directions == 6
                  else fibonacci_directions(args.num_directions))

    # Dispatch to full-cycle branch — returns before the single-state body runs.
    if args.full_cycle:
        _run_full_cycle(args, save_dir, directions)
        return

    balloon = SphericalBalloon(center=CENTER, radius_vx=args.radius_vx,
                                pressure=args.pressure)
    sigma   = stress_tensor_sphere(balloon, N)

    # G_base = intrinsic (no acoustoelastic) — the tensor field encodes σ.
    G_base = np.full((N, N, N), G_BG, dtype=np.float64)
    G_base[balloon.mask(N)] = G_LESION

    # For reference: the isotropic-scalar effective G (what the memoryless
    # demo uses) — this is the "everyone-agrees" baseline for μ_conv.
    G_iso = make_effective_G_3d(N, balloon, G_BG, G_LESION, A_COEFF,
                                 stiffening_exponent=args.stiffening_exponent,
                                 G_max_pa=500000.0,
                                 constitutive=args.constitutive)

    # Direction-specific G_DI and |u| stacks.
    di_maps = []
    amp_maps = []

    if args.method == "solves":
        # N independent solves, each with a direction-dependent G_eff.
        for khat, name, face in directions:
            print(f"solving direction {name}  k̂={khat.tolist()}")
            amp, G_DI = solve_one_direction(khat, face, balloon, sigma, G_base,
                                             stiffening_exponent=args.stiffening_exponent,
                                             viscosity=args.viscosity)
            di_maps.append(G_DI)
            amp_maps.append(amp)
    else:
        # method == "filter": ONE solve on the isotropic-average G_eff with
        # broadband multi-face sources, then k-space directional filter to
        # isolate each k̂ component before DI. This is Yin's algorithm.
        print("solving ONE broadband multi-face source on isotropic-avg G_eff …")
        src = multi_face_broadband_sources(N, radius_frac=DRIVER_R,
                                            faces=("iN", "jN", "j0", "kN", "k0"))
        u_full = helmholtz_solve_3d(G_iso, freq=FREQ, rho=RHO, dx=DX,
                                     damping=DAMPING, sources=src,
                                     top_free=False, viscosity=args.viscosity)
        print(f"  |u_full|max = {np.max(np.abs(u_full)):.3f}")
        for khat, name, _face in directions:
            u_k = directional_filter_3d(u_full, khat=khat,
                                         angular_width=args.wedge_width)
            G_DI = direct_inversion_3d(u_k, freq=FREQ, rho=RHO, dx=DX)
            print(f"filtered {name}  |u_k|max = {np.max(np.abs(u_k)):.4f}")
            di_maps.append(G_DI)
            amp_maps.append(np.abs(u_k))

    di_stack  = np.stack(di_maps,  axis=0)   # (6, N, N, N)
    amp_stack = np.stack(amp_maps, axis=0)

    # Amplitude gate: mask G_DI to NaN where |u| < threshold * per-direction
    # peak. Matches Yin: "amplitude cutoff was applied prior to MIP to
    # exclude unreliable inversions in low-amplitude regions".
    if args.amp_threshold > 0.0:
        per_dir_peak = amp_stack.reshape(len(directions), -1).max(axis=1)
        thresh = per_dir_peak[:, None, None, None] * args.amp_threshold
        di_stack = np.where(amp_stack >= thresh, di_stack, np.nan)

    # μ_conv = amplitude-weighted average (Yin's "conventional" combiner).
    w = amp_stack ** 2
    with np.errstate(invalid="ignore"):
        # nansum with amp-weighting: skip NaN voxels per direction.
        num = np.nansum(np.where(np.isnan(di_stack), 0.0, w * di_stack), axis=0)
        den = np.nansum(np.where(np.isnan(di_stack), 0.0, w),            axis=0)
        mu_conv = num / (den + 1e-30)
        mu_conv[den == 0] = np.nan
    # μ_TSM = voxelwise max across directions (Yin's MIP TSM combiner).
    mu_tsm = np.nanmax(di_stack, axis=0)

    # Ring statistics.
    shell = perilesional_shell_3d(balloon.mask(N), shell_mm=SHELL_MM, dx=DX)

    def _mean_shell(field):
        vals = field[shell]
        vals = vals[np.isfinite(vals)]
        lo, hi = np.percentile(vals, [10, 90]) if vals.size else (0, 0)
        trimmed = vals[(vals >= lo) & (vals <= hi)]
        return float(np.mean(trimmed)) if trimmed.size else float("nan")

    ring_iso  = _mean_shell(G_iso)
    ring_conv = _mean_shell(mu_conv)
    ring_tsm  = _mean_shell(mu_tsm)

    print("\n─── Perilesional shell mean stiffness ───")
    print(f"  G_true (isotropic acoustoelastic)  = {ring_iso:>8.0f} Pa")
    print(f"  μ_conv (amplitude-weighted mean)   = {ring_conv:>8.0f} Pa")
    print(f"  μ_TSM  (MIP over 6 directions)     = {ring_tsm:>8.0f} Pa")
    print(f"  TSM / conv ratio (Yin's key signal)= {ring_tsm/ring_conv:>8.2f}")

    # ── Figure: mid-slice comparison ────────────────────────────────────────
    mid = N // 2
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    vmin_iso, vmax_iso = np.percentile(G_iso[mid], [5, 99])
    vmin_tsm = min(vmin_iso, np.nanpercentile(mu_tsm[mid], 5))
    vmax_tsm = max(vmax_iso, np.nanpercentile(mu_tsm[mid], 99))
    for ax, field, title in [
        (axes[0, 0], G_iso[mid],   "G_true  (isotropic acoustoelastic)"),
        (axes[0, 1], mu_conv[mid], "μ_conv  (amp-weighted mean)"),
        (axes[0, 2], mu_tsm[mid],  "μ_TSM   (MIP over 6 directions)"),
    ]:
        im = ax.imshow(field, cmap="hot", vmin=vmin_tsm, vmax=vmax_tsm)
        ax.axis("off"); ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Per-direction G_DI mid-slice for a spread of directions.
    show_idx = [0, len(directions) // 2, len(directions) - 1]
    for col, dir_idx in enumerate(show_idx):
        khat, name, _ = directions[dir_idx]
        ax = axes[1, col]
        di_mid = di_maps[dir_idx][mid]
        finite = di_mid[np.isfinite(di_mid)]
        vmn, vmx = np.percentile(finite, [5, 99]) if finite.size else (0, 1)
        im = ax.imshow(di_mid, cmap="hot", vmin=vmn, vmax=vmx)
        ax.axis("off"); ax.set_title(f"G_DI along {name}", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(
        f"Yin-style TSM pipeline — spherical balloon @ p={args.pressure:.0f} Pa, "
        f"m={args.stiffening_exponent}\n"
        f"Top: isotropic reference vs μ_conv vs μ_TSM (ring should live in μ_TSM only). "
        f"Bottom: 3 of 6 direction-specific DI maps.",
        fontsize=10,
    )
    plt.tight_layout()
    out_fig = save_dir / "tsm_comparison.png"
    fig.savefig(out_fig, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_fig}")

    lines = [
        "3D TSM Pipeline Demo — spherical balloon @ single inflation state",
        "=" * 78,
        f"Grid: {N}³, dx={DX*1000:.0f} mm    p = {args.pressure} Pa    "
        f"r_vx = {args.radius_vx}    m = {args.stiffening_exponent}",
        f"Frequency: {FREQ:.0f} Hz    η = {args.viscosity or 0} Pa·s "
        f"(damping ξ = {DAMPING})",
        f"Directions: {len(directions)} " + (
            "face normals (±i, ±j, ±k)" if args.num_directions == 6
            else f"Fibonacci-lattice unit vectors (Yin-style {args.num_directions}-direction DF)"),
        f"Method: {args.method}" + (" (N solves with direction-dependent G_eff)"
                                    if args.method == "solves"
                                    else f" (1 broadband solve + k-space wedge filter, σ={args.wedge_width})"),
        f"Amplitude gate: |u| >= {args.amp_threshold*100:.0f}% of per-direction "
        f"peak (Yin-style)",
        "",
        f"{'Field':<40} {'ring mean (Pa)':>18}",
        "-" * 62,
        f"{'G_true (isotropic acoustoelastic)':<40} {ring_iso:>18.0f}",
        f"{'μ_conv (amplitude-weighted mean)':<40} {ring_conv:>18.0f}",
        f"{'μ_TSM  (MIP over directions)':<40} {ring_tsm:>18.0f}",
        "-" * 62,
        f"{'TSM / conv ratio':<40} {ring_tsm/ring_conv:>18.2f}",
        "",
        "Notes:",
        "  - Radial-propagation directions (into the balloon) sense the",
        "    softened radial pre-stress → lower G_DI along those axes.",
        "  - Tangential-propagation directions sense the stiffened",
        "    circumferential pre-stress → higher G_DI in the perilesional",
        "    shell. MIP over directions picks up those high values.",
        "  - μ_conv (weighted mean) averages away most of the anisotropy",
        "    signal, matching what conventional MRE inversion reports.",
        "  - This is the essence of Yin's TSM signature: the ring exists",
        "    ONLY when you combine direction-specific inversions with MIP.",
    ]
    (save_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
