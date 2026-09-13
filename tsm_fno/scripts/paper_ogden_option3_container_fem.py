#!/usr/bin/env python3
"""Option 3: FEM baseline for the container-confined balloon problem.

Directly solves the static elastostatic BVP inside Yin's actual
15×15×18 cm container (5 fixed walls + free top), for the balloon
sitting at the origin. Extracts ring-mean radial displacement, compares
to analytical infinite-matrix Lamé, applies Ogden ex-post, reports
ring-mean G_θ.

This replaces the earlier analytically-derived confinement factor
cf=1.174 (applied as a scalar multiplier on λ_θ) with the actual
displacement field from a numerical solve. It answers: does container
confinement AMPLIFY or REDUCE the ring stretch?

Method:
  1. Small-inflation FDM solve inside Yin's container.
  2. Compare ring-mean u_r_container against analytical Lamé u_r_infinite
     (which we already have from earlier scripts).
  3. Container-effect ratio R = u_r_container / u_r_infinite.
  4. Apply Ogden with rescaled stretch:
        λ_θ_container = 1 + R · (λ_θ_Lamé − 1)
  5. Ring-mean using the same 5 mm shell + 3 mm inner offset as prior scripts.

Compares against pure infinite-matrix Ogden (paper_ogden_only_baseline.py)
and Yin's Fig 6 (as reference, NOT ground truth).

Limitations:
- Small-strain linear elasticity used for the container-effect factor R;
  finite-strain container effect could differ by O(strain²).
- Free-top BC approximated as mirror (∂u/∂z = 0), not exact σ·n=0.
- No hyperelastic/nonlinear FEM (would need FEniCS + Newton).
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

from src.solver.elastostatic_fd_3d import elastostatic_solve_3d
from src.phantom.geometry_3d import perilesional_shell_3d


# ── Grid & container geometry ────────────────────────────────────────
DX_M = 0.006  # 6 mm voxels → tractable solve
NX = NY = 25  # 25 × 6mm = 15.0 cm
NZ = 30       # 30 × 6mm = 18.0 cm
CENTER = (NX // 2, NY // 2, NZ // 2)
CONTAINER_L_CM = NX * DX_M * 100
CONTAINER_H_CM = NZ * DX_M * 100

# ── Balloon volumes ─────────────────────────────────────────────────
BALLOON_VOLUMES_ML = [50, 100, 150, 200, 250]

def _r_vx(vol_ml):
    return ((3 * vol_ml * 1e-6 / (4 * math.pi)) ** (1 / 3)) / DX_M

BALLOON_RADII_VX = [_r_vx(v) for v in BALLOON_VOLUMES_ML]
A0_VX = BALLOON_RADII_VX[0]

# ── Ogden parameters ────────────────────────────────────────────────
PHANTOMS = [
    dict(name="Phantom 1 — 10% bovine gelatin",
         mu=[1800.0, 700.0], alpha=[2.5, 3.0],
         yin_tsm=[3.5, 3.8, 3.9, 4.2, 4.4]),
    dict(name="Phantom 2 — 8% gel + 7% cellulose",
         mu=[2500.0, 1500.0], alpha=[2.5, 5.0],
         yin_tsm=[3.5, 3.9, 4.2, 4.9, 5.15]),
]


def _make_balloon_dirichlet(a0_vx, du_r_m):
    """Internal Dirichlet BC on balloon interior: u = (r/a₀)·du_r·r̂."""
    dirichlet = []
    for i in range(NX):
        for j in range(NY):
            for k in range(NZ):
                dxv = i - CENTER[0]
                dyv = j - CENTER[1]
                dzv = k - CENTER[2]
                r_vx = math.sqrt(dxv * dxv + dyv * dyv + dzv * dzv)
                if 0 < r_vx <= a0_vx:
                    scale = (r_vx / a0_vx) * du_r_m / r_vx
                    dirichlet.append((i, j, k, 0, scale * dxv))
                    dirichlet.append((i, j, k, 1, scale * dyv))
                    dirichlet.append((i, j, k, 2, scale * dzv))
    return dirichlet


def _balloon_and_shell(a0_vx):
    balloon_mask = np.zeros((NX, NY, NZ), dtype=bool)
    for i in range(NX):
        for j in range(NY):
            for k in range(NZ):
                dxv = i - CENTER[0]; dyv = j - CENTER[1]; dzv = k - CENTER[2]
                if math.sqrt(dxv * dxv + dyv * dyv + dzv * dzv) <= a0_vx:
                    balloon_mask[i, j, k] = True
    shell = perilesional_shell_3d(balloon_mask, shell_mm=5.0, dx=DX_M,
                                    inner_offset_mm=3.0)
    return balloon_mask, shell


def _ring_ur_mean(u_vec, shell):
    ur = []
    for i, j, k in zip(*np.where(shell)):
        dxv = i - CENTER[0]; dyv = j - CENTER[1]; dzv = k - CENTER[2]
        r_vx = math.sqrt(dxv * dxv + dyv * dyv + dzv * dzv)
        if r_vx < 1e-6:
            continue
        r_hat = np.array([dxv, dyv, dzv]) / r_vx
        ur.append(float(np.dot(u_vec[i, j, k, :], r_hat)))
    if not ur:
        return float("nan")
    v = np.array(ur)
    lo, hi = np.percentile(v, [10, 90])
    return float(v[(v >= lo) & (v <= hi)].mean())


def analytical_lame_ur_ring(a0_vx, a_vx, shell):
    """Ring-mean u_r at balloon size a_vx from infinite-matrix Lamé."""
    a_m  = a_vx  * DX_M
    a0_m = a0_vx * DX_M
    ur = []
    for i, j, k in zip(*np.where(shell)):
        dxv = i - CENTER[0]; dyv = j - CENTER[1]; dzv = k - CENTER[2]
        r_vx = math.sqrt(dxv * dxv + dyv * dyv + dzv * dzv)
        r_m = r_vx * DX_M
        R = (r_m ** 3 - a0_m ** 3 + a_m ** 3) ** (1.0 / 3.0)
        ur.append(R - r_m)
    v = np.array(ur)
    lo, hi = np.percentile(v, [10, 90])
    return float(v[(v >= lo) & (v <= hi)].mean())


def ogden_G_ring(a_vx, mu, alpha, R_container):
    """Ring-mean Ogden G_θ [kPa] with container-effect scaling.

    λ_θ_c = 1 + R · (λ_θ_∞ − 1),  where λ_θ_∞ = R_Lamé(r) / r.
    """
    _mask, shell = _balloon_and_shell(A0_VX)
    a_m  = a_vx  * DX_M
    a0_m = A0_VX * DX_M
    G_vals = []
    for i, j, k in zip(*np.where(shell)):
        dxv = i - CENTER[0]; dyv = j - CENTER[1]; dzv = k - CENTER[2]
        r_vx = math.sqrt(dxv * dxv + dyv * dyv + dzv * dzv)
        r_m = r_vx * DX_M
        R_lame = (r_m ** 3 - a0_m ** 3 + a_m ** 3) ** (1.0 / 3.0)
        lam_theta_inf = R_lame / r_m
        lam_theta_c   = 1.0 + R_container * (lam_theta_inf - 1.0)
        G_vals.append(sum(m * lam_theta_c ** (a - 2.0)
                            for m, a in zip(mu, alpha)))
    v = np.array(G_vals)
    lo, hi = np.percentile(v, [10, 90])
    return float(v[(v >= lo) & (v <= hi)].mean()) / 1000  # kPa


def main():
    out_dir = ROOT / "results" / "paper_ogden_option3_container_fem"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Container: {CONTAINER_L_CM:.1f}×{CONTAINER_L_CM:.1f}×"
          f"{CONTAINER_H_CM:.1f} cm at dx={DX_M*1000:.1f} mm "
          f"→ {NX}×{NY}×{NZ} = {NX*NY*NZ} voxels ({3*NX*NY*NZ} DOF)")
    print()

    # ── Step 1: probe container-effect ratio R via small-inflation FDM ──
    # Linear elasticity is scale-invariant, so R depends only on geometry.
    _mask, shell = _balloon_and_shell(A0_VX)
    du_probe_m = 0.001  # 1 mm probe displacement at balloon surface

    print("Running probe FDM (small-inflation)...")
    dirichlet = _make_balloon_dirichlet(A0_VX, du_probe_m)
    u_container = elastostatic_solve_3d(
        NX, NY, NZ, DX_M, mu=1000.0, lam=1.0e6,
        fixed_faces=("-x", "+x", "-y", "+y", "-z"),
        mirror_faces=("+z",),
        dirichlet=dirichlet,
    )

    ur_container = _ring_ur_mean(u_container, shell)

    # For comparison: compute u_r_infinite at same ring voxels from Lamé
    # using the same probe magnitude.
    # Small-inflation Lamé: u_r(r) ≈ (a-a₀)·a₀²/r² for r > a₀
    ur_infinite = 0.0
    n_ok = 0
    for i, j, k in zip(*np.where(shell)):
        dxv = i - CENTER[0]; dyv = j - CENTER[1]; dzv = k - CENTER[2]
        r_vx = math.sqrt(dxv * dxv + dyv * dyv + dzv * dzv)
        if r_vx < 1e-6:
            continue
        r_m = r_vx * DX_M
        a0_m = A0_VX * DX_M
        ur_infinite += du_probe_m * a0_m * a0_m / (r_m * r_m)
        n_ok += 1
    ur_infinite /= n_ok

    R_ratio = ur_container / ur_infinite
    print()
    print(f"Ring-mean u_r (probe = {du_probe_m*1000:.1f} mm at balloon surface):")
    print(f"  Container FDM:       {ur_container*1e3:.4f} mm")
    print(f"  Infinite Lamé (∼):   {ur_infinite*1e3:.4f} mm")
    print(f"  Ratio R = C/∞:       {R_ratio:.4f}")
    if R_ratio < 1:
        print(f"  → Container REDUCES ring displacement by {(1-R_ratio)*100:.1f}%")
        print(f"  → λ_θ_container < λ_θ_infinite  → LOWER G_θ")
    else:
        print(f"  → Container AMPLIFIES ring displacement by {(R_ratio-1)*100:.1f}%")
        print(f"  → λ_θ_container > λ_θ_infinite  → HIGHER G_θ")
    print()

    # ── Step 2: Compute Ogden G_θ ring curves ────────────────────────
    for p in PHANTOMS:
        p["G_inf"] = [ogden_G_ring(a, p["mu"], p["alpha"], R_container=1.0)
                      for a in BALLOON_RADII_VX]
        p["G_c"] = [ogden_G_ring(a, p["mu"], p["alpha"], R_container=R_ratio)
                    for a in BALLOON_RADII_VX]

    # ── Report ────────────────────────────────────────────────────────
    lines = [
        "Option 3 — FEM container baseline (pure Ogden + numerical geometry)",
        "=" * 76,
        f"Grid: {NX}×{NY}×{NZ}, dx={DX_M*1000:.1f} mm",
        f"Container: {CONTAINER_L_CM:.1f}×{CONTAINER_L_CM:.1f}×"
        f"{CONTAINER_H_CM:.1f} cm (target: 15×15×18)",
        f"BCs: 5 rigid walls (bottom + 4 sides), free top (mirror approx.)",
        "",
        f"Container-effect ratio R = u_r_container / u_r_infinite = {R_ratio:.4f}",
        "  (from linear-elastic small-inflation FDM; geometry-only)",
        "",
        "IMPORTANT: Yin's Fig 6 values are NOT ground truth — they are her",
        "MIP-based MRE readouts. This report shows raw Ogden physics with",
        "container geometry, with NO fitting to Yin, NO MIP-bias correction.",
        "",
    ]
    for p in PHANTOMS:
        lines.append(f"{p['name']}")
        lines.append(f"  Ogden: μ = {p['mu']} Pa, α = {p['alpha']}, "
                     f"G₀ = {sum(p['mu']):.0f} Pa")
        lines.append(f"  Volume(mL):        {'  '.join(f'{v:>5d}' for v in BALLOON_VOLUMES_ML)}")
        lines.append(f"  Yin ref:           {'  '.join(f'{v:5.2f}' for v in p['yin_tsm'])}")
        lines.append(f"  Ogden ∞-matrix:    {'  '.join(f'{v:5.2f}' for v in p['G_inf'])}")
        lines.append(f"  Ogden + container: {'  '.join(f'{v:5.2f}' for v in p['G_c'])}")
        delta_pct = [(c - i) / i * 100 for c, i in zip(p["G_c"], p["G_inf"])]
        lines.append(f"  Container Δ (%):   {'  '.join(f'{v:+5.1f}' for v in delta_pct)}")
        lines.append("")

    # ── Figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    for ax, p in zip(axes, PHANTOMS):
        ax.plot(BALLOON_VOLUMES_ML, p["yin_tsm"], "*-", color="black",
                ms=14, lw=1.5, alpha=0.55,
                label="Yin μ_TSM (measured, reference only)")
        ax.plot(BALLOON_VOLUMES_ML, p["G_inf"], "s--", color="tab:blue",
                ms=8, lw=1.8, alpha=0.85,
                label="Ogden ∞-matrix Lamé (option 1)")
        ax.plot(BALLOON_VOLUMES_ML, p["G_c"], "o-", color="tab:red",
                ms=9, lw=2.4,
                label=f"Ogden + container FEM (option 3), R={R_ratio:.3f}")

        info = (f"Ogden N=2: μ = {p['mu']} Pa\n"
                f"           α = {p['alpha']}\n"
                f"G₀ = {sum(p['mu']):.0f} Pa (from composition)\n"
                f"Container FEM: {NX}×{NY}×{NZ} at dx={DX_M*1000:.1f} mm")
        ax.text(0.02, 0.97, info, transform=ax.transAxes,
                 fontsize=8, va="top", fontfamily="monospace",
                 bbox=dict(facecolor="white", alpha=0.85, pad=4,
                           edgecolor="lightgray"))

        ax.set_title(p["name"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Balloon water volume (mL)", fontsize=11)
        ax.set_ylabel("Ring stiffness [kPa]", fontsize=11)
        ax.set_xticks(BALLOON_VOLUMES_ML)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper left")

    R_verdict = ("REDUCES" if R_ratio < 1 else "AMPLIFIES")
    plt.suptitle(
        f"Option 3 — pure Ogden + container FEM (no MIP-bias, no fitting)\n"
        f"Container-effect factor R = {R_ratio:.3f} ({R_verdict} ring stretch)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out_fig = out_dir / "ogden_option3_container_fem.png"
    fig.savefig(out_fig, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_fig}")

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
