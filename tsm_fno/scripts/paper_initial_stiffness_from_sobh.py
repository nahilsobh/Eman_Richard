#!/usr/bin/env python3
"""Initial stiffness field G(x) for P1 and P2 at 250 mL inflation.

Builds the perilesional shear modulus map µ_app,θθ(x) implied by the
Sobh–Ehman finite-deformation model (paper_lesion_expansion.pdf) with
mean-of-band material constants, on a 3D grid representing Yin's
15 × 15 × 18 cm container.  Serves as the INITIAL CONDITION (initial
stiffness field) for downstream wave-propagation / TSM simulations of
the balloon-inflated phantoms.

Constitutive laws (eqs 6 & 8 of the paper)
-------------------------------------------
P1 (plain gelatin, neo-Hookean):
    µ_app,θθ = µ · λ_θ²                                       (eq 6)

P2 (fiber-reinforced, matrix + exponential hoop-strain fiber):
    ε = λ_θ − 1
    µ_app,θθ = µ · λ_θ² + 2 k₁ ε exp(k₂ ε²) · 4(λ_θ − λ_θ⁻⁵) · λ_θ²
                                                             (eq 8)

Mean-of-band constants (Table 2):
    µ  = 2.75 kPa   (mean of [2.5, 3.0])
    k₁ = 3.25 kPa   (mean of [1.5, 5.0])     — P2 only
    k₂ = 1.25       (mean of [0.5, 2.0])     — P2 only

Kinematics (eqs 1–2)
--------------------
Incompressible spherical cavity expansion from A = 2.285 cm (50 mL cast
state) to a = 3.91 cm (250 mL peak):

    R(r)  = (r³ − a³ + A³)^(1/3)     — reference radius mapping
    λ_θ   = r / R                     — tangential stretch
    λ_r   = 1 / λ_θ²                  — radial stretch (incompressible)

The paper's §2 argument (top of p.3): for an OPEN-TOPPED container that
permits inflation (as here: fixed 4 sides + bottom, traction-free top),
the infinite-domain Lamé map is EXACT — the finite container geometry
does not scale the prestrain field, only permits or forbids inflation.

Container BCs (documented in the output; not applied to the stretch
field per the paper's own reasoning):
    x = ±L/2, y = ±L/2, z = −H/2  : u = 0 (rigid walls, gel clamped)
    z = +H/2                      : σ·n = 0 (free surface, escape path)

Output
------
- G_P1(x), G_P2(x) : 3D scalar fields on the grid                    [Pa]
- λ_θ(x)           : scalar tangential stretch field                 [-]
- Midplane PNG showing G_P1 vs G_P2 side by side
- Shell-averaged values comparable to the paper's Table 3 250-mL row
- .npz snapshot for downstream forward-model consumption
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


# ── Container geometry (Yin 15 × 15 × 18 cm) ────────────────────────
CONTAINER_L_M = 0.15
CONTAINER_H_M = 0.18

# ── Balloon cavity kinematics ───────────────────────────────────────
A_REF_M = 0.02285      # 50 mL reference cavity radius (paper §2)
V_INJECTION_ML = 250   # target injected volume
_a = ((3 * V_INJECTION_ML * 1e-6) / (4 * math.pi)) ** (1/3)
A_INFL_M = _a           # 250 mL current cavity radius (≈ 3.91 cm)

# ── Mean-of-band material constants (Table 2) ──────────────────────
MU_MEAN   = 2750.0     # Pa  matrix shear modulus (P1 and P2 matrix)
K1_MEAN   = 3250.0     # Pa  fiber reinforcement scaling (P2 only)
K2_MEAN   = 1.25       # –   nonlinear stiffening rate (P2 only)

# ── Grid resolution ─────────────────────────────────────────────────
DX_M = 0.003          # 3 mm ≈ Yin acquisition voxel
NX = int(round(CONTAINER_L_M / DX_M))
NY = int(round(CONTAINER_L_M / DX_M))
NZ = int(round(CONTAINER_H_M / DX_M))


def _lame_stretch(r_m: np.ndarray, a: float, A: float) -> np.ndarray:
    """Return λ_θ(r) via incompressible cavity expansion (eqs 1–2)."""
    r_safe = np.where(r_m < 1e-12, 1e-12, r_m)
    R = np.cbrt(r_safe ** 3 - a ** 3 + A ** 3)
    lam_theta = r_safe / R
    return lam_theta


def _mu_P1(lam_theta: np.ndarray) -> np.ndarray:
    """P1 neo-Hookean acoustoelastic modulus (eq 6)."""
    return MU_MEAN * lam_theta ** 2


def _mu_P2(lam_theta: np.ndarray) -> np.ndarray:
    """P2 matrix + fiber acoustoelastic modulus (paper eq 8).

    Derived rigorously (verified by sympy against Ogden's µ_ij formula):
        µ_fiber = k₁·λ⁷·(λ−1)·exp(k₂(λ−1)²) / [2·(λ⁶ − 1)]

    Limit λ → 1: k₁/12 (matches the paper's underbraced annotation).
    Equivalent form (paper text): k₁·ε·exp(k₂ε²)·λ² / [2·(λ − λ⁻⁵)].
    """
    eps = lam_theta - 1.0
    matrix_term = MU_MEAN * lam_theta ** 2
    lam6_m1 = lam_theta ** 6 - 1.0
    # avoid division by zero at λ = 1 exactly (use L'Hôpital: → k1/12)
    safe = np.where(np.abs(lam6_m1) < 1e-12, 1e-12, lam6_m1)
    fiber_term = (K1_MEAN * lam_theta ** 7 * eps
                    * np.exp(K2_MEAN * eps ** 2) / (2.0 * safe))
    # Explicit fix at λ=1: μ_fiber = k₁/12
    fiber_term = np.where(np.abs(lam_theta - 1.0) < 1e-8,
                           K1_MEAN / 12.0, fiber_term)
    return matrix_term + fiber_term


def build_fields():
    """Build 3D fields on the container grid; balloon centred, z-axis vertical."""
    # Container-centred coordinates
    x = (np.arange(NX) - (NX - 1) / 2) * DX_M
    y = (np.arange(NY) - (NY - 1) / 2) * DX_M
    z = (np.arange(NZ) - (NZ - 1) / 2) * DX_M
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)

    # Boolean masks
    in_balloon = r < A_INFL_M
    in_gel = ~in_balloon

    lam_theta = np.ones_like(r)
    lam_theta[in_gel] = _lame_stretch(r[in_gel], A_INFL_M, A_REF_M)

    # Gel-region stiffness fields
    G_P1 = np.zeros_like(r)
    G_P2 = np.zeros_like(r)
    G_P1[in_gel] = _mu_P1(lam_theta[in_gel])
    G_P2[in_gel] = _mu_P2(lam_theta[in_gel])

    # Balloon interior: not gel material — mark with 0 Pa (or leave for
    # downstream code to treat as water/void). Rationale: initial-stiffness
    # field represents the tissue phase only; interior is water.
    G_P1[in_balloon] = 0.0
    G_P2[in_balloon] = 0.0
    lam_theta[in_balloon] = np.nan

    return {
        "coords": {"x": x, "y": y, "z": z},
        "r": r,
        "lam_theta": lam_theta,
        "G_P1": G_P1,
        "G_P2": G_P2,
        "in_gel": in_gel,
        "in_balloon": in_balloon,
    }


def shell_average(G, r, in_gel, r_in, r_out):
    """Shell-averaged G (eq 9): 3/(r_out³−r_in³) ∫_{r_in}^{r_out} G(r) r² dr,
    approximated by simple mean of gel voxels with r ∈ [r_in, r_out]."""
    mask = in_gel & (r >= r_in) & (r <= r_out)
    if not mask.any():
        return float("nan"), 0
    return float(G[mask].mean()) / 1000.0, int(mask.sum())   # kPa, count


def main():
    out_dir = ROOT / "results" / "paper_initial_stiffness_sobh"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Grid: {NX}×{NY}×{NZ} at dx={DX_M*1000:.1f} mm "
          f"→ container {NX*DX_M*100:.1f}×{NY*DX_M*100:.1f}×{NZ*DX_M*100:.1f} cm")
    print(f"Cavity: A={A_REF_M*100:.3f} cm (50 mL), "
          f"a={A_INFL_M*100:.3f} cm ({V_INJECTION_ML} mL)")
    print(f"Mean-of-band constants: µ={MU_MEAN} Pa, k1={K1_MEAN} Pa, k2={K2_MEAN}")
    print()

    fields = build_fields()

    # Shell-averaged G to compare with paper Table 3 at 250 mL
    r_in  = A_INFL_M
    r_out = A_INFL_M + 0.012
    G_P1_shell, n_p1 = shell_average(fields["G_P1"], fields["r"], fields["in_gel"], r_in, r_out)
    G_P2_shell, n_p2 = shell_average(fields["G_P2"], fields["r"], fields["in_gel"], r_in, r_out)

    # Central-axis radial profile (for sanity, matching paper Fig 1 bottom)
    center = (NX // 2, NY // 2, NZ // 2)
    r_line = fields["coords"]["x"][NX // 2:]
    lam_line = fields["lam_theta"][NX // 2:, center[1], center[2]]
    G_P1_line = fields["G_P1"][NX // 2:, center[1], center[2]]
    G_P2_line = fields["G_P2"][NX // 2:, center[1], center[2]]

    # ── Save numpy snapshot ──────────────────────────────────────────
    np.savez_compressed(
        out_dir / "initial_stiffness.npz",
        G_P1=fields["G_P1"],
        G_P2=fields["G_P2"],
        lam_theta=fields["lam_theta"],
        x=fields["coords"]["x"], y=fields["coords"]["y"], z=fields["coords"]["z"],
        dx_m=DX_M,
        container_L_m=CONTAINER_L_M, container_H_m=CONTAINER_H_M,
        A_ref_m=A_REF_M, a_infl_m=A_INFL_M, V_ml=V_INJECTION_ML,
        mu_mean_Pa=MU_MEAN, k1_mean_Pa=K1_MEAN, k2_mean=K2_MEAN,
    )
    print(f"Saved {out_dir / 'initial_stiffness.npz'}")

    # ── Midplane figure ──────────────────────────────────────────────
    z_mid = NZ // 2
    slice_p1 = fields["G_P1"][:, :, z_mid] / 1000.0   # kPa
    slice_p2 = fields["G_P2"][:, :, z_mid] / 1000.0
    # Mask balloon interior for display
    for s in (slice_p1, slice_p2):
        s[fields["in_balloon"][:, :, z_mid]] = np.nan

    extent = [-CONTAINER_L_M*50, CONTAINER_L_M*50,
              -CONTAINER_L_M*50, CONTAINER_L_M*50]  # cm

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
    vmax = max(np.nanmax(slice_p1), np.nanmax(slice_p2))
    im0 = axes[0].imshow(slice_p1.T, origin="lower", extent=extent,
                          vmin=MU_MEAN/1000, vmax=vmax, cmap="viridis")
    axes[0].set_title(f"P1 (neo-Hookean)\nµ={MU_MEAN/1000:.2f} kPa\n"
                       f"⟨G_shell⟩ = {G_P1_shell:.2f} kPa "
                       f"(paper band: 4.28–5.14)",
                       fontsize=11)
    plt.colorbar(im0, ax=axes[0], label="µ_app,θθ [kPa]", shrink=0.85)

    im1 = axes[1].imshow(slice_p2.T, origin="lower", extent=extent,
                          vmin=MU_MEAN/1000, vmax=vmax, cmap="viridis")
    axes[1].set_title(f"P2 (matrix + fiber)\nµ={MU_MEAN/1000:.2f}, "
                       f"k₁={K1_MEAN/1000:.2f} kPa, k₂={K2_MEAN}\n"
                       f"⟨G_shell⟩ = {G_P2_shell:.2f} kPa "
                       f"(paper band: 4.70–6.94)",
                       fontsize=11)
    plt.colorbar(im1, ax=axes[1], label="µ_app,θθ [kPa]", shrink=0.85)

    for ax in axes:
        # draw balloon outline (in cm)
        ang = np.linspace(0, 2 * np.pi, 200)
        ax.plot(A_INFL_M * 100 * np.cos(ang), A_INFL_M * 100 * np.sin(ang),
                 "w--", lw=1.2)
        # draw 12 mm shell outer edge
        ax.plot((A_INFL_M + 0.012) * 100 * np.cos(ang),
                (A_INFL_M + 0.012) * 100 * np.sin(ang),
                 "w:", lw=0.9)
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")
        ax.set_aspect("equal")

    plt.suptitle(f"Initial stiffness field at 250 mL — Sobh-Ehman "
                 f"mean-of-band material constants\n"
                 f"container 15×15×18 cm, 5 rigid walls + free top "
                 f"(BCs permit inflation → Lamé exact)",
                 fontsize=11, fontweight="bold")
    fig.savefig(out_dir / "initial_stiffness_midplane.png", dpi=140,
                 bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'initial_stiffness_midplane.png'}")

    # ── Radial profile figure ────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.2))
    r_cm = r_line * 100
    # only plot from r > 0
    mask_plot = np.isfinite(lam_line) & (r_cm > 0)
    ax.plot(r_cm[mask_plot], G_P1_line[mask_plot] / 1000, "o-",
             color="tab:red", ms=4, lw=1.8,
             label=f"P1 (neo-Hookean)")
    ax.plot(r_cm[mask_plot], G_P2_line[mask_plot] / 1000, "s-",
             color="tab:blue", ms=4, lw=1.8,
             label=f"P2 (matrix + fiber)")
    ax.axvline(A_INFL_M * 100, color="k", ls="--", alpha=0.4,
                label=f"cavity edge a = {A_INFL_M*100:.2f} cm")
    ax.axvline((A_INFL_M + 0.012) * 100, color="gray", ls=":", alpha=0.6,
                label="shell outer r_out = a + 12 mm")
    ax.axhline(MU_MEAN / 1000, color="tab:red", ls=":", alpha=0.4,
                label="µ (unstretched)")
    ax.set_xlabel("Radial distance from balloon center [cm]", fontsize=11)
    ax.set_ylabel("µ_app,θθ [kPa]", fontsize=11)
    ax.set_title("Initial stiffness radial profile at 250 mL "
                  "(central-axis line)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(0, CONTAINER_L_M * 100 / 2)
    fig.savefig(out_dir / "initial_stiffness_radial.png", dpi=140,
                 bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'initial_stiffness_radial.png'}")

    # ── Text summary ─────────────────────────────────────────────────
    lines = [
        "Initial stiffness field at 250 mL — Sobh-Ehman mean-of-band",
        "=" * 72,
        f"Grid:      {NX}×{NY}×{NZ} hex at dx = {DX_M*1000:.1f} mm",
        f"Container: {NX*DX_M*100:.1f}×{NY*DX_M*100:.1f}×{NZ*DX_M*100:.1f} cm",
        f"Cavity:    A_ref = {A_REF_M*100:.3f} cm  (50 mL cast state)",
        f"           a_infl = {A_INFL_M*100:.3f} cm  ({V_INJECTION_ML} mL)",
        "",
        "Container BCs (per paper §2 argument, they only PERMIT/FORBID",
        "inflation — they do not modify the Lamé prestrain field):",
        "  x = ±L/2, y = ±L/2, z = −H/2  →  u = 0     (rigid walls)",
        "  z = +H/2                       →  σ · n = 0 (traction-free top)",
        "",
        "Mean-of-band material constants (Table 2):",
        f"  µ (matrix, both phantoms) = {MU_MEAN:.0f} Pa   (mean of [2.5, 3.0] kPa)",
        f"  k₁ (fiber, P2 only)       = {K1_MEAN:.0f} Pa   (mean of [1.5, 5.0] kPa)",
        f"  k₂ (nonlinear, P2 only)   = {K2_MEAN}      (mean of [0.5, 2.0])",
        "",
        "Shell-averaged µ_app,θθ (r ∈ [a, a+12 mm]) at 250 mL:",
        f"  P1: ⟨G⟩ = {G_P1_shell:.3f} kPa   ({n_p1} voxels)   "
        f"paper band 4.28–5.14  ({'inside' if 4.28 <= G_P1_shell <= 5.14 else 'outside'})",
        f"  P2: ⟨G⟩ = {G_P2_shell:.3f} kPa   ({n_p2} voxels)   "
        f"paper band 4.70–6.94  ({'inside' if 4.70 <= G_P2_shell <= 6.94 else 'outside'})",
        "",
        "Cavity-edge stretches (analytical):",
        f"  λ_θ at r = a       = {A_INFL_M/A_REF_M:.3f}   (paper Table 1: 1.710)",
        f"  λ_θ at r = a+12 mm = "
        f"{(A_INFL_M+0.012)/np.cbrt((A_INFL_M+0.012)**3 - A_INFL_M**3 + A_REF_M**3):.3f}   "
        f"(paper Table 1: 1.159)",
        "",
        "Files:",
        "  initial_stiffness.npz            — G_P1, G_P2, λ_θ 3D arrays",
        "  initial_stiffness_midplane.png   — G_P1 vs G_P2 z-midplane heatmaps",
        "  initial_stiffness_radial.png     — radial profile along central axis",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
