#!/usr/bin/env python3
"""
Synthetic reproduction of Yin et al. 2026 phantom experiments.

Creates a gelatin-background phantom with a circular pressurised inclusion
at six inflation states (analogous to the paper's balloon: 0→250 mL) and
runs the trained FNO_TSM to predict G and ε_latent at each state.

Also produces an Expanding vs Control comparison matching the paper's
Phantom 2 (nonlinear gelatin, expanding) vs Phantom 3 (control, p=0).

Outputs
-------
results/paper_demo/inflation_series.png   – 6-row × 5-col inflation figure
results/paper_demo/expanding_vs_control.png – side-by-side at peak pressure
results/paper_demo/summary.txt            – quantitative table
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phantom.geometry import LesionGeometry, compute_lame_field, perilesional_shell
from src.phantom.acoustoelastic import make_effective_G, make_latent_strain
from src.solver.helmholtz_fd import solve_two_frequencies, random_sources
from src.model.fno_tsm import FNO_TSM

# ── Paper-matched phantom parameters ────────────────────────────────────────
N    = 80
DX   = 0.003          # 3 mm/voxel → 240 mm FOV  (matches paper: 80×80 matrix, 3 mm)
RHO  = 1000.0
DAMPING = 0.05

# Gelatin background: 2–3 kPa in the paper
G_BG     = 2500.0     # Pa  (mid 2–3 kPa)
G_LESION = 2000.0     # Pa  (balloon ≈ fluid, slightly softer than gel)
A_COEFF  = 5.0        # acoustoelastic constant (within training range 2–8)

# Balloon centred in the grid.
# Radius grows with volume: r = (3V/4π)^(1/3) cm, converted to voxels at dx=3mm.
# Paper states: 0, 50, 100, 150, 200, 250 mL
CX, CY = 40.0, 40.0   # centred in 80×80 grid

import math as _math
def _balloon_vx(vol_ml: float, dx: float = DX) -> float:
    """Balloon radius in voxels for a given inflation volume in mL."""
    r_m = (3 * vol_ml * 1e-6 / (4 * _math.pi)) ** (1/3)
    return r_m / dx

BALLOON_VOLUMES_ML = [0,   50,   100,   150,   200,   250]
# Pressure scales roughly linearly with balloon over-pressure
PRESSURE_STATES    = [0, 1000,  2000,  3000,  5000,  7000]   # Pa
STATE_LABELS       = [
    "Baseline (0 mL)",
    "+50 mL  (r≈23mm, p=1kPa)",
    "+100 mL (r≈29mm, p=2kPa)",
    "+150 mL (r≈33mm, p=3kPa)",
    "+200 mL (r≈36mm, p=5kPa)",
    "+250 mL (r≈39mm, p=7kPa)",
]
# Balloon radius in voxels at each state (0 mL → use 5 vx baseline)
BALLOON_RADII_VX = [5.0] + [_balloon_vx(v) for v in BALLOON_VOLUMES_ML[1:]]

SHELL_MM = 5.0        # perilesional shell thickness (matches training)
RNG_SEED = 7          # fixed seed → reproducible wave fields


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_geometry(pressure: float, radius_vx: float) -> LesionGeometry:
    return LesionGeometry(
        center=(CX, CY),
        semi_axes=(radius_vx, radius_vx),
        angle=0.0,
        pressure=float(pressure),
    )


def build_input(geom: LesionGeometry, rng: np.random.Generator) -> np.ndarray:
    """Assemble X (6, N, N) for one phantom state."""
    G_eff = make_effective_G(N, DX, geom, G_BG, G_LESION, A_COEFF)
    eps   = make_latent_strain(N, DX, geom, G_BG)

    sources = random_sources(N, rng)
    u60, u120 = solve_two_frequencies(
        G_eff, freq1=60.0, freq2=120.0,
        rho=RHO, dx=DX, damping=DAMPING, sources=sources,
    )
    u_max = max(float(np.max(np.abs(u60))), float(np.max(np.abs(u120))), 1e-12)
    u60  /= u_max
    u120 /= u_max

    lame_prior = eps.copy()
    dist = geom.distance_field(N, DX)
    dist_norm = dist / (dist.max() + 1e-12)

    X = np.stack([
        u60.real.astype(np.float32),
        u60.imag.astype(np.float32),
        u120.real.astype(np.float32),
        u120.imag.astype(np.float32),
        lame_prior.astype(np.float32),
        dist_norm.astype(np.float32),
    ], axis=0)
    return X, G_eff, eps


def load_model(run_dir: Path, device: torch.device) -> FNO_TSM:
    cfg_path  = run_dir / "args.json"
    ckpt_path = run_dir / "best.pt"
    cfg = json.load(open(cfg_path)) if cfg_path.exists() else {}
    model = FNO_TSM(
        in_channels=6,
        modes1=cfg.get("modes", 12),
        modes2=cfg.get("modes", 12),
        width=cfg.get("width", 48),
        n_layers=cfg.get("n_layers", 4),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict(model, X_np: np.ndarray, device: torch.device):
    X_t = torch.tensor(X_np[None], dtype=torch.float32, device=device)
    G_p, eps_p, A_p = model(X_t)
    return (G_p[0].cpu().numpy(),
            eps_p[0].cpu().numpy(),
            float(A_p[0].cpu()))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    save_dir = ROOT / "results" / "paper_demo"
    save_dir.mkdir(parents=True, exist_ok=True)

    run_dir = ROOT / "runs" / "tsm_v2"
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(run_dir, device)
    print(f"Loaded checkpoint from {run_dir / 'best.pt'}")

    # ── Build all inflation states ──────────────────────────────────────────
    states = []
    for pressure, label, r_vx in zip(PRESSURE_STATES, STATE_LABELS, BALLOON_RADII_VX):
        rng   = np.random.default_rng(RNG_SEED)
        geom  = make_geometry(pressure, r_vx)
        X, G_true, eps_true = build_input(geom, rng)
        G_pred, eps_pred, A_pred = predict(model, X, device)
        ring  = perilesional_shell(geom.mask(N), shell_mm=SHELL_MM, dx=DX)
        states.append(dict(
            label=label, pressure=pressure,
            X=X, G_true=G_true, eps_true=eps_true,
            G_pred=G_pred, eps_pred=eps_pred, A_pred=A_pred,
            ring=ring,
        ))
        max_eps_ring = float(eps_pred[ring].max()) if ring.any() else 0.0
        print(f"  {label:35s}  max(ε_pred in ring)={max_eps_ring:.4f}  A={A_pred:.3f}")

    # ── Control phantom (same geometry, p=0) ───────────────────────────────
    rng_ctrl = np.random.default_rng(RNG_SEED)
    geom_ctrl = make_geometry(pressure=0, radius_vx=BALLOON_RADII_VX[0])
    X_ctrl, G_true_ctrl, eps_true_ctrl = build_input(geom_ctrl, rng_ctrl)
    G_pred_ctrl, eps_pred_ctrl, A_pred_ctrl = predict(model, X_ctrl, device)
    ring_ctrl = perilesional_shell(geom_ctrl.mask(N), shell_mm=SHELL_MM, dx=DX)

    # ── Figure 1: Inflation series ──────────────────────────────────────────
    n_states = len(states)
    # Shared colour limits across all rows
    G_vmin = min(s["G_true"].min() for s in states)
    G_vmax = max(s["G_true"].max() for s in states)
    e_vmax = max(max(s["eps_pred"].max() for s in states),
                 max(s["eps_true"].max() for s in states), 0.01)

    fig, axes = plt.subplots(n_states, 5, figsize=(17, 3.2 * n_states))
    col_titles = ["Re(u 60 Hz)\n[input]",
                  "G_pred [Pa]", "G_true [Pa]",
                  "ε_pred (KEY)", "ε_true"]

    for ax, t in zip(axes[0], col_titles):
        ax.set_title(t, fontsize=9, fontweight="bold")

    for row, s in enumerate(states):
        ax = axes[row]
        max_eps_ring = float(s["eps_pred"][s["ring"]].max()) if s["ring"].any() else 0.0

        ax[0].imshow(s["X"][0], cmap="RdBu_r"); ax[0].axis("off")
        ax[0].set_ylabel(s["label"], fontsize=7.5, rotation=0,
                          labelpad=160, va="center")

        im1 = ax[1].imshow(s["G_pred"], cmap="hot", vmin=G_vmin, vmax=G_vmax)
        ax[1].axis("off")
        plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

        im2 = ax[2].imshow(s["G_true"], cmap="hot", vmin=G_vmin, vmax=G_vmax)
        ax[2].axis("off")
        plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

        im3 = ax[3].imshow(s["eps_pred"], cmap="magma", vmin=0, vmax=e_vmax)
        ax[3].axis("off")
        ax[3].set_title(f"ε_pred  max_ring={max_eps_ring:.3f}", fontsize=8)
        plt.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)

        im4 = ax[4].imshow(s["eps_true"], cmap="magma", vmin=0, vmax=e_vmax)
        ax[4].axis("off")
        plt.colorbar(im4, ax=ax[4], fraction=0.046, pad=0.04)

    plt.suptitle(
        "Simulated balloon-phantom inflation series  (Yin et al. 2026 protocol)\n"
        f"G_bg={G_BG:.0f} Pa, circular lesion r=20 mm, A_coeff={A_COEFF}",
        fontsize=10,
    )
    plt.tight_layout()
    out1 = save_dir / "inflation_series.png"
    fig.savefig(out1, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out1}")

    # ── Figure 2: Expanding vs Control ─────────────────────────────────────
    peak = states[-1]   # p = 7000 Pa
    G_vmin2 = min(peak["G_true"].min(), G_true_ctrl.min())
    G_vmax2 = max(peak["G_true"].max(), G_true_ctrl.max())
    e_vmax2 = max(peak["eps_pred"].max(), eps_pred_ctrl.max(), 0.01)

    fig2, axes2 = plt.subplots(2, 4, figsize=(14, 6))
    for col, (tag, Gp, ep, Gt, et) in enumerate([
        ("EXPANDING  (Phantom 2 analogue)",
         peak["G_pred"], peak["eps_pred"], peak["G_true"], peak["eps_true"]),
        ("CONTROL  (Phantom 3 analogue, p=0)",
         G_pred_ctrl, eps_pred_ctrl, G_true_ctrl, eps_true_ctrl),
    ]):
        axes2[0, col * 2].imshow(Gp, cmap="hot", vmin=G_vmin2, vmax=G_vmax2)
        axes2[0, col * 2].set_title(f"{tag}\nG_pred [Pa]", fontsize=8)
        axes2[0, col * 2].axis("off")

        axes2[0, col * 2 + 1].imshow(Gt, cmap="hot", vmin=G_vmin2, vmax=G_vmax2)
        axes2[0, col * 2 + 1].set_title("G_true [Pa]", fontsize=8)
        axes2[0, col * 2 + 1].axis("off")

        im_ep = axes2[1, col * 2].imshow(ep, cmap="magma", vmin=0, vmax=e_vmax2)
        axes2[1, col * 2].set_title("ε_pred  (KEY)", fontsize=8)
        axes2[1, col * 2].axis("off")
        plt.colorbar(im_ep, ax=axes2[1, col * 2], fraction=0.046, pad=0.04)

        im_et = axes2[1, col * 2 + 1].imshow(et, cmap="magma", vmin=0, vmax=e_vmax2)
        axes2[1, col * 2 + 1].set_title("ε_true", fontsize=8)
        axes2[1, col * 2 + 1].axis("off")
        plt.colorbar(im_et, ax=axes2[1, col * 2 + 1], fraction=0.046, pad=0.04)

    plt.suptitle(
        "Expanding (Phantom 2) vs Control (Phantom 3)  —  peak pressure p=7 kPa\n"
        "Ring should be visible in ε_pred for EXPANDING; absent for CONTROL",
        fontsize=10,
    )
    plt.tight_layout()
    out2 = save_dir / "expanding_vs_control.png"
    fig2.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {out2}")

    # ── Summary table ───────────────────────────────────────────────────────
    lines = [
        "Paper Phantom Demo — Yin et al. 2026 Simulation",
        "=" * 70,
        f"G_bg={G_BG:.0f} Pa  G_lesion={G_LESION:.0f} Pa  A_coeff={A_COEFF}",
        f"Lesion: circular, r={A_VX * DX * 1000:.0f} mm, centred at ({CX},{CY})",
        f"Grid: {N}×{N}, dx={DX*1000:.0f} mm  →  FOV {N*DX*100:.1f}×{N*DX*100:.1f} cm  (paper: 24×24 cm)",
        "",
        f"{'State':<38} {'p (Pa)':>7}  {'max(ε_pred ring)':>17}  "
        f"{'mean(G_pred ring) Pa':>21}  {'A_pred':>7}",
        "-" * 100,
    ]
    for s in states:
        ring = s["ring"]
        max_eps = float(s["eps_pred"][ring].max()) if ring.any() else float("nan")
        mean_G  = float(s["G_pred"][ring].mean()) if ring.any() else float("nan")
        lines.append(
            f"{s['label']:<38} {s['pressure']:>7.0f}  {max_eps:>17.4f}  "
            f"{mean_G:>21.1f}  {s['A_pred']:>7.3f}"
        )
    lines += [
        "-" * 100,
        "",
        "CONTROL (p=0, Phantom 3 analogue):",
        f"  max(ε_pred in ring) = {float(eps_pred_ctrl[ring_ctrl].max()):.4f}  "
        f"(target ≈ 0)",
        "",
        "Notes:",
        "  - Frequency: 60+120 Hz (training distribution; paper uses 80 Hz)",
        f"  - Resolution: {DX*1000:.0f} mm/voxel (paper: 0.9 mm clinical)",
        "  - Lamé prior is provided exactly (advantage vs real MRE)",
    ]

    summary_path = save_dir / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"Saved {summary_path}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
