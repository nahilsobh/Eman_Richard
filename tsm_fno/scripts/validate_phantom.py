#!/usr/bin/env python3
"""Phase A phantom validation — dry run on a synthetic phantom-like stand-in.

Builds a deterministic single-inclusion phantom with NO pressure (p=0, static),
solves the forward Helmholtz problem at 80 Hz, runs the trained TSM-FNO and
the DI baseline, and reports Phase A pass/fail.

Phase A success criteria (see tsm_fno/README.md):
    1. G_pred Pearson r >= 0.85 in inclusion ROI (vs ground truth here;
       vs NLI when real data arrives)
    2. eps_pred ~= 0 everywhere (static phantom — no expansion)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.generate_tsm import RHO, DX, FREQ_1, DAMPING, N_DEFAULT
from src.model.fno_tsm import FNO_TSM
from src.phantom.geometry import LesionGeometry
from src.solver.helmholtz_fd import helmholtz_eshelby_solve, random_sources


def _load_di():
    """Load mre_pipeline/src/direct_inversion.py without polluting sys.path."""
    path = (Path(__file__).resolve().parents[2]
            / "mre_pipeline" / "src" / "direct_inversion.py")
    spec = importlib.util.spec_from_file_location("mre_di", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.direct_inversion

direct_inversion = _load_di()


PHASE_A_R_THRESHOLD = 0.85
PHASE_A_EPS_THRESHOLD = 0.05   # ring-region mean |eps_pred| must be below this


def build_phantom(N: int, dx: float, G_bg: float, G_lesion: float,
                  a_vox: float, b_vox: float, angle: float = 0.0,
                  seed: int = 0) -> tuple[np.ndarray, np.ndarray, LesionGeometry]:
    """Return (G_true, X_input, geometry) for a static (p=0) single-inclusion phantom."""
    geom = LesionGeometry(
        center=(N / 2.0, N / 2.0),
        semi_axes=(a_vox, b_vox),
        angle=angle,
        pressure=0.0,           # static phantom — Phase A success criterion #2
    )

    G_true = np.full((N, N), G_bg, dtype=np.float64)
    G_true[geom.mask(N)] = G_lesion

    # Static phantom: eps_star = 0 everywhere
    eps_star = np.zeros((N, N), dtype=np.float64)

    rng = np.random.default_rng(seed)
    sources = random_sources(N, rng)

    u80 = helmholtz_eshelby_solve(
        G_true, eps_star, center=(N // 2, N // 2),
        freq=FREQ_1, rho=RHO, dx=dx, damping=DAMPING, sources=sources,
    )

    u_max = float(np.max(np.abs(u80)))
    if u_max > 0:
        u80 = u80 / u_max

    # Static phantom — Lamé pre-stress field is identically zero
    lame_prior = np.zeros((N, N), dtype=np.float64)

    dist_vox = distance_transform_edt(~geom.mask(N)).astype(np.float64)
    dist_m = dist_vox * dx
    dist_norm = dist_m / dist_m.max() if dist_m.max() > 0 else dist_m

    X = np.stack([
        u80.real.astype(np.float32),
        u80.imag.astype(np.float32),
        lame_prior.astype(np.float32),
        dist_norm.astype(np.float32),
    ], axis=0)

    return G_true.astype(np.float32), X, geom


def load_model(checkpoint: Path, device: torch.device) -> FNO_TSM:
    cfg_path = checkpoint.parent / "args.json"
    cfg = json.load(open(cfg_path)) if cfg_path.exists() else {}
    model = FNO_TSM(
        in_channels=cfg.get("in_channels", 4),
        modes1=cfg.get("modes", 16),
        modes2=cfg.get("modes", 16),
        width=cfg.get("width", 48),
        n_layers=cfg.get("n_layers", 4),
    )
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


def masked_pearson(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    av, bv = a[mask].ravel(), b[mask].ravel()
    if av.std() < 1e-10 or bv.std() < 1e-10:
        return float("nan")
    r, _ = pearsonr(av, bv)
    return float(r)


def rl2(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        pred = pred[mask]; target = target[mask]
    return float(np.linalg.norm(pred - target) / (np.linalg.norm(target) + 1e-12))


def figure_panel(out: Path, G_true, G_fno, G_di, eps_pred, lesion_mask):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    vmin, vmax = float(G_true.min()), float(G_true.max())

    im0 = axes[0].imshow(G_true, vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0].set_title("Ground truth G [Pa]")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(G_fno, vmin=vmin, vmax=vmax, cmap="viridis")
    axes[1].set_title("FNO G_pred [Pa]")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(G_di, vmin=vmin, vmax=vmax, cmap="viridis")
    axes[2].set_title("DI baseline G [Pa]")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(eps_pred, cmap="magma")
    axes[3].set_title(f"FNO eps_pred  (max={eps_pred.max():.3f})")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    for ax in axes:
        ax.contour(lesion_mask.astype(float), levels=[0.5], colors="white", linewidths=0.8)
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/tsm_80hz/best.pt")
    ap.add_argument("--save_dir",   default="results/phantom_phaseA")
    ap.add_argument("--N",          type=int,   default=N_DEFAULT)
    ap.add_argument("--dx",         type=float, default=DX)
    ap.add_argument("--G_bg",       type=float, default=1500.0)
    ap.add_argument("--G_lesion",   type=float, default=5000.0)
    ap.add_argument("--a_vox",      type=float, default=12.0)
    ap.add_argument("--b_vox",      type=float, default=12.0)
    ap.add_argument("--seed",       type=int,   default=0)
    args = ap.parse_args()

    save = Path(args.save_dir); save.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    G_true, X, geom = build_phantom(
        N=args.N, dx=args.dx, G_bg=args.G_bg, G_lesion=args.G_lesion,
        a_vox=args.a_vox, b_vox=args.b_vox, seed=args.seed,
    )
    lesion_mask = geom.mask(args.N)
    print(f"Phantom: N={args.N}, G_bg={args.G_bg:.0f} Pa, "
          f"G_lesion={args.G_lesion:.0f} Pa, a={args.a_vox:.1f} vox, "
          f"pressure=0 (static)")

    model = load_model(Path(args.checkpoint), device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    with torch.no_grad():
        Xt = torch.from_numpy(X[None]).to(device)
        G_pred_t, eps_pred_t, A_pred_t = model(Xt)
    G_fno = G_pred_t[0].cpu().numpy()
    eps_pred = eps_pred_t[0].cpu().numpy()
    A_pred = float(A_pred_t[0].cpu().numpy())

    u = X[0] + 1j * X[1]
    G_di = direct_inversion(u, freq=FREQ_1, rho=RHO, dx=args.dx)

    inside = lesion_mask
    outside = ~lesion_mask

    metrics = {
        "phantom": {
            "G_bg": args.G_bg, "G_lesion": args.G_lesion,
            "a_vox": args.a_vox, "b_vox": args.b_vox,
            "pressure": 0.0, "freq_hz": FREQ_1, "N": args.N,
        },
        "fno": {
            "pearson_r_whole":   masked_pearson(G_fno, G_true, np.ones_like(lesion_mask, dtype=bool)),
            "pearson_r_inside":  masked_pearson(G_fno, G_true, inside),
            "pearson_r_outside": masked_pearson(G_fno, G_true, outside),
            "rl2_whole":         rl2(G_fno, G_true),
            "rl2_inside":        rl2(G_fno, G_true, inside),
            "G_pred_inside_mean":  float(G_fno[inside].mean()),
            "G_pred_outside_mean": float(G_fno[outside].mean()),
        },
        "di": {
            "pearson_r_whole":  masked_pearson(G_di, G_true, np.ones_like(lesion_mask, dtype=bool)),
            "pearson_r_inside": masked_pearson(G_di, G_true, inside),
            "rl2_whole":        rl2(G_di, G_true),
        },
        "fno_vs_di": {
            "pearson_r_whole": masked_pearson(G_fno, G_di, np.ones_like(lesion_mask, dtype=bool)),
        },
        "eps": {
            "max_abs":     float(np.abs(eps_pred).max()),
            "mean_abs":    float(np.abs(eps_pred).mean()),
            "outside_mean_abs": float(np.abs(eps_pred[outside]).mean()),
        },
        "A_pred": A_pred,
    }

    r_fno = metrics["fno"]["pearson_r_whole"]
    eps_outside = metrics["eps"]["outside_mean_abs"]
    pass1 = r_fno >= PHASE_A_R_THRESHOLD
    pass2 = eps_outside <= PHASE_A_EPS_THRESHOLD
    metrics["phase_a_pass"] = {
        "criterion_1_r_geq_0p85":              {"passed": bool(pass1), "value": r_fno},
        "criterion_2_eps_outside_leq_0p05":    {"passed": bool(pass2), "value": eps_outside},
        "overall_pass": bool(pass1 and pass2),
    }

    with open(save / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    figure_panel(save / "phantom_panel.png",
                 G_true, G_fno, G_di, eps_pred, lesion_mask)

    print("\n========== Phase A phantom validation ==========")
    print(f"FNO  whole-field Pearson r vs GT : {r_fno:.4f}   "
          f"(criterion >= {PHASE_A_R_THRESHOLD})  "
          f"{'PASS' if pass1 else 'FAIL'}")
    print(f"FNO  inside-lesion Pearson r     : {metrics['fno']['pearson_r_inside']:.4f}")
    print(f"FNO  whole-field RL2 vs GT       : {metrics['fno']['rl2_whole']:.4f}")
    print(f"FNO  G_pred mean inside / outside: "
          f"{metrics['fno']['G_pred_inside_mean']:.1f} / "
          f"{metrics['fno']['G_pred_outside_mean']:.1f} Pa   "
          f"(GT inside={args.G_lesion:.0f}, outside={args.G_bg:.0f})")
    print(f"DI   whole-field Pearson r vs GT : {metrics['di']['pearson_r_whole']:.4f}")
    print(f"DI   whole-field RL2 vs GT       : {metrics['di']['rl2_whole']:.4f}")
    print(f"FNO vs DI Pearson r              : {metrics['fno_vs_di']['pearson_r_whole']:.4f}")
    print(f"eps_pred mean |outside lesion|   : {eps_outside:.4f}   "
          f"(criterion <= {PHASE_A_EPS_THRESHOLD})  "
          f"{'PASS' if pass2 else 'FAIL'}")
    print(f"eps_pred max |val|               : {metrics['eps']['max_abs']:.4f}")
    print(f"A_pred (scalar)                  : {A_pred:.3f}")
    print("================================================")
    print(f"Overall Phase A: {'PASS' if metrics['phase_a_pass']['overall_pass'] else 'FAIL'}")
    print(f"Artifacts: {save}/metrics.json, {save}/phantom_panel.png")


if __name__ == "__main__":
    main()
