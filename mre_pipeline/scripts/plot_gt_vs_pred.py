#!/usr/bin/env python3
"""GT vs prediction stiffness plots from current best.pt.
Runs on CPU on the login node — no GPU needed for a handful of samples."""
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fno_model import FNO2d


DX_CM = 0.2   # 2 mm per voxel = 0.2 cm
N = 64
EXTENT = [0, N * DX_CM, 0, N * DX_CM]   # cm


def main(n_samples=6, seed=42,
         data_path="data/mre_v2_50000.h5",
         ckpt_path="runs/phase0_v2/best.pt",
         out="runs/phase0_v2/gt_vs_pred.png"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    stats = ckpt["stats"]
    epoch = ckpt["epoch"]

    n_in = stats["X_mean"].shape[1]
    model = FNO2d(modes1=12, modes2=12, width=32, n_layers=4, in_channels=n_in)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    X_mean = stats["X_mean"]
    X_std  = stats["X_std"]
    log_min = stats["Y_log_min"]
    log_max = stats["Y_log_max"]

    def denorm_Y(y_n):
        return (y_n * (log_max - log_min) + log_min).exp()

    rng = np.random.default_rng(seed)
    f = h5py.File(data_path, "r")
    n_total = f["X"].shape[0]
    val_start = int(n_total * 0.9)
    idxs = sorted(rng.choice(np.arange(val_start, n_total), n_samples, replace=False))

    fig, axes = plt.subplots(n_samples, 4, figsize=(15, 3.2 * n_samples))
    fig.suptitle(f"GT vs FNO prediction — best checkpoint at epoch {epoch}\n"
                 f"Grid 64 × 64,  voxel 2 mm,  domain 12.8 cm × 12.8 cm",
                 fontsize=14, y=1.002)

    for row, idx in enumerate(idxs):
        X_raw = torch.tensor(f["X"][idx:idx + 1], dtype=torch.float32)
        Y_true = f["Y"][idx]
        Xn = (X_raw - X_mean) / X_std
        with torch.no_grad():
            Y_pred = denorm_Y(model(Xn)).squeeze().numpy()

        rl2  = np.linalg.norm(Y_pred - Y_true) / np.linalg.norm(Y_true)
        contrast = Y_true.max() / Y_true.min()
        vmin = min(Y_true.min(), Y_pred.min())
        vmax = max(Y_true.max(), Y_pred.max())
        err = np.abs(Y_pred - Y_true)

        # Column 0 — Input wave
        u_ext = max(abs(X_raw[0, 0].min().item()), abs(X_raw[0, 0].max().item()))
        im0 = axes[row, 0].imshow(X_raw[0, 0], cmap="RdBu_r", origin="lower",
                                   extent=EXTENT, vmin=-u_ext, vmax=u_ext)
        axes[row, 0].set_title(f"Re(u) at 60 Hz  [normalised]\nsample {idx}",
                                fontsize=10)
        cb = plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)
        cb.set_label("u  [—]", fontsize=9)

        # Column 1 — Ground truth G
        im1 = axes[row, 1].imshow(Y_true, cmap="hot", origin="lower",
                                   extent=EXTENT, vmin=vmin, vmax=vmax)
        axes[row, 1].set_title(f"Ground truth  G  [Pa]\ncontrast {contrast:.1f}×",
                                fontsize=10)
        cb = plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)
        cb.set_label("G  [Pa]", fontsize=9)

        # Column 2 — Prediction
        im2 = axes[row, 2].imshow(Y_pred, cmap="hot", origin="lower",
                                   extent=EXTENT, vmin=vmin, vmax=vmax)
        axes[row, 2].set_title(f"FNO prediction  G  [Pa]\nRL² = {rl2:.3f}",
                                fontsize=10)
        cb = plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)
        cb.set_label("G  [Pa]", fontsize=9)

        # Column 3 — Absolute error
        im3 = axes[row, 3].imshow(err, cmap="viridis", origin="lower",
                                   extent=EXTENT)
        axes[row, 3].set_title(f"|GT − Pred|  [Pa]\nmax error {err.max():.0f} Pa",
                                fontsize=10)
        cb = plt.colorbar(im3, ax=axes[row, 3], fraction=0.046, pad=0.04)
        cb.set_label("Δ G  [Pa]", fontsize=9)

        for ax in axes[row]:
            ax.set_xlabel("x [cm]", fontsize=9)
            ax.set_ylabel("y [cm]", fontsize=9)
            ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved {out}  (epoch {epoch}, {n_samples} val samples)")


if __name__ == "__main__":
    main()
