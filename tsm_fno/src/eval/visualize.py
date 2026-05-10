"""6-panel benchmark figure for TSM-FNO."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def benchmark_figure(
    X: np.ndarray,
    h5_path: str | Path,
    val_start: int,
    G_pred: np.ndarray,
    eps_pred: np.ndarray,
    n_samples: int = 6,
    save_path: str | Path = "benchmark.png",
):
    """Plot n_samples × 6 panels.

    Panels per row:
      1. Re(u₆₀)        — input wave
      2. Lamé prior     — channel 4 of input X
      3. G_pred         — predicted stiffness
      4. G_true         — ground truth stiffness
      5. ε_pred         — predicted latent strain (KEY OUTPUT)
      6. ε_true         — ground truth latent strain
    """
    save_path = Path(save_path)

    rng = np.random.default_rng(0)
    with h5py.File(h5_path, "r") as f:
        n_val = G_pred.shape[0]
        # Cap n_samples to what's actually available
        n_samples = min(n_samples, n_val)
        is_exp_v = f["meta/is_expanding"][val_start:].astype(bool)
        exp_idx = np.where(is_exp_v)[0]
        ctl_idx = np.where(~is_exp_v)[0]
        n_exp = min(n_samples - n_samples // 2, len(exp_idx))
        n_ctl = min(n_samples - n_exp, len(ctl_idx))
        chosen_exp = rng.choice(exp_idx, n_exp, replace=False) \
                       if n_exp > 0 else np.array([], int)
        chosen_ctl = rng.choice(ctl_idx, n_ctl, replace=False) \
                       if n_ctl > 0 else np.array([], int)
        idxs = np.concatenate([chosen_exp, chosen_ctl]).astype(int)
        n_samples = len(idxs)
        if n_samples == 0:
            print("benchmark_figure: no samples to plot")
            return

        X_v   = f["X"][val_start:][idxs]
        G_v   = f["Y_G"][val_start:][idxs]
        eps_v = f["Y_epsilon"][val_start:][idxs]

    fig, axes = plt.subplots(n_samples, 6, figsize=(18, 3.0 * n_samples))
    if n_samples == 1:
        axes = axes[None, :]

    chosen_exp_set = set(chosen_exp.tolist())
    for row in range(n_samples):
        i = int(idxs[row])
        is_exp = i in chosen_exp_set
        ax = axes[row]

        n_ch = X_v.shape[1]
        lame_ch = 2 if n_ch == 4 else 4   # 4-ch: Re,Im,Lame,dist  6-ch: Re60,Im60,Re120,Im120,Lame,dist
        freq_label = "80Hz" if n_ch == 4 else "60Hz"

        ax[0].imshow(X_v[row, 0], cmap="RdBu_r"); ax[0].axis("off")
        ax[0].set_title(f"Re(u {freq_label}) {'EXPANDING' if is_exp else 'control'}",
                         fontsize=9)

        ax[1].imshow(X_v[row, lame_ch], cmap="viridis"); ax[1].axis("off")
        ax[1].set_title("Lamé prior  Δσ/G_bg", fontsize=9)

        vmin = min(float(G_v[row].min()), float(G_pred[i].min()))
        vmax = max(float(G_v[row].max()), float(G_pred[i].max()))
        ax[2].imshow(G_pred[i], cmap="hot", vmin=vmin, vmax=vmax); ax[2].axis("off")
        ax[2].set_title("G_pred [Pa]", fontsize=9)

        ax[3].imshow(G_v[row], cmap="hot", vmin=vmin, vmax=vmax); ax[3].axis("off")
        ax[3].set_title("G_true [Pa]", fontsize=9)

        eps_lo = min(float(eps_v[row].min()), float(eps_pred[i].min()))
        eps_hi = max(float(eps_v[row].max()), float(eps_pred[i].max()))
        ax[4].imshow(eps_pred[i], cmap="magma", vmin=eps_lo, vmax=eps_hi)
        ax[4].axis("off"); ax[4].set_title("ε_pred  (KEY)", fontsize=9)

        im5 = ax[5].imshow(eps_v[row], cmap="magma", vmin=eps_lo, vmax=eps_hi)
        ax[5].axis("off"); ax[5].set_title("ε_true", fontsize=9)
        plt.colorbar(im5, ax=ax[5], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")
