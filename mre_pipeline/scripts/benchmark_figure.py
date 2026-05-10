#!/usr/bin/env python3
"""Benchmark figure: FNO predictions vs ground truth across stiffness contrasts.
Run after training completes and runs/phase0/best.pt exists."""
import sys
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
    from src.fno_model import FNO2d
except ImportError as e:
    print(f"Import error: {e}\nRun training first to produce src/fno_model.py and best.pt.")
    sys.exit(1)

run_dir = "runs/phase0"
ckpt_path = f"{run_dir}/best.pt"

model = FNO2d()
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt["model_state"])
model.eval()

f = h5py.File("data/mre_synthetic_50000.h5", "r")

# 6 test cases spanning stiffness contrast range
test_ids = [45000, 45200, 45500, 46000, 47000, 48000]

fig, axes = plt.subplots(6, 3, figsize=(9, 18))

for row, idx in enumerate(test_ids):
    X = torch.tensor(f["X"][idx:idx + 1])
    Y_true = f["Y"][idx]

    with torch.no_grad():
        Y_pred = model(X).squeeze().numpy()

    rl2 = np.linalg.norm(Y_pred - Y_true) / np.linalg.norm(Y_true)
    vmin, vmax = Y_true.min(), Y_true.max()
    contrast = Y_true.max() / Y_true.min()

    axes[row, 0].imshow(X[0, 0], cmap="RdBu_r")
    axes[row, 0].set_title("Re(u) — input" if row == 0 else f"contrast {contrast:.1f}×")

    axes[row, 1].imshow(Y_pred, cmap="hot", vmin=vmin, vmax=vmax)
    axes[row, 1].set_title(f"FNO prediction  RL²={rl2:.3f}" if row == 0 else f"RL²={rl2:.3f}")

    axes[row, 2].imshow(Y_true, cmap="hot", vmin=vmin, vmax=vmax)
    axes[row, 2].set_title("Ground truth" if row == 0 else "")

    for ax in axes[row]:
        ax.axis("off")

plt.tight_layout()
out = f"{run_dir}/benchmark_figure.png"
plt.savefig(out, dpi=200)
print(f"Saved {out}")
