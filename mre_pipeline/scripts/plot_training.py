#!/usr/bin/env python3
"""Plot training curves from runs/phase0/history.json after training completes."""
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

run_dir = "runs/phase0"
history_path = f"{run_dir}/history.json"

try:
    h = json.load(open(history_path))
except FileNotFoundError:
    print(f"No history found at {history_path} — run training first.")
    sys.exit(1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(h["train_loss"], label="train")
ax1.plot(h["val_rl2"],    label="val RL²")
ax1.axhline(0.05, color="red", linestyle="--", label="target RL²=0.05")
ax1.set_title("Relative L² Loss")
ax1.set_xlabel("Epoch")
ax1.legend()

ax2.plot(h["val_ssim"])
ax2.axhline(0.90, color="red", linestyle="--", label="target SSIM=0.90")
ax2.set_title("Validation SSIM")
ax2.set_xlabel("Epoch")
ax2.legend()

plt.tight_layout()
out = f"{run_dir}/training_curves.png"
plt.savefig(out, dpi=150)
print(f"Saved {out}")
