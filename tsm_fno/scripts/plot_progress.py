#!/usr/bin/env python3
"""Plot training curves and GT vs prediction panels from current best.pt."""
import sys, json, argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.fno_tsm import FNO_TSM


def plot_loss(history, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ep = range(1, len(history["train_L_G"]) + 1)

    axes[0].plot(ep, history["train_L_G"], label="train L_G", color="steelblue")
    axes[0].plot(ep, history["val_L_G"],   label="val L_G",   color="steelblue", linestyle="--")
    axes[0].set_title("G Relative-L²"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_ylim(bottom=0)

    axes[1].plot(ep, history["train_L_ring"], label="train L_ring", color="darkorange")
    axes[1].plot(ep, history["val_L_ring"],   label="val L_ring",   color="darkorange", linestyle="--")
    axes[1].axhline(0.10, color="red", linestyle=":", linewidth=1.5, label="target < 0.10")
    axes[1].set_title("Ring Relative-L² (ε)"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_ylim(bottom=0)

    axes[2].plot(ep, history["val_auc"], color="green", label="val AUC")
    axes[2].axhline(0.90, color="red", linestyle=":", linewidth=1.5, label="target > 0.90")
    axes[2].set_title("Expansion AUC"); axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0, 1.05); axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_predictions(ckpt_path, data_path, save_path, n_rows=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Build model from args.json
    args_path = Path(ckpt_path).parent / "args.json"
    cfg = json.load(open(args_path)) if args_path.exists() else {}
    model = FNO_TSM(
        in_channels=6,
        modes1=cfg.get("modes", 12),
        modes2=cfg.get("modes", 12),
        width=cfg.get("width", 48),
        n_layers=cfg.get("n_layers", 4),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rng = np.random.default_rng(42)
    with h5py.File(data_path, "r") as f:
        n = len(f["Y_G"])
        is_exp = f["meta/is_expanding"][:].astype(bool)
        exp_idx  = np.where(is_exp)[0]
        ctrl_idx = np.where(~is_exp)[0]
        n_exp  = min(n_rows // 2 + n_rows % 2, len(exp_idx))
        n_ctrl = min(n_rows - n_exp, len(ctrl_idx))
        chosen = np.sort(np.concatenate([
            rng.choice(exp_idx,  n_exp,  replace=False),
            rng.choice(ctrl_idx, n_ctrl, replace=False),
        ]))
        is_exp_chosen = is_exp[chosen]
        X_np   = f["X"][chosen]
        G_np   = f["Y_G"][chosen]
        eps_np = f["Y_epsilon"][chosen]
        ring_np = f["Y_ring"][chosen]

    X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        G_p, eps_p, A_p = model(X_t)
    G_p   = G_p.cpu().numpy()
    eps_p = eps_p.cpu().numpy()
    A_p   = A_p.cpu().numpy()

    n_rows = len(chosen)
    fig, axes = plt.subplots(n_rows, 6, figsize=(18, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    titles = ["Re(u 60Hz)\n[input]", "Lamé prior\n[input]",
              "G_pred [Pa]", "G_true [Pa]",
              "ε_pred", "ε_true"]
    cmaps  = ["RdBu_r", "viridis", "hot", "hot", "magma", "magma"]

    for row in range(n_rows):
        i = row
        is_e = is_exp_chosen[i]
        label = "EXPANDING" if is_e else "control"

        panels = [
            X_np[i, 0],
            X_np[i, 4],
            G_p[i],
            G_np[i],
            eps_p[i],
            eps_np[i],
        ]
        # Shared colour limits for paired panels
        vG = (min(G_np[i].min(), G_p[i].min()), max(G_np[i].max(), G_p[i].max()))
        ve = (0, max(eps_np[i].max(), eps_p[i].max(), 0.01))
        vlims = [None, None, vG, vG, ve, ve]

        for col, (data, title, cmap, vl) in enumerate(zip(panels, titles, cmaps, vlims)):
            kw = dict(vmin=vl[0], vmax=vl[1]) if vl else {}
            im = axes[row, col].imshow(data, cmap=cmap, **kw)
            axes[row, col].axis("off")
            tstr = f"{title}\n[{label}]" if col == 0 else title
            axes[row, col].set_title(tstr, fontsize=8)
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

        # Annotate A prediction on ε_pred panel
        axes[row, 4].set_title(f"ε_pred  A={A_p[i]:.2f}", fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir",   default="runs/tsm_v2")
    p.add_argument("--data_path", default="data/tsm_50000.h5")
    p.add_argument("--save_dir",  default="results/tsm_v2")
    args = p.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history_path = Path(args.run_dir) / "history.json"
    ckpt_path    = Path(args.run_dir) / "best.pt"

    if history_path.exists():
        history = json.load(open(history_path))
        plot_loss(history, save_dir / "loss_curves.png")

    if ckpt_path.exists():
        plot_predictions(ckpt_path, args.data_path, save_dir / "gt_vs_pred.png")
