#!/usr/bin/env python3
"""Extract latent embeddings from trained FNO_TSM and visualize expanding vs control.

Outputs (all in --save_dir):
  embeddings.h5          – global_vec (N,48), labels, pressures, etc.
  embed_pca.png          – PCA 2D scatter
  embed_tsne.png         – t-SNE 2D scatter
  embed_umap.png         – UMAP 2D scatter (if umap-learn installed)
  spatial_pca_grid.png   – per-pixel PCA component 1 for 6 sample images
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import TSMDataset
from src.model.fno_tsm import FNO_TSM


# ── Hook helper ──────────────────────────────────────────────────────────────

class LatentHook:
    """Register on any nn.Module to capture its output tensor during forward."""

    def __init__(self, module: torch.nn.Module):
        self.output = None
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self.output = output.detach().cpu()

    def remove(self):
        self._handle.remove()


# ── Scatter plot helper ───────────────────────────────────────────────────────

def scatter2d(xy: np.ndarray, labels: np.ndarray, pressures: np.ndarray,
              title: str, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: binary expanding/control
    colors = np.where(labels, "#e63946", "#457b9d")
    for ax, (c, t) in zip(axes, [
        (colors, "Expanding (red) vs Control (blue)"),
        (pressures, "Pressure (Pa)"),
    ]):
        if isinstance(c, np.ndarray) and c.dtype != object:
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, s=6, alpha=0.5, cmap="plasma")
            plt.colorbar(sc, ax=ax, label="Pressure (Pa)")
        else:
            ax.scatter(xy[:, 0], xy[:, 1], c=c, s=6, alpha=0.5)
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
        ax.axis("equal")

    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Spatial PCA grid ─────────────────────────────────────────────────────────

def spatial_pca_grid(spatial_samples: list[np.ndarray],
                     labels: list[bool], pressures: list[float],
                     path: Path):
    """Plot first PCA component of spatial feature map for a few samples."""
    n = len(spatial_samples)
    # Stack all pixels across samples to fit PCA
    flat = np.concatenate([s.reshape(s.shape[0], -1).T for s in spatial_samples], axis=0)
    pca = PCA(n_components=3)
    pca.fit(flat)

    fig, axes = plt.subplots(3, n, figsize=(3 * n, 8))
    comp_labels = ["PC1", "PC2", "PC3"]
    for comp in range(3):
        for col, (s, is_exp, p) in enumerate(zip(spatial_samples, labels, pressures)):
            H, W = s.shape[1], s.shape[2]
            proj = pca.transform(s.reshape(s.shape[0], -1).T)[:, comp].reshape(H, W)
            axes[comp, col].imshow(proj, cmap="RdBu_r")
            axes[comp, col].axis("off")
            if comp == 0:
                tag = f"{'EXP' if is_exp else 'CTL'} p={p:.0f}Pa"
                axes[comp, col].set_title(tag, fontsize=8)
            if col == 0:
                axes[comp, col].set_ylabel(comp_labels[comp], fontsize=9)

    plt.suptitle("Spatial feature map — PCA components (per-pixel latent space)",
                 fontsize=10)
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="runs/tsm_80hz/best.pt")
    p.add_argument("--data_path",  default="data/tsm_50000_80hz.h5")
    p.add_argument("--save_dir",   default="results/tsm_80hz/embeddings")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_samples", type=int, default=5000,
                   help="Cap val samples for speed (0 = all)")
    args = p.parse_args()

    save = Path(args.save_dir); save.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    cfg_path = Path(args.checkpoint).parent / "args.json"
    cfg = json.load(open(cfg_path)) if cfg_path.exists() else {}
    model = FNO_TSM(
        in_channels=cfg.get("in_channels", 4),
        modes1=cfg.get("modes", 16),
        modes2=cfg.get("modes", 16),
        width=cfg.get("width", 48),
        n_layers=cfg.get("n_layers", 4),
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    width = cfg.get("width", 48)

    # ── Register hook on last Fourier block (alternative to encode()) ────────
    # This demonstrates the hook API — encode() is used below for the main loop
    hook = LatentHook(model.blocks[-1])
    print("Hook registered on model.blocks[-1]")

    # Dataset
    ds = TSMDataset(args.data_path, augment=False)
    n = len(ds)
    val_start = int(n * 0.9)
    val_idx = list(range(val_start, n))
    if args.max_samples > 0:
        val_idx = val_idx[:args.max_samples]
    print(f"Val samples: {len(val_idx)}")

    loader = DataLoader(Subset(ds, val_idx), batch_size=args.batch_size,
                        shuffle=False, num_workers=2)

    # Load meta
    with h5py.File(args.data_path, "r") as f:
        is_exp_all  = f["meta/is_expanding"][val_start:len(val_idx)+val_start].astype(bool)
        pressure_all = f["meta/p"][val_start:len(val_idx)+val_start]

    # Extract embeddings
    global_vecs = np.empty((len(val_idx), width), dtype=np.float32)
    spatial_list: list[np.ndarray] = []   # small sample for spatial PCA

    with torch.no_grad():
        offset = 0
        for batch in loader:
            X = batch["X"].to(device)
            _, _, _, spatial, gvec = model.encode(X)
            bs = len(X)
            global_vecs[offset:offset+bs] = gvec.cpu().numpy()
            # Store a few spatial maps for visualization
            if len(spatial_list) < 6:
                spatial_list.append(spatial[0].cpu().numpy())  # (width, H, W)
            offset += bs

    hook.remove()
    print(f"Hook output shape (last batch): {hook.output.shape}")
    print(f"Global embeddings shape: {global_vecs.shape}")

    # Save embeddings
    emb_path = save / "embeddings.h5"
    with h5py.File(emb_path, "w") as f:
        f.create_dataset("global_vec",  data=global_vecs)
        f.create_dataset("is_expanding", data=is_exp_all[:len(val_idx)])
        f.create_dataset("pressure",     data=pressure_all[:len(val_idx)])
    print(f"Saved {emb_path}")

    labels    = is_exp_all[:len(val_idx)]
    pressures = pressure_all[:len(val_idx)]

    # ── PCA ──────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(global_vecs)

    pca = PCA(n_components=2)
    xy_pca = pca.fit_transform(X_scaled)
    print(f"PCA variance explained: {pca.explained_variance_ratio_}")
    scatter2d(xy_pca, labels, pressures,
              f"PCA  (var={pca.explained_variance_ratio_.sum()*100:.1f}%)",
              save / "embed_pca.png")

    # ── t-SNE ────────────────────────────────────────────────────────────────
    print("Running t-SNE …")
    tsne = TSNE(n_components=2, perplexity=40, random_state=0, max_iter=1000)
    xy_tsne = tsne.fit_transform(X_scaled[:3000])   # cap for speed
    scatter2d(xy_tsne, labels[:3000], pressures[:3000],
              "t-SNE  (first 3000 val samples)",
              save / "embed_tsne.png")

    # ── UMAP (optional) ──────────────────────────────────────────────────────
    try:
        import umap
        print("Running UMAP …")
        reducer = umap.UMAP(n_components=2, random_state=0)
        xy_umap = reducer.fit_transform(X_scaled)
        scatter2d(xy_umap, labels, pressures,
                  "UMAP", save / "embed_umap.png")
    except ImportError:
        print("umap-learn not installed — skipping UMAP (pip install umap-learn)")

    # ── Spatial PCA grid ─────────────────────────────────────────────────────
    sp_labels   = [bool(labels[i]) for i in range(min(6, len(spatial_list)))]
    sp_pressures = [float(pressures[i]) for i in range(min(6, len(spatial_list)))]
    spatial_pca_grid(
        spatial_list[:6], sp_labels, sp_pressures,
        save / "spatial_pca_grid.png",
    )

    # ── PCA loadings bar chart ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(range(width), np.abs(pca.components_[0]), label="PC1")
    ax.bar(range(width), np.abs(pca.components_[1]), alpha=0.6, label="PC2")
    ax.set_xlabel("Latent dimension (0–47)")
    ax.set_ylabel("|loading|")
    ax.set_title("PCA loadings — which latent dims drive expanding/control separation")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save / "pca_loadings.png", dpi=130)
    plt.close(fig)
    print(f"Saved {save}/pca_loadings.png")

    print(f"\nAll outputs in {save}/")


if __name__ == "__main__":
    main()
