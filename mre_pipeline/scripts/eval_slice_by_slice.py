#!/usr/bin/env python3
"""Slice-by-slice 3D MRE inversion using the v2 2D FNO.

For each of N 3D phantoms:
    1. Generate G(x,y,z) — a 3D ellipsoidal inclusion on a uniform background.
    2. For each axial slice z, run the 2D Helmholtz solver with random
       multi-source excitation + variable damping to produce u_z(x,y).
    3. Add complex Gaussian noise (SNR uniform 15-30 dB) per slice.
    4. Apply the v2 FNO independently to each slice.
    5. Stack slice predictions back into a 3D stiffness volume.

Reports:
    - Pearson R between predicted and true G at the inclusion-centre voxel
      across all volumes (directly comparable to ILI Table 1: R = 0.940).
    - Mean RL² and SSIM across all 2D slices.
    - Saves a 3-orthogonal-view figure for one representative volume.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fno_model import FNO2d
from src.phantom_3d import generate_phantom_3d
from src.fem_solver import helmholtz_solve, random_sources
from src.dataset import RHO, DX, FREQ, DAMPING_MIN, DAMPING_MAX, add_noise


def slice_inputs(G_3d, rng):
    """For a (N,N,N) stiffness volume return X:(N,2,N,N) and per-slice noisy u."""
    N = G_3d.shape[0]
    X = np.empty((N, 2, N, N), dtype=np.float32)
    for z in range(N):
        G_slice = G_3d[z]
        damping = float(rng.uniform(DAMPING_MIN, DAMPING_MAX))
        n_patches = int(rng.integers(1, 11))
        sources = random_sources(N, rng, n_min=n_patches, n_max=n_patches)
        u = helmholtz_solve(G_slice, freq=FREQ, rho=RHO, dx=DX,
                            damping=damping, sources=sources)
        snr = float(rng.uniform(15, 30))
        u = add_noise(u, snr, rng)
        scale = np.max(np.abs(u))
        if scale > 0:
            u = u / scale
        X[z, 0] = u.real.astype(np.float32)
        X[z, 1] = u.imag.astype(np.float32)
    return X


def ssim_2d(pred, target, C1=1e-4, C2=9e-4):
    mu_p, mu_t = pred.mean(), target.mean()
    sig_p, sig_t = pred.var(), target.var()
    sig_pt = ((pred - mu_p) * (target - mu_t)).mean()
    return ((2 * mu_p * mu_t + C1) * (2 * sig_pt + C2)) / \
           ((mu_p ** 2 + mu_t ** 2 + C1) * (sig_p + sig_t + C2))


def find_3d_inclusion_centre(G_3d):
    """Return (z, y, x) at the centroid of the brightest connected region."""
    G_max = G_3d.max()
    bg = np.median(G_3d)
    threshold = bg + 0.5 * (G_max - bg)
    mask = G_3d >= threshold
    if mask.sum() == 0:
        N = G_3d.shape[0]
        return N // 2, N // 2, N // 2
    zs, ys, xs = np.where(mask)
    return int(np.round(zs.mean())), int(np.round(ys.mean())), int(np.round(xs.mean()))


def main(
    n_volumes=10,
    N=64,
    ckpt_path="runs/phase0_v2/best.pt",
    out_dir="runs/phase0_v2",
    seed=2026,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    stats = ckpt["stats"]
    n_in = stats["X_mean"].shape[1]
    model = FNO2d(modes1=12, modes2=12, width=32, n_layers=4, in_channels=n_in)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    X_mean = stats["X_mean"]
    X_std  = stats["X_std"]
    log_min = stats["Y_log_min"]
    log_max = stats["Y_log_max"]

    def denorm_Y(yn):
        return (yn * (log_max - log_min) + log_min).exp()

    rng = np.random.default_rng(seed)

    rl2_per_slice = []
    ssim_per_slice = []
    inc_true_vals, inc_pred_vals = [], []          # centroid of inclusion (per volume)
    geom_true_vals, geom_pred_vals = [], []        # geometric center (per slice; ILI metric)
    per_volume_rl2 = []
    g_true_dense, g_pred_dense = [], []
    keep_one = None

    DENSE_PER_VOLUME = 20000
    sub_rng = np.random.default_rng(0)
    geom_c = N // 2

    t0 = time.time()
    for v in range(n_volumes):
        G_3d = generate_phantom_3d(N=N, rng=rng).astype(np.float32)
        print(f"\nVolume {v+1}/{n_volumes}  G range "
              f"{G_3d.min():.0f}-{G_3d.max():.0f} Pa", flush=True)

        X = slice_inputs(G_3d, rng)
        Xn = (torch.tensor(X) - X_mean) / X_std

        with torch.no_grad():
            Yp_n = model(Xn)
            Yp = denorm_Y(Yp_n).numpy()  # (N, N, N) — (z, y, x)

        # 2D metrics, slice-by-slice + ILI-style geometric-centre Pearson R
        for z in range(N):
            Y_z = G_3d[z]
            Yp_z = Yp[z]
            num = np.linalg.norm(Yp_z - Y_z)
            den = np.linalg.norm(Y_z) + 1e-30
            rl2_per_slice.append(num / den)
            ssim_per_slice.append(float(ssim_2d(Yp_z, Y_z)))
            geom_true_vals.append(float(Y_z[geom_c, geom_c]))
            geom_pred_vals.append(float(Yp_z[geom_c, geom_c]))

        # Per-volume RL² and inclusion-centroid voxel
        num = np.linalg.norm(Yp - G_3d)
        den = np.linalg.norm(G_3d) + 1e-30
        per_volume_rl2.append(num / den)

        cz, cy, cx = find_3d_inclusion_centre(G_3d)
        inc_true_vals.append(float(G_3d[cz, cy, cx]))
        inc_pred_vals.append(float(Yp[cz, cy, cx]))

        # Dense voxel subsample for population-level Pearson R
        idxs = sub_rng.choice(N ** 3, size=DENSE_PER_VOLUME, replace=False)
        g_true_dense.append(G_3d.ravel()[idxs])
        g_pred_dense.append(Yp.ravel()[idxs])

        if keep_one is None:
            keep_one = (G_3d.copy(), Yp.copy(), (cz, cy, cx), v)

    elapsed = time.time() - t0
    rl2_per_slice = np.array(rl2_per_slice)
    ssim_per_slice = np.array(ssim_per_slice)
    inc_true_vals = np.array(inc_true_vals)
    inc_pred_vals = np.array(inc_pred_vals)
    geom_true_vals = np.array(geom_true_vals)
    geom_pred_vals = np.array(geom_pred_vals)
    per_volume_rl2 = np.array(per_volume_rl2)
    g_true_dense = np.concatenate(g_true_dense)
    g_pred_dense = np.concatenate(g_pred_dense)

    R_inc,    _ = pearsonr(inc_pred_vals,  inc_true_vals)
    R_geom,   _ = pearsonr(geom_pred_vals, geom_true_vals)
    R_voxel,  _ = pearsonr(g_pred_dense,   g_true_dense)

    print()
    print("=" * 60)
    print(f"  Slice-by-slice 3D inversion — {n_volumes} volumes ({n_volumes*N} slices)")
    print("=" * 60)
    print(f"  Mean RL² per slice           = {rl2_per_slice.mean():.4f}")
    print(f"  Mean SSIM per slice          = {ssim_per_slice.mean():.4f}")
    print(f"  Mean RL² per volume          = {per_volume_rl2.mean():.4f}")
    print()
    print(f"  Pearson R — geom-centre (ILI metric)  = {R_geom:.4f}    "
          f"({len(geom_true_vals)} slices)")
    print(f"  Pearson R — dense voxels             = {R_voxel:.4f}    "
          f"({len(g_true_dense):,} voxels)")
    print(f"  Pearson R — inclusion centroid       = {R_inc:.4f}    "
          f"({n_volumes} per-volume)")
    print()
    print("  ILI reference (Scott/Murphy 2020, noisy 9³):")
    print("    DI = 0.685   HLI = 0.798   ILI = 0.940")
    print(f"  Wall time: {elapsed:.1f} s")
    print("=" * 60)

    # 3-orthogonal-view figure for one volume
    G_3d, Yp, (cz, cy, cx), vid = keep_one
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(f"Slice-by-slice 3D inversion — volume {vid}  "
                 f"(centre voxel z={cz}, y={cy}, x={cx})", fontsize=13)
    vmin = min(G_3d.min(), Yp.min())
    vmax = max(G_3d.max(), Yp.max())

    titles = ["axial  (z = const)", "coronal  (y = const)", "sagittal  (x = const)"]
    slices = [(G_3d[cz], Yp[cz]),
              (G_3d[:, cy, :], Yp[:, cy, :]),
              (G_3d[:, :, cx], Yp[:, :, cx])]
    for col, (title, (gt, pr)) in enumerate(zip(titles, slices)):
        im0 = axes[0, col].imshow(gt, cmap="hot", origin="lower",
                                    vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f"{title}\nGround truth", fontsize=10)
        axes[0, col].axis("off")
        plt.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)

        im1 = axes[1, col].imshow(pr, cmap="hot", origin="lower",
                                    vmin=vmin, vmax=vmax)
        axes[1, col].set_title("FNO slice-by-slice", fontsize=10)
        axes[1, col].axis("off")
        plt.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_png = out_dir / "slice_by_slice_3d.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"\nSaved {out_png}")

    # Scatter plot — use geometric-centre (ILI metric)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(geom_true_vals, geom_pred_vals, s=8, alpha=0.4)
    lim = [0, max(geom_true_vals.max(), geom_pred_vals.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("True G at slice geometric centre [Pa]")
    ax.set_ylabel("Predicted G [Pa]")
    ax.set_title(f"Slice-by-slice 3D inversion (ILI metric)\n"
                 f"Pearson R = {R_geom:.3f}  "
                 f"({len(geom_true_vals):,} slices, {n_volumes} volumes)\n"
                 f"vs ILI published R = 0.940")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_png2 = out_dir / "slice_by_slice_scatter.png"
    plt.savefig(out_png2, dpi=140, bbox_inches="tight")
    print(f"Saved {out_png2}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_volumes", type=int, default=10)
    p.add_argument("--N",         type=int, default=64)
    p.add_argument("--ckpt",      default="runs/phase0_v2/best.pt")
    p.add_argument("--out_dir",   default="runs/phase0_v2")
    args = p.parse_args()
    main(n_volumes=args.n_volumes, N=args.N,
         ckpt_path=args.ckpt, out_dir=args.out_dir)
