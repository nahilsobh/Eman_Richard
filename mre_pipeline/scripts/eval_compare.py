#!/usr/bin/env python3
"""Compare FNO (best.pt) vs Direct Inversion baseline on the v2 validation set.

Reports ILI-style Pearson R at inclusion-center voxels plus RL², SSIM, and
contrast-stratified accuracy. Designed to read the same val split used during
training (last 10% of the dataset)."""
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fno_model import FNO2d
from src.direct_inversion import direct_inversion_from_X


def ssim_per_sample(pred, target, C1=1e-4, C2=9e-4):
    out = []
    for i in range(len(pred)):
        p, t = pred[i], target[i]
        mu_p, mu_t = p.mean(), t.mean()
        sig_p, sig_t = p.var(), t.var()
        sig_pt = ((p - mu_p) * (t - mu_t)).mean()
        s = ((2 * mu_p * mu_t + C1) * (2 * sig_pt + C2)) / \
            ((mu_p ** 2 + mu_t ** 2 + C1) * (sig_p + sig_t + C2))
        out.append(float(s))
    return np.array(out)


def rl2_per_sample(pred, target):
    num = np.linalg.norm(pred - target, axis=(-2, -1))
    den = np.linalg.norm(target, axis=(-2, -1)) + 1e-30
    return num / den


def inclusion_center(Y):
    """Return (i, j) of the centroid of the brightest connected stiffness region."""
    G_max = Y.max()
    bg = np.median(Y)
    threshold = bg + 0.5 * (G_max - bg)
    mask = Y >= threshold
    if mask.sum() == 0:
        return Y.shape[0] // 2, Y.shape[1] // 2
    ys, xs = np.where(mask)
    return int(np.round(ys.mean())), int(np.round(xs.mean()))


def main(
    data_path="data/mre_v2_50000.h5",
    ckpt_path="runs/phase0_v2/best.pt",
    val_frac=0.1,
    out_dir="runs/phase0_v2",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    stats = ckpt["stats"]
    epoch = ckpt["epoch"]
    n_in = stats["X_mean"].shape[1]
    print(f"Loaded checkpoint: epoch {epoch}, input channels = {n_in}")

    model = FNO2d(modes1=12, modes2=12, width=32, n_layers=4, in_channels=n_in)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    log_min = stats["Y_log_min"]
    log_max = stats["Y_log_max"]
    def denorm_Y(yn):
        return (yn * (log_max - log_min) + log_min).exp()
    X_mean = stats["X_mean"]
    X_std  = stats["X_std"]

    f = h5py.File(data_path, "r")
    n_total = f["X"].shape[0]
    val_start = int(n_total * (1 - val_frac))
    n_val = n_total - val_start
    print(f"Eval set: {n_val} samples (idx {val_start}–{n_total - 1})")

    BATCH = 128
    fno_pred  = np.empty((n_val, 64, 64), dtype=np.float32)
    di_pred   = np.empty((n_val, 64, 64), dtype=np.float32)
    Y_true    = np.empty((n_val, 64, 64), dtype=np.float32)
    contrasts = np.empty(n_val,           dtype=np.float32)

    with torch.no_grad():
        for k0 in range(0, n_val, BATCH):
            k1 = min(k0 + BATCH, n_val)
            sl = slice(val_start + k0, val_start + k1)
            Xb = f["X"][sl]
            Yb = f["Y"][sl]
            Xt = (torch.tensor(Xb) - X_mean) / X_std
            Yp_n = model(Xt)
            Yp = denorm_Y(Yp_n).numpy()
            fno_pred[k0:k1] = Yp
            for j in range(k1 - k0):
                di_pred[k0 + j] = direct_inversion_from_X(Xb[j])
            Y_true[k0:k1] = Yb
            contrasts[k0:k1] = Yb.max(axis=(-2, -1)) / np.maximum(Yb.min(axis=(-2, -1)), 1.0)
            if k0 % (BATCH * 8) == 0:
                print(f"  evaluated {k1}/{n_val}", flush=True)

    rl2_fno  = rl2_per_sample(fno_pred,  Y_true)
    rl2_di   = rl2_per_sample(di_pred,   Y_true)
    ssim_fno = ssim_per_sample(fno_pred, Y_true)
    ssim_di  = ssim_per_sample(di_pred,  Y_true)

    centers = np.array([inclusion_center(Y_true[i]) for i in range(n_val)])
    ci, cj = centers[:, 0], centers[:, 1]
    g_true_c = Y_true [np.arange(n_val), ci, cj]
    g_fno_c  = fno_pred[np.arange(n_val), ci, cj]
    g_di_c   = di_pred [np.arange(n_val), ci, cj]
    R_fno, _ = pearsonr(g_fno_c, g_true_c)
    R_di,  _ = pearsonr(g_di_c,  g_true_c)

    print()
    print("=" * 60)
    print(f"  Phase 0 v2 — FNO vs DI on {n_val} val samples")
    print("=" * 60)
    print(f"  Mean RL²       FNO = {rl2_fno.mean():.4f}    DI = {rl2_di.mean():.4f}")
    print(f"  Mean SSIM      FNO = {ssim_fno.mean():.4f}    DI = {ssim_di.mean():.4f}")
    print(f"  Pearson R (inclusion center)  FNO = {R_fno:.4f}    DI = {R_di:.4f}")
    print()
    print("  ILI reference (Scott/Murphy 2020, noisy 9³, 60 Hz):")
    print("    DI  = 0.685     HLI = 0.798     ILI = 0.940")
    print("=" * 60)

    bins = [(1, 2), (2, 5), (5, 10), (10, 20), (20, 60)]
    print(f"\n  Stratified by contrast (max/min):")
    print(f"  {'bin':>10}  {'N':>6}  {'RL² FNO':>9}  {'RL² DI':>9}  "
          f"{'SSIM FNO':>10}  {'SSIM DI':>10}  {'R FNO':>8}  {'R DI':>8}")
    for lo, hi in bins:
        m = (contrasts >= lo) & (contrasts < hi)
        if m.sum() < 5:
            continue
        rfno = pearsonr(g_fno_c[m], g_true_c[m])[0]
        rdi  = pearsonr(g_di_c[m],  g_true_c[m])[0]
        print(f"  {lo:2d}×–{hi:2d}×  {m.sum():>6}  "
              f"{rl2_fno[m].mean():>9.4f}  {rl2_di[m].mean():>9.4f}  "
              f"{ssim_fno[m].mean():>10.4f}  {ssim_di[m].mean():>10.4f}  "
              f"{rfno:>8.3f}  {rdi:>8.3f}")

    # Save scatter
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, gp, lab, R in [
        (axes[0], g_di_c,  "Direct Inversion", R_di),
        (axes[1], g_fno_c, "FNO",              R_fno),
    ]:
        ax.scatter(g_true_c, gp, s=4, alpha=0.4)
        lim = [0, max(g_true_c.max(), gp.max()) * 1.05]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.set_xlabel("True G at inclusion centre [Pa]")
        ax.set_ylabel("Predicted G [Pa]")
        ax.set_title(f"{lab}    R = {R:.3f}")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    out_png = out_dir / "scatter_fno_vs_di.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"\n  Saved {out_png}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="data/mre_v2_50000.h5")
    p.add_argument("--ckpt",      default="runs/phase0_v2/best.pt")
    p.add_argument("--out_dir",   default="runs/phase0_v2")
    args = p.parse_args()
    main(data_path=args.data_path, ckpt_path=args.ckpt, out_dir=args.out_dir)
