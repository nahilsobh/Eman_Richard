#!/usr/bin/env python3
"""Run the full evaluation suite on a trained TSM-FNO checkpoint."""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import TSMDataset
from src.model.fno_tsm import FNO_TSM
from src.eval.metrics import (relative_l2_np, ssim_np, ring_rl2,
                                expansion_auc, recover_acoustoelastic_constant,
                                stratified_table)
from src.eval.visualize import benchmark_figure


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="runs/tsm/best.pt")
    p.add_argument("--data_path",  default="data/tsm_50000.h5")
    p.add_argument("--save_dir",   default="results/tsm")
    p.add_argument("--n_figure_samples", type=int, default=6)
    args = p.parse_args()

    save = Path(args.save_dir); save.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg_path = Path(args.checkpoint).parent / "args.json"
    cfg = json.load(open(cfg_path)) if cfg_path.exists() else {}
    model = FNO_TSM(
        in_channels=cfg.get("in_channels", 6),
        modes1=cfg.get("modes", 12),
        modes2=cfg.get("modes", 12),
        width=cfg.get("width", 48),
        n_layers=cfg.get("n_layers", 4),
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    print(f"Model: in_channels={cfg.get('in_channels',6)}, modes={cfg.get('modes',12)}, "
          f"width={cfg.get('width',48)}", flush=True)

    ds = TSMDataset(args.data_path, augment=False)
    with h5py.File(args.data_path, "r") as f:
        is_exp = f["meta/is_expanding"][:]
        G_bg_all = f["meta/G_bg"][:]
        A_all    = f["meta/A"][:]
        p_arr    = f["meta/p"][:]
        Glesion  = f["meta/G_lesion"][:]
        a_eq     = f["meta/a_eq_m"][:]

    n = len(ds)
    val_start = int(n * 0.9)
    val_idx = list(range(val_start, n))
    print(f"Eval set: {len(val_idx)} samples")

    loader = DataLoader(Subset(ds, val_idx), batch_size=32,
                         shuffle=False, num_workers=2)

    _sample = ds[val_idx[0]]
    _N = _sample["G"].shape[-1]
    G_pred_all = np.empty((len(val_idx), _N, _N), dtype=np.float32)
    eps_pred_all = np.empty_like(G_pred_all)
    A_pred_all = np.empty(len(val_idx), dtype=np.float32)

    with torch.no_grad():
        for k, batch in enumerate(loader):
            X = batch["X"].to(device)
            G_p, eps_p, A_p = model(X)
            sl = slice(k * 32, k * 32 + len(X))
            G_pred_all[sl]   = G_p.cpu().numpy()
            eps_pred_all[sl] = eps_p.cpu().numpy()
            A_pred_all[sl]   = A_p.cpu().numpy()

    # Build numpy arrays of GT for the val slice
    with h5py.File(args.data_path, "r") as f:
        G_true   = f["Y_G"][val_start:]
        eps_true = f["Y_epsilon"][val_start:]
        ring     = f["Y_ring"][val_start:]
        G_bg_v   = G_bg_all[val_start:]
        A_v      = A_all[val_start:]
        is_exp_v = is_exp[val_start:].astype(bool)
        p_v      = p_arr[val_start:]
        Gle_v    = Glesion[val_start:]
        a_eq_v   = a_eq[val_start:]

    # Metrics
    rl2_G  = relative_l2_np(G_pred_all,   G_true)
    rl2_e  = relative_l2_np(eps_pred_all, eps_true)
    ring_e = ring_rl2(eps_pred_all, eps_true, ring)
    ssim_G = ssim_np(G_pred_all, G_true)
    ssim_e = ssim_np(eps_pred_all, eps_true)
    auc    = expansion_auc(eps_pred_all, ring, is_exp_v)
    A_pred_used = np.abs(A_pred_all)
    A_recovered, A_err_rel = recover_acoustoelastic_constant(
        G_pred_all, eps_pred_all, G_bg_v, ring, A_v
    )

    summary = {
        "n_eval":          int(len(val_idx)),
        "rl2_G_mean":      float(rl2_G.mean()),
        "rl2_eps_mean":    float(rl2_e.mean()),
        "ring_rl2_mean":   float(ring_e.mean()),
        "ssim_G_mean":     float(ssim_G.mean()),
        "ssim_eps_mean":   float(ssim_e.mean()),
        "expansion_auc":   float(auc),
        "A_recovered_mean":  float(np.nanmean(A_recovered)),
        "A_relative_err":    float(np.nanmean(A_err_rel)),
    }
    print("=" * 56)
    for k, v in summary.items():
        print(f"  {k:24s} = {v}")
    print("=" * 56)
    json.dump(summary, open(save / "summary.json", "w"), indent=2)

    # Stratified table
    contrast = Gle_v / np.maximum(G_bg_v, 1.0)
    a_eq_mm = a_eq_v * 1000.0
    stratified_table(
        rl2_G=rl2_G, rl2_eps=rl2_e, ring_rl2=ring_e,
        pressure=p_v, contrast=contrast, size_mm=a_eq_mm,
        path=save / "stratified.txt",
    )

    # Benchmark figure
    benchmark_figure(
        X=ds[val_start]["X"].numpy()[None],   # placeholder; viz reads from h5
        h5_path=args.data_path, val_start=val_start,
        G_pred=G_pred_all, eps_pred=eps_pred_all,
        n_samples=args.n_figure_samples,
        save_path=save / "benchmark.png",
    )

    print(f"\nResults saved to {save}/")


if __name__ == "__main__":
    main()
