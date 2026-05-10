#!/usr/bin/env python3
"""SLURM-array entry point: generate one chunk of TSM training pairs."""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.generate_tsm import make_tsm_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--grid_size", type=int, default=64)
    parser.add_argument("--output_dir", type=Path, default=Path("data/chunks"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"tsm_chunk_{args.task_id:04d}.h5"

    rng = np.random.default_rng(args.task_id * 1000)

    n = args.chunk_size
    N = args.grid_size

    # Determine channel count from first sample
    _X0, _, _ = make_tsm_pair(N=N, rng=np.random.default_rng(0))
    n_channels = _X0.shape[0]

    X_all   = np.empty((n, n_channels, N, N), dtype=np.float32)
    G_all   = np.empty((n, N, N),    dtype=np.float32)
    eps_all = np.empty((n, N, N),    dtype=np.float32)
    ring_all = np.empty((n, N, N),   dtype=np.bool_)

    p_arr           = np.empty(n, dtype=np.float32)
    A_arr           = np.empty(n, dtype=np.float32)
    G_bg_arr        = np.empty(n, dtype=np.float32)
    G_lesion_arr    = np.empty(n, dtype=np.float32)
    a_eq_arr        = np.empty(n, dtype=np.float32)
    is_exp_arr      = np.empty(n, dtype=np.bool_)
    snr_arr         = np.empty(n, dtype=np.float32)

    for i in tqdm(range(n), desc=f"task {args.task_id}"):
        X, Y, meta = make_tsm_pair(N=N, rng=rng)
        X_all[i]    = X
        G_all[i]    = Y["G"]
        eps_all[i]  = Y["epsilon"]
        ring_all[i] = Y["ring"]
        p_arr[i]        = meta["p"]
        A_arr[i]        = meta["A"]
        G_bg_arr[i]     = meta["G_bg"]
        G_lesion_arr[i] = meta["G_lesion"]
        a_eq_arr[i]     = meta["a_eq_m"]
        is_exp_arr[i]   = meta["is_expanding"]
        snr_arr[i]      = meta["snr_db"]

    with h5py.File(out_path, "w") as f:
        f.create_dataset("X",         data=X_all,   dtype=np.float32)
        f.create_dataset("Y_G",       data=G_all,   dtype=np.float32)
        f.create_dataset("Y_epsilon", data=eps_all, dtype=np.float32)
        f.create_dataset("Y_ring",    data=ring_all)
        m = f.create_group("meta")
        m.create_dataset("p",            data=p_arr)
        m.create_dataset("A",            data=A_arr)
        m.create_dataset("G_bg",         data=G_bg_arr)
        m.create_dataset("G_lesion",     data=G_lesion_arr)
        m.create_dataset("a_eq_m",       data=a_eq_arr)
        m.create_dataset("is_expanding", data=is_exp_arr)
        m.create_dataset("snr_db",       data=snr_arr)
        f.attrs["chunk_id"] = args.task_id
        f.attrs["N"] = N
        f.attrs["channels"] = "Re_u80,Im_u80,Lame,dist" if n_channels == 4 \
            else "Re_u60,Im_u60,Re_u120,Im_u120,Lame,dist"

    print(f"Saved {n} samples to {out_path}")


if __name__ == "__main__":
    main()
