#!/usr/bin/env python3
"""Generate 1D eigenstrain training pairs and save to HDF5."""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.generator_1d import make_1d_pair, N_DEFAULT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=20000)
    p.add_argument("--output",    default="data/1d_pairs_{n}.h5")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--N",         type=int, default=N_DEFAULT)
    args = p.parse_args()

    out = Path(args.output.format(n=args.n_samples))
    out.parent.mkdir(parents=True, exist_ok=True)
    dx = 2 * 0.10 / args.N

    rng = np.random.default_rng(args.seed)

    with h5py.File(out, "w") as f:
        X_ds    = f.create_dataset("X",            shape=(args.n_samples, 5, args.N), dtype="f4")
        eps_t   = f.create_dataset("Y_eps_true",   shape=(args.n_samples, args.N),    dtype="f4")
        eps_a   = f.create_dataset("Y_eps_analytic", shape=(args.n_samples, args.N),  dtype="f4")
        meta_grp = f.create_group("meta")
        sb_ds   = meta_grp.create_dataset("sigma_bar",    shape=(args.n_samples,), dtype="f4")
        ac_ds   = meta_grp.create_dataset("A_coeff",      shape=(args.n_samples,), dtype="f4")
        eb_ds   = meta_grp.create_dataset("E_bg",         shape=(args.n_samples,), dtype="f4")
        e0_ds   = meta_grp.create_dataset("eps0",         shape=(args.n_samples,), dtype="f4")
        el_ds   = meta_grp.create_dataset("ell",          shape=(args.n_samples,), dtype="f4")
        sn_ds   = meta_grp.create_dataset("snr_db",       shape=(args.n_samples,), dtype="f4")
        ie_ds   = meta_grp.create_dataset("is_expanding", shape=(args.n_samples,), dtype="bool")

        for i in tqdm(range(args.n_samples), desc="Generating"):
            X, Y, _ = make_1d_pair(N=args.N, dx=dx, rng=rng)
            X_ds[i]  = X
            eps_t[i] = Y["eps_star_true"]
            eps_a[i] = Y["eps_star_analytic"]
            sb_ds[i] = Y["sigma_bar"]
            ac_ds[i] = Y["A_coeff"]
            eb_ds[i] = Y["E_bg"]
            e0_ds[i] = Y["eps0"]
            el_ds[i] = Y["ell"]
            sn_ds[i] = Y["snr_db"]
            ie_ds[i] = Y["is_expanding"]

    print(f"Saved {args.n_samples} samples to {out}")


if __name__ == "__main__":
    main()
