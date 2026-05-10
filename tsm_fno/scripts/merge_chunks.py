#!/usr/bin/env python3
"""Merge per-task TSM chunks into a single dataset HDF5."""
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    chunk_dir = Path("data/chunks_80hz")
    out_path = Path("data/tsm_50000_80hz.h5")

    chunks = sorted(chunk_dir.glob("tsm_chunk_*.h5"))
    if not chunks:
        raise RuntimeError(f"No chunks found in {chunk_dir}")
    print(f"Merging {len(chunks)} chunks into {out_path}")

    Xs, Gs, EPSs, Rings = [], [], [], []
    metas = {k: [] for k in
             ["p", "A", "G_bg", "G_lesion", "a_eq_m", "is_expanding", "snr_db"]}

    for cf in chunks:
        with h5py.File(cf, "r") as f:
            Xs.append(f["X"][:])
            Gs.append(f["Y_G"][:])
            EPSs.append(f["Y_epsilon"][:])
            Rings.append(f["Y_ring"][:])
            for k in metas:
                metas[k].append(f["meta/" + k][:])

    X_all = np.concatenate(Xs, axis=0)
    G_all = np.concatenate(Gs, axis=0)
    EPS_all = np.concatenate(EPSs, axis=0)
    Ring_all = np.concatenate(Rings, axis=0)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("X",         data=X_all,    dtype=np.float32)
        f.create_dataset("Y_G",       data=G_all,    dtype=np.float32)
        f.create_dataset("Y_epsilon", data=EPS_all,  dtype=np.float32)
        f.create_dataset("Y_ring",    data=Ring_all)
        m = f.create_group("meta")
        for k, arrs in metas.items():
            m.create_dataset(k, data=np.concatenate(arrs, axis=0))
        f.attrs["n_total"] = len(X_all)
        f.attrs["channels"] = "Re_u80,Im_u80,Lame,dist"

    n = len(X_all)
    size_mb = out_path.stat().st_size / 1e6
    print(f"Total samples       : {n:,}")
    print(f"File size           : {size_mb:.1f} MB")
    print(f"X shape             : {X_all.shape}")
    print(f"Expanding fraction  : {metas['is_expanding'][0].dtype} "
          f"{np.concatenate(metas['is_expanding']).mean():.2%}")
    print(f"G range             : {G_all.min():.0f} – {G_all.max():.0f} Pa")
    print(f"ε range             : {EPS_all.min():.3f} – {EPS_all.max():.3f}")
    p_arr = np.concatenate(metas['p'])
    print(f"Pressure range      : {p_arr.min():.0f} – {p_arr.max():.0f} Pa")

    for cf in chunks:
        cf.unlink()
    print("Chunk files deleted.")


if __name__ == "__main__":
    main()
