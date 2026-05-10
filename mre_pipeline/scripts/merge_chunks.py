#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import h5py


def main():
    chunk_dir = Path("data/chunks")
    out_path = Path("data/mre_v3_50000.h5")
    expected_chunks = 100
    expected_per_chunk = 500

    chunk_files = sorted(chunk_dir.glob("chunk_*.h5"))
    if len(chunk_files) != expected_chunks:
        raise RuntimeError(
            f"Expected {expected_chunks} chunks, found {len(chunk_files)}"
        )

    X_list, Y_list, damp_list, src_list, snr_list = [], [], [], [], []
    for cf in chunk_files:
        with h5py.File(cf, "r") as f:
            n = f["X"].shape[0]
            if n != expected_per_chunk:
                raise RuntimeError(f"{cf}: expected {expected_per_chunk} samples, got {n}")
            X_list.append(f["X"][:])
            Y_list.append(f["Y"][:])
            damp_list.append(f["damping"][:])
            src_list.append(f["n_src"][:])
            snr_list.append(f["snr_db"][:])

    X_all   = np.concatenate(X_list, axis=0)
    Y_all   = np.concatenate(Y_list, axis=0)
    damp_all = np.concatenate(damp_list, axis=0)
    src_all  = np.concatenate(src_list, axis=0)
    snr_all  = np.concatenate(snr_list, axis=0)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("X",       data=X_all,    dtype=np.float32)
        f.create_dataset("Y",       data=Y_all,    dtype=np.float32)
        f.create_dataset("damping", data=damp_all, dtype=np.float32)
        f.create_dataset("n_src",   data=src_all,  dtype=np.int32)
        f.create_dataset("snr_db",  data=snr_all,  dtype=np.float32)
        f.attrs["n_total"] = len(X_all)
        f.attrs["channels"] = "Re(u_60Hz),Im(u_60Hz)"

    total = len(X_all)
    size_mb = out_path.stat().st_size / 1e6
    print(f"Total samples : {total}")
    print(f"File size     : {size_mb:.1f} MB")
    print(f"X shape       : {X_all.shape}")
    print(f"Y shape       : {Y_all.shape}")
    print(f"G min/max     : {Y_all.min():.1f} / {Y_all.max():.1f} Pa")
    print(f"damping range : {damp_all.min():.3f} – {damp_all.max():.3f}")
    print(f"n_src range   : {src_all.min()} – {src_all.max()}")
    print(f"SNR range     : {snr_all.min():.1f} – {snr_all.max():.1f} dB")

    for cf in chunk_files:
        cf.unlink()
    print("Chunk files deleted.")


if __name__ == "__main__":
    main()
