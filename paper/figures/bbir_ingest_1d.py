"""Extract 1D voxel lines from the BBIR UDel 2D HDF5.

Each 2D axial slice (160x160 downsampled to 80x80) yields:
  - 80 lines along x (rows, fixed y)
  - 80 lines along y (cols, fixed x)

Lines whose brain-mask coverage is below ``min_brain_frac`` are dropped.
The output HDF5 has flat arrays:
  X       (N, 2, 80)  float32   complex displacement (Re, Im)
  Y       (N, 2, 80)  float32   complex G (Re, Im)  / G_scale at write time? no -- raw Pa
  mask    (N,   80)   bool      brain mask along the line
  meta/subject  bytes(N)
  meta/freq     int32(N)
  meta/z_idx    int32(N)
  meta/axis     int8(N)         0 = x, 1 = y
  meta/transverse_idx int32(N)   y for x-lines, x for y-lines

Notes
-----
* Z-direction lines (length 80) are out of scope here -- they would
  require a separate ingest from the source 3D NIfTI volumes.
* G label values are in Pa.  The training script divides by G_scale.
* The 2D HDF5 carries 80x80 in-plane resolution, ~3 mm per voxel.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


SRC_DEFAULT = Path("/projects/bfid/sobh/data/bbir_udel_2d_slices.h5")
DST_DEFAULT = Path("/projects/bfid/sobh/data/bbir_udel_1d_lines.h5")


def extract_lines(src_path: Path = SRC_DEFAULT,
                   dst_path: Path = DST_DEFAULT,
                   min_brain_frac: float = 0.30,
                   verbose: bool = True) -> dict:
    """Walk the 2D HDF5, extract row+col lines, write a new HDF5."""
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(src_path, "r") as h_in, h5py.File(dst_path, "w") as h_out:
        n_slices = h_in["X"].shape[0]
        L = h_in["X"].shape[-1]                            # 80
        subj_arr = h_in["meta/subject"][:].astype("S32")
        freq_arr = h_in["meta/freq"][:].astype(np.int32)
        zix_arr  = h_in["meta/z_idx"][:].astype(np.int32)

        # Pre-allocate growable datasets
        chunk_n = 1024
        ds_X  = h_out.create_dataset("X", shape=(0, 2, L), maxshape=(None, 2, L),
                                      dtype=np.float32, chunks=(chunk_n, 2, L),
                                      compression="gzip", compression_opts=4)
        ds_Y  = h_out.create_dataset("Y", shape=(0, 2, L), maxshape=(None, 2, L),
                                      dtype=np.float32, chunks=(chunk_n, 2, L),
                                      compression="gzip", compression_opts=4)
        ds_M  = h_out.create_dataset("mask", shape=(0, L), maxshape=(None, L),
                                      dtype=np.bool_, chunks=(chunk_n, L),
                                      compression="gzip", compression_opts=4)
        ds_s  = h_out.create_dataset("meta/subject", shape=(0,),
                                      maxshape=(None,), dtype="S32")
        ds_f  = h_out.create_dataset("meta/freq", shape=(0,),
                                      maxshape=(None,), dtype=np.int32)
        ds_z  = h_out.create_dataset("meta/z_idx", shape=(0,),
                                      maxshape=(None,), dtype=np.int32)
        ds_a  = h_out.create_dataset("meta/axis", shape=(0,),
                                      maxshape=(None,), dtype=np.int8)
        ds_t  = h_out.create_dataset("meta/transverse_idx", shape=(0,),
                                      maxshape=(None,), dtype=np.int32)

        n_total = 0
        n_kept_per_slice = []

        for i in range(n_slices):
            X2 = np.asarray(h_in["X"][i],       dtype=np.float32)     # (2, 80, 80)
            G_re = np.asarray(h_in["Y_G_re"][i], dtype=np.float32)
            G_im = np.asarray(h_in["Y_G_im"][i], dtype=np.float32)
            mask = np.asarray(h_in["mask"][i],  dtype=np.bool_)
            subj = subj_arr[i]; f = freq_arr[i]; z = zix_arr[i]

            buf_X, buf_Y, buf_M = [], [], []
            buf_s, buf_f, buf_z, buf_a, buf_t = [], [], [], [], []

            # x-lines: for each y row, line along x
            for y in range(L):
                m_line = mask[y, :]
                if m_line.mean() < min_brain_frac:
                    continue
                buf_X.append(X2[:, y, :])              # (2, L)
                buf_Y.append(np.stack([G_re[y, :], G_im[y, :]]))
                buf_M.append(m_line)
                buf_s.append(subj); buf_f.append(f); buf_z.append(z)
                buf_a.append(0); buf_t.append(y)
            # y-lines: for each x col, line along y
            for x in range(L):
                m_line = mask[:, x]
                if m_line.mean() < min_brain_frac:
                    continue
                buf_X.append(X2[:, :, x])              # (2, L)
                buf_Y.append(np.stack([G_re[:, x], G_im[:, x]]))
                buf_M.append(m_line)
                buf_s.append(subj); buf_f.append(f); buf_z.append(z)
                buf_a.append(1); buf_t.append(x)

            if not buf_X:
                n_kept_per_slice.append(0)
                continue
            n_new = len(buf_X)
            for ds in (ds_X, ds_Y, ds_M):
                ds.resize((ds.shape[0] + n_new,) + ds.shape[1:])
            for ds in (ds_s, ds_f, ds_z, ds_a, ds_t):
                ds.resize((ds.shape[0] + n_new,))
            ds_X[-n_new:]    = np.stack(buf_X)
            ds_Y[-n_new:]    = np.stack(buf_Y)
            ds_M[-n_new:]    = np.stack(buf_M)
            ds_s[-n_new:]    = np.array(buf_s, dtype="S32")
            ds_f[-n_new:]    = np.array(buf_f, dtype=np.int32)
            ds_z[-n_new:]    = np.array(buf_z, dtype=np.int32)
            ds_a[-n_new:]    = np.array(buf_a, dtype=np.int8)
            ds_t[-n_new:]    = np.array(buf_t, dtype=np.int32)
            n_total += n_new
            n_kept_per_slice.append(n_new)
            if verbose and (i % 500 == 0 or i == n_slices - 1):
                print(f"  slice {i+1:5d}/{n_slices}: {n_new:3d} lines "
                       f"(running total {n_total})", flush=True)

        h_out.attrs["min_brain_frac"] = min_brain_frac
        h_out.attrs["L"]              = L
        h_out.attrs["n_subjects"]     = len(set(subj_arr.tolist()))

    return {
        "n_lines":    n_total,
        "n_subjects": len(set(subj_arr.tolist())),
        "out_path":   str(dst_path),
        "mean_per_slice": float(np.mean(n_kept_per_slice)) if n_kept_per_slice else 0.0,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC_DEFAULT))
    ap.add_argument("--dst", default=str(DST_DEFAULT))
    ap.add_argument("--min_brain_frac", type=float, default=0.30)
    args = ap.parse_args()
    summary = extract_lines(Path(args.src), Path(args.dst),
                             min_brain_frac=args.min_brain_frac)
    print()
    print(f"Wrote {summary['out_path']}")
    print(f"  {summary['n_subjects']} subjects, {summary['n_lines']:,} lines  "
          f"({summary['mean_per_slice']:.1f} lines per slice on average)")


if __name__ == "__main__":
    main()
