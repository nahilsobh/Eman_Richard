"""Extract per-voxel x/y line pairs from the BBIR UDel 2D HDF5 for
multi-view contrastive SSL.

For each axial slice, sample K voxels inside the brain mask.  For each
sampled voxel (row i, col j), extract:
  - the x-line through it: slice[:, i, :]   (2 channels, length L)
  - the y-line through it: slice[:, :, j]   (2 channels, length L)
  - the ground-truth G at that voxel:       (2,)  -- needed only for
                                                     downstream eval

These two lines pass through the *same* underlying tissue point and so
share the unknown G(r) at that point.  Contrastive SSL pulls their
representations together while pushing apart representations from
different voxels.

Output HDF5 schema:
  X_x       (N_pairs, 2, L)   float32   x-line displacement (Re, Im)
  X_y       (N_pairs, 2, L)   float32   y-line displacement
  G_at_vox  (N_pairs, 2)       float32   Re/Im G at the shared voxel (Pa)
  meta/subject (N_pairs,) S32
  meta/freq    (N_pairs,) int32
  meta/z_idx   (N_pairs,) int32
  meta/i       (N_pairs,) int16   row of the shared voxel (y-coordinate)
  meta/j       (N_pairs,) int16   col of the shared voxel (x-coordinate)

Defaults sample K = 8 voxels per slice, yielding ~140k pairs over 17,535
slices.  Pairs are restricted to voxels at least 8 voxels from any edge
(line still has lots of brain on both sides).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


SRC_DEFAULT = Path("/projects/bfid/sobh/data/bbir_udel_2d_slices.h5")
DST_DEFAULT = Path("/projects/bfid/sobh/data/bbir_udel_multiview.h5")


def extract(src_path: Path = SRC_DEFAULT, dst_path: Path = DST_DEFAULT,
             k_per_slice: int = 8, edge_margin: int = 8,
             min_line_brain_frac: float = 0.30,
             seed: int = 0, verbose: bool = True) -> dict:
    src_path, dst_path = Path(src_path), Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Preload the entire source 2D HDF5 into memory.  Per-slice random
    # access through gzip-compressed chunks is the bottleneck (~5% CPU,
    # almost all spent on inflate); one bulk read decompresses each chunk
    # exactly once.
    print("Preloading source 2D HDF5 into memory ...", flush=True)
    import time as _time
    _t0 = _time.time()
    with h5py.File(src_path, "r") as h_in:
        all_X     = np.asarray(h_in["X"][:],      dtype=np.float32)
        all_G_re  = np.asarray(h_in["Y_G_re"][:], dtype=np.float32)
        all_G_im  = np.asarray(h_in["Y_G_im"][:], dtype=np.float32)
        all_mask  = np.asarray(h_in["mask"][:],   dtype=np.bool_)
        subj_arr  = h_in["meta/subject"][:].astype("S32")
        freq_arr  = h_in["meta/freq"][:].astype(np.int32)
        zix_arr   = h_in["meta/z_idx"][:].astype(np.int32)
    n_slices = all_X.shape[0]
    L        = all_X.shape[-1]
    print(f"  loaded in {_time.time()-_t0:.1f}s "
           f"(X {all_X.nbytes/1e9:.2f}GB + Y {all_G_re.nbytes*2/1e9:.2f}GB)",
           flush=True)

    with h5py.File(dst_path, "w") as h_out:

        chunk_n = 256
        ds_Xx   = h_out.create_dataset("X_x", shape=(0, 2, L),
                                        maxshape=(None, 2, L),
                                        dtype=np.float32,
                                        chunks=(chunk_n, 2, L),
                                        compression="gzip", compression_opts=4)
        ds_Xy   = h_out.create_dataset("X_y", shape=(0, 2, L),
                                        maxshape=(None, 2, L),
                                        dtype=np.float32,
                                        chunks=(chunk_n, 2, L),
                                        compression="gzip", compression_opts=4)
        ds_G    = h_out.create_dataset("G_at_vox", shape=(0, 2),
                                        maxshape=(None, 2),
                                        dtype=np.float32)
        ds_s = h_out.create_dataset("meta/subject", shape=(0,),
                                     maxshape=(None,), dtype="S32")
        ds_f = h_out.create_dataset("meta/freq",    shape=(0,),
                                     maxshape=(None,), dtype=np.int32)
        ds_z = h_out.create_dataset("meta/z_idx",   shape=(0,),
                                     maxshape=(None,), dtype=np.int32)
        ds_i = h_out.create_dataset("meta/i",       shape=(0,),
                                     maxshape=(None,), dtype=np.int16)
        ds_j = h_out.create_dataset("meta/j",       shape=(0,),
                                     maxshape=(None,), dtype=np.int16)

        n_total = 0
        skipped_slices = 0

        for s in range(n_slices):
            X2   = all_X[s]
            G_re = all_G_re[s]
            G_im = all_G_im[s]
            mask = all_mask[s]

            # Candidate voxels: in brain AND not too close to edges
            valid_i, valid_j = np.where(mask)
            keep = ((valid_i >= edge_margin) & (valid_i < L - edge_margin) &
                    (valid_j >= edge_margin) & (valid_j < L - edge_margin))
            valid_i = valid_i[keep]; valid_j = valid_j[keep]
            if len(valid_i) == 0:
                skipped_slices += 1
                continue

            # Further filter by requiring the two lines through (i,j) to be
            # brain-rich enough
            pick_mask = np.array([
                (mask[i, :].mean() >= min_line_brain_frac) and
                (mask[:, j].mean() >= min_line_brain_frac)
                for i, j in zip(valid_i, valid_j)
            ])
            valid_i = valid_i[pick_mask]; valid_j = valid_j[pick_mask]
            if len(valid_i) == 0:
                skipped_slices += 1
                continue

            n_pick = min(k_per_slice, len(valid_i))
            sel = rng.choice(len(valid_i), n_pick, replace=False)
            chosen_i = valid_i[sel]; chosen_j = valid_j[sel]

            buf_Xx, buf_Xy, buf_G = [], [], []
            for i, j in zip(chosen_i, chosen_j):
                buf_Xx.append(X2[:, i, :])                  # (2, L)
                buf_Xy.append(X2[:, :, j])                  # (2, L)
                buf_G.append(np.array([G_re[i, j], G_im[i, j]]))

            n_new = len(buf_Xx)
            for ds in (ds_Xx, ds_Xy):
                ds.resize((ds.shape[0] + n_new,) + ds.shape[1:])
            for ds in (ds_G,):
                ds.resize((ds.shape[0] + n_new, 2))
            for ds in (ds_s, ds_f, ds_z, ds_i, ds_j):
                ds.resize((ds.shape[0] + n_new,))
            ds_Xx[-n_new:]  = np.stack(buf_Xx)
            ds_Xy[-n_new:]  = np.stack(buf_Xy)
            ds_G[-n_new:]   = np.stack(buf_G)
            ds_s[-n_new:]   = np.full(n_new, subj_arr[s], dtype="S32")
            ds_f[-n_new:]   = np.full(n_new, freq_arr[s], dtype=np.int32)
            ds_z[-n_new:]   = np.full(n_new, zix_arr[s],  dtype=np.int32)
            ds_i[-n_new:]   = chosen_i.astype(np.int16)
            ds_j[-n_new:]   = chosen_j.astype(np.int16)
            n_total += n_new

            if verbose and (s % 1000 == 0 or s == n_slices - 1):
                print(f"  slice {s+1:5d}/{n_slices}: +{n_new:2d} pairs  "
                       f"(running total {n_total:,})", flush=True)

        h_out.attrs["k_per_slice"]         = k_per_slice
        h_out.attrs["edge_margin"]         = edge_margin
        h_out.attrs["min_line_brain_frac"] = min_line_brain_frac
        h_out.attrs["L"]                   = L

    return {"n_pairs": n_total, "skipped_slices": skipped_slices,
             "out_path": str(dst_path)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC_DEFAULT))
    ap.add_argument("--dst", default=str(DST_DEFAULT))
    ap.add_argument("--k_per_slice", type=int, default=8)
    ap.add_argument("--edge_margin", type=int, default=8)
    ap.add_argument("--min_line_brain_frac", type=float, default=0.30)
    args = ap.parse_args()
    s = extract(Path(args.src), Path(args.dst),
                 k_per_slice=args.k_per_slice,
                 edge_margin=args.edge_margin,
                 min_line_brain_frac=args.min_line_brain_frac)
    print(f"\nWrote {s['out_path']}")
    print(f"  {s['n_pairs']:,} multi-view pairs  "
          f"({s['skipped_slices']} slices skipped)")


if __name__ == "__main__":
    main()
