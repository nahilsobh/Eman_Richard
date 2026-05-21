#!/usr/bin/env python3
"""Visual check that BBIR slices look reasonable.

Reads the extracted HDF5 file (default /tmp/bbir_udel_smoke.h5) and plots,
for the first subject in the file, one axial slice per frequency along
with: displacement (Re, Im), shear modulus (Re, Im), brain mask, and
anatomical-ish overlay.

Output:
  paper/figures/bbir_inspect.png
"""
from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5",  type=str,
                    default="/tmp/bbir_udel_smoke.h5")
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).parent / "bbir_inspect.png"))
    ap.add_argument("--z_target", type=int, default=40,
                    help="Preferred axial slice index (will pick the slice "
                          "whose z_idx is closest)")
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as h:
        X    = h["X"][:]
        Yre  = h["Y_G_re"][:]
        Yim  = h["Y_G_im"][:]
        mask = h["mask"][:]
        subj = h["meta/subject"][:].astype(str)
        freq = h["meta/freq"][:]
        z    = h["meta/z_idx"][:]
        print(f"loaded {len(X)} slices  "
              f"({len(set(subj))} subjects, freqs={sorted(set(freq.tolist()))})")

    first_subj = subj[0]
    sel_mask   = (subj == first_subj)
    sel_idx    = np.where(sel_mask)[0]

    rows = []
    for f in sorted(set(freq[sel_idx].tolist())):
        rows_f = sel_idx[freq[sel_idx] == f]
        if len(rows_f) == 0:
            continue
        # pick z closest to z_target
        z_here = z[rows_f]
        pick = rows_f[np.argmin(np.abs(z_here - args.z_target))]
        rows.append((f, pick))

    fig, axes = plt.subplots(len(rows), 5, figsize=(13, 2.6 * len(rows)),
                               sharex=False, sharey=False)
    if len(rows) == 1:
        axes = axes[None, :]

    for r, (f, idx) in enumerate(rows):
        u_re = X[idx, 0]
        u_im = X[idx, 1]
        gr   = Yre[idx] / 1000.0       # kPa
        gi   = Yim[idx] / 1000.0
        m    = mask[idx]
        ttls = [
            f"Re u (μm)\n{first_subj} {f} Hz z={z[idx]}",
            "Im u (μm)",
            "Re G (kPa)",
            "Im G (kPa)",
            "brain mask",
        ]
        # Use a symmetric vmin/vmax for displacement, fixed range for G
        amax = max(np.abs(u_re).max(), np.abs(u_im).max(), 1.0)
        ims  = [
            axes[r, 0].imshow(u_re,        cmap="seismic",  vmin=-amax, vmax=amax),
            axes[r, 1].imshow(u_im,        cmap="seismic",  vmin=-amax, vmax=amax),
            axes[r, 2].imshow(gr,          cmap="viridis",  vmin=0, vmax=6),
            axes[r, 3].imshow(gi,          cmap="viridis",  vmin=0, vmax=2),
            axes[r, 4].imshow(m.astype(float), cmap="gray", vmin=0, vmax=1),
        ]
        for c, (ax, t, im) in enumerate(zip(axes[r], ttls, ims)):
            ax.set_title(t, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

    fig.suptitle(f"BBIR UDel slice inspection: subject {first_subj}",
                  fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
