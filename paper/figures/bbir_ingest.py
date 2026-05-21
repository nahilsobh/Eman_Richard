"""Ingest BBIR UDel brain MRE volumes into 2D training slices.

The BBIR UDel cohort stores each subject × frequency as a directory with
NIfTI volumes at 160 × 160 × 80 voxels, 1.5 mm isotropic.  Each voxel
has:

    disp_re[..., 3], disp_im[..., 3]            complex 3-vector wave (microns)
    curl_re[..., 3], curl_im[..., 3]            complex 3-vector curl (shear)
    props_shear_real, props_shear_imag          scalar complex G  (Pa)
    props_shear_stiff                            scalar |G|        (Pa)
    strain_re[..., 6], strain_im[..., 6]        complex sym strain (xx xy xz yy yz zz)
    anat                                         T1 reference
    plus a brain mask under register_to_MRE/.

For 2D operator training we extract axial slices, apply the brain mask,
and downsample in-plane from 160×160 to 80×80 (matches the TSM-FNO grid).
The simplest two-channel input is one component of complex displacement;
the three-channel option uses the in-plane curl magnitude (shear only,
longitudinal removed).  The label is the scalar complex G on the slice.

The output is a single HDF5 file with arrays

    X        (N_slices, 2, 80, 80)  float32   complex displacement
    Y_G_re   (N_slices,    80, 80)  float32   Re G in Pa
    Y_G_im   (N_slices,    80, 80)  float32   Im G in Pa
    mask     (N_slices,    80, 80)  bool      brain mask
    meta/subject  (N_slices,)        bytes    subject id
    meta/freq     (N_slices,)        int      MRE driver frequency in Hz
    meta/z_idx    (N_slices,)        int      axial slice index in the volume

so any downstream training loop can be a straight HDF5 dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import csv
import h5py
import nibabel as nib
import numpy as np


BBIR_ROOT_DEFAULT = Path("/projects/bfid/sobh/BBIR/U01_UDEL_v5a-All")
INDEX_CSV_DEFAULT = Path("/projects/bfid/sobh/BBIR/udel_index_out/"
                          "mre_dataset_index.csv")

UDEL_FREQS = (30, 50, 70)              # Hz, the BBIR UDel protocol
TARGET_GRID = 80                       # in-plane size used by tsm_fno
DEFAULT_DISP_CHANNEL = 2               # 0 = x, 1 = y, 2 = z (through-slice)
MIN_BRAIN_FRAC = 0.05                  # drop slices with <5% brain voxels


# ── per-acquisition I/O ───────────────────────────────────────────────────────

@dataclass
class UDelAcquisition:
    """Paths to the NIfTI volumes for one (subject, frequency) pair."""
    subject: str
    freq_hz: int
    disp_re: Path
    disp_im: Path
    shear_re: Path
    shear_im: Path
    shear_stiff: Path
    mask: Path
    anat: Path | None = None


def _acq_paths(root: Path, subject: str, freq_hz: int) -> UDelAcquisition:
    subj_dir = root / f"{subject}_v5"
    mre_dir  = subj_dir / f"{subject}_MRE_AP_{freq_hz}Hz"
    reg_dir  = subj_dir / f"{subject}_register_to_MRE"
    mre_pfx  = mre_dir / f"{subject}_MRE_AP_{freq_hz}Hz"
    return UDelAcquisition(
        subject     = subject,
        freq_hz     = freq_hz,
        disp_re     = mre_pfx.with_name(mre_pfx.name + "_disp_re.nii.gz"),
        disp_im     = mre_pfx.with_name(mre_pfx.name + "_disp_im.nii.gz"),
        shear_re    = mre_pfx.with_name(mre_pfx.name + "_props_shear_real.nii.gz"),
        shear_im    = mre_pfx.with_name(mre_pfx.name + "_props_shear_imag.nii.gz"),
        shear_stiff = mre_pfx.with_name(mre_pfx.name + "_props_shear_stiff.nii.gz"),
        mask        = reg_dir / f"{subject}_MREreg_brainmask.nii.gz",
        anat        = mre_pfx.with_name(mre_pfx.name + "_anat.nii.gz"),
    )


def list_udel_subjects(root: Path = BBIR_ROOT_DEFAULT) -> list[str]:
    """All subject directories present on disk (not QC-filtered)."""
    return sorted(p.name.rsplit("_v5", 1)[0]
                   for p in root.iterdir() if p.is_dir())


def read_index_csv(path: Path = INDEX_CSV_DEFAULT) -> list[str]:
    """Subjects that passed the index pre-filter (n_kept = 82)."""
    with open(path) as f:
        rdr = csv.reader(f)
        next(rdr)              # skip header
        return [row[0] for row in rdr]


def load_acquisition(acq: UDelAcquisition
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one acquisition.

    Returns
    -------
    u    : (160, 160, 80, 3) complex64 -- displacement vector field
    G    : (160, 160, 80)    complex64 -- shear modulus
    mask : (160, 160, 80)    bool      -- brain mask
    voxel_mm : (3,) float    -- voxel size in mm (x, y, z)
    """
    disp_re = np.asarray(nib.load(acq.disp_re).dataobj, dtype=np.float32)
    disp_im = np.asarray(nib.load(acq.disp_im).dataobj, dtype=np.float32)
    u = (disp_re + 1j * disp_im).astype(np.complex64)

    G_re = np.asarray(nib.load(acq.shear_re).dataobj, dtype=np.float32)
    G_im = np.asarray(nib.load(acq.shear_im).dataobj, dtype=np.float32)
    G = (G_re + 1j * G_im).astype(np.complex64)

    mask = np.asarray(nib.load(acq.mask).dataobj, dtype=bool)

    voxel_mm = np.asarray(nib.load(acq.disp_re).header.get_zooms()[:3],
                           dtype=np.float32)
    return u, G, mask, voxel_mm


# ── 2D slice extraction ───────────────────────────────────────────────────────

def _crop_or_resize(slice_2d: np.ndarray, target: int) -> np.ndarray:
    """Resize a 2D (H, W) slice to (target, target).

    Uses scipy.ndimage.zoom with order 1 (bilinear for real-valued; the
    caller handles real/imag separately for complex arrays).
    """
    from scipy.ndimage import zoom
    H, W = slice_2d.shape[:2]
    zh, zw = target / H, target / W
    return zoom(slice_2d, (zh, zw) + (1.0,) * (slice_2d.ndim - 2),
                 order=1, prefilter=False)


def _resize_complex(slice_2d_complex: np.ndarray, target: int) -> np.ndarray:
    re = _crop_or_resize(slice_2d_complex.real, target)
    im = _crop_or_resize(slice_2d_complex.imag, target)
    return (re + 1j * im).astype(np.complex64)


def _resize_bool(mask_2d: np.ndarray, target: int) -> np.ndarray:
    return _crop_or_resize(mask_2d.astype(np.float32), target) > 0.5


def axial_slices(acq: UDelAcquisition, disp_channel: int = DEFAULT_DISP_CHANNEL,
                 min_brain_frac: float = MIN_BRAIN_FRAC,
                 target_grid: int = TARGET_GRID,
                 ) -> Iterator[dict]:
    """Yield brain-containing axial slices from one acquisition.

    Each yielded record is a dict with arrays of shape (target_grid,
    target_grid) suitable for direct stacking into a training tensor.
    """
    u, G, mask, _ = load_acquisition(acq)
    u_comp = u[..., disp_channel]                # pick one displacement component
    n_total = u_comp.shape[-1]
    for z in range(n_total):
        m2 = mask[..., z]
        if m2.mean() < min_brain_frac:
            continue
        u_slice = u_comp[..., z]
        G_slice = G[..., z]
        u_slice_re = _resize_complex(u_slice,  target_grid)
        G_slice_re = _resize_complex(G_slice,  target_grid)
        m_slice    = _resize_bool   (m2,       target_grid)

        # Apply mask: outside-brain voxels are set to zero on the displacement
        # (real MRE has no signal there) and to zero on the modulus (no
        # tissue to invert).  The mask itself is also stored so downstream
        # loss functions can ignore non-brain pixels.
        u_slice_re = u_slice_re * m_slice
        G_slice_re = G_slice_re * m_slice

        yield {
            "u_re": u_slice_re.real.astype(np.float32),
            "u_im": u_slice_re.imag.astype(np.float32),
            "G_re": G_slice_re.real.astype(np.float32),
            "G_im": G_slice_re.imag.astype(np.float32),
            "mask": m_slice,
            "subject": acq.subject,
            "freq_hz": acq.freq_hz,
            "z_idx":   z,
        }


# ── batch extraction to HDF5 ──────────────────────────────────────────────────

def extract_to_h5(out_path: Path, subjects: list[str],
                  freqs: tuple[int, ...] = UDEL_FREQS,
                  root: Path = BBIR_ROOT_DEFAULT,
                  disp_channel: int = DEFAULT_DISP_CHANNEL,
                  target_grid: int = TARGET_GRID,
                  min_brain_frac: float = MIN_BRAIN_FRAC,
                  verbose: bool = True,
                  ) -> dict:
    """Iterate (subject, freq), extract axial slices, write to HDF5.

    The HDF5 has fixed-size datasets (chunked) under the names listed in
    the module docstring.  Returns a dict of summary statistics.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Two-pass for clean preallocation: first count slices, then write.
    # The acquisitions are independent so we just iterate.
    records: list[dict] = []
    skipped: list[tuple[str, int, str]] = []
    for subj in subjects:
        for f in freqs:
            acq = _acq_paths(root, subj, f)
            missing = [name for name, p in
                        [("disp_re", acq.disp_re),
                         ("disp_im", acq.disp_im),
                         ("shear_re", acq.shear_re),
                         ("shear_im", acq.shear_im),
                         ("mask",    acq.mask)]
                       if not p.exists()]
            if missing:
                skipped.append((subj, f, ",".join(missing)))
                continue
            try:
                for rec in axial_slices(
                        acq, disp_channel=disp_channel,
                        min_brain_frac=min_brain_frac,
                        target_grid=target_grid):
                    records.append(rec)
                if verbose:
                    print(f"  {subj}  {f} Hz : appended  (running total "
                          f"{len(records)} slices, "
                          f"{len(skipped)} acquisitions skipped)",
                          flush=True)
            except Exception as e:
                skipped.append((subj, f, repr(e)))

    n = len(records)
    if n == 0:
        raise RuntimeError("No slices extracted")

    X      = np.stack([np.stack([r["u_re"], r["u_im"]]) for r in records])
    Y_re   = np.stack([r["G_re"] for r in records])
    Y_im   = np.stack([r["G_im"] for r in records])
    mask   = np.stack([r["mask"] for r in records])
    subj_b = np.array([r["subject"].encode("ascii") for r in records],
                       dtype="S32")
    freq   = np.array([r["freq_hz"] for r in records], dtype=np.int32)
    z_idx  = np.array([r["z_idx"]   for r in records], dtype=np.int32)

    with h5py.File(out_path, "w") as h:
        h.create_dataset("X",      data=X,    compression="gzip",
                          compression_opts=4, chunks=(64, 2, target_grid, target_grid))
        h.create_dataset("Y_G_re", data=Y_re, compression="gzip",
                          compression_opts=4, chunks=(64, target_grid, target_grid))
        h.create_dataset("Y_G_im", data=Y_im, compression="gzip",
                          compression_opts=4, chunks=(64, target_grid, target_grid))
        h.create_dataset("mask",   data=mask, compression="gzip",
                          compression_opts=4, chunks=(64, target_grid, target_grid))
        g = h.create_group("meta")
        g.create_dataset("subject", data=subj_b)
        g.create_dataset("freq",    data=freq)
        g.create_dataset("z_idx",   data=z_idx)
        h.attrs["target_grid"]  = target_grid
        h.attrs["disp_channel"] = disp_channel
        h.attrs["min_brain_frac"] = min_brain_frac
        h.attrs["udel_freqs"]   = list(freqs)
        h.attrs["n_subjects"]   = len(set(r["subject"] for r in records))

    summary = {
        "n_slices":    n,
        "n_subjects":  len(set(r["subject"] for r in records)),
        "n_acquisitions_skipped": len(skipped),
        "skipped":     skipped,
        "out_path":    str(out_path),
    }
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",    type=str, required=True,
                    help="Output HDF5 file path")
    ap.add_argument("--limit",  type=int, default=None,
                    help="Only process the first N subjects (smoke-test)")
    ap.add_argument("--freqs",  type=int, nargs="+", default=list(UDEL_FREQS))
    ap.add_argument("--disp_channel", type=int, default=DEFAULT_DISP_CHANNEL,
                    choices=[0, 1, 2])
    ap.add_argument("--target_grid",  type=int, default=TARGET_GRID)
    ap.add_argument("--min_brain_frac", type=float, default=MIN_BRAIN_FRAC)
    args = ap.parse_args()

    subjects = read_index_csv()
    if args.limit is not None:
        subjects = subjects[:args.limit]
    print(f"Processing {len(subjects)} subjects × {len(args.freqs)} freqs ...")
    summary = extract_to_h5(
        Path(args.out), subjects, tuple(args.freqs),
        disp_channel=args.disp_channel,
        target_grid=args.target_grid,
        min_brain_frac=args.min_brain_frac,
    )
    print()
    print(f"Wrote {summary['out_path']}")
    print(f"  {summary['n_subjects']} subjects, "
          f"{summary['n_slices']} slices, "
          f"{summary['n_acquisitions_skipped']} acquisitions skipped")
    if summary["skipped"]:
        print("  Skipped:")
        for s, f, why in summary["skipped"][:10]:
            print(f"    {s} {f} Hz : {why}")
        if len(summary["skipped"]) > 10:
            print(f"    ... ({len(summary['skipped']) - 10} more)")


if __name__ == "__main__":
    main()
