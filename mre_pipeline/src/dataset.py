import numpy as np
import h5py
from pathlib import Path

from .phantom import generate_phantom
from .fem_solver import helmholtz_solve, random_sources

RHO = 1000.0
DX = 0.002
FREQ = 60.0          # single clinical drive frequency

# Damping range matches ILI (Scott/Murphy 2020): U(0, 0.7).
# Lower bound ε avoids the singular ξ=0 case in our complex-shear-modulus formulation.
DAMPING_MIN = 0.005
DAMPING_MAX = 0.70


def add_noise(u: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add complex Gaussian noise to displacement field at given SNR (dB)."""
    signal_power = np.mean(np.abs(u) ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    sigma = np.sqrt(noise_power / 2.0)
    noise = sigma * (rng.standard_normal(u.shape) + 1j * rng.standard_normal(u.shape))
    return u + noise


def make_training_pair(
    N: int = 64,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return X:(2,N,N) float32 (Re/Im at 60 Hz) and Y:(N,N) float32.

    Per-sample randomisation:
      - random damping ξ ∈ [DAMPING_MIN, DAMPING_MAX]
      - 1–10 random point sources on the boundary
      - random SNR ∈ [15, 30] dB
    """
    if rng is None:
        rng = np.random.default_rng()

    G = generate_phantom(N, rng)
    damping = float(rng.uniform(DAMPING_MIN, DAMPING_MAX))
    n_patches = int(rng.integers(1, 11))
    sources = random_sources(N, rng, n_min=n_patches, n_max=n_patches)

    u = helmholtz_solve(G, freq=FREQ, rho=RHO, dx=DX,
                         damping=damping, sources=sources)

    snr = float(rng.uniform(15, 30))
    u = add_noise(u, snr, rng)

    # Normalise so |u| ≤ 1 globally — each sample independently
    scale = np.max(np.abs(u))
    if scale > 0:
        u = u / scale

    X = np.stack([u.real, u.imag], axis=0).astype(np.float32)
    Y = G.astype(np.float32)

    meta = {"damping": damping, "n_sources": n_patches, "snr_db": snr}
    return X, Y, meta


def write_chunk(
    samples: list[tuple[np.ndarray, np.ndarray, dict]],
    output_path: Path,
    chunk_id: int,
    N: int = 64,
):
    """Write a list of (X, Y, meta) tuples to an HDF5 chunk file."""
    n = len(samples)
    X_all = np.stack([s[0] for s in samples], axis=0)
    Y_all = np.stack([s[1] for s in samples], axis=0)
    damping = np.array([s[2]["damping"]   for s in samples], dtype=np.float32)
    n_src   = np.array([s[2]["n_sources"] for s in samples], dtype=np.int32)
    snr_db  = np.array([s[2]["snr_db"]    for s in samples], dtype=np.float32)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("X",       data=X_all,   dtype=np.float32)
        f.create_dataset("Y",       data=Y_all,   dtype=np.float32)
        f.create_dataset("damping", data=damping, dtype=np.float32)
        f.create_dataset("n_src",   data=n_src,   dtype=np.int32)
        f.create_dataset("snr_db",  data=snr_db,  dtype=np.float32)

        a = f.attrs
        a["freq"]        = FREQ
        a["dx"]          = DX
        a["rho"]         = RHO
        a["damping_min"] = DAMPING_MIN
        a["damping_max"] = DAMPING_MAX
        a["N"]           = N
        a["G_floor_pa"]  = 500.0
        a["G_ceil_pa"]   = 15000.0
        a["G_range_max"] = 14500.0
        a["inclusion_prob"] = 0.5
        a["snr_min_db"]  = 15.0
        a["snr_max_db"]  = 30.0
        a["src_min"]     = 1
        a["src_max"]     = 10
        a["n_total"]     = n
        a["chunk_id"]    = chunk_id
        a["channels"]    = "Re(u_60Hz),Im(u_60Hz)"
