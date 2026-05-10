import sys
from pathlib import Path
import numpy as np
import h5py
import tempfile
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.phantom import generate_phantom
from src.fem_solver import helmholtz_solve
from src.dataset import make_training_pair, add_noise, write_chunk


def test_phantom_shape():
    N = 64
    rng = np.random.default_rng(42)
    G = generate_phantom(N, rng)
    assert G.shape == (N, N)
    assert G.dtype == np.float64
    assert G.min() >= 500.0
    assert G.max() <= 15000.0


def test_solver_runs():
    N = 16
    rng = np.random.default_rng(0)
    G = generate_phantom(N, rng)
    u = helmholtz_solve(G, freq=60.0)
    assert u.shape == (N, N)
    assert np.iscomplexobj(u)
    # Left column should be close to 1.0 (Dirichlet source)
    assert np.allclose(u[:, 0], 1.0 + 0j, atol=1e-8)
    # Interior should be non-trivial
    assert np.any(np.abs(u[:, 1:-1]) > 1e-6)


def test_dataset_pair():
    N = 16
    rng = np.random.default_rng(7)
    X, Y, meta = make_training_pair(N=N, rng=rng)
    assert X.shape == (2, N, N)
    assert Y.shape == (N, N)
    assert X.dtype == np.float32
    assert Y.dtype == np.float32
    assert Y.min() >= 500.0
    assert Y.max() <= 15000.0
    assert 0.005 <= meta["damping"] <= 0.70
    assert 1 <= meta["n_sources"] <= 10
    assert 15.0 <= meta["snr_db"] <= 30.0


def test_noise_snr():
    rng = np.random.default_rng(99)
    N = 32
    G = generate_phantom(N, rng)
    u = helmholtz_solve(G, freq=60.0)
    target_snr = 20.0
    u_noisy = add_noise(u, target_snr, rng)
    signal_power = np.mean(np.abs(u) ** 2)
    noise_power = np.mean(np.abs(u_noisy - u) ** 2)
    actual_snr = 10 * np.log10(signal_power / noise_power)
    assert abs(actual_snr - target_snr) < 2.0


def test_chunk_hdf5():
    N = 16
    rng = np.random.default_rng(123)
    n_samples = 5
    samples = [make_training_pair(N=N, rng=rng) for _ in range(n_samples)]

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        tmp_path = Path(f.name)

    write_chunk(samples, tmp_path, chunk_id=0, N=N)

    with h5py.File(tmp_path, "r") as f:
        assert "X" in f and "Y" in f
        assert f["X"].shape == (n_samples, 2, N, N)
        assert f["Y"].shape == (n_samples, N, N)
        assert f["damping"].shape == (n_samples,)
        assert f["n_src"].shape == (n_samples,)
        assert f["snr_db"].shape == (n_samples,)
        assert f.attrs["freq"] == 60.0
        assert f.attrs["N"] == N
        assert f.attrs["n_total"] == n_samples

    tmp_path.unlink()
