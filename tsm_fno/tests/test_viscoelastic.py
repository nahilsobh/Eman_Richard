"""Tests for the SLS quasi-static viscoelastic response."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.phantom.viscoelastic import sls_effective_pressures


def test_full_equilibration_limit():
    # tau <<< pause → p_eff must track applied exactly.
    applied = [0.0, 1000.0, 2000.0, 3000.0]
    pauses  = [60.0] * 4
    p_eff   = sls_effective_pressures(applied, pauses, tau=0.01)
    assert np.allclose(p_eff, applied, atol=1e-6), \
        f"tau=0.01, pauses=60 → expected p_eff==applied, got {p_eff}"


def test_frozen_limit():
    # tau >>> pause → gel barely moves; p_eff stays close to p_eff_init.
    applied = [1000.0, 2000.0, 3000.0]
    pauses  = [1.0] * 3
    p_eff   = sls_effective_pressures(applied, pauses, tau=1e6, p_eff_init=0.0)
    # With decay ≈ 1, p_eff[k] ≈ p_eff[k-1] + tiny correction.
    assert np.all(p_eff < 100.0), \
        f"tau=1e6, pauses=1 → gel should barely respond, got {p_eff}"


def test_hysteresis_inflation_below_deflation():
    # Inflation 0→peak, deflation peak→0. At the same applied p (mid-way),
    # the deflation branch's p_eff must be HIGHER than the inflation branch's
    # p_eff, because it's still relaxing down from the peak.
    peak = 5000.0
    applied_inf = [0.0, 1000.0, 2000.0, 3000.0, 4000.0, peak]
    applied_def = [4000.0, 3000.0, 2000.0, 1000.0, 0.0]
    pauses = [30.0] * (len(applied_inf) + len(applied_def))
    schedule = list(applied_inf) + list(applied_def)
    p_eff = sls_effective_pressures(schedule, pauses, tau=60.0)

    inf_p_eff = p_eff[:len(applied_inf)]
    def_p_eff = p_eff[len(applied_inf):]
    # Compare mid-cycle: applied=3000 appears at inf[3] and def[1].
    inf_at_3k = inf_p_eff[3]      # applied[3] = 3000
    def_at_3k = def_p_eff[1]      # applied[1 of def] = 3000
    assert def_at_3k > inf_at_3k + 100.0, \
        f"deflation p_eff ({def_at_3k:.1f}) should exceed inflation ({inf_at_3k:.1f})"


def test_analytical_single_step():
    # Single-step exact-value check: from p_eff_init=0, apply p=1000, wait dt=τ.
    # p_eff should be 1000·(1 - 1/e) ≈ 632.12.
    p_eff = sls_effective_pressures([1000.0], [30.0], tau=30.0, p_eff_init=0.0)
    assert math.isclose(p_eff[0], 1000.0 * (1.0 - 1.0/math.e), rel_tol=1e-9)


def test_returns_to_zero_on_deflate_to_zero():
    # After full inflation + full deflation with enough scan pauses to
    # equilibrate, p_eff at the final zero-pressure state must return to
    # (near) zero — the gel eventually relaxes back.
    schedule = [0, 1000, 2000, 3000, 4000, 5000, 4000, 3000, 2000, 1000, 0]
    pauses = [300.0] * len(schedule)   # 5-min scan pauses
    p_eff = sls_effective_pressures(schedule, pauses, tau=30.0)
    assert p_eff[-1] < 1.0, f"expected p_eff→0 after long deflation, got {p_eff[-1]:.3f}"


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="matching length"):
        sls_effective_pressures([0, 1000], [60.0], tau=30.0)


def test_bad_tau_raises():
    with pytest.raises(ValueError, match="tau must be > 0"):
        sls_effective_pressures([1000], [60], tau=0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
