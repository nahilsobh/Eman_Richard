"""Viscoelastic time-domain response of the pre-stress field.

Models the gel around the balloon as a lumped standard linear solid (SLS)
whose effective pre-stress magnitude relaxes toward the applied pressure
with a single characteristic time τ. The MRE wave physics remains
frequency-domain (the solver's `damping` term handles wave-scale losses);
this module handles the *quasi-static* creep between inflation states,
which is where the hysteresis between the Yin Fig. 6 inflation and
deflation branches actually lives.

For a piecewise-constant applied-pressure protocol
    (p_1, dt_1), (p_2, dt_2), ..., (p_N, dt_N)
where dt_k is the wait (scan pause) after stepping to p_k, the effective
pressure the gel has "seen" by the time step k is measured is

    p_eff[k] = p_k + (p_eff[k-1] - p_k) * exp(-dt_k / tau)

with p_eff[0]_init = 0. This is the analytical exponential-approach
solution of a first-order relaxation ODE, which is the SLS's response
under quasi-static loading with the elastic branch equilibrated first
and the Maxwell arm relaxing on timescale τ.

Limits
------
- τ → 0: `exp(-dt/τ) → 0` → `p_eff[k] = p_k` (instantaneous equilibration,
  no memory, no hysteresis).
- τ → ∞: `exp(-dt/τ) → 1` → `p_eff[k] = p_eff[k-1]` (frozen; gel never
  responds).
- τ ~ dt: partial equilibration → inflation `p_eff` lags below applied,
  deflation `p_eff` lags above applied → hysteresis loop.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def sls_effective_pressures(applied: Sequence[float],
                            pauses: Sequence[float],
                            tau: float,
                            p_eff_init: float = 0.0) -> np.ndarray:
    """Effective pre-stress-driving pressure per step of the loading protocol.

    Parameters
    ----------
    applied : length-N sequence of applied pressures at each step [Pa].
    pauses  : length-N sequence of wait times after stepping to each
              applied[k] before the MRE measurement is made [s].
    tau     : SLS relaxation time [s]. Must be > 0 (or use +inf).
    p_eff_init : effective pressure before step 1 [Pa]. Default 0.

    Returns
    -------
    p_eff : (N,) ndarray of effective pressures at measurement times.
    """
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    if len(applied) != len(pauses):
        raise ValueError(f"applied ({len(applied)}) and pauses ({len(pauses)}) "
                         "must have matching length")
    p_eff = np.empty(len(applied), dtype=float)
    prev = float(p_eff_init)
    for k, (p_new, dt) in enumerate(zip(applied, pauses)):
        decay = np.exp(-float(dt) / float(tau))
        prev = float(p_new) + (prev - float(p_new)) * decay
        p_eff[k] = prev
    return p_eff
