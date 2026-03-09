"""Shared wrapper around D-Wave neal SimulatedAnnealingSampler.

Usage from any page::

    from core.neal_sa import run_neal
    from core.neal_sidebar import NealParams

    result = run_neal(bqm, neal_params)
    best_x   = result["best_x"]       # np.ndarray of ints
    penalty  = result["penalty"]       # λ(Δsum)² — 0 means perfect
    qubo_raw = result["qubo_raw"]      # x^T Q x  (without BQM offset)
    all_energies = result["all_energies"]  # per-read energies (includes offset)
"""

from __future__ import annotations

from dataclasses import dataclass

import dimod
import neal
import numpy as np

from core.neal_sidebar import NealParams


@dataclass
class NealResult:
    """Return type of run_neal()."""

    best_x: np.ndarray
    """Best binary solution vector (integer ndarray, 0/1)."""

    penalty: float
    """λ(Δsum)² = x^T Q x + offset.  0 means perfect partition."""

    qubo_raw: float
    """x^T Q x  (BQM offset excluded)."""

    all_energies: np.ndarray
    """Per-read energies including offset (one scalar per read)."""

    elapsed_sec: float
    """Wall-clock time of sampler.sample() call."""


def run_neal(
    bqm: dimod.BinaryQuadraticModel,
    params: NealParams,
) -> NealResult:
    """
    Run D-Wave neal SA on a dimod BQM and return structured results.

    Parameters
    ----------
    bqm    : dimod.BinaryQuadraticModel — problem to solve.
    params : NealParams from neal_sidebar().

    Returns
    -------
    NealResult
        best_x       : Best binary solution (ndarray[int], sorted by variable index)
        penalty      : λ(Δsum)² = best energy + offset  (≥ 0; 0 = perfect)
        qubo_raw     : x^T Q x without constant offset
        all_energies : Per-read energies (includes offset)
        elapsed_sec  : Solver wall-clock time in seconds
    """
    import time

    sampler = neal.SimulatedAnnealingSampler()

    t0 = time.perf_counter()
    sample_set = sampler.sample(bqm, **params.sampler_kwargs)
    elapsed = time.perf_counter() - t0

    best_datum = sample_set.first
    # dimod SampleSet energies include the BQM offset, so:
    #   best_datum.energy == x^T Q x + offset == λ(Δsum)²
    penalty: float = best_datum.energy # type: ignore
    qubo_raw: float = best_datum.energy - bqm.offset # type: ignore

    n_vars = len(bqm.variables)
    best_x = np.array(
        [best_datum.sample[v] for v in range(n_vars)], # type: ignore
        dtype=int,
    )
    all_energies: np.ndarray = sample_set.record["energy"]

    return NealResult(
        best_x=best_x,
        penalty=penalty,
        qubo_raw=qubo_raw,
        all_energies=all_energies,
        elapsed_sec=elapsed,
    )
