"""Number Partitioning — QUBO formulation module (dimod)."""

from __future__ import annotations

import dimod
import numpy as np

# QUBO parameter definition
# Each entry holds: type / label / default / min / max / step
PARAMS: dict = {
    "lam": {
        "type": "float",
        "label": "Penalty coefficient λ",
        "default": 1.0,
        "min": 0.1,
        "max": 10.0,
        "step": 0.1,
    },
}


def build_qubo(numbers: list[float], lam: float = 1.0) -> dimod.BinaryQuadraticModel:
    """
    Build a BinaryQuadraticModel for the number partitioning problem.

    Objective: λ * (Σ_i (2x_i - 1) * n_i)^2

    Expanding (with x_i^2 = x_i for binary variables):
        linear bias   : λ * 4 * n_i * (n_i - S)
        quadratic bias: λ * 8 * n_i * n_j   (i < j)
        offset        : λ * S^2

    where S = Σ_i n_i.
    """
    nums = np.array(numbers, dtype=float)
    S = float(nums.sum())
    n = len(nums)

    bqm = dimod.BinaryQuadraticModel(vartype=dimod.BINARY)

    # Linear (diagonal) terms
    for i in range(n):
        bqm.add_variable(i, lam * 4.0 * float(nums[i]) * (float(nums[i]) - S))

    # Quadratic (off-diagonal) terms
    for i in range(n):
        for j in range(i + 1, n):
            bqm.add_interaction(i, j, lam * 8.0 * float(nums[i]) * float(nums[j]))

    # Constant offset (λ S²) — not part of Q but useful for true energy
    bqm.offset += lam * S * S

    return bqm


def bqm_to_numpy(bqm: dimod.BinaryQuadraticModel) -> np.ndarray:
    """
    Convert a BinaryQuadraticModel to a symmetric QUBO numpy matrix.

    Variables are sorted and mapped to row/column indices.
    The offset (constant term) is not included.
    """
    labels = sorted(bqm.variables)  # type: ignore[type-var]
    idx = {v: i for i, v in enumerate(labels)}
    n = len(labels)
    Q = np.zeros((n, n))

    for v, bias in bqm.linear.items():
        Q[idx[v], idx[v]] = bias

    for (u, v), bias in bqm.quadratic.items():
        i, j = idx[u], idx[v]
        Q[i, j] = bias
        Q[j, i] = bias

    return Q
