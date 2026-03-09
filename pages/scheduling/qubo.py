"""Scheduling — QUBO formulation module (dimod).

Binary variable x[(p, f, t)] = 1 iff worker p is assigned to task f at time slot t.
Variables are linearised into a flat numpy array in the order produced by
``make_var_list``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import dimod
import numpy as np

# ── QUBO parameter slider definitions ───────────────────────────
PARAMS: dict[str, dict] = {
    "alpha": {
        "label": "Makespan weight α",
        "default": 1.0,
        "min": 0.0,
        "max": 5.0,
        "step": 0.1,
    },
    "lambda_1": {
        "label": "Effort constraint λ₁",
        "default": 15.0,
        "min": 0.0,
        "max": 50.0,
        "step": 0.5,
    },
    "lambda_2": {
        "label": "Continuity constraint λ₂",
        "default": 2.0,
        "min": 0.0,
        "max": 20.0,
        "step": 0.5,
    },
    "lambda_3": {
        "label": "No-multitasking λ₃",
        "default": 5.0,
        "min": 0.0,
        "max": 30.0,
        "step": 0.5,
    },
    "lambda_4": {
        "label": "Unassignable period λ₄",
        "default": 10.0,
        "min": 0.0,
        "max": 30.0,
        "step": 0.5,
    },
    "lambda_5": {
        "label": "Precedence constraint λ₅",
        "default": 10.0,
        "min": 0.0,
        "max": 30.0,
        "step": 0.5,
    },
}


@dataclass
class SchedulingConfig:
    """All data needed to build the scheduling QUBO."""

    workers: list[str]
    tasks: list[str]
    T_max: int
    d_t: float
    # A_f[f] : person-month effort required for task f
    A_f: dict[str, float]
    # d_p_f[(p, f)] : productivity (default 1.0)
    d_p_f: dict[tuple[str, str], float] = field(default_factory=dict)
    # D_f_t[(f, t)] : 1 = assignable, 0 = blocked
    D_f_t: dict[tuple[str, int], int] = field(default_factory=dict)
    # list of (pred, succ) task name pairs
    precedences: list[tuple[str, str]] = field(default_factory=list)
    # QUBO weights
    alpha: float = 1.0
    lambda_1: float = 15.0
    lambda_2: float = 2.0
    lambda_3: float = 5.0
    lambda_4: float = 10.0
    lambda_5: float = 10.0

    def __post_init__(self) -> None:
        P, F, T_list = self.workers, self.tasks, list(range(self.T_max))
        if not self.d_p_f:
            self.d_p_f = {(p, f): 1.0 for p in P for f in F}
        if not self.D_f_t:
            self.D_f_t = {(f, t): 1 for f in F for t in T_list}


DEFAULT_CONFIG = SchedulingConfig(
    workers=["Worker_A", "Worker_B"],
    tasks=["Task_1", "Task_2", "Task_3"],
    T_max=10,
    d_t=0.1,
    A_f={"Task_1": 0.05, "Task_2": 0.03, "Task_3": 0.02},
    precedences=[("Task_1", "Task_2")],
)


# ── Variable helpers ─────────────────────────────────────────────

def make_var_list(
    workers: list[str], tasks: list[str], T_max: int
) -> list[tuple[str, str, int]]:
    """Return the canonical variable ordering [(p, f, t), ...]."""
    T_list = list(range(T_max))
    return [(p, f, t) for p in workers for f in tasks for t in T_list]


# ── QUBO builder ─────────────────────────────────────────────────

def build_qubo_matrix(cfg: SchedulingConfig) -> tuple[np.ndarray, list[tuple[str, str, int]]]:
    """Build the scheduling QUBO matrix from *cfg*.

    Returns
    -------
    Q       : (n × n) float64 upper-triangular QUBO matrix
    var_list: ordered variable list for decoding the solution vector
    """
    P = cfg.workers
    F = cfg.tasks
    T_list = list(range(cfg.T_max))

    var_list = make_var_list(P, F, cfg.T_max)
    var_idx: dict[tuple[str, str, int], int] = {v: i for i, v in enumerate(var_list)}
    n = len(var_list)

    Q_dict: dict[tuple[int, int], float] = defaultdict(float)

    def add_Q(v1: tuple, v2: tuple, val: float) -> None:
        i, j = var_idx[v1], var_idx[v2]
        key = (min(i, j), max(i, j))
        Q_dict[key] += val

    # ── E_alpha : Makespan minimisation ──────────────────────────
    for p in P:
        for f in F:
            for t in T_list:
                v = (p, f, t)
                add_Q(v, v, cfg.alpha * t)

    # ── E_1 : Effort constraint  (Σ d·x − C_f)² ─────────────────
    for f in F:
        C_f = (20.0 / cfg.d_t) * cfg.A_f[f]
        pt_combos = [(p, t) for p in P for t in T_list]
        for idx_i, (p1, t1) in enumerate(pt_combos):
            v1 = (p1, f, t1)
            d1 = cfg.d_p_f[(p1, f)]
            add_Q(v1, v1, cfg.lambda_1 * (d1**2 - 2.0 * C_f * d1))
            for idx_j in range(idx_i + 1, len(pt_combos)):
                p2, t2 = pt_combos[idx_j]
                v2 = (p2, f, t2)
                d2 = cfg.d_p_f[(p2, f)]
                add_Q(v1, v2, cfg.lambda_1 * 2.0 * d1 * d2)

    # ── E_2 : Task continuity (penalise gaps) ────────────────────
    for p in P:
        for f in F:
            for t in range(cfg.T_max - 1):
                v1 = (p, f, T_list[t])
                v2 = (p, f, T_list[t + 1])
                add_Q(v1, v1, cfg.lambda_2 * 1.0)
                add_Q(v2, v2, cfg.lambda_2 * 1.0)
                add_Q(v1, v2, cfg.lambda_2 * -2.0)

    # ── E_3 : No multitasking – at most 1 task per (worker, slot) ─
    for p in P:
        for t in T_list:
            for fi in range(len(F)):
                for fj in range(fi + 1, len(F)):
                    v1 = (p, F[fi], t)
                    v2 = (p, F[fj], t)
                    add_Q(v1, v2, cfg.lambda_3 * 2.0)

    # ── E_4 : Unassignable period ────────────────────────────────
    for f in F:
        for t in T_list:
            if cfg.D_f_t.get((f, t), 1) == 0:
                for p in P:
                    v = (p, f, t)
                    add_Q(v, v, cfg.lambda_4 * 1.0)

    # ── E_5 : Task precedence ────────────────────────────────────
    for (fi, fj) in cfg.precedences:
        for t in T_list:
            for p in P:
                v_j = (p, fj, t)
                for p_prime in P:
                    for t_prime in T_list:
                        if t_prime >= t:
                            v_i = (p_prime, fi, t_prime)
                            add_Q(v_j, v_i, cfg.lambda_5 * 1.0)

    # Convert dict → dense upper-triangular numpy matrix
    Q_mat = np.zeros((n, n), dtype=np.float64)
    for (i, j), val in Q_dict.items():
        Q_mat[i, j] += val

    return Q_mat, var_list


def build_bqm(cfg: SchedulingConfig) -> tuple[dimod.BinaryQuadraticModel, list[tuple[str, str, int]]]:
    """Build a dimod BinaryQuadraticModel for the scheduling problem.

    Returns
    -------
    bqm      : dimod.BinaryQuadraticModel
    var_list : ordered variable list for decoding the solution vector
    """
    Q_mat, var_list = build_qubo_matrix(cfg)
    n = len(var_list)

    bqm = dimod.BinaryQuadraticModel(vartype=dimod.BINARY)
    for i in range(n):
        bqm.add_variable(i, 0.0)

    for i in range(n):
        for j in range(i, n):
            val = Q_mat[i, j]
            if val == 0.0:
                continue
            if i == j:
                bqm.add_variable(i, val)
            else:
                bqm.add_interaction(i, j, val)

    return bqm, var_list
