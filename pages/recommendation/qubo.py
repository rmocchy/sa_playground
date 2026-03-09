"""Recommendation System — QUBO formulation module (dimod)."""

from __future__ import annotations

import dimod
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pages.recommendation.items_data import Item

# (i, j) -> float
QuboDict = dict[tuple[int, int], float]



# ── QUBO parameter definitions (for slider UI) ──────────────────
PARAMS: dict[str, dict] = {
    "lambda_required": {
        "type": "float",
        "label": "Required Category λ (λ_req)",
        "default": 5.0,
        "min": 0.1,
        "max": 20.0,
        "step": 0.1,
    },
    "lambda_optional": {
        "type": "float",
        "label": "Optional Category λ (λ_opt)",
        "default": 5.0,
        "min": 0.1,
        "max": 20.0,
        "step": 0.1,
    },
    "lambda_budget": {
        "type": "float",
        "label": "Budget Penalty λ (×10⁻⁶)",
        "default": 1.0,
        "min": 0.0,
        "max": 50.0,
        "step": 0.5,
    },
    "lambda_score": {
        "type": "float",
        "label": "Score Weight λ (λ_score)",
        "default": 1.0,
        "min": 0.0,
        "max": 10.0,
        "step": 0.1,
    },
}


def formulate_recommendation_qubo(
    items: list["Item"],
    required_categories: list[str],
    optional_categories: list[str],
    budget_target: float,
    lambda_required: float = 5.0,
    lambda_optional: float = 5.0,
    lambda_budget: float = 0.0001,
    lambda_score: float = 1.0,
) -> QuboDict:
    """
    Build the recommendation QUBO as a dictionary.

    Parameters
    ----------
    items               : List of products (Item dataclass)
    required_categories : Categories that must have at least 1 item selected
    optional_categories : Categories where selecting more than 1 item is penalised (soft at-most-one)
    budget_target       : Budget limit
    lambda_required     : Penalty strength for missing a required category
    lambda_optional     : Penalty strength for missing an optional category
    lambda_budget       : Penalty strength for exceeding the budget
    lambda_score        : Weight for product score (higher = prefer highly rated items)

    Returns
    -------
    QuboDict : { (i, j): coefficient }
    """
    Q: QuboDict = {}

    def add(i: int, j: int, v: float) -> None:
        Q[(i, j)] = Q.get((i, j), 0.0) + v

    n = len(items)

    # ── Required category constraint ───────────────────────────────
    # Soft exactly-one constraint per required category: (Σ x_i − 1)²
    # Penalises both 0 selections and ≥2 selections.
    for cat in required_categories:
        idx = [i for i, it in enumerate(items) if it.category == cat]
        if not idx:
            continue
        # Expanded: λ * (Σ_i Σ_j x_i x_j - 2Σ_i x_i + 1)
        for i in idx:
            for j in idx:
                add(i, j, lambda_required)
            add(i, i, -2 * lambda_required)

    # For optional categories: derived from λ(Σx_i - 1/2)² minus the constant λ/4.
    # Results in Q[i,i]=0, Q[i,j]=+λ (i≠j) → cost=0 for k=0 or k=1, penalises k≥2.
    for cat in optional_categories:
        idx = [i for i, it in enumerate(items) if it.category == cat]
        if not idx:
            continue
        for i in idx:
            for j in idx:
                add(i, j, lambda_optional)
            add(i, i, -1 * lambda_optional)

    # ── Budget constraint: expand (Σ price_i * x_i - budget)^2 ─────────
    lb = lambda_budget * 1e-6  # UI uses ×10⁻⁶ scale
    for i in range(n):
        for j in range(n):
            add(i, j, lb * items[i].price * items[j].price)
        add(i, i, -2 * lb * items[i].price * budget_target)

    # ── Score maximization (reduce diagonal entries) ─────────────────
    for i in range(n):
        add(i, i, -lambda_score * items[i].score)

    return Q


def build_bqm(
    items: list["Item"],
    required_categories: list[str],
    optional_categories: list[str],
    budget_target: float,
    params: dict,
) -> dimod.BinaryQuadraticModel:
    """Build a dimod BinaryQuadraticModel for the recommendation problem."""
    q_dict = formulate_recommendation_qubo(
        items=items,
        required_categories=required_categories,
        optional_categories=optional_categories,
        budget_target=budget_target,
        lambda_required=params.get("lambda_required", 5.0),
        lambda_optional=params.get("lambda_optional", 5.0),
        lambda_budget=params.get("lambda_budget", 1.0),
        lambda_score=params.get("lambda_score", 1.0),
    )

    n = len(items)
    bqm = dimod.BinaryQuadraticModel(vartype=dimod.BINARY)
    for i in range(n):
        bqm.add_variable(i, 0.0)

    for (i, j), v in q_dict.items():
        if i == j:
            bqm.add_variable(i, v)
        elif i < j:
            bqm.add_interaction(i, j, v)
        else:
            bqm.add_interaction(j, i, v)

    return bqm


def build_qubo_matrix(
    items: list["Item"],
    required_categories: list[str],
    optional_categories: list[str],
    budget_target: float,
    params: dict,
) -> np.ndarray:
    """Convert QuboDict to np.ndarray and return it (kept for QUBO preview)."""
    q_dict = formulate_recommendation_qubo(
        items=items,
        required_categories=required_categories,
        optional_categories=optional_categories,
        budget_target=budget_target,
        lambda_required=params.get("lambda_required", 5.0),
        lambda_optional=params.get("lambda_optional", 5.0),
        lambda_budget=params.get("lambda_budget", 1.0),
        lambda_score=params.get("lambda_score", 1.0),
    )

    n = len(items)
    Q = np.zeros((n, n))
    for (i, j), v in q_dict.items():
        Q[i, j] += v
    return Q
