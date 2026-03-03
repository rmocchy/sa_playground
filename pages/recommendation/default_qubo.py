"""レコメンドシステム — QUBO 定式化モジュール。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pages.recommendation.items_data import Item

# (i, j) -> float
QuboDict = dict[tuple[int, int], float]


# ── QUBO パラメータ定義 (スライダー UI 用) ──────────────────
PARAMS: dict[str, dict] = {
    "lambda_required": {
        "type": "float",
        "label": "必須カテゴリ λ (λ_req)",
        "default": 5.0,
        "min": 0.1,
        "max": 20.0,
        "step": 0.1,
    },
    "lambda_optional": {
        "type": "float",
        "label": "任意カテゴリ λ (λ_opt)",
        "default": 5.0,
        "min": 0.1,
        "max": 20.0,
        "step": 0.1,
    },
    "lambda_budget": {
        "type": "float",
        "label": "予算ペナルティ λ (×10⁻⁶)",
        "default": 1.0,
        "min": 0.0,
        "max": 50.0,
        "step": 0.5,
    },
    "lambda_score": {
        "type": "float",
        "label": "スコア重み λ (λ_score)",
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
    レコメンドシステムの QUBO を辞書形式で構築する。

    Parameters
    ----------
    items               : 商品リスト (Item dataclass)
    required_categories : 必ず 1 つ以上含めるべきカテゴリ
    optional_categories : あれば嬉しいカテゴリ (任意)
    budget_target       : 予算上限 (円)
    lambda_required     : 必須カテゴリ未選択ペナルティ強度
    lambda_optional     : 任意カテゴリ不足ペナルティ強度
    lambda_budget       : 予算超過ペナルティ強度
    lambda_score        : 商品スコア重み (大きいほど高評価品を優先)

    Returns
    -------
    QuboDict : { (i, j): coefficient }
    """
    Q: QuboDict = {}

    def add(i: int, j: int, v: float) -> None:
        Q[(i, j)] = Q.get((i, j), 0.0) + v

    n = len(items)

    # ── 必須カテゴリ制約 ─────────────────────────────────────
    # 各必須カテゴリから「少なくとも 1 件」を選ぶよう強制
    # (Σ x_i - 1)^2 → 展開して QUBO 係数に加算
    for cat in required_categories:
        idx = [i for i, it in enumerate(items) if it.category == cat]
        if not idx:
            continue
        # 二乗展開: λ * (Σ x_i - 1)^2 = λ * (Σ_i Σ_j x_i x_j - 2 Σ_i x_i + 1)
        for i in idx:
            for j in idx:
                add(i, j, lambda_required)
            add(i, i, -2 * lambda_required)

    # 必須以外のについて2つ以上選ばないようにする
    for cat in optional_categories:
        idx = [i for i, it in enumerate(items) if it.category == cat]
        if not idx:
            continue
        for i in idx:
            for j in idx:
                add(i, j, lambda_optional)
            add(i, i, -1 * lambda_optional)

    # ── 予算制約: (Σ price_i * x_i - budget)^2 展開 ──────────
    lb = lambda_budget * 1e-6  # UI では ×10⁻⁶ スケールで入力
    for i in range(n):
        for j in range(n):
            add(i, j, lb * items[i].price * items[j].price)
        add(i, i, -2 * lb * items[i].price * budget_target)

    # ── スコア最大化 (対角成分を減らす) ─────────────────────
    for i in range(n):
        add(i, i, -lambda_score * items[i].score)

    return Q


def build_qubo_matrix(
    items: list["Item"],
    required_categories: list[str],
    optional_categories: list[str],
    budget_target: float,
    params: dict,
) -> "np.ndarray":  # type: ignore[name-defined]
    """QuboDict を np.ndarray に変換して返す。"""
    import numpy as np

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
