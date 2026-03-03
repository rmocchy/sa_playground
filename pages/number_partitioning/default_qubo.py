"""数分割問題のデフォルト QUBO コードテンプレート。"""

DEFAULT_QUBO_CODE = '''\
# ============================================================
# QUBO 定式化コード — 数分割問題
#
# ルール:
#   1. PARAMS 辞書でチューニング可能なパラメータを定義する。
#      各エントリのキーがパラメータ名、値は以下の辞書:
#        type    : "float" | "int"
#        label   : サイドバーに表示するラベル
#        default : デフォルト値
#        min     : 最小値
#        max     : 最大値
#        step    : ステップ幅
#   2. build_qubo(numbers, params) を定義する。
#      - numbers : list[float]  入力数列
#      - params  : dict         PARAMS の各キーに現在値が入る
#      - return  : np.ndarray   n×n QUBO 行列
# ============================================================

import numpy as np

PARAMS = {
    "lam": {
        "type": "float",
        "label": "ペナルティ係数 λ",
        "default": 1.0,
        "min": 0.1,
        "max": 10.0,
        "step": 0.1,
    },
}


def build_qubo(numbers, params):
    """
    数分割問題の QUBO 行列を構築する。

    目的関数: λ * (Σ_i (2x_i - 1) * n_i)^2

    Q[i][i] = λ * 4 * n_i * (n_i - S)
    Q[i][j] = λ * 8 * n_i * n_j   (i ≠ j)
    """
    lam = params["lam"]
    nums = np.array(numbers, dtype=float)
    n = len(nums)
    S = nums.sum()
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] = lam * 4.0 * nums[i] * (nums[i] - S)
        for j in range(i + 1, n):
            Q[i, j] = lam * 8.0 * nums[i] * nums[j]
            Q[j, i] = Q[i, j]
    return Q
'''
