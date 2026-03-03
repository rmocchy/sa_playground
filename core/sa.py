"""汎用シミュレーテッドアニーリングソルバー。"""

from __future__ import annotations

import math
import random

import numpy as np


def qubo_energy(x: np.ndarray, Q: np.ndarray) -> float:
    """QUBO エネルギー E = x^T Q x を計算する。"""
    return float(x @ Q @ x)


def simulated_annealing(
    Q: np.ndarray,
    T_init: float,
    T_min: float,
    cooling_rate: float,
    max_iter: int,
    seed: int | None = None,
) -> dict:
    """
    シミュレーテッドアニーリングで QUBO を最小化する。

    Parameters
    ----------
    Q            : QUBO 行列 (n×n)
    T_init       : 初期温度
    T_min        : 終了温度
    cooling_rate : 冷却率 α (T ← α·T)
    max_iter     : 最大イテレーション数
    seed         : 乱数シード (None でランダム)

    Returns
    -------
    dict:
        best_x          : 最良解 (ndarray)
        best_energy     : 最良エネルギー
        energy_history  : 各ステップのエネルギー推移
        best_history    : 各ステップのベストエネルギー推移
        temp_history    : 各ステップの温度推移
        n_iter          : 実際のイテレーション数
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    n = Q.shape[0]
    x = np_rng.integers(0, 2, n).astype(float)
    current_energy = qubo_energy(x, Q)
    best_x = x.copy()
    best_energy = current_energy

    T = T_init
    energy_history: list[float] = [current_energy]
    best_history: list[float] = [best_energy]
    temp_history: list[float] = [T]

    for _ in range(max_iter):
        if T <= T_min:
            break

        flip_idx = rng.randint(0, n - 1)
        x_new = x.copy()
        x_new[flip_idx] = 1.0 - x_new[flip_idx]

        new_energy = qubo_energy(x_new, Q)
        delta_E = new_energy - current_energy

        if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
            x = x_new
            current_energy = new_energy
            if current_energy < best_energy:
                best_x = x.copy()
                best_energy = current_energy

        T *= cooling_rate
        energy_history.append(current_energy)
        best_history.append(best_energy)
        temp_history.append(T)

    return {
        "best_x": best_x,
        "best_energy": best_energy,
        "energy_history": energy_history,
        "best_history": best_history,
        "temp_history": temp_history,
        "n_iter": len(energy_history) - 1,
    }
