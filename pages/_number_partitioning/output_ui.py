"""数分割問題 — SA 実行 & 出力 UI。"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.sa import simulated_annealing
from core.sa_sidebar import SAParams
from core.sa_viz import plot_sa_detail


# ── 数分割問題専用グラフ ─────────────────────────────────────
def _plot_partition(numbers: Sequence[float], best_x: np.ndarray) -> go.Figure:
    """グループ A / B への分割結果を棒グラフで表示する。"""
    labels = [f"n_{i}" for i in range(len(numbers))]
    group_labels = ["グループ A (x=1)" if xi == 1 else "グループ B (x=0)" for xi in best_x]
    colors = ["#636EFA" if xi == 1 else "#EF553B" for xi in best_x]

    fig = go.Figure()
    seen: set[str] = set()
    for i, (num, color, grp) in enumerate(zip(numbers, colors, group_labels)):
        fig.add_trace(
            go.Bar(
                x=[labels[i]], y=[num],
                name=grp, marker_color=color,
                showlegend=grp not in seen,
                legendgroup=grp,
            )
        )
        seen.add(grp)

    sum_A = sum(n for n, xi in zip(numbers, best_x) if xi == 1)
    sum_B = sum(n for n, xi in zip(numbers, best_x) if xi == 0)
    fig.update_layout(
        title=f"Σ A = {sum_A:.2f}  /  Σ B = {sum_B:.2f}  /  差 = {abs(sum_A - sum_B):.4f}",
        xaxis_title="変数", yaxis_title="値",
        height=340, barmode="group",
        margin=dict(t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ── メイン出力 UI ────────────────────────────────────────────
def render_output(
    numbers: list[float],
    Q: np.ndarray,
    sa_params: SAParams,
) -> None:
    """
    SA を実行し、結果・グラフを描画する。

    Parameters
    ----------
    numbers   : 入力数列
    Q         : QUBO 行列
    sa_params : sa_sidebar() が返す SAParams
    """
    if not sa_params.run:
        st.info("👈 サイドバーで SA パラメータを設定し、**SA を実行** ボタンを押してください。")
        return

    with st.spinner("シミュレーテッドアニーリングを実行中…"):
        t0 = time.perf_counter()
        result = simulated_annealing(
            Q=Q,
            T_init=sa_params.T_init,
            T_min=sa_params.T_min,
            cooling_rate=sa_params.cooling_rate,
            max_iter=sa_params.max_iter,
            seed=sa_params.seed,
        )
        elapsed = time.perf_counter() - t0

    best_x = result["best_x"].astype(int)
    best_energy: float = result["best_energy"]
    n_iter: int = result["n_iter"]

    sum_A = sum(n for n, xi in zip(numbers, best_x) if xi == 1)
    sum_B = sum(n for n, xi in zip(numbers, best_x) if xi == 0)
    diff = abs(sum_A - sum_B)

    # ── 共通: 最小エネルギー・実行時間 ───────────────────────
    st.subheader("📊 実行結果")
    m1, m2 = st.columns(2)
    m1.metric("最良エネルギー E*", f"{best_energy:.4f}")
    m2.metric("実行時間", f"{elapsed * 1000:.1f} ms")

    # ── 共通: エネルギー推移・温度推移 ───────────────────────
    st.plotly_chart(
        plot_sa_detail(result["energy_history"], result["best_history"], result["temp_history"]),
        use_container_width=True,
    )

    st.divider()

    # ── 数分割専用 UI ────────────────────────────────────────
    st.subheader("✂️ 分割結果")
    col_table, col_chart = st.columns([1, 2])
    with col_table:
        st.markdown("**変数の割り当て**")
        rows = [
            {"変数": f"n_{i}", "値": numbers[i], "グループ": "A" if xi == 1 else "B"}
            for i, xi in enumerate(best_x)
        ]
        df = pd.DataFrame(rows)
        st.dataframe(
            df.style.apply(
                lambda row: [
                    "background-color: #d0d8ff" if row["グループ"] == "A"
                    else "background-color: #ffd0d0"
                ] * len(row),
                axis=1,
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.markdown(f"**Σ A** = `{sum_A:.2f}`　　**Σ B** = `{sum_B:.2f}`　　**差** = `{diff:.4f}`")

    with col_chart:
        st.plotly_chart(_plot_partition(numbers, best_x), use_container_width=True)
