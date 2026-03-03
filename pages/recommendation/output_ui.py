"""レコメンドシステム — SA 実行 & EC サイト風リッチ出力 UI。"""

from __future__ import annotations

import time

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.sa import simulated_annealing
from core.sa_sidebar import SAParams
from core.sa_viz import plot_sa_detail
from pages.recommendation.cards import compact_card_html, item_card_html
from pages.recommendation.items_data import Item

# ── メイン出力 UI ────────────────────────────────────────────
def render_output(
    items: list[Item],
    budget: float,
    Q: np.ndarray,
    sa_params: SAParams,
) -> None:
    """
    SA を実行し、EC サイト風リッチ画面で結果を表示する。
    """
    if not sa_params.run:
        st.info("👈 サイドバーで **SA を実行** ボタンを押してください。")
        return

    # ── SA 実行 ──────────────────────────────────────────────
    with st.spinner("🤖 SA でレコメンドを計算中…"):
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

    recommended = [it for it, xi in zip(items, best_x) if xi == 1]
    not_recommended = [it for it, xi in zip(items, best_x) if xi == 0]

    total_price = sum(it.price for it in recommended)

    # ── KPI バー ─────────────────────────────────────────────
    k1, k2, k3 = st.columns(3)
    k1.metric("🛒 推奨アイテム数", f"{len(recommended)} 件")
    k2.metric("💴 合計金額", f"¥{total_price:,}")
    k3.metric("⚡ SA 実行時間", f"{elapsed * 1000:.0f} ms")

    st.divider()

    # ── 推奨商品グリッド ─────────────────────────────────────
    st.subheader(f"🛒 推奨商品 ({len(recommended)} 件)")

    if not recommended:
        st.warning("推奨商品がありません。SA パラメータまたは QUBO 係数を調整してください。")
    else:
        # スコア順に並べて表示
        rec_ranked = sorted(recommended, key=lambda x: -x.score)
        cols_per_row = min(4, len(rec_ranked))
        for row_start in range(0, len(rec_ranked), cols_per_row):
            row_items = rec_ranked[row_start: row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, it in zip(cols, row_items):
                with col:
                    st.markdown(
                        item_card_html(it),
                        unsafe_allow_html=True,
                    )
            st.write("")  # gap

    st.divider()

    # ── カート合計バー ────────────────────────────────────────
    budget_pct = min(total_price / budget * 100, 150) if budget > 0 else 0
    bar_color = "#e74c3c" if total_price > budget else "#2ecc71"
    st.markdown(f"""
<div style="background:#f8f9fa;border-radius:12px;padding:16px 20px;margin-bottom:16px;">
  <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
    <span style="font-weight:700;color:#333;">🛒 カート合計</span>
    <span style="font-size:20px;font-weight:800;color:{bar_color};">¥{total_price:,}</span>
  </div>
  <div style="background:#e9ecef;border-radius:8px;height:12px;overflow:hidden;">
    <div style="
      background:linear-gradient(90deg,{bar_color},{bar_color}aa);
      width:{min(budget_pct,100):.1f}%;height:12px;border-radius:8px;
      transition:width .4s;
    "></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:12px;color:#888;margin-top:4px;">
    <span>¥0</span>
    <span>予算上限 ¥{budget:,.0f}</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── SA 収束グラフ ────────────────────────────────────────
    with st.expander("📈 SA 収束グラフを表示", expanded=False):
        st.plotly_chart(
            plot_sa_detail(result["energy_history"], result["best_history"], result["temp_history"]),
            use_container_width=True,
        )
        c1, c2 = st.columns(2)
        c1.metric("最良エネルギー E*", f"{best_energy:.4f}")
        c2.metric("イテレーション数", f"{result['n_iter']:,}")