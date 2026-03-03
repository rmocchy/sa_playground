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


# ── スコア vs 価格 散布図 ────────────────────────────────────
def _plot_scatter(items: list[Item], best_x: np.ndarray) -> go.Figure:
    names = [it.name for it in items]
    prices = [it.price for it in items]
    scores = [it.score for it in items]
    cats = [it.category for it in items]
    selected = ["✅ 推奨" if xi == 1 else "❌ 非推奨" for xi in best_x]
    sizes = [22 if xi == 1 else 10 for xi in best_x]

    fig = px.scatter(
        x=prices, y=scores,
        color=selected,
        size=sizes,
        hover_name=names,
        hover_data={"カテゴリ": cats, "価格": prices, "スコア": scores},
        title="商品マップ: スコア vs 価格",
        labels={"x": "価格 (¥)", "y": "評価スコア", "color": "推奨状態"},
        color_discrete_map={"✅ 推奨": "#2ecc71", "❌ 非推奨": "#bdc3c7"},
        template="plotly_white",
    )
    fig.update_layout(
        height=360,
        margin=dict(t=50, b=40, l=50, r=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ── カテゴリ充足チャート ─────────────────────────────────────
def _plot_category_coverage(
    items: list[Item],
    best_x: np.ndarray,
    required_cats: list[str],
    optional_cats: list[str],
) -> go.Figure:
    from collections import Counter
    sel_cats = Counter(it.category for it, xi in zip(items, best_x) if xi == 1)
    all_cats = sorted({it.category for it in items})

    counts = [sel_cats.get(c, 0) for c in all_cats]
    colors = []
    for c in all_cats:
        if c in required_cats:
            colors.append("#e74c3c" if sel_cats.get(c, 0) == 0 else "#2ecc71")
        elif c in optional_cats:
            colors.append("#f39c12" if sel_cats.get(c, 0) == 0 else "#3498db")
        else:
            colors.append("#bdc3c7")

    fig = go.Figure(go.Bar(
        x=all_cats, y=counts,
        marker_color=colors,
        text=counts,
        textposition="outside",
    ))
    fig.update_layout(
        title="カテゴリ別 選択数",
        xaxis_title="カテゴリ",
        yaxis_title="選択された商品数",
        height=320,
        margin=dict(t=50, b=80, l=40, r=20),
        xaxis_tickangle=-30,
        template="plotly_white",
        showlegend=False,
    )
    fig.add_annotation(
        text="🟢必須充足 🔴必須不足 🔵任意充足 🟡任意不足 ⬜その他",
        xref="paper", yref="paper",
        x=0.5, y=-0.35, showarrow=False,
        font=dict(size=11), xanchor="center",
    )
    return fig

# ── メイン出力 UI ────────────────────────────────────────────
def render_output(
    items: list[Item],
    required_cats: list[str],
    optional_cats: list[str],
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

    # ── チャート 2 列 ─────────────────────────────────────────
    col_scatter, col_cat = st.columns(2)
    with col_scatter:
        st.plotly_chart(_plot_scatter(items, best_x), use_container_width=True)
    with col_cat:
        st.plotly_chart(
            _plot_category_coverage(items, best_x, required_cats, optional_cats),
            use_container_width=True,
        )

    # ── SA 収束グラフ ────────────────────────────────────────
    with st.expander("📈 SA 収束グラフを表示", expanded=False):
        st.plotly_chart(
            plot_sa_detail(result["energy_history"], result["best_history"], result["temp_history"]),
            use_container_width=True,
        )
        c1, c2 = st.columns(2)
        c1.metric("最良エネルギー E*", f"{best_energy:.4f}")
        c2.metric("イテレーション数", f"{result['n_iter']:,}")

    st.divider()

    # ── 非推奨商品 (コンパクト) ──────────────────────────────
    if not_recommended:
        with st.expander(f"👀 今回 SA が選ばなかった商品 ({len(not_recommended)} 件)", expanded=False):
            nr_cols = st.columns(2)
            for idx, it in enumerate(not_recommended):
                with nr_cols[idx % 2]:
                    st.markdown(compact_card_html(it), unsafe_allow_html=True)
                    st.write("")

    # ── 詳細テーブル ─────────────────────────────────────────
    with st.expander("📋 全商品の結果テーブル", expanded=False):
        import pandas as pd
        rows = []
        for it, xi in zip(items, best_x):
            rows.append({
                "推奨": "✅" if xi == 1 else "❌",
                "商品名": f"{it.emoji} {it.name}",
                "カテゴリ": it.category,
                "価格 (¥)": it.price,
                "スコア": it.score,
            })
        df = pd.DataFrame(rows)
        st.dataframe(
            df.style.apply(
                lambda row: [
                    "background-color:#d4edda" if row["推奨"] == "✅"
                    else "background-color:#f8f9fa"
                ] * len(row),
                axis=1,
            ),
            hide_index=True,
            use_container_width=True,
        )