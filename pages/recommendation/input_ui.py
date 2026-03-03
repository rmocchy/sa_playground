"""レコメンドシステム — 入力 UI。

商品カタログ表示・条件設定・QUBO 構築を担当する。
"""

from __future__ import annotations

import traceback

import numpy as np
import pandas as pd
import streamlit as st

from core.qubo_editor import qubo_editor, render_qubo_params
from pages.recommendation.cards import item_card_html
from pages.recommendation.default_qubo import DEFAULT_QUBO_CODE
from pages.recommendation.items_data import (
    ALL_CATEGORIES,
    DEFAULT_ITEMS,
    Item,
)

_QUBO_DESCRIPTION = """\
`PARAMS` 辞書にパラメータを追加すると、スライダーが**自動で更新**されます。  
`build_qubo(items, required_categories, optional_categories, budget_target, params)` \
関数を書き換えて定式化を変更してください。  
編集後は **Ctrl+Enter** またはエディタ外クリックで自動適用されます。
"""

def render_input(
    items: list[Item] | None = None,
) -> tuple[list[Item], list[str], list[str], float, dict, np.ndarray] | None:
    """
    商品カタログ・条件設定・QUBO 構築UIを描画する。

    Returns
    -------
    (items, required_cats, optional_cats, budget, qubo_params, Q_matrix)
    または None (構築エラー時)
    """
    if items is None:
        items = DEFAULT_ITEMS

    # ── 商品カタログ ─────────────────────────────────────────
    with st.expander(f"🛍️ 商品カタログ ({len(items)} 件)", expanded=False):
        # カテゴリフィルター
        filter_cats = st.multiselect(
            "カテゴリで絞り込み (空欄 = 全表示)",
            options=ALL_CATEGORIES,
            default=[],
            key="rec_filter_cats",
        )
        filtered_items = [it for it in items if (not filter_cats or it.category in filter_cats)]

        # 並び替え
        sort_opt = st.selectbox(
            "並び替え",
            ["価格 (安順)", "価格 (高順)", "評価 (高順)", "カテゴリ"],
            key="rec_sort",
            label_visibility="collapsed",
        )
        if sort_opt == "価格 (安順)":
            filtered_items = sorted(filtered_items, key=lambda x: x.price)
        elif sort_opt == "価格 (高順)":
            filtered_items = sorted(filtered_items, key=lambda x: -x.price)
        elif sort_opt == "評価 (高順)":
            filtered_items = sorted(filtered_items, key=lambda x: -x.score)
        else:
            filtered_items = sorted(filtered_items, key=lambda x: x.category)

        # 商品カードグリッド (4列)
        cols_per_row = 4
        for row_start in range(0, len(filtered_items), cols_per_row):
            row_items = filtered_items[row_start: row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, it in zip(cols, row_items):
                with col:
                    st.markdown(item_card_html(it), unsafe_allow_html=True)
                    st.caption(f"ID: {it.id}")

    st.divider()

    # ── QUBO パラメータ ──────────────────────────────────────
    st.subheader("🔢 QUBO パラメータ")
    PARAMS, build_qubo_fn = qubo_editor(
        default_code=DEFAULT_QUBO_CODE,
        session_prefix="recommendation",
        description=_QUBO_DESCRIPTION,
    )
    with st.expander("スライダーで調整", expanded=False):
        qubo_params = render_qubo_params(PARAMS, key_prefix="rec", n_cols=2)

    st.divider()

    # ── レコメンド条件 ────────────────────────────────────────
    st.subheader("⚙️ レコメンド条件")
    col_req, col_bud = st.columns([3, 1])

    with col_req:
        st.markdown("**必須カテゴリ** (最低 1 件は必ず含める)")
        required_cats: list[str] = st.multiselect(
            "必須カテゴリを選択",
            options=ALL_CATEGORIES,
            default=["スマホ・PC"],
            key="rec_required_cats",
            label_visibility="collapsed",
        )
        optional_cats: list[str] = [c for c in ALL_CATEGORIES if c not in required_cats]
        if optional_cats:
            st.caption(
                "ソフト報酬 (SA が選びやすくなる): " + "、".join(optional_cats)
            )

    with col_bud:
        st.markdown("**予算上限**")
        budget: float = st.number_input(
            "予算 (円)",
            min_value=1000,
            max_value=1_000_000,
            value=80000,
            step=5000,
            format="%d",
            label_visibility="collapsed",
            key="rec_budget",
        )
        st.caption(f"¥{budget:,}")

    # ── QUBO 行列構築 ────────────────────────────────────────
    try:
        Q = build_qubo_fn(items, required_cats, optional_cats, budget, qubo_params)  # type: ignore[operator]
        if not isinstance(Q, np.ndarray):
            raise TypeError("build_qubo は np.ndarray を返す必要があります。")
    except Exception:
        st.error(f"**QUBO 構築エラー:**\n```\n{traceback.format_exc()}\n```")
        return None

    n = len(items)

    # ── メトリクス表示 ──────────────────────────────────────
    c1, c2, c4 = st.columns(3)
    c1.metric("商品数", n)
    c2.metric("必須カテゴリ", len(required_cats))
    c4.metric("予算上限", f"¥{budget:,}")

    with st.expander("📐 QUBO 行列プレビュー", expanded=False):
        from core.sa_viz import plot_qubo_matrix
        labels = [f"{it.emoji}{it.id}" for it in items]
        tab_heat, tab_raw = st.tabs(["ヒートマップ", "生の値"])
        with tab_heat:
            st.plotly_chart(plot_qubo_matrix(Q, var_labels=labels), use_container_width=True)
        with tab_raw:
            df_q = pd.DataFrame(Q, index=labels, columns=labels)
            st.dataframe(df_q.style.format("{:.2f}"), use_container_width=True)

    return items, required_cats, optional_cats, budget, qubo_params, Q
