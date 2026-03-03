"""Streamlit ページ: レコメンドシステム — QUBO × シミュレーテッドアニーリング。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from core.sa_sidebar import sa_sidebar
from pages.recommendation.input_ui import render_input
from pages.recommendation.items_data import DEFAULT_ITEMS
from pages.recommendation.output_ui import render_output

st.title("🛍️ レコメンドシステム — QUBO × シミュレーテッドアニーリング")
st.markdown(
    """
商品カタログから **必須カテゴリ** を満たしつつ **予算** に収まる最適な商品セットを、  
QUBO × シミュレーテッドアニーリング (SA) でレコメンドします。
"""
)

# SA パラメータ (サイドバー)
sa_params = sa_sidebar()

st.divider()

# 入力 UI + QUBO 構築
input_result = render_input(items=DEFAULT_ITEMS)

# 出力 UI (EC サイト風)
if input_result is not None:
    items, required_cats, optional_cats, budget, _qubo_params, Q = input_result
    st.divider()
    st.subheader("結果")
    render_output(
        items=items,
        budget=budget,
        Q=Q,
        sa_params=sa_params,
    )
