"""Recommendation System — Input UI.

Handles product catalog display, condition settings, and QUBO construction.
"""

from __future__ import annotations

import dimod
import numpy as np
import pandas as pd
import streamlit as st

from core.sa_viz import plot_qubo_matrix
from pages.recommendation.cards import item_card_html
from pages.recommendation.qubo import PARAMS, build_bqm, build_qubo_matrix
from pages.recommendation.items_data import (
    ALL_CATEGORIES,
    DEFAULT_ITEMS,
    Item,
)

def render_input(
    items: list[Item] | None = None,
) -> tuple[list[Item], list[str], list[str], float, dimod.BinaryQuadraticModel, np.ndarray] | None:
    """
    Render the product catalog, condition settings, and QUBO construction UI.

    Returns
    -------
    (items, required_cats, optional_cats, budget, bqm, Q_matrix)
    or None on construction error.
    """
    if items is None:
        items = DEFAULT_ITEMS

    # ── Product catalog ──────────────────────────────────
    with st.expander(f"🛍️ Product Catalog ({len(items)} items)", expanded=False):
        # Category filter
        filter_cats = st.multiselect(
            "Filter by category (leave empty to show all)",
            options=ALL_CATEGORIES,
            default=[],
            key="rec_filter_cats",
        )
        filtered_items = [it for it in items if (not filter_cats or it.category in filter_cats)]

        # Sort
        sort_opt = st.selectbox(
            "Sort by",
            ["Price (low to high)", "Price (high to low)", "Rating (high to low)", "Category"],
            key="rec_sort",
            label_visibility="collapsed",
        )
        if sort_opt == "Price (low to high)":
            filtered_items = sorted(filtered_items, key=lambda x: x.price)
        elif sort_opt == "Price (high to low)":
            filtered_items = sorted(filtered_items, key=lambda x: -x.price)
        elif sort_opt == "Rating (high to low)":
            filtered_items = sorted(filtered_items, key=lambda x: -x.score)
        else:
            filtered_items = sorted(filtered_items, key=lambda x: x.category)

        # Product card grid (4 columns)
        cols_per_row = 4
        for row_start in range(0, len(filtered_items), cols_per_row):
            row_items = filtered_items[row_start: row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, it in zip(cols, row_items):
                with col:
                    st.markdown(item_card_html(it), unsafe_allow_html=True)
                    st.caption(f"ID: {it.id}")

    st.divider()

    # ── QUBO parameter sliders ───────────────────────────
    st.subheader("🔢 QUBO Parameters")
    cols = st.columns(2)
    qubo_params: dict = {}
    for idx, (key, spec) in enumerate(PARAMS.items()):
        with cols[idx % 2]:
            qubo_params[key] = st.slider(
                spec["label"],
                min_value=float(spec["min"]),
                max_value=float(spec["max"]),
                value=float(spec["default"]),
                step=float(spec["step"]),
                key=f"rec__{key}",
            )

    st.divider()

    # ── Recommendation settings ───────────────────────────────
    st.subheader("⚙️ Recommendation Settings")
    col_req, col_bud = st.columns([3, 1])

    with col_req:
        st.markdown("**Required Categories** (at least 1 must be included)")
        required_cats: list[str] = st.multiselect(
            "Select required categories",
            options=ALL_CATEGORIES,
            default=["Phones & PCs"],
            key="rec_required_cats",
            label_visibility="collapsed",
        )
        optional_cats: list[str] = [c for c in ALL_CATEGORIES if c not in required_cats]
        if optional_cats:
            st.caption(
                "Optional (SA will lean toward these): " + ", ".join(optional_cats)
            )

    with col_bud:
        st.markdown("**Budget Limit**")
        budget: float = st.number_input(
            "Budget",
            min_value=1000,
            max_value=1_000_000,
            value=80000,
            step=5000,
            format="%d",
            label_visibility="collapsed",
            key="rec_budget",
        )
        st.caption(f"${budget:,}")

    # ── Build QUBO (BQM for neal, numpy for preview) ────────────────────────────────
    bqm = build_bqm(items, required_cats, optional_cats, budget, qubo_params)
    Q = build_qubo_matrix(items, required_cats, optional_cats, budget, qubo_params)
    n = len(items)

    # ── Metrics ───────────────────────────────────────
    c1, c2, c4 = st.columns(3)
    c1.metric("Products", n)
    c2.metric("Required Categories", len(required_cats))
    c4.metric("Budget Limit", f"${budget:,}")

    with st.expander("📐 QUBO Matrix Preview", expanded=False):
        labels = [f"{it.emoji}{it.id}" for it in items]
        tab_heat, tab_raw = st.tabs(["Heatmap", "Raw Values"])
        with tab_heat:
            st.plotly_chart(plot_qubo_matrix(Q, var_labels=labels), use_container_width=True)
        with tab_raw:
            df_q = pd.DataFrame(Q, index=labels, columns=labels)
            st.dataframe(df_q.style.format("{:.2f}"), use_container_width=True)

    return items, required_cats, optional_cats, budget, bqm, Q
