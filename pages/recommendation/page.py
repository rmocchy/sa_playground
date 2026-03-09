"""Streamlit page: Recommendation System — QUBO × Simulated Annealing."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from core.sa_sidebar import sa_sidebar
from pages.recommendation.input_ui import render_input
from pages.recommendation.items_data import DEFAULT_ITEMS
from pages.recommendation.output_ui import render_output

st.title("🛍️ Recommendation System — QUBO × Simulated Annealing")
st.markdown(
    """
Find the best set of products from a catalog that fits your **required categories** and **budget**,  
using QUBO × Simulated Annealing (SA).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1Mn286V_Rbad--ZDMTUEpLL5F7X8Qokrt/view?usp=sharing)
"""
)

# SA parameters (sidebar)
sa_params = sa_sidebar()

st.divider()

# Input UI + QUBO construction
input_result = render_input(items=DEFAULT_ITEMS)

# Output UI
if input_result is not None:
    items, required_cats, optional_cats, budget, Q = input_result
    st.divider()
    st.subheader("Results")
    render_output(
        items=items,
        budget=budget,
        Q=Q,
        sa_params=sa_params,
    )
