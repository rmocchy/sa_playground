"""Streamlit page: Scheduling Optimizer"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from core.sa_sidebar import sa_sidebar
from pages.scheduling.input_ui import render_input
from pages.scheduling.output_ui import render_output

st.title("📅 Scheduling Optimizer")
st.markdown(
    """
Assign **workers** to **tasks** across discrete **time slots** to minimise makespan,  
while respecting effort requirements, continuity, no-multitasking and precedence constraints —  
encoded as a QUBO problem and solved with Simulated Annealing (SA).
"""
)

# SA parameters (sidebar)
sa_params = sa_sidebar()

st.divider()

# Input UI + QUBO construction
result = render_input()

# Output UI
if result is not None:
    cfg, Q, var_list = result
    st.divider()
    st.subheader("Results")
    render_output(cfg=cfg, Q=Q, var_list=var_list, sa_params=sa_params)
