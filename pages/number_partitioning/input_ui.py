"""Number Partitioning — QUBO variable input UI.

Handles number sequence input, QUBO parameter adjustment, and QUBO matrix construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.sa_viz import plot_qubo_matrix
from pages.number_partitioning.qubo import PARAMS, build_qubo

_PRESETS: dict[str, list[float]] = {
    "Small: [3, 1, 4, 1, 5]": [3, 1, 4, 1, 5],
    "Medium: [10, 7, 3, 6, 9, 2, 8, 4]": [10, 7, 3, 6, 9, 2, 8, 4],
    "Large: [17, 5, 22, 11, 8, 14, 3, 19, 6, 13]": [17, 5, 22, 11, 8, 14, 3, 19, 6, 13],
}


def render_input() -> tuple[list[float], np.ndarray] | None:
    """
    Render number sequence input, QUBO parameter input, and QUBO matrix construction.

    Returns
    -------
    (numbers, Q) or None on construction error.
    """
    # ── Number sequence input ────────────────────────────────
    st.subheader("📥 Input Data")

    preset_options = ["Custom Input"] + list(_PRESETS.keys())
    preset = st.selectbox("Preset numbers", options=preset_options)

    numbers: list[float]
    if preset == "Custom Input":
        raw = st.text_input("Number sequence (comma-separated)", value="3, 8, 5, 2, 7, 4")
        try:
            numbers = [float(v.strip()) for v in raw.split(",") if v.strip()]
            if len(numbers) < 2:
                st.error("Please enter at least 2 numbers.")
                numbers = [3.0, 8.0, 5.0, 2.0, 7.0, 4.0]
        except ValueError:
            st.error("Could not parse as numbers. Please check your input.")
            numbers = [3.0, 8.0, 5.0, 2.0, 7.0, 4.0]
    else:
        numbers = _PRESETS[preset]
        st.info(f"Numbers: {numbers}")

    # ── QUBO parameter sliders ──────────────────────────────
    st.subheader("🔢 QUBO Parameters")
    spec = PARAMS["lam"]
    lam: float = st.slider(
        spec["label"],
        min_value=float(spec["min"]),
        max_value=float(spec["max"]),
        value=float(spec["default"]),
        step=float(spec["step"]),
        key="np__lam",
    )

    # ── QUBO matrix construction ──────────────────────
    Q = build_qubo(numbers, lam=lam)

    # ── Metrics & matrix display ──────────────────────
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Variables (bits)", len(numbers))
    c2.metric("Sum \u03a3", f"{sum(numbers):.2f}")
    c3.metric("Target group sum", f"{sum(numbers) / 2:.2f}")

    with st.expander("📐 Check QUBO Matrix", expanded=False):
        tab_heat, tab_raw = st.tabs(["Heatmap", "Raw Values"])
        labels = [f"n_{i}" for i in range(len(numbers))]
        with tab_heat:
            st.plotly_chart(plot_qubo_matrix(Q, var_labels=labels), use_container_width=True)
        with tab_raw:
            st.dataframe(pd.DataFrame(Q, index=labels, columns=labels))

    return numbers, Q
