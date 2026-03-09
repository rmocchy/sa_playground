"""Number Partitioning — neal SA execution & output UI."""

from __future__ import annotations

from collections.abc import Sequence

import dimod
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.neal_sa import run_neal
from core.neal_sidebar import NealParams


# ── Number partitioning chart ─────────────────────────────────
def _plot_partition(numbers: Sequence[float], best_x: np.ndarray) -> go.Figure:
    """Bar chart showing the partition result for Group A / B."""
    labels = [f"n_{i}" for i in range(len(numbers))]
    group_labels = ["Group A (x=1)" if xi == 1 else "Group B (x=0)" for xi in best_x]
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
        title=f"Σ A = {sum_A:.2f}  /  Σ B = {sum_B:.2f}  /  Diff = {abs(sum_A - sum_B):.4f}",
        xaxis_title="Variable", yaxis_title="Value",
        height=340, barmode="group",
        margin=dict(t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ── Energy distribution across reads ─────────────────────────
def _plot_energy_distribution(energies: np.ndarray, best_energy: float) -> go.Figure:
    """Bar chart of per-read energies (sorted), highlighting the best."""
    sorted_e = np.sort(energies)
    colors = ["#EF553B" if e == best_energy else "#636EFA" for e in sorted_e]
    fig = go.Figure(
        go.Bar(
            x=list(range(1, len(sorted_e) + 1)),
            y=sorted_e,
            marker_color=colors,
            name="Energy per read",
        )
    )
    fig.add_hline(
        y=best_energy,
        line_dash="dash",
        line_color="#EF553B",
        annotation_text=f"Best: {best_energy:.4f}",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title="Energy Distribution across Reads (sorted)",
        xaxis_title="Read (sorted by energy)",
        yaxis_title="Energy (xᵀ Q x)",
        height=340,
        margin=dict(t=60, b=40),
    )
    return fig


# ── Main output UI ────────────────────────────────────────────
def render_output(
    numbers: list[float],
    bqm: dimod.BinaryQuadraticModel,
    neal_params: NealParams,
) -> None:
    """
    Run neal SA and render results and graphs.

    Parameters
    ----------
    numbers     : Input number sequence
    bqm         : dimod BinaryQuadraticModel built by build_qubo()
    neal_params : NealParams returned by neal_sidebar()
    """
    if not neal_params.run:
        st.info("👈 Set SA parameters in the sidebar, then press the **Run SA** button.")
        return

    with st.spinner("Running Neal Simulated Annealing…"):
        result = run_neal(bqm, neal_params)

    penalty      = result.penalty
    qubo_raw     = result.qubo_raw
    best_x       = result.best_x
    all_energies = result.all_energies
    elapsed      = result.elapsed_sec

    sum_A = sum(n for n, xi in zip(numbers, best_x) if xi == 1)
    sum_B = sum(n for n, xi in zip(numbers, best_x) if xi == 0)
    diff = abs(sum_A - sum_B)

    # ── Metrics ───────────────────────────────────────────────
    st.subheader("📊 Results")
    m1, m2, m3 = st.columns(3)
    m1.metric("Best QUBO Energy (xᵀQx, no const)", f"{qubo_raw:.4f}")
    m2.metric("λ(Δsum)²  (penalty; 0 = perfect)", f"{penalty:.4f}")
    m3.metric("Runtime", f"{elapsed * 1000:.1f} ms")

    # ── Energy distribution across reads ──────────────────────
    st.plotly_chart(
        _plot_energy_distribution(all_energies, penalty),
        use_container_width=True,
    )

    st.divider()

    # ── Number partitioning specific UI ───────────────────────
    st.subheader("✂️ Partition Results")
    col_table, col_chart = st.columns([1, 2])
    with col_table:
        st.markdown("**Variable Assignment**")
        rows = [
            {"Variable": f"n_{i}", "Value": numbers[i], "Group": "A" if xi == 1 else "B"}
            for i, xi in enumerate(best_x)
        ]
        df = pd.DataFrame(rows)
        st.dataframe(
            df.style.apply(
                lambda row: [
                    "background-color: #d0d8ff" if row["Group"] == "A"
                    else "background-color: #ffd0d0"
                ] * len(row),
                axis=1,
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.markdown(f"**Σ A** = `{sum_A:.2f}`    **Σ B** = `{sum_B:.2f}`    **Diff** = `{diff:.4f}`")

    with col_chart:
        st.plotly_chart(_plot_partition(numbers, best_x), use_container_width=True)
