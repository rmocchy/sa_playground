"""Scheduling — Neal SA execution & output UI.

Runs Neal SA, decodes the solution, and displays results including a
Plotly-based Gantt chart.
"""

from __future__ import annotations

import dimod
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.neal_sa import run_neal
from core.neal_sidebar import NealParams
from pages.scheduling.qubo import SchedulingConfig

# Colour palette for tasks (matching the blueprint's matplotlib colours)
_TASK_COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _task_colour_map(tasks: list[str]) -> dict[str, str]:
    return {f: _TASK_COLOURS[i % len(_TASK_COLOURS)] for i, f in enumerate(tasks)}


def _decode_solution(
    best_x: np.ndarray,
    var_list: list[tuple[str, str, int]],
) -> dict[tuple[str, str, int], int]:
    """Return a dict {(p, f, t): 0|1} from the flat solution vector."""
    return {v: int(best_x[i]) for i, v in enumerate(var_list)}


def _gantt_chart(
    cfg: SchedulingConfig,
    sample: dict[tuple[str, str, int], int],
) -> go.Figure:
    """Build an interactive Plotly Gantt chart from the SA solution."""
    task_color = _task_colour_map(cfg.tasks)
    T_list = list(range(cfg.T_max))

    fig = go.Figure()

    # Add one invisible scatter trace per task for the legend
    for f in cfg.tasks:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=12, color=task_color[f], symbol="square"),
            name=f,
            showlegend=True,
        ))

    for i, p in enumerate(cfg.workers):
        y_centre = i
        for t in T_list:
            for f in cfg.tasks:
                if sample.get((p, f, t), 0) == 1:
                    fig.add_shape(
                        type="rect",
                        x0=t, x1=t + 1,
                        y0=y_centre - 0.38, y1=y_centre + 0.38,
                        fillcolor=task_color[f],
                        line=dict(width=1.5, color="white"),
                        layer="below",
                    )

    fig.update_layout(
        title="Optimized Schedule (Gantt Chart)",
        xaxis=dict(
            title="Time Slot",
            tickmode="linear",
            tick0=0,
            dtick=1,
            range=[-0.5, cfg.T_max + 0.5],
            gridcolor="#e0e0e0",
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(cfg.workers))),
            ticktext=cfg.workers,
            range=[-0.7, len(cfg.workers) - 0.3],
        ),
        height=max(280, 120 * len(cfg.workers)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#f8f9fa",
        margin=dict(t=80, b=50, l=100),
    )
    return fig


def _before_chart(cfg: SchedulingConfig) -> go.Figure:
    """Bar chart showing estimated required slots per task (Before state)."""
    required_slots = {f: int((20.0 / cfg.d_t) * cfg.A_f[f]) for f in cfg.tasks}
    task_color = _task_colour_map(cfg.tasks)
    tasks_rev = list(reversed(cfg.tasks))

    fig = go.Figure(go.Bar(
        x=[required_slots[f] for f in tasks_rev],
        y=tasks_rev,
        orientation="h",
        marker_color=[task_color[f] for f in tasks_rev],
        text=[f"{required_slots[f]} slots" for f in tasks_rev],
        textposition="inside",
        insidetextanchor="end",
        textfont=dict(color="white", size=12),
    ))
    fig.update_layout(
        title="Task Pool — Required Slots (Before Optimization)",
        xaxis_title="Required Time Slots",
        height=280,
        plot_bgcolor="#f8f9fa",
        margin=dict(t=50, b=40, l=100),
    )
    return fig

# ── Main output UI ────────────────────────────────────────────────

def render_output(
    cfg: SchedulingConfig,
    bqm: dimod.BinaryQuadraticModel,
    var_list: list[tuple[str, str, int]],
    neal_params: NealParams,
) -> None:
    """Run Neal SA and display the full results panel."""
    if not neal_params.run:
        st.info("👈 Press the **Run SA** button in the sidebar.")
        return

    # Before chart
    with st.expander("📊 Task Pool (Before Optimization)", expanded=False):
        st.plotly_chart(_before_chart(cfg), use_container_width=True)

    # ── Run Neal SA ────────────────────────────────────────────────────
    with st.spinner("🤖 Running Neal Simulated Annealing…"):
        result = run_neal(bqm, neal_params)

    best_x = result.best_x
    best_energy: float = result.penalty
    elapsed = result.elapsed_sec
    sample = _decode_solution(best_x, var_list)

    # ── KPI bar ───────────────────────────────────────────────────
    T_list = list(range(cfg.T_max))
    total_assigned = sum(1 for v, a in sample.items() if a == 1)
    # makespan = last time slot where any task is assigned
    used_slots = [t for (_, _, t), a in sample.items() if a == 1]
    makespan = max(used_slots) + 1 if used_slots else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("✅ Assigned Slots", f"{total_assigned}")
    k2.metric("⏱ Makespan", f"{makespan} / {cfg.T_max} slots")
    k3.metric("⚡ SA Runtime", f"{elapsed * 1000:.0f} ms")
    k4.metric("Best Energy E*", f"{best_energy:.2f}")

    st.divider()

    # ── Gantt chart ───────────────────────────────────────────────
    st.subheader("📅 Optimized Schedule")
    st.plotly_chart(_gantt_chart(cfg, sample), use_container_width=True)

    # ── Text schedule (like bootstrap output) ─────────────────────
    with st.expander("📋 Text Schedule", expanded=False):
        for p in cfg.workers:
            st.markdown(f"**{p}**")
            rows = []
            for t in T_list:
                for f in cfg.tasks:
                    if sample.get((p, f, t), 0) == 1:
                        rows.append({"Time Slot": t, "Task": f})
            if rows:
                import pandas as pd
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            else:
                st.caption("No tasks assigned.")

    st.divider()

    # ── Energy distribution across reads ──────────────────────────────
    with st.expander("📊 Energy Distribution across Reads", expanded=False):
        sorted_e = np.sort(result.all_energies)
        colors = ["#EF553B" if e == best_energy else "#636EFA" for e in sorted_e]
        fig_e = go.Figure(
            go.Bar(
                x=list(range(1, len(sorted_e) + 1)),
                y=sorted_e,
                marker_color=colors,
                name="Energy per read",
            )
        )
        fig_e.add_hline(
            y=best_energy,
            line_dash="dash",
            line_color="#EF553B",
            annotation_text=f"Best: {best_energy:.4f}",
            annotation_position="bottom right",
        )
        fig_e.update_layout(
            title="Energy Distribution across Reads (sorted)",
            xaxis_title="Read (sorted by energy)",
            yaxis_title="Energy",
            height=320,
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig_e, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.metric("Best Energy E*", f"{best_energy:.4f}")
        c2.metric("Num reads", f"{neal_params.num_reads}")

