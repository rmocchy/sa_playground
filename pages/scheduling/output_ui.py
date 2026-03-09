"""Scheduling — SA execution & output UI.

Runs SA, decodes the solution, and displays results including a
Plotly-based Gantt chart.
"""

from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.sa import simulated_annealing
from core.sa_sidebar import SAParams
from core.sa_viz import plot_sa_detail
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
    Q: np.ndarray,
    var_list: list[tuple[str, str, int]],
    sa_params: SAParams,
) -> None:
    """Run SA and display the full results panel."""
    if not sa_params.run:
        st.info("👈 Press the **Run SA** button in the sidebar.")
        return

    # Before chart
    with st.expander("📊 Task Pool (Before Optimization)", expanded=False):
        st.plotly_chart(_before_chart(cfg), use_container_width=True)

    # ── Run SA ────────────────────────────────────────────────────
    with st.spinner("🤖 Running Simulated Annealing…"):
        t0 = time.perf_counter()
        result = simulated_annealing(
            Q=Q,
            T_init=sa_params.T_init,
            T_min=sa_params.T_min,
            cooling_rate=sa_params.cooling_rate,
            max_iter=sa_params.max_iter,
            seed=sa_params.seed,
            timeout_sec=sa_params.timeout_sec,
        )
        elapsed = time.perf_counter() - t0

    if result["timed_out"]:
        st.warning(f"⏱ SA stopped due to timeout ({sa_params.timeout_sec:.0f}s). Showing best solution so far.")

    best_x = result["best_x"].astype(int)
    best_energy: float = result["best_energy"]
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

    # ── SA convergence ────────────────────────────────────────────
    with st.expander("📈 SA Convergence Graph", expanded=False):
        st.plotly_chart(
            plot_sa_detail(
                result["energy_history"],
                result["best_history"],
                result["temp_history"],
            ),
            use_container_width=True,
        )
        c1, c2 = st.columns(2)
        c1.metric("Best Energy E*", f"{best_energy:.4f}")
        c2.metric("Iterations", f"{result['n_iter']:,}")
