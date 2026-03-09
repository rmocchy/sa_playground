"""Scheduling — Input UI.

Handles worker/task configuration, QUBO parameter sliders,
and QUBO matrix construction.
"""

from __future__ import annotations

import dimod
import numpy as np
import streamlit as st

from pages.scheduling.qubo import (
    PARAMS,
    DEFAULT_CONFIG,
    SchedulingConfig,
    build_bqm,
    build_qubo_matrix,
)


def render_input() -> tuple[SchedulingConfig, dimod.BinaryQuadraticModel, np.ndarray, list[tuple[str, str, int]]] | None:
    """Render all input widgets and return (config, bqm, Q_matrix, var_list) or None on error."""

    # ── Problem Configuration ────────────────────────────────────
    st.subheader("⚙️ Problem Configuration")

    col_left, col_right = st.columns(2)

    with col_left:
        workers_raw = st.text_input(
            "Workers (comma-separated)",
            value=", ".join(DEFAULT_CONFIG.workers),
            key="sch_workers",
        )
        workers = [w.strip() for w in workers_raw.split(",") if w.strip()]

        tasks_raw = st.text_input(
            "Tasks (comma-separated)",
            value=", ".join(DEFAULT_CONFIG.tasks),
            key="sch_tasks",
        )
        tasks = [f.strip() for f in tasks_raw.split(",") if f.strip()]

    with col_right:
        T_max = st.slider("Time slots T_max", min_value=5, max_value=20, value=DEFAULT_CONFIG.T_max, step=1, key="sch_T_max")
        d_t = st.number_input("Time unit d_t", min_value=0.01, max_value=1.0, value=DEFAULT_CONFIG.d_t, step=0.01, format="%.2f", key="sch_d_t")

    if not workers:
        st.error("Please enter at least one worker.")
        return None
    if not tasks:
        st.error("Please enter at least one task.")
        return None

    # ── Task Effort (A_f) ────────────────────────────────────────
    with st.expander("📋 Task Effort (A_f — person-months)", expanded=True):
        st.caption(
            "Each task requires `(20 / d_t) × A_f` time slots of total effort. "
            "With the current d_t this equals the number next to each slider."
        )
        A_f: dict[str, float] = {}
        default_a = {f: DEFAULT_CONFIG.A_f.get(f, 0.03) for f in tasks}
        cols = st.columns(min(len(tasks), 4))
        for idx, f in enumerate(tasks):
            with cols[idx % len(cols)]:
                val = st.slider(
                    f"{f}",
                    min_value=0.01,
                    max_value=0.30,
                    value=default_a[f],
                    step=0.01,
                    format="%.2f",
                    key=f"sch_A_{f}",
                )
                required_slots = int((20.0 / d_t) * val)
                st.caption(f"→ {required_slots} slots needed")
                A_f[f] = val

    # ── Precedence Constraints ────────────────────────────────────
    with st.expander("🔀 Precedence Constraints", expanded=True):
        st.caption("Select pairs where the **left task must finish before** the right task starts.")
        all_pairs = [(fi, fj) for i, fi in enumerate(tasks) for fj in tasks[i + 1:]]
        pair_labels = [f"{a} → {b}" for a, b in all_pairs]
        default_prec = DEFAULT_CONFIG.precedences
        default_labels = [f"{a} → {b}" for a, b in default_prec if a in tasks and b in tasks]
        selected_labels: list[str] = st.multiselect(
            "Active precedence pairs",
            options=pair_labels,
            default=[lbl for lbl in default_labels if lbl in pair_labels],
            key="sch_precedences",
            label_visibility="collapsed",
        )
        label_to_pair = {f"{a} → {b}": (a, b) for a, b in all_pairs}
        precedences: list[tuple[str, str]] = [label_to_pair[lbl] for lbl in selected_labels]

    st.divider()

    # ── QUBO Parameters ──────────────────────────────────────────
    st.subheader("🔢 QUBO Hyperparameters")
    st.caption("Adjust penalty weights to tune constraint enforcement vs. objective strength.")
    qubo_params: dict[str, float] = {}
    cols2 = st.columns(3)
    for idx, (key, spec) in enumerate(PARAMS.items()):
        with cols2[idx % 3]:
            qubo_params[key] = st.slider(
                spec["label"],
                min_value=float(spec["min"]),
                max_value=float(spec["max"]),
                value=float(spec["default"]),
                step=float(spec["step"]),
                key=f"sch__{key}",
            )

    # ── Build QUBO ───────────────────────────────────────────────
    n_vars = len(workers) * len(tasks) * T_max
    st.caption(f"Problem size: {len(workers)} workers × {len(tasks)} tasks × {T_max} slots = **{n_vars} binary variables**")

    if n_vars > 500:
        st.warning(f"⚠️ {n_vars} variables is large — SA may be slow or inaccurate. Consider reducing T_max or the number of tasks.")

    cfg = SchedulingConfig(
        workers=workers,
        tasks=tasks,
        T_max=T_max,
        d_t=d_t,
        A_f=A_f,
        precedences=precedences,
        **qubo_params,  # type: ignore[arg-type]
    )

    try:
        bqm, var_list = build_bqm(cfg)
        Q, _ = build_qubo_matrix(cfg)
    except Exception as exc:
        st.error(f"QUBO construction failed: {exc}")
        return None

    return cfg, bqm, Q, var_list
