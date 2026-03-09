"""Shared SA visualization helpers.

Provides reusable charts for any QUBO problem.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go

def plot_qubo_matrix(Q: np.ndarray, var_labels: Sequence[str] | None = None) -> go.Figure:
    """Display a QUBO matrix as a heatmap."""
    n = Q.shape[0]
    labels = list(var_labels) if var_labels is not None else [f"x_{i}" for i in range(n)]
    fig = go.Figure(
        go.Heatmap(
            z=Q,
            x=labels,
            y=labels,
            colorscale="RdBu",
            zmid=0,
            text=np.round(Q, 1),
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title="QUBO Matrix Q",
        height=380,
        margin=dict(t=60, b=40),
        yaxis=dict(autorange="reversed"),
    )
    return fig
