"""Streamlit page: Number Partitioning"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from core.openjij_sidebar import openjij_sidebar
from pages.number_partitioning.input_ui import render_input
from pages.number_partitioning.output_ui import render_output

st.title("✂️ Number Partitioning")
st.markdown(
    """
Given a list of numbers, find a way to split them into two groups (A / B)  
so that the sum of each group is as equal as possible.  
The problem is encoded as QUBO and solved with Simulated Annealing (SA) in the browser.

📄 QUBO implementation: [pages/number_partitioning/qubo.py](https://github.com/rmocchy/sa_playground/blob/main/pages/number_partitioning/qubo.py)
"""
)

# ── Formulation ──────────────────────────────────────────────────
st.subheader("📐 QUBO Formulation")
st.markdown(
    r"""
**Variable** : $x_i \in \{0, 1\}$ — assign $n_i$ to **Group B** if $x_i = 1$, else to **Group A**

**Objective** : minimise the squared difference between the two group sums
"""
)
st.latex(
    r"\min_{\mathbf{x}} \; E = \lambda \left(\sum_{i=1}^{N} (2x_i - 1)\, n_i\right)^2"
)
st.markdown("**QUBO coefficients** ($S = \\sum_i n_i$)")
st.latex(
    r"""
    Q_{ii} = \lambda \cdot 4\, n_i\,(n_i - S),
    \qquad
    Q_{ij} = \lambda \cdot 8\, n_i\, n_j
    \quad (i \neq j)
    """
)

with st.expander("🔍 See detailed derivation"):
    st.markdown(
        r"""
#### Problem setting

Given $N$ numbers $n_1, \ldots, n_N$, partition them into two groups A and B to minimise

$$\left|\text{sum}_A - \text{sum}_B\right|$$

---

#### Derivation

Letting the sum of group B be $\sum_i x_i n_i$ and group A be $\sum_i (1-x_i) n_i$,
their difference is

$$\Delta = \text{sum}_B - \text{sum}_A = \sum_{i=1}^{N} (2x_i - 1)\, n_i$$

Squaring this gives the cost function:

$$H = \lambda \left(\sum_{i} (2x_i - 1) n_i\right)^2$$

Since $x_i \in \{0,1\}$ implies $x_i^2 = x_i$, expanding yields

$$H = \sum_i Q_{ii}\, x_i + \sum_{i < j} Q_{ij}\, x_i x_j + \text{const}$$

where $S = \sum_i n_i$ and

$$Q_{ii} = \lambda \cdot 4\, n_i(n_i - S), \qquad Q_{ij} = \lambda \cdot 8\, n_i n_j \quad (i \neq j)$$

---

#### Role of the penalty coefficient $\lambda$

| Large $\lambda$ | Strongly enforces equal sums — may make SA harder to converge |
|---|---|
| Small $\lambda$ | Weaker constraint — solution quality may degrade |
        """
    )

# OpenJij SA parameters (sidebar)
openjij_params = openjij_sidebar()

st.divider()

# Input UI
input_result = render_input()

# Output UI
if input_result is not None:
    numbers, bqm = input_result
    st.divider()
    render_output(numbers, bqm, openjij_params)

