"""Streamlit page: Recommendation System"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from core.neal_sidebar import neal_sidebar
from pages.recommendation.input_ui import render_input
from pages.recommendation.items_data import DEFAULT_ITEMS
from pages.recommendation.output_ui import render_output

st.title("🛍️ Recommendation System")
st.markdown(
    """
Find the best set of products from a catalog that fits your **required categories** and **budget**,  
using QUBO × Simulated Annealing (SA).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1Mn286V_Rbad--ZDMTUEpLL5F7X8Qokrt/view?usp=sharing)
"""
)

# ── Formulation ──────────────────────────────────────────────────
st.subheader("📐 QUBO Formulation")
st.markdown(
    r"""
**Variable** : $x_i \in \{0, 1\}$ — select item $i$ if $x_i = 1$

**Objective** : minimise the sum of four terms
"""
)
st.latex(
    r"""
    \min_{\mathbf{x}} \;
    \underbrace{\lambda_{\mathrm{req}} \sum_{c \in C_{\mathrm{req}}}
        \left(\sum_{i \in c} x_i - 1\right)^{\!2}}_{E_{\mathrm{req}}}
    + \underbrace{\lambda_{\mathrm{opt}} \sum_{c \in C_{\mathrm{opt}}}
        \left(\sum_{i \in c} x_i - \tfrac{1}{2}\right)^{\!2}}_{E_{\mathrm{opt}}}
    + \underbrace{\lambda_{\mathrm{bud}} \left(\sum_i p_i x_i - B\right)^{\!2}}_{E_{\mathrm{bud}}}
    - \underbrace{\lambda_{s} \sum_i s_i x_i}_{E_{\mathrm{score}}}
    """
)

with st.expander("🔍 See details for each term"):
    st.markdown(
        r"""
#### $E_{\mathrm{req}}$ — Required category constraint

$$E_{\mathrm{req}} = \lambda_{\mathrm{req}} \sum_{c \in C_{\mathrm{req}}} \left(\sum_{i \in c} x_i - 1\right)^2$$

Forces exactly one item to be selected from each required category $c$.
A penalty is incurred if the selection count is 0 or $\geq 2$.

QUBO expansion ($S_c = \sum_{i \in c} x_i$):

| Coefficient | Value |
|---|---|
| $Q_{ii}$ (within category $c$) | $-\lambda_{\mathrm{req}}$ |
| $Q_{ij}$ ($i \neq j$, same category) | $+\lambda_{\mathrm{req}}$ |

---

#### $E_{\mathrm{opt}}$ — Optional category constraint (at-most-one)

$$E_{\mathrm{opt}} = \lambda_{\mathrm{opt}} \sum_{c \in C_{\mathrm{opt}}} \sum_{\substack{i,j\in c\\i\neq j}} x_i x_j$$

Derived from $\lambda_{\mathrm{opt}}\!\left(\displaystyle\sum_{i\in c} x_i - \tfrac{1}{2}\right)^{\!2}$ by dropping the constant $\frac{\lambda_{\mathrm{opt}}}{4}$.
Expanding and using $x_i^2 = x_i$:

$$\lambda_{\mathrm{opt}}\!\left(\sum_i x_i - \tfrac{1}{2}\right)^{\!2} = \lambda_{\mathrm{opt}}\!\sum_i\sum_j x_i x_j - \lambda_{\mathrm{opt}}\!\sum_i x_i + \tfrac{\lambda_{\mathrm{opt}}}{4}$$

The linear term $-\lambda_{\mathrm{opt}}\sum_i x_i$ cancels the diagonal contributions from $\sum_i\sum_j$, leaving $Q_{ii}=0$ and $Q_{ij}=+\lambda_{\mathrm{opt}}$ for $i\neq j$.

The energy minimum of the original square is at $\sum x_i = \tfrac{1}{2}$, so **both $k=0$ and $k=1$ are equally optimal** (cost = 0), and $k\geq 2$ is penalised:

| Items selected $k$ | Cost |
|---|---|
| 0 | $0$ |
| 1 | $0$ |
| 2 | $2\lambda_{\mathrm{opt}}$ |
| 3 | $6\lambda_{\mathrm{opt}}$ |

QUBO coefficients:

| Coefficient | Value |
|---|---|
| $Q_{ii}$ (within optional category $c$) | $0$ |
| $Q_{ij}$ ($i \neq j$, same category) | $+\lambda_{\mathrm{opt}}$ |

---

#### $E_{\mathrm{bud}}$ — Budget constraint

$$E_{\mathrm{bud}} = \lambda_{\mathrm{bud}} \left(\sum_i p_i x_i - B\right)^2$$

The UI coefficient is on a $\times 10^{-6}$ scale. The penalty grows as total spend exceeds budget $B$.

$$Q_{ij} += \lambda_{\mathrm{bud}} \cdot p_i p_j, \qquad Q_{ii} -= 2\,\lambda_{\mathrm{bud}} \cdot p_i B$$

---

#### $E_{\mathrm{score}}$ — Score maximisation

$$E_{\mathrm{score}} = \lambda_s \sum_i s_i x_i$$

Lowers the diagonal entries for highly-rated items, encouraging their selection.
        """
    )

# SA parameters (sidebar)
neal_params = neal_sidebar()

st.divider()

# Input UI + QUBO construction
input_result = render_input(items=DEFAULT_ITEMS)

# Output UI
if input_result is not None:
    items, required_cats, optional_cats, budget, bqm, Q = input_result
    st.divider()
    st.subheader("Results")
    render_output(
        items=items,
        budget=budget,
        bqm=bqm,
        neal_params=neal_params,
    )
