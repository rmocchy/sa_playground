from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from core.openjij_sidebar import openjij_sidebar
from pages.task_scheduling.input_ui import render_input
from pages.task_scheduling.output_ui import render_output

st.title("📅 Task Scheduling Optimizer")
st.markdown(
    """
Assign **workers** to **tasks** across discrete **time slots** to minimise makespan,  
while respecting effort requirements, continuity, no-multitasking and precedence constraints —  
encoded as a QUBO problem and solved with Simulated Annealing (SA).

📄 QUBO implementation: [pages/task_scheduling/qubo.py](https://github.com/rmocchy/sa_playground/blob/main/pages/task_scheduling/qubo.py)
"""
)

# ── Formulation ──────────────────────────────────────────────────
st.subheader("📐 QUBO Formulation")
st.markdown(
    r"""
**Variable** : $x_{p,f,t} \in \{0, 1\}$ — 1 if worker $p$ is assigned to task $f$ at time slot $t$

**Objective** : makespan minimisation + sum of five constraint penalties
"""
)
st.latex(
    r"""
    \min_{\mathbf{x}} \;
    \underbrace{\alpha \sum_{p,f,t} t\, x_{p,f,t}}_{E_\alpha}
    + \underbrace{\lambda_1 \sum_f \left(\sum_{p,t} d_{p,f}\, x_{p,f,t} - C_f\right)^{\!2}}_{E_1}
    + \underbrace{\lambda_2 \sum_{p,f,t} (x_{p,f,t} - x_{p,f,t+1})^2}_{E_2}
    + \underbrace{\lambda_3 \sum_{p,t}\sum_{f < g} x_{p,f,t}\,x_{p,g,t}}_{E_3}
    + \underbrace{\lambda_4 \sum_{\substack{f,t:\\D_{f,t}=0}}\sum_p x_{p,f,t}}_{E_4}
    + \underbrace{\lambda_5 \sum_{(f_i \to f_j)} \sum_{p,t}\sum_{p',t' \geq t} x_{p,f_j,t}\,x_{p',f_i,t'}}_{E_5}
    """
)

with st.expander("🔍 See details for each term"):
    st.markdown(
        r"""
#### $E_\alpha$ — Makespan minimisation

$$E_\alpha = \alpha \sum_{p,f,t} t\, x_{p,f,t}$$

Using the slot index $t$ as a weight penalises late assignments,
encouraging the schedule to finish as early as possible.

---

#### $E_1$ — Effort constraint

$$E_1 = \lambda_1 \sum_f \left(\sum_{p,t} d_{p,f}\, x_{p,f,t} - C_f\right)^2$$

Forces the total assigned effort for each task $f$ to equal its required capacity $C_f$ (in person-slot units).
$d_{p,f}$ is worker $p$'s productivity on task $f$ (default 1.0).

$$C_f = \frac{20}{\Delta t} \cdot A_f$$

where $A_f$ is the person-month effort and $\Delta t$ is the duration of one slot (in months).

---

#### $E_2$ — Task continuity constraint

$$E_2 = \lambda_2 \sum_{p,f,t} (x_{p,f,t} - x_{p,f,t+1})^2$$

Encourages consecutive slot assignments, penalising gaps (interrupted work).

---

#### $E_3$ — No-multitasking constraint

$$E_3 = \lambda_3 \sum_{p,t} \sum_{f < g} x_{p,f,t}\, x_{p,g,t}$$

Prohibits the same worker from being assigned to more than one task in the same time slot.

---

#### $E_4$ — Unassignable period constraint

$$E_4 = \lambda_4 \sum_{\substack{f,t:\,D_{f,t}=0}} \sum_p x_{p,f,t}$$

Applies a penalty for any assignment to a slot marked $D_{f,t} = 0$ (unavailable).

---

#### $E_5$ — Precedence constraint

$$E_5 = \lambda_5 \sum_{(f_i \to f_j)} \sum_{p,t} \sum_{p',\, t' \geq t} x_{p,f_j,t}\, x_{p',f_i,t'}$$

Prohibits task $f_j$ from starting before its predecessor $f_i$ has finished.
($f_i \to f_j$ means "$f_i$ must precede $f_j$".)
        """
    )

# SA parameters (sidebar)
openjij_params = openjij_sidebar()

st.divider()

# Input UI + QUBO construction
result = render_input()

# Output UI
if result is not None:
    cfg, bqm, Q, var_list = result
    st.divider()
    st.subheader("Results")
    render_output(cfg=cfg, bqm=bqm, var_list=var_list, openjij_params=openjij_params)
