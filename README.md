# 🌡️ QUBO SA Simulator

A Streamlit application that solves combinatorial optimization problems formulated as  
**QUBO (Quadratic Unconstrained Binary Optimization)** using **Simulated Annealing (SA)** — all in the browser.

Each problem page lets you tweak SA parameters (initial temperature, cooling rate, max iterations, etc.)  
from the sidebar and interactively inspect the energy convergence graph and QUBO matrix heatmap.

---

## Setup & Running

### Prerequisites

- Python 3.13
- [direnv](https://direnv.net/) — `brew install direnv`

### Steps

```bash
# 1. Clone the repository
git clone <repository-url>
cd recommend_system

# 2. Allow direnv (first time only)
direnv allow

# 3. Install dependencies
make install

# 4. Start the app
make run
```

Open http://localhost:8501 in your browser.

### `make` Commands

| Command | Description |
|---------|-------------|
| `make run` | Start the Streamlit app |
| `make install` | Install packages from `requirements.txt` |
| `make lint` | Lint with ruff |
| `make fmt` | Format with ruff |
| `make clean` | Remove `__pycache__` and Streamlit cache |

---

<details>
<summary>📁 File Structure &amp; Descriptions</summary>

> Reference this section when forking the repo to add a new problem.

```
.
├── app.py                          # Entry point & page navigation
├── makefile                        # Developer commands
├── requirements.txt                # Python dependencies
├── pyrightconfig.json              # Pyright / VS Code type-check config
│
├── core/                           # Problem-agnostic shared modules
│   ├── sa.py                       # Simulated Annealing solver
│   ├── sa_sidebar.py               # SA parameter sidebar component
│   └── sa_viz.py                   # Visualization helpers (convergence graph, QUBO heatmap)
│
└── pages/                          # Per-problem page implementations
    ├── number_partitioning/
    │   ├── page.py                 # Page entry point
    │   ├── qubo.py                 # QUBO formulation
    │   ├── input_ui.py             # Input UI (number list, penalty slider)
    │   └── output_ui.py            # Output UI (results, charts)
    │
    └── recommendation/
        ├── page.py                 # Page entry point
        ├── qubo.py                 # QUBO formulation
        ├── input_ui.py             # Input UI (product catalog, conditions)
        ├── output_ui.py            # Output UI (results, charts)
        ├── items_data.py           # Product catalog data definitions
        └── cards.py                # Product card HTML component
```

---

### `app.py` — Entry Point

Defines Streamlit multi-page navigation.  
The home screen displays a card list of all registered problems.  
To add a new problem page, create an `st.Page(...)` entry and register it in `st.navigation()`.

---

### `core/sa.py` — SA Solver

```
simulated_annealing(Q, T_init, T_min, cooling_rate, max_iter, seed, timeout_sec)
  → { best_x, best_energy, energy_history, best_history, temp_history, n_iter, timed_out }
```

A general-purpose SA solver. Accepts a QUBO matrix `Q` (n×n) and searches for the binary vector  
$\mathbf{x} \in \{0,1\}^n$ that minimizes energy $E = \mathbf{x}^\top Q\mathbf{x}$.  
A configurable timeout prevents the UI from freezing on large problems.

---

### `core/sa_sidebar.py` — SA Parameter Sidebar

```python
from core.sa_sidebar import sa_sidebar

sa_params = sa_sidebar()
if sa_params.run:
    result = simulated_annealing(Q, **sa_params.sa)
```

Renders the SA parameter controls in the left sidebar and returns a `SAParams` dataclass.  
Shared across all pages — simply call it from any `page.py`.

---

### `core/sa_viz.py` — Visualization Helpers

| Function | Description |
|----------|-------------|
| `plot_sa_detail(energy_history, best_history, temp_history)` | 2-panel Plotly figure: energy convergence + temperature |
| `plot_qubo_matrix(Q, var_labels)` | QUBO matrix heatmap |

---

### `pages/<problem>/page.py` — Page Entry Point

Root file for each problem page. Follows this structure:

1. Call `sa_sidebar()` to render the sidebar and get SA parameters
2. Call `render_input()` to render the input UI and build the QUBO matrix
3. Call `render_output(...)` to run SA and display results

---

### `pages/<problem>/qubo.py` — QUBO Formulation

Contains the QUBO matrix builder and a `PARAMS` dict for Streamlit slider definitions.  
**This is the most important file when adding a new problem.**

Entries in `PARAMS` follow the schema `{ "label", "default", "min", "max", "step" }`,  
allowing `input_ui.py` to auto-generate sliders without additional boilerplate.

---

### `pages/<problem>/input_ui.py` — Input UI

Accepts user input (numbers, dropdowns, sliders) and calls `qubo.py` to build the QUBO matrix.  
Returns a tuple `(inputs..., Q)` that `page.py` forwards to `render_output()`.

---

### `pages/<problem>/output_ui.py` — Output UI

Runs SA when `sa_params.run` is `True` and displays the results.  
Uses `core/sa_viz.py` to render the energy convergence graph and QUBO heatmap.

---

### `pages/recommendation/items_data.py` — Product Catalog

Defines the `Item` dataclass and the `DEFAULT_ITEMS` list.  
Edit this file to add or modify products in the catalog.

### `pages/recommendation/cards.py` — Product Card UI

An HTML-based card component for displaying products.  
Called from `input_ui.py`.

</details>

---

<details>
<summary>➕ Adding a New Problem</summary>

1. Create `pages/<new_problem>/` directory
2. Implement `qubo.py` — QUBO formulation and `PARAMS` definition
3. Implement `input_ui.py`, `output_ui.py`, and `page.py`
4. Register the page in `app.py` via `st.Page(...)` and `st.navigation()`

`pages/number_partitioning/` is the simplest existing example to reference.

</details>
