"""QUBO Simulated Annealing — Entry point."""

import streamlit as st

st.set_page_config(
    page_title="QUBO SA Simulator",
    page_icon="🌡️",
    layout="wide",
)

# ── Page definitions ──────────────────────────
number_partitioning = st.Page(
    "pages/number_partitioning/page.py",
    title="Number Partitioning",
    icon="✂️",
    url_path="number_partitioning",
)
recommendation = st.Page(
    "pages/recommendation/page.py",
    title="Recommendation System",
    icon="🛍️",
    url_path="recommendation",
)
scheduling = st.Page(
    "pages/scheduling/page.py",
    title="Scheduling Optimizer",
    icon="📅",
    url_path="scheduling",
)


# ── Home page ─────────────────────
def home() -> None:
    st.title("🌡️ QUBO Simulated Annealing")
    st.markdown(
        """
A collection of simulators that encode various combinatorial optimization problems as QUBO  
and solve them with Simulated Annealing (SA) — all in the browser.

Click a problem card to open its simulator.
"""
    )
    st.divider()

    PROBLEMS = [
        {
            "name": "✂️ Number Partitioning",
            "desc": "Find a way to split a list of numbers into two groups with equal sums.",
            "qubo": "minimize (Σ (2xᵢ − 1) nᵢ)²",
            "page": number_partitioning,
            "status": "✅ Available",
        },
        {
            "name": "🛍️ Recommendation System",
            "desc": "Use SA to find the best cart from 21 products that fits your required categories and budget.",
            "qubo": "minimize λ_req·Req + λ_opt·Opt + λ_budget·Budget − λ_score·Score",
            "page": recommendation,
            "status": "✅ Available",
        },
        {
            "name": "📅 Scheduling Optimizer",
            "desc": "Assign workers to tasks over discrete time slots — minimising makespan while meeting effort, continuity, no-multitasking and precedence constraints.",
            "qubo": "minimize α·makespan + λ₁·Effort + λ₂·Continuity + λ₃·NoMultitask + λ₄·Blocked + λ₅·Precedence",
            "page": scheduling,
            "status": "✅ Available",
        },
    ]

    for p in PROBLEMS:
        with st.container(border=True):
            col_title, col_status = st.columns([4, 1])
            col_title.markdown(f"### {p['name']}")
            col_status.markdown(f"<br>{p['status']}", unsafe_allow_html=True)
            st.markdown(p["desc"])
            st.markdown(f"**QUBO Objective**: `{p['qubo']}`")
            st.page_link(p["page"], label="▶ Open Simulator", icon="🔗")

    st.divider()
    st.caption("Coming soon: Max-Cut, Graph Coloring, Traveling Salesman Problem, and more.")


# ── Navigation ─────────────────────────────
pg = st.navigation(
    {
        "Home": [st.Page(home, title="Home", icon="🌡️", default=True)],
        "Problems": [number_partitioning, recommendation, scheduling],
    }
)
pg.run()
