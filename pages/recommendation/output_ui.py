"""Recommendation System — Neal SA execution & rich output UI."""

from __future__ import annotations

import dimod
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.neal_sa import run_neal
from core.neal_sidebar import NealParams
from pages.recommendation.cards import compact_card_html, item_card_html
from pages.recommendation.items_data import Item

# ── Main output UI ────────────────────────────────────
def render_output(
    items: list[Item],
    budget: float,
    bqm: dimod.BinaryQuadraticModel,
    neal_params: NealParams,
) -> None:
    """
    Run Neal SA and display results in a rich shopping-site style layout.
    """
    if not neal_params.run:
        st.info("👈 Press the **Run SA** button in the sidebar.")
        return

    # ── Run Neal SA ────────────────────────────────────────
    with st.spinner("🤖 Computing recommendations with Neal SA…"):
        result = run_neal(bqm, neal_params)

    best_x = result.best_x
    best_energy: float = result.penalty
    elapsed = result.elapsed_sec

    recommended = [it for it, xi in zip(items, best_x) if xi == 1]
    not_recommended = [it for it, xi in zip(items, best_x) if xi == 0]

    total_price = sum(it.price for it in recommended)

    # ── KPI bar ──────────────────────────────────────────
    k1, k2, k3 = st.columns(3)
    k1.metric("🛒 Recommended Items", f"{len(recommended)}")
    k2.metric("💰 Total Price", f"${total_price:,}")
    k3.metric("⚡ SA Runtime", f"{elapsed * 1000:.0f} ms")

    st.divider()

    # ── Recommended product grid ──────────────────────────────
    st.subheader(f"🛒 Recommended Products ({len(recommended)})")

    if not recommended:
        st.warning("No products were recommended. Try adjusting the SA parameters or QUBO settings.")
    else:
        # Sort by rating and display
        rec_ranked = sorted(recommended, key=lambda x: -x.score)
        cols_per_row = min(4, len(rec_ranked))
        for row_start in range(0, len(rec_ranked), cols_per_row):
            row_items = rec_ranked[row_start: row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, it in zip(cols, row_items):
                with col:
                    st.markdown(
                        item_card_html(it),
                        unsafe_allow_html=True,
                    )
            st.write("")  # gap

    st.divider()

    # ── Cart total bar ─────────────────────────────────────────
    budget_pct = min(total_price / budget * 100, 150) if budget > 0 else 0
    bar_color = "#e74c3c" if total_price > budget else "#2ecc71"
    st.markdown(f"""
<div style="background:#f8f9fa;border-radius:12px;padding:16px 20px;margin-bottom:16px;">
  <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
    <span style="font-weight:700;color:#333;">🛒 Cart Total</span>
    <span style="font-size:20px;font-weight:800;color:{bar_color};">${total_price:,}</span>
  </div>
  <div style="background:#e9ecef;border-radius:8px;height:12px;overflow:hidden;">
    <div style="
      background:linear-gradient(90deg,{bar_color},{bar_color}aa);
      width:{min(budget_pct,100):.1f}%;height:12px;border-radius:8px;
      transition:width .4s;
    "></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:12px;color:#888;margin-top:4px;">
    <span>$0</span>
    <span>Budget Limit ${budget:,.0f}</span>
  </div>
</div>
""", unsafe_allow_html=True)

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