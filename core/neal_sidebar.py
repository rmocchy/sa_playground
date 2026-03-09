"""Neal (D-Wave) SA parameter sidebar component.

Call this from any page as follows:

    from core.neal_sidebar import NealParams, neal_sidebar

    neal_params = neal_sidebar()
    if neal_params.run:
        sampler = neal.SimulatedAnnealingSampler()
        sample_set = sampler.sample(bqm, **neal_params.sampler_kwargs)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import streamlit as st


@dataclass
class NealParams:
    num_reads: int
    num_sweeps: int
    beta_min: float   # 1/T_start  (low beta = high temperature = exploration)
    beta_max: float   # 1/T_end    (high beta = low temperature = exploitation)
    beta_schedule_type: str
    seed: int | None
    run: bool

    @property
    def sampler_kwargs(self) -> dict:
        """Return kwargs ready to pass to neal sampler.sample()."""
        return {
            "num_reads": self.num_reads,
            "num_sweeps": self.num_sweeps,
            "beta_range": [self.beta_min, self.beta_max],
            "beta_schedule_type": self.beta_schedule_type,
            **({"seed": self.seed} if self.seed is not None else {}),
        }


def neal_sidebar() -> NealParams:
    """
    Render Neal SA parameters in the left sidebar and return a NealParams object.

    Returns
    -------
    NealParams
    """
    with st.sidebar:
        st.header("🔧 Neal SA Parameters")

        num_reads = st.number_input(
            "Num reads (independent runs)",
            min_value=1, max_value=200, value=20, step=1,
            help="Number of independent SA runs. Best solution is reported.",
        )
        num_sweeps = st.number_input(
            "Num sweeps (steps per read)",
            min_value=100, max_value=100_000, value=1_000, step=100,
        )
        st.markdown("**Temperature range** (β = 1/T)")
        beta_min = st.number_input(
            "β_min  (1/T_start, high T = exploration)",
            min_value=1e-6, max_value=1.0, value=0.001, format="%.4f",
        )
        beta_max = st.number_input(
            "β_max  (1/T_end, low T = exploitation)",
            min_value=0.1, max_value=1000.0, value=10.0, step=1.0,
        )
        beta_schedule_type = st.selectbox(
            "β schedule type",
            options=["geometric", "linear"],
            index=0,
        )
        seed_raw = st.number_input(
            "Random seed (0 = random)",
            min_value=0, max_value=99999, value=42, step=1,
        )
        st.divider()
        run = st.button("▶ Run SA", type="primary", use_container_width=True)

    return NealParams(
        num_reads=int(num_reads),
        num_sweeps=int(num_sweeps),
        beta_min=float(beta_min),
        beta_max=float(beta_max),
        beta_schedule_type=str(beta_schedule_type),
        seed=int(seed_raw) if seed_raw != 0 else None,
        run=run,
    )
