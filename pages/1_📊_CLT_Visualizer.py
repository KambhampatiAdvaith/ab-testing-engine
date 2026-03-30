import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.unit1_2_random_variables.distributions import (
    BinomialDistribution,
    NormalDistribution,
    UniformDistribution,
)
from src.unit1_2_random_variables.traffic_simulator import (
    simulate_clicks,
    simulate_session_times,
    simulate_traffic,
)

st.set_page_config(page_title="CLT Visualizer", page_icon="📊", layout="wide")

st.title("📊 CLT Visualizer")
st.caption("Unit I & II — Random Variables & the Central Limit Theorem")

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Simulation Controls")

dist_name = st.sidebar.selectbox(
    "Population distribution",
    ["Uniform", "Exponential (via Uniform)", "Binomial"],
)

n_simulations = st.sidebar.slider(
    "Number of simulations", min_value=100, max_value=10_000, value=1_000, step=100
)

sample_size_options = [5, 10, 30, 100, 500]
selected_sizes = st.sidebar.multiselect(
    "Sample sizes to display",
    options=sample_size_options,
    default=[5, 30, 100, 500],
)
if not selected_sizes:
    selected_sizes = [5, 30]
selected_sizes = sorted(selected_sizes)

run_btn = st.sidebar.button("▶ Run Simulation", type="primary", use_container_width=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _build_distribution(dist_name: str):
    """Return a distribution object and a display name."""
    if dist_name == "Uniform":
        return UniformDistribution(a=0, b=1), "Uniform(0, 1)"
    elif dist_name == "Exponential (via Uniform)":
        # Wrap Uniform so CLT demo works; we generate exponential samples
        # by inverse transform: X = -ln(U) ≈ Exp(1).  We can't subclass,
        # so we return a lightweight adapter.
        class _ExpAdapter:
            def sample(self, n):
                u = np.random.uniform(0, 1, n)
                return -np.log(np.maximum(u, 1e-300))
            def mean(self):
                return 1.0
            def variance(self):
                return 1.0
        return _ExpAdapter(), "Exponential(1)"
    else:  # Binomial
        return BinomialDistribution(n=20, p=0.3), "Binomial(20, 0.3)"


@st.cache_data(show_spinner=False)
def _run_clt(dist_name: str, sample_sizes: tuple, n_simulations: int):
    dist, label = _build_distribution(dist_name)
    pop_mean = dist.mean()
    pop_var = dist.variance()
    results = {}
    for n in sample_sizes:
        means = np.array([np.mean(dist.sample(n)) for _ in range(n_simulations)])
        results[n] = means
    return results, pop_mean, pop_var, label


def _compute_skew_kurt(arr: np.ndarray):
    mu = arr.mean()
    sigma = arr.std()
    if sigma == 0:
        return 0.0, 0.0
    z = (arr - mu) / sigma
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4) - 3)
    return skew, kurt


# ── Session state ─────────────────────────────────────────────────────────────
if "clt_results" not in st.session_state or run_btn:
    with st.spinner("Running CLT simulation…"):
        clt_results, pop_mean, pop_var, dist_label = _run_clt(
            dist_name, tuple(selected_sizes), n_simulations
        )
    st.session_state["clt_results"] = clt_results
    st.session_state["clt_meta"] = (pop_mean, pop_var, dist_label)
else:
    clt_results = st.session_state["clt_results"]
    pop_mean, pop_var, dist_label = st.session_state["clt_meta"]

# ── Population distribution plot ─────────────────────────────────────────────
st.subheader("Population Distribution")

fig_pop, ax_pop = plt.subplots(figsize=(7, 3))
pop_samples = list(clt_results.values())[0]  # use first key's raw means as proxy
# Draw a fresh sample for the population plot
dist_obj, _ = _build_distribution(dist_name)
pop_raw = dist_obj.sample(5000)
ax_pop.hist(pop_raw, bins=50, density=True, color="steelblue", alpha=0.7,
            label=f"Population: {dist_label}")
ax_pop.set_xlabel("Value")
ax_pop.set_ylabel("Density")
ax_pop.set_title(f"Population — {dist_label}")
ax_pop.legend()
plt.tight_layout()
st.pyplot(fig_pop)
plt.close(fig_pop)

# ── CLT panels ────────────────────────────────────────────────────────────────
st.subheader("Sampling Distribution of the Mean")
st.caption(
    "Each panel shows the distribution of sample means for a given sample size *n*, "
    "with the theoretical Normal curve (CLT prediction) overlaid in red."
)

n_cols = len(selected_sizes)
cols = st.columns(n_cols)
convergence_rows = []

for col, n in zip(cols, selected_sizes):
    means = clt_results[n]
    theoretical_std = np.sqrt(pop_var / n)

    skew, kurt = _compute_skew_kurt(means)
    convergence_rows.append({"n": n, "skewness": round(skew, 4), "excess kurtosis": round(kurt, 4)})

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(means, bins=40, density=True, alpha=0.65, color="steelblue",
            label="Sample means")
    x = np.linspace(means.min(), means.max(), 300)
    pdf = (1 / (theoretical_std * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x - pop_mean) / theoretical_std) ** 2
    )
    ax.plot(x, pdf, "r-", lw=2,
            label=f"N({pop_mean:.2f}, {theoretical_std**2:.4f})")
    ax.set_title(f"n = {n}")
    ax.set_xlabel("Sample Mean")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
    plt.tight_layout()
    col.pyplot(fig)
    plt.close(fig)

# ── Convergence metrics ───────────────────────────────────────────────────────
st.subheader("Convergence Metrics")
st.caption(
    "For a perfect Normal distribution, skewness → 0 and excess kurtosis → 0 as *n* increases."
)
st.dataframe(
    convergence_rows,
    use_container_width=True,
    column_config={
        "n": st.column_config.NumberColumn("Sample Size (n)"),
        "skewness": st.column_config.NumberColumn("Skewness (→ 0)", format="%.4f"),
        "excess kurtosis": st.column_config.NumberColumn("Excess Kurtosis (→ 0)", format="%.4f"),
    },
)

with st.expander("📐 Mathematical Background"):
    st.markdown(
        r"""
        **Central Limit Theorem**

        Let $X_1, X_2, \ldots, X_n$ be i.i.d. random variables with mean $\mu$ and
        variance $\sigma^2 < \infty$.  As $n \to \infty$:

        $$\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{d} \mathcal{N}\!\left(\mu,\, \frac{\sigma^2}{n}\right)$$

        The *rate* of convergence depends on the skewness of the population distribution
        (Berry–Esséen theorem).
        """
    )

st.divider()

# ── Traffic Simulator ─────────────────────────────────────────────────────────
st.subheader("🚦 Traffic Simulator")

sim_col1, sim_col2 = st.columns(2)
with sim_col1:
    n_users_sim = st.number_input(
        "Number of users", min_value=10, max_value=10_000, value=500, step=50
    )
with sim_col2:
    click_prob = st.slider(
        "Click probability", min_value=0.01, max_value=0.99, value=0.15, step=0.01
    )

sim_btn = st.button("🚀 Simulate Traffic", type="primary")

if sim_btn or "traffic_data" in st.session_state:
    if sim_btn:
        traffic = simulate_traffic(n_users=int(n_users_sim), n_days=1)
        st.session_state["traffic_data"] = traffic
        st.session_state["traffic_clicks"] = simulate_clicks(
            int(n_users_sim), float(click_prob)
        )
        st.session_state["traffic_sessions"] = simulate_session_times(
            int(n_users_sim), distribution="normal", params={"mu": 5.0, "sigma": 2.0}
        )

    traffic = st.session_state["traffic_data"]
    clicks = st.session_state["traffic_clicks"]
    sessions = st.session_state["traffic_sessions"]

    df_traffic = pd.DataFrame(
        {
            "user_id": traffic["user_id"][:50],
            "clicks": traffic["clicks"][:50],
            "session_time (min)": np.round(traffic["session_time"][:50], 2),
            "page_views": traffic["page_views"][:50],
        }
    )
    st.markdown("**Sample of simulated traffic (first 50 rows)**")
    st.dataframe(df_traffic, use_container_width=True)

    t_col1, t_col2 = st.columns(2)

    with t_col1:
        fig_c, ax_c = plt.subplots(figsize=(5, 3))
        unique, counts = np.unique(clicks, return_counts=True)
        ax_c.bar(unique, counts / counts.sum(), color="steelblue", alpha=0.8)
        ax_c.set_xlabel("Clicks (0 or 1)")
        ax_c.set_ylabel("Proportion")
        ax_c.set_title(f"Click Distribution\n(p={click_prob:.2f}, n={int(n_users_sim)})")
        plt.tight_layout()
        t_col1.pyplot(fig_c)
        plt.close(fig_c)

    with t_col2:
        fig_s, ax_s = plt.subplots(figsize=(5, 3))
        ax_s.hist(sessions, bins=40, density=True, color="coral", alpha=0.8)
        ax_s.set_xlabel("Session Time (min)")
        ax_s.set_ylabel("Density")
        ax_s.set_title("Session Time Distribution\nN(5, 4)")
        plt.tight_layout()
        t_col2.pyplot(fig_s)
        plt.close(fig_s)
