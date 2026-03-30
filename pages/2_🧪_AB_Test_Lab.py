import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from math import lgamma
import streamlit as st

from src.unit3_hypothesis_testing.ab_test_engine import ABTestEngine
from src.unit3_hypothesis_testing.bayesian_estimation import BayesianABTest

st.set_page_config(page_title="A/B Test Lab", page_icon="🧪", layout="wide")

st.title("🧪 A/B Test Lab")
st.caption("Unit III — Hypothesis Testing & Bayesian Estimation")

# ── Shared inputs in sidebar ──────────────────────────────────────────────────
st.sidebar.header("⚙️ Experiment Inputs")
ctrl_clicks = st.sidebar.number_input("Control — conversions", min_value=1, value=200, step=1)
ctrl_total = st.sidebar.number_input("Control — total visitors", min_value=1, value=2000, step=1)
var_clicks = st.sidebar.number_input("Variant — conversions", min_value=1, value=240, step=1)
var_total = st.sidebar.number_input("Variant — total visitors", min_value=1, value=2000, step=1)
alpha = st.sidebar.slider("Significance level (α)", min_value=0.01, max_value=0.10,
                           value=0.05, step=0.01)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_freq, tab_bayes, tab_ssc = st.tabs(
    ["📊 Frequentist Z-Test", "🎲 Bayesian Estimation", "📏 Sample Size Calculator"]
)

# ════════════════════════════════════════════════════════════════════════════════
# Tab 1 — Frequentist
# ════════════════════════════════════════════════════════════════════════════════
with tab_freq:
    st.subheader("Two-Proportion Z-Test")

    run_ztest = st.button("▶ Run Z-Test", type="primary", key="ztest_btn")

    if run_ztest or "ztest_result" in st.session_state:
        if run_ztest:
            engine = ABTestEngine()
            result = engine.run_ztest(
                int(ctrl_clicks), int(ctrl_total),
                int(var_clicks), int(var_total),
                alpha=float(alpha),
            )
            st.session_state["ztest_result"] = result

        result = st.session_state["ztest_result"]
        sig = result["is_significant"]

        # ── Metrics ──────────────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Z-Statistic", f"{result['z_statistic']:.4f}")
        m2.metric("P-Value", f"{result['p_value']:.4f}")
        m3.metric("Control Rate", f"{result['control_rate']:.2%}")
        m4.metric(
            "Variant Rate",
            f"{result['variant_rate']:.2%}",
            delta=f"{result['relative_uplift']:+.2%} lift",
        )

        # ── Result card ───────────────────────────────────────────────────────
        ci_lo, ci_hi = result["confidence_interval"]
        ci_str = f"[{ci_lo:+.4f}, {ci_hi:+.4f}]"
        if sig:
            st.success(
                f"✅ **Significant** at α={alpha} — {result['conclusion']}\n\n"
                f"95% CI for difference: {ci_str}"
            )
        else:
            st.error(
                f"❌ **Not Significant** at α={alpha} — {result['conclusion']}\n\n"
                f"95% CI for difference: {ci_str}"
            )

        # ── Bar chart ─────────────────────────────────────────────────────────
        ctrl_rate = result["control_rate"]
        var_rate = result["variant_rate"]
        ctrl_se = np.sqrt(ctrl_rate * (1 - ctrl_rate) / int(ctrl_total))
        var_se = np.sqrt(var_rate * (1 - var_rate) / int(var_total))

        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(
            ["Control", "Variant"],
            [ctrl_rate, var_rate],
            color=["steelblue", "coral" if not sig else "seagreen"],
            alpha=0.85,
            width=0.5,
        )
        ax.errorbar(
            [0, 1],
            [ctrl_rate, var_rate],
            yerr=[1.96 * ctrl_se, 1.96 * var_se],
            fmt="none",
            color="black",
            capsize=6,
            linewidth=2,
        )
        ax.set_ylabel("Conversion Rate")
        ax.set_title("Conversion Rates ± 95% CI")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        with st.expander("📐 Mathematical Details"):
            st.markdown(
                r"""
                **Two-proportion z-test**

                $$p_1 = \frac{c_1}{n_1},\quad p_2 = \frac{c_2}{n_2}$$

                Pooled proportion:
                $$\hat{p} = \frac{c_1 + c_2}{n_1 + n_2}$$

                Pooled standard error:
                $$\text{SE}_{\text{pooled}} = \sqrt{\hat{p}(1-\hat{p})\!\left(\frac{1}{n_1}+\frac{1}{n_2}\right)}$$

                Z-statistic:
                $$z = \frac{p_2 - p_1}{\text{SE}_{\text{pooled}}}$$

                Two-tailed p-value:
                $$p = 2\,(1 - \Phi(|z|))$$
                """
            )

# ════════════════════════════════════════════════════════════════════════════════
# Tab 2 — Bayesian
# ════════════════════════════════════════════════════════════════════════════════
with tab_bayes:
    st.subheader("Bayesian A/B Test — Beta-Binomial Conjugate Model")

    run_bayes = st.button("▶ Run Bayesian Test", type="primary", key="bayes_btn")

    if run_bayes or "bayes_result" in st.session_state:
        if run_bayes:
            btest = BayesianABTest()
            btest.update("A", int(ctrl_clicks), int(ctrl_total) - int(ctrl_clicks))
            btest.update("B", int(var_clicks), int(var_total) - int(var_clicks))
            prob_b = btest.probability_b_beats_a()
            loss_a = btest.expected_loss("A")
            loss_b = btest.expected_loss("B")
            posteriors = {
                "A": btest.get_posterior("A"),
                "B": btest.get_posterior("B"),
            }
            st.session_state["bayes_result"] = {
                "prob_b": prob_b,
                "loss_a": loss_a,
                "loss_b": loss_b,
                "posteriors": posteriors,
            }

        br = st.session_state["bayes_result"]
        prob_b = br["prob_b"]

        b1, b2, b3 = st.columns(3)
        b1.metric("P(B beats A)", f"{prob_b:.4f}")
        b2.metric("Expected Loss (A)", f"{br['loss_a']:.6f}")
        b3.metric("Expected Loss (B)", f"{br['loss_b']:.6f}")

        if prob_b >= 0.95:
            st.success(f"✅ **Recommend Variant B** — P(B > A) = {prob_b:.2%} ≥ 95%")
        elif prob_b <= 0.05:
            st.success(f"✅ **Recommend keeping Control A** — P(B > A) = {prob_b:.2%} ≤ 5%")
        else:
            st.warning(
                f"⚠️ **Inconclusive** — P(B > A) = {prob_b:.2%}. Collect more data."
            )

        # ── Posterior plot ────────────────────────────────────────────────────
        x = np.linspace(0.001, 0.999, 1000)
        fig, ax = plt.subplots(figsize=(8, 4))

        def _beta_pdf(x, a, b):
            log_norm = lgamma(a + b) - lgamma(a) - lgamma(b)
            return np.exp(log_norm + (a - 1) * np.log(np.maximum(x, 1e-300))
                          + (b - 1) * np.log(np.maximum(1 - x, 1e-300)))

        for variant, color in [("A", "steelblue"), ("B", "coral")]:
            a, b = br["posteriors"][variant]
            pdf = _beta_pdf(x, a, b)
            ax.plot(x, pdf, lw=2, color=color,
                    label=f"Variant {variant}: Beta({a:.0f}, {b:.0f})")
            ax.fill_between(x, pdf, alpha=0.2, color=color)

        ax.set_xlabel("Conversion Rate θ")
        ax.set_ylabel("Posterior Density")
        ax.set_title(f"Posterior Distributions — P(B > A) = {prob_b:.4f}")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Live simulation ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔴 Live Posterior Update Simulation")
    st.caption(
        "Watch the posteriors update as simulated clicks arrive one-by-one "
        "(true rates: A = 10%, B = 12%)."
    )

    live_steps = st.slider("Simulation steps", 10, 300, 80, step=10)
    start_live = st.button("▶ Start Live Simulation", key="live_btn")

    if start_live:
        live_test = BayesianABTest()
        true_a, true_b = 0.10, 0.12
        prob_history = []

        placeholder = st.empty()
        prog_metric = st.empty()

        for step in range(live_steps):
            click_a = int(np.random.rand() < true_a)
            click_b = int(np.random.rand() < true_b)
            live_test.update("A", click_a, 1 - click_a)
            live_test.update("B", click_b, 1 - click_b)

            p = live_test.probability_b_beats_a(n_simulations=10_000)
            prob_history.append(p)

            if (step + 1) % 5 == 0 or step == live_steps - 1:
                fig_live, axes = plt.subplots(1, 2, figsize=(10, 3))

                # Posterior curves
                x = np.linspace(0.001, 0.999, 500)
                for variant, color in [("A", "steelblue"), ("B", "coral")]:
                    a, bv = live_test.get_posterior(variant)
                    pdf = _beta_pdf(x, a, bv)
                    axes[0].plot(x, pdf, lw=2, color=color,
                                 label=f"Var {variant} Beta({a:.0f},{bv:.0f})")
                    axes[0].fill_between(x, pdf, alpha=0.2, color=color)
                axes[0].set_title(f"Posteriors after {step+1} steps")
                axes[0].legend(fontsize=8)
                axes[0].set_xlabel("θ")
                axes[0].set_ylabel("Density")

                # P(B>A) history
                axes[1].plot(prob_history, color="purple", lw=2)
                axes[1].axhline(0.95, color="green", linestyle="--", alpha=0.7,
                                label="95% threshold")
                axes[1].axhline(0.5, color="grey", linestyle=":", alpha=0.5)
                axes[1].set_ylim(0, 1)
                axes[1].set_title("P(B > A) over time")
                axes[1].set_xlabel("Step")
                axes[1].set_ylabel("P(B > A)")
                axes[1].legend(fontsize=8)

                plt.tight_layout()
                placeholder.pyplot(fig_live)
                plt.close(fig_live)
                prog_metric.metric("P(B > A)", f"{p:.4f}")

        st.success(f"Simulation complete — final P(B > A) = {prob_history[-1]:.4f}")

    with st.expander("📐 Mathematical Details"):
        st.markdown(
            r"""
            **Beta-Binomial Conjugate Model**

            Prior: $\theta \sim \text{Beta}(\alpha_0, \beta_0)$ (default: uniform $\alpha_0=\beta_0=1$)

            After observing $s$ successes and $f$ failures:

            $$\theta \mid \text{data} \sim \text{Beta}(\alpha_0 + s,\; \beta_0 + f)$$

            **P(B > A)** via Monte Carlo:

            $$\hat{P}(\theta_B > \theta_A) = \frac{1}{N}\sum_{i=1}^N \mathbf{1}[\theta_B^{(i)} > \theta_A^{(i)}]$$

            where $\theta_A^{(i)} \sim \text{Beta}(\alpha_A, \beta_A)$ and similarly for B.

            **Expected Loss:**

            $$E[\text{loss}(A)] = E[\max(\theta_B - \theta_A, 0)]$$
            """
        )

# ════════════════════════════════════════════════════════════════════════════════
# Tab 3 — Sample Size Calculator
# ════════════════════════════════════════════════════════════════════════════════
with tab_ssc:
    st.subheader("Sample Size Calculator")
    st.caption("How many users do you need before running the experiment?")

    sc1, sc2 = st.columns(2)
    with sc1:
        baseline = st.slider("Baseline conversion rate", 0.01, 0.50, 0.10, 0.01)
        mde = st.slider("Minimum detectable effect (MDE)", 0.001, 0.10, 0.02, 0.001,
                        format="%.3f")
    with sc2:
        ssc_alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01, key="ssc_alpha")
        power = st.slider("Statistical power (1 - β)", 0.70, 0.99, 0.80, 0.01)

    calc_btn = st.button("📐 Calculate Sample Size", type="primary")

    if calc_btn or "ssc_result" in st.session_state:
        if calc_btn:
            engine = ABTestEngine()
            n = engine.calculate_sample_size(
                baseline_rate=float(baseline),
                min_detectable_effect=float(mde),
                alpha=float(ssc_alpha),
                power=float(power),
            )
            st.session_state["ssc_result"] = n

        n = st.session_state["ssc_result"]
        st.metric("Required sample size per group", f"{n:,}")
        st.info(f"ℹ️ You need **{n:,} users per group** ({2*n:,} total) to detect a "
                f"{mde:.1%} lift from a {baseline:.1%} baseline with "
                f"α={ssc_alpha} and power={power:.0%}.")

        # ── Power curve ───────────────────────────────────────────────────────
        engine2 = ABTestEngine()
        mde_range = np.linspace(max(0.001, mde * 0.2), mde * 3, 40)
        ns = [engine2.calculate_sample_size(float(baseline), float(m),
                                             float(ssc_alpha), float(power))
              for m in mde_range]

        fig_pw, ax_pw = plt.subplots(figsize=(7, 4))
        ax_pw.plot(mde_range * 100, ns, "b-o", markersize=4)
        ax_pw.axvline(x=mde * 100, color="red", linestyle="--", label=f"Selected MDE={mde:.1%}")
        ax_pw.set_xlabel("Minimum Detectable Effect (%)")
        ax_pw.set_ylabel("Required Sample Size per Group")
        ax_pw.set_title("Sample Size vs. MDE (Power Curve)")
        ax_pw.legend()
        ax_pw.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_pw)
        plt.close(fig_pw)

        with st.expander("📐 Formula"):
            st.markdown(
                r"""
                $$n = \frac{(z_{\alpha/2} + z_\beta)^2\,(p_1(1-p_1) + p_2(1-p_2))}{(p_2 - p_1)^2}$$

                where $p_2 = p_1 + \text{MDE}$, $z_{\alpha/2}$ is the critical value for
                significance and $z_\beta$ is the critical value for power.
                """
            )
