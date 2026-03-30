import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.unit4_regression.linear_regression import MultipleLinearRegression
from src.unit4_regression.hypothesis_tests import t_test_coefficients, f_test_overall
from src.unit4_regression.lifetime_value import generate_ltv_dataset

st.set_page_config(page_title="LTV Predictor", page_icon="📈", layout="wide")

st.title("📈 LTV Predictor")
st.caption("Unit IV — Multiple Linear Regression for Customer Lifetime Value")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Model Settings")
n_users = st.sidebar.slider(
    "Synthetic users to generate", min_value=100, max_value=10_000, value=1_000, step=100
)
train_btn = st.sidebar.button("🏋️ Train Model", type="primary", use_container_width=True)

# ── Train / load model ────────────────────────────────────────────────────────
if train_btn or "ltv_model" in st.session_state:
    if train_btn:
        with st.spinner("Training regression model…"):
            X, y, feature_names = generate_ltv_dataset(n_users=int(n_users))
            model = MultipleLinearRegression()
            model.fit(X, y)
            t_res = t_test_coefficients(model)
            f_res = f_test_overall(model)
        st.session_state["ltv_model"] = model
        st.session_state["ltv_features"] = feature_names
        st.session_state["ltv_t"] = t_res
        st.session_state["ltv_f"] = f_res
        st.session_state["ltv_X"] = X
        st.session_state["ltv_y"] = y
        st.success("✅ Model trained successfully!")

    model: MultipleLinearRegression = st.session_state["ltv_model"]
    feature_names = st.session_state["ltv_features"]
    t_res = st.session_state["ltv_t"]
    f_res = st.session_state["ltv_f"]
    X = st.session_state["ltv_X"]
    y = st.session_state["ltv_y"]

    # ── Model metrics ─────────────────────────────────────────────────────────
    st.subheader("Model Performance")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("R²", f"{model.r_squared():.4f}")
    mc2.metric("Adjusted R²", f"{model.adjusted_r_squared():.4f}")
    mc3.metric("F-Statistic", f"{f_res['f_statistic']:.2f}")
    mc4.metric("F p-value", f"{f_res['p_value']:.4e}")

    # ── Coefficient table ─────────────────────────────────────────────────────
    st.subheader("Regression Coefficients")

    names = ["Intercept"] + feature_names
    coefs = model.coefficients_
    ses = t_res["std_errors"]
    tsts = t_res["t_statistics"]
    pvs = t_res["p_values"]

    rows = []
    for name, coef, se, t, p in zip(names, coefs, ses, tsts, pvs):
        rows.append({
            "Feature": name,
            "Coefficient": round(float(coef), 4),
            "Std Error": round(float(se), 4),
            "t-Statistic": round(float(t), 4),
            "p-Value": round(float(p), 4),
            "Significant": "✅" if p < 0.05 else "❌",
        })

    df_coefs = pd.DataFrame(rows)
    st.dataframe(
        df_coefs,
        use_container_width=True,
        column_config={
            "p-Value": st.column_config.NumberColumn(format="%.4f"),
            "Significant": st.column_config.TextColumn(),
        },
    )

    with st.expander("📐 Mathematical Background"):
        st.markdown(
            r"""
            **Normal Equation**

            $$\hat{\boldsymbol{\beta}} = (X^\top X)^{-1} X^\top \mathbf{y}$$

            **Coefficient t-test** (H₀: βⱼ = 0):

            $$t_j = \frac{\hat{\beta}_j}{\text{SE}(\hat{\beta}_j)}, \quad
              \text{SE}(\hat{\beta}_j) = \sqrt{\text{MSE} \cdot [(X^\top X)^{-1}]_{jj}}$$

            **Overall F-test** (H₀: all βⱼ = 0 for j ≥ 1):

            $$F = \frac{SS_\text{reg}/p}{SS_\text{res}/(n-p-1)}$$
            """
        )

    # ── Diagnostic plots ──────────────────────────────────────────────────────
    st.subheader("Diagnostic Plots")

    diag_col1, diag_col2, diag_col3 = st.columns(3)

    y_pred = model.X_with_bias_ @ model.coefficients_
    residuals = model.residuals()

    # Residuals vs Fitted
    with diag_col1:
        fig_rv, ax_rv = plt.subplots(figsize=(5, 4))
        ax_rv.scatter(y_pred, residuals, alpha=0.3, s=10, color="steelblue")
        ax_rv.axhline(0, color="red", lw=1.5, linestyle="--")
        ax_rv.set_xlabel("Fitted Values")
        ax_rv.set_ylabel("Residuals")
        ax_rv.set_title("Residuals vs Fitted")
        ax_rv.grid(True, alpha=0.3)
        plt.tight_layout()
        diag_col1.pyplot(fig_rv)
        plt.close(fig_rv)

    # Q-Q plot
    with diag_col2:
        sorted_res = np.sort(residuals)
        n = len(sorted_res)
        # Theoretical normal quantiles via inverse-CDF approximation (no random sampling needed)
        probs = (np.arange(1, n + 1) - 0.5) / n
        # Rational approximation to the standard-normal inverse CDF (Abramowitz & Stegun)
        def _norm_ppf(p):
            p = np.clip(p, 1e-10, 1 - 1e-10)
            sign = np.where(p < 0.5, -1.0, 1.0)
            t = np.sqrt(-2 * np.log(np.minimum(p, 1 - p)))
            c0, c1, c2 = 2.515517, 0.802853, 0.010328
            d1, d2, d3 = 1.432788, 0.189269, 0.001308
            num = c0 + c1 * t + c2 * t ** 2
            den = 1 + d1 * t + d2 * t ** 2 + d3 * t ** 3
            return sign * (t - num / den)
        theoretical_q = _norm_ppf(probs)
        fig_qq, ax_qq = plt.subplots(figsize=(5, 4))
        ax_qq.scatter(theoretical_q, sorted_res, alpha=0.3, s=10, color="coral")
        lo = min(theoretical_q.min(), sorted_res.min())
        hi = max(theoretical_q.max(), sorted_res.max())
        ax_qq.plot([lo, hi], [lo, hi], "r--", lw=1.5)
        ax_qq.set_xlabel("Theoretical Quantiles")
        ax_qq.set_ylabel("Sample Quantiles")
        ax_qq.set_title("Q-Q Plot of Residuals")
        ax_qq.grid(True, alpha=0.3)
        plt.tight_layout()
        diag_col2.pyplot(fig_qq)
        plt.close(fig_qq)

    # Feature importance by |t-stat|
    with diag_col3:
        fi_names = feature_names
        fi_t = np.abs(t_res["t_statistics"][1:])  # skip intercept
        sorted_idx = np.argsort(fi_t)
        fig_fi, ax_fi = plt.subplots(figsize=(5, 4))
        colors = ["seagreen" if t_res["p_values"][i + 1] < 0.05 else "salmon"
                  for i in sorted_idx]
        ax_fi.barh([fi_names[i] for i in sorted_idx],
                   fi_t[sorted_idx], color=colors, alpha=0.85)
        ax_fi.set_xlabel("|t-Statistic|")
        ax_fi.set_title("Feature Importance\n(green = significant at α=0.05)")
        ax_fi.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        diag_col3.pyplot(fig_fi)
        plt.close(fig_fi)

    st.divider()

    # ── Predict LTV for a new user ────────────────────────────────────────────
    st.subheader("🔮 Predict LTV for a New User")

    region_map = {"North America": (1, 0), "Europe": (0, 1), "Asia / Other": (0, 0)}
    referral_map = {"Paid": (1, 0), "Social": (0, 1), "Organic / Other": (0, 0)}
    device_map = {"Mobile": (1, 0), "Tablet": (0, 1), "Desktop": (0, 0)}

    pred_col1, pred_col2, pred_col3 = st.columns(3)
    with pred_col1:
        region = pred_col1.selectbox("Geographic region", list(region_map.keys()))
        referral = pred_col1.selectbox("Referral source", list(referral_map.keys()))
        device = pred_col1.selectbox("Device type", list(device_map.keys()))
    with pred_col2:
        avg_session = pred_col2.slider("Avg session duration (min)", 1.0, 60.0, 8.0, 0.5)
        pages = pred_col2.slider("Pages per visit", 1, 20, 5)
    with pred_col3:
        days = pred_col3.slider("Days since signup", 1, 365, 90)

    predict_btn = st.button("🔮 Predict LTV", type="primary")

    if predict_btn:
        region_na, region_eu = region_map[region]
        ref_paid, ref_social = referral_map[referral]
        dev_mobile, dev_tablet = device_map[device]

        user_X = np.array([[
            region_na, region_eu,
            ref_paid, ref_social,
            dev_mobile, dev_tablet,
            avg_session, float(pages), float(days),
        ]])

        predicted_ltv = float(model.predict(user_X)[0])

        # Approximate prediction interval using MSE
        residuals_all = model.residuals()
        mse = float(np.sum(residuals_all ** 2) / (model.n_ - model.p_ - 1))
        rmse = np.sqrt(mse)

        st.metric("Predicted Lifetime Value", f"${predicted_ltv:,.2f}")
        st.info(
            f"ℹ️ Approximate 95% prediction interval: "
            f"**${max(0, predicted_ltv - 2*rmse):,.2f}** — "
            f"**${predicted_ltv + 2*rmse:,.2f}**"
        )

else:
    st.info("👈 Set the number of users in the sidebar and click **Train Model** to begin.")
