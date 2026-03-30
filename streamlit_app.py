import streamlit as st

st.set_page_config(
    page_title="A/B Testing & User Segmentation Engine",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("🔬 Navigation")
st.sidebar.markdown(
    """
    Use the **pages** listed below to explore each module:

    - 📊 **CLT Visualizer** — Central Limit Theorem demo & traffic simulator
    - 🧪 **A/B Test Lab** — Frequentist Z-test & Bayesian estimation
    - 📈 **LTV Predictor** — Multiple linear regression for lifetime value
    - 🎯 **User Personas** — K-Means clustering & persona discovery
    """
)
st.sidebar.divider()
st.sidebar.markdown(
    """
    **About**

    All statistical algorithms are implemented from scratch using only
    **NumPy** — no sklearn, no scipy, no statsmodels for core logic.
    """
)

# ── Hero section ─────────────────────────────────────────────────────────────
st.title("🔬 A/B Testing & User Segmentation Engine")
st.subheader("Advanced statistical engine built from scratch — no sklearn, no scipy")

st.divider()

# ── Project stats ─────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Statistical Modules", "5")
col2.metric("Tests", "32+")
col3.metric("ML Library Dependencies", "0")
col4.metric("Lines of Core Math", "1,000+")

st.divider()

# ── Unit cards ────────────────────────────────────────────────────────────────
st.header("📚 Modules")

card_col1, card_col2 = st.columns(2)

with card_col1:
    with st.container(border=True):
        st.markdown("### 📊 Unit I & II — CLT Visualizer")
        st.markdown(
            """
            **Distributions from scratch:** Binomial (log-factorial PMF, inverse-transform
            sampling), Normal (Box-Muller), Uniform.

            **Traffic simulator:** Bernoulli clicks, Normal/Uniform session times, full
            synthetic dataset with timestamps.

            **CLT demo:** Watch sample-mean distributions converge to Normal as *n* grows,
            with theoretical overlay.
            """
        )
        st.page_link("pages/1_📊_CLT_Visualizer.py", label="Open CLT Visualizer →", icon="📊")

    with st.container(border=True):
        st.markdown("### 📈 Unit IV — LTV Predictor")
        st.markdown(
            """
            **Multiple Linear Regression** via the Normal Equation
            *β = (XᵀX)⁻¹Xᵀy*.

            t-tests for each coefficient, F-test for overall model significance,
            R² and Adjusted R², residual diagnostics, feature importance ranking.

            Predict **customer lifetime value** from region, device, referral source,
            and behavioural signals.
            """
        )
        st.page_link("pages/3_📈_LTV_Predictor.py", label="Open LTV Predictor →", icon="📈")

with card_col2:
    with st.container(border=True):
        st.markdown("### 🧪 Unit III — A/B Test Lab")
        st.markdown(
            """
            **Frequentist:** Two-proportion z-test (pooled SE, z-score, p-value, CI)
            and sample-size calculator.

            **Bayesian:** Beta-Binomial conjugate model — Monte Carlo P(B > A),
            expected loss, live posterior animation.

            Interactive inputs for conversion data, significance level, and power.
            """
        )
        st.page_link("pages/2_🧪_AB_Test_Lab.py", label="Open A/B Test Lab →", icon="🧪")

    with st.container(border=True):
        st.markdown("### 🎯 Unit V — User Personas")
        st.markdown(
            """
            **K-Means++** clustering (custom centroid init, WCSS, silhouette score,
            elbow method) — all from scratch.

            PCA via eigendecomposition for 2D visualization.

            Auto-label clusters as *Power User*, *Window Shopper*, *Bot*, or
            *Casual User* and display styled persona cards.
            """
        )
        st.page_link("pages/4_🎯_User_Personas.py", label="Open User Personas →", icon="🎯")

st.divider()

# ── Quick-start ───────────────────────────────────────────────────────────────
st.header("⚡ Quick Start")
st.code(
    """\
# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run streamlit_app.py

# Or run the CLI demos
python main.py --unit 3   # A/B testing demo
python main.py --all      # Run all units
""",
    language="bash",
)

st.info(
    "👈 Select a page from the sidebar (or click a card above) to get started.",
    icon="ℹ️",
)
