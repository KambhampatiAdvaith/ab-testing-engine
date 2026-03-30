# Advanced A/B Testing & User Segmentation Engine

![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)

A complete statistical engine for A/B testing, Bayesian inference, regression analysis, and user segmentation — built from scratch using only NumPy and Matplotlib.

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Mathematical Details](#mathematical-details)
- [Technologies](#technologies)
- [License](#license)

## Description

This engine provides a full pipeline for data-driven experimentation and user segmentation:

- **Distributions**: Binomial, Normal, Uniform — with custom PMF/PDF/CDF and sampling
- **A/B Testing**: Two-proportion z-test, confidence intervals, sample size calculation
- **Bayesian Estimation**: Beta-Binomial conjugate model, P(B>A), expected loss
- **Regression**: Multiple linear regression via Normal Equation, t-tests, F-test
- **Clustering**: K-Means++ with silhouette score and elbow method for user personas

## Installation

```bash
pip install -r requirements.txt
```

## 🌐 Web Dashboard

An interactive Streamlit dashboard is included on top of the backend engine.

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501` with four interactive pages:

| Page | Description |
|------|-------------|
| 📊 CLT Visualizer | Central Limit Theorem demo & traffic simulator |
| 🧪 A/B Test Lab | Frequentist Z-test, Bayesian estimation, live posterior animation |
| 📈 LTV Predictor | Regression training, diagnostics, and LTV prediction form |
| 🎯 User Personas | K-Means clustering, elbow method, persona cards & radar charts |

## Usage

```bash
python main.py --unit 1   # CLT & distributions demo
python main.py --unit 3   # A/B testing demo
python main.py --unit 4   # Regression & LTV demo
python main.py --unit 5   # Clustering & personas demo
python main.py --all      # Run all demos
```

Outputs (plots) are saved to the `outputs/` directory.

## Project Structure

```
ab-testing-engine/
├── requirements.txt
├── setup.py
├── main.py
├── src/
│   ├── unit1_2_random_variables/
│   │   ├── distributions.py       # Binomial, Normal, Uniform distributions
│   │   ├── traffic_simulator.py   # Website traffic simulation
│   │   └── clt_visualizer.py      # Central Limit Theorem demonstration
│   ├── unit3_hypothesis_testing/
│   │   ├── ab_test_engine.py      # Two-proportion z-test, sample size
│   │   └── bayesian_estimation.py # Beta-Binomial conjugate model
│   ├── unit4_regression/
│   │   ├── linear_regression.py   # Multiple linear regression (Normal Equation)
│   │   ├── hypothesis_tests.py    # t-tests and F-test for regression
│   │   └── lifetime_value.py      # LTV dataset generation & feature importance
│   ├── unit5_clustering/
│   │   ├── kmeans.py              # K-Means++ clustering
│   │   └── user_personas.py       # User persona discovery & visualization
│   └── utils/
│       ├── data_generator.py      # Synthetic data generation
│       └── visualization.py       # Plotting utilities
├── tests/
└── outputs/
```

## Mathematical Details

### Unit 1-2: Distributions & CLT

**Binomial PMF** (log-space for numerical stability):
`log P(X=k) = log C(n,k) + k·log(p) + (n-k)·log(1-p)`

**Normal PDF/CDF** using Abramowitz & Stegun erf approximation (max error 1.5e-7):
`CDF(x) = 0.5 · (1 + erf((x - μ) / (σ√2)))`

**Box-Muller Transform**: `Z = √(-2·ln U₁) · cos(2π U₂)`

**CLT**: For iid X₁,...,Xₙ: `X̄ₙ → N(μ, σ²/n)` as n → ∞

### Unit 3: Hypothesis Testing & Bayesian A/B Testing

**Two-proportion z-test**: `z = (p̂₂ - p̂₁) / SE_pooled`, `p-value = 2·(1 - Φ(|z|))`

**Beta-Binomial Conjugate**: `θ | data ~ Beta(α + successes, β + failures)`

**P(B > A)** estimated via Monte Carlo sampling from posteriors.

### Unit 4: Multiple Linear Regression

**Normal Equation**: `β̂ = (XᵀX)⁻¹ Xᵀy`

**R²**: `1 - SS_res/SS_tot`; **Adjusted R²**: `1 - (1-R²)·(n-1)/(n-p-1)`

**t-test**: `t_j = β̂_j / SE(β̂_j)` where `SE = √(MSE · [(XᵀX)⁻¹]_{jj})`

**F-test**: `F = (SS_reg/p) / (SS_res/(n-p-1))`

### Unit 5: K-Means Clustering

**K-Means++ Init**: select centroids with probability ∝ d(x, nearest centroid)²

**WCSS**: `Σ_k Σ_{x∈Cₖ} ||x - μₖ||²`

**Silhouette**: `s(i) = (b(i) - a(i)) / max(a(i), b(i))`

## Technologies

- **NumPy** ≥ 1.21.0 — all numerical computations
- **Matplotlib** ≥ 3.4.0 — visualizations
- **pytest** ≥ 7.0.0 — testing

## License

MIT License — free to use, modify, and distribute.