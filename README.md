# A/B Testing & User Segmentation Engine

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)
![React](https://img.shields.io/badge/React-19-61dafb)
![TypeScript](https://img.shields.io/badge/TypeScript-5.8-blue)

An enterprise-grade full-stack A/B Testing & User Segmentation platform. The backend wraps a
custom statistical math engine (NumPy/SciPy) in a production FastAPI microservice with
PostgreSQL persistence and real-time WebSocket streaming. The frontend is a React TypeScript
dashboard with Recharts visualisations and Zustand state management.

---

## Architecture

```
ab-testing-engine/
├── backend/
│   ├── main.py              ← FastAPI app (CORS, lifespan, all routers)
│   ├── config.py            ← pydantic-settings (DATABASE_URL, CORS, etc.)
│   ├── database.py          ← Async SQLAlchemy engine + session
│   ├── models.py            ← ORM models (UUID PKs, SQLAlchemy 2.0)
│   ├── cli.py               ← Legacy CLI entry point
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── api/
│   │   ├── routes_ab_testing.py   ← POST /test/frequentist|bayesian|sample-size
│   │   ├── routes_clustering.py   ← POST /clustering/personas|kmeans
│   │   ├── routes_clt.py          ← POST /clt/demonstrate
│   │   ├── routes_experiments.py  ← CRUD /experiments
│   │   └── routes_websocket.py    ← WS /ws/v1/experiment-stream/{id}
│   └── src/                 ← Math engine (Z-tests, Bayesian, K-Means, CLT)
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   ├── types/index.ts          ← TypeScript interfaces
│   │   ├── api/client.ts           ← Axios + React Query setup
│   │   ├── hooks/useWebSocket.ts   ← Reconnecting WebSocket hook
│   │   ├── store/useAppStore.ts    ← Zustand global store
│   │   └── components/
│   │       ├── layout/AppLayout.tsx
│   │       ├── ABTestPanel.tsx
│   │       ├── CLTVisualizer.tsx
│   │       ├── PersonaScatter.tsx
│   │       └── ExperimentList.tsx
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml
└── tests/
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend API | FastAPI, Uvicorn, Pydantic v2 |
| Database | PostgreSQL 16 (async via asyncpg) |
| ORM | SQLAlchemy 2.0 (async) |
| Math Engine | NumPy, SciPy (custom implementations) |
| Real-time | WebSockets |
| Frontend | React 19, TypeScript 5, Vite 8 |
| Styling | Tailwind CSS v4 |
| Charts | Recharts |
| State | Zustand |
| HTTP Client | Axios + TanStack Query |
| CI/CD | GitHub Actions |

---

## Running with Docker Compose

```bash
docker compose up --build
```

- API docs: http://localhost:8000/docs
- React dashboard: http://localhost:80

---

## Running Locally

### Backend

```bash
cd backend
pip install -r requirements.txt

# Start PostgreSQL, then:
export DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/ab_testing
uvicorn backend.main:app --reload
```

API available at `http://localhost:8000/docs`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Dashboard available at `http://localhost:5173`

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/test/frequentist` | Two-proportion Z-test |
| POST | `/api/v1/test/bayesian` | Bayesian Beta-Binomial test |
| POST | `/api/v1/test/sample-size` | Sample size calculator |
| POST | `/api/v1/clt/demonstrate` | CLT simulation |
| POST | `/api/v1/clustering/personas` | User persona discovery |
| POST | `/api/v1/clustering/kmeans` | K-Means clustering |
| GET/POST | `/api/v1/experiments` | Experiment CRUD |
| WS | `/ws/v1/experiment-stream/{id}` | Real-time traffic stream |

---

## Running Tests

```bash
pip install -r backend/requirements.txt
pytest tests/ -v
```

All 43 tests run against an in-memory SQLite database — no PostgreSQL required.


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