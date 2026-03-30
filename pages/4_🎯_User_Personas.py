import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.unit5_clustering.kmeans import KMeans
from src.unit5_clustering.user_personas import (
    generate_user_behavior_data,
    discover_personas,
    analyze_personas,
    _normalize,
    _pca_2d,
)

st.set_page_config(page_title="User Personas", page_icon="🎯", layout="wide")

st.title("🎯 User Personas")
st.caption("Unit V — K-Means Clustering & User Persona Discovery")

FEATURE_NAMES = [
    "pages/session",
    "avg_session_dur",
    "bounce_rate",
    "purchase_freq",
    "support_tickets",
    "login_freq",
    "feature_depth",
]

PERSONA_EMOJI = {
    "Power User": "⚡",
    "Window Shopper": "🛍️",
    "Bot": "🤖",
    "Casual User": "😊",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Clustering Controls")
n_users = st.sidebar.slider(
    "Synthetic users", min_value=200, max_value=5_000, value=800, step=100
)
k = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=8, value=4)
discover_btn = st.sidebar.button("🔍 Discover Personas", type="primary", use_container_width=True)

# ── Run clustering ────────────────────────────────────────────────────────────
if discover_btn or "personas_data" in st.session_state:
    if discover_btn:
        with st.spinner("Generating data and clustering users…"):
            data = generate_user_behavior_data(n_users=int(n_users))
            labels = discover_personas(data, k=int(k))
            personas = analyze_personas(data, labels)
        st.session_state["personas_data"] = data
        st.session_state["personas_labels"] = labels
        st.session_state["personas_info"] = personas
        st.session_state["personas_k"] = int(k)

    data = st.session_state["personas_data"]
    labels = st.session_state["personas_labels"]
    personas = st.session_state["personas_info"]

    unique_labels = np.unique(labels)

    # ── Elbow method ──────────────────────────────────────────────────────────
    st.subheader("📐 Elbow Method")
    with st.spinner("Computing inertias for k = 2…8 (this may take a moment)…"):
        k_range = range(2, 9)
        normalized_data = _normalize(data)

        @st.cache_data(show_spinner=False)
        def _compute_elbow(data_bytes: bytes, _k_range):
            arr = np.frombuffer(data_bytes, dtype=np.float64).reshape(-1, len(FEATURE_NAMES))
            inertias = []
            silhouettes = []
            for kk in _k_range:
                km = KMeans(k=kk, max_iterations=100)
                km.fit(arr)
                inertias.append(km.inertia())
                sil = km.silhouette_score(arr) if len(arr) <= 500 else float("nan")
                silhouettes.append(sil)
            return inertias, silhouettes

        inertias, silhouettes = _compute_elbow(normalized_data.tobytes(), tuple(k_range))

    el_col1, el_col2 = st.columns(2)

    with el_col1:
        fig_elbow, ax_elbow = plt.subplots(figsize=(5, 3))
        ax_elbow.plot(list(k_range), inertias, "bo-", markersize=7)
        ax_elbow.axvline(x=st.session_state["personas_k"], color="red",
                         linestyle="--", label=f"Selected k={st.session_state['personas_k']}")
        ax_elbow.set_xlabel("k")
        ax_elbow.set_ylabel("Inertia (WCSS)")
        ax_elbow.set_title("Elbow Curve")
        ax_elbow.legend()
        ax_elbow.grid(True, alpha=0.3)
        plt.tight_layout()
        el_col1.pyplot(fig_elbow)
        plt.close(fig_elbow)

    with el_col2:
        if not all(np.isnan(s) for s in silhouettes):
            fig_sil, ax_sil = plt.subplots(figsize=(5, 3))
            valid = [(kk, s) for kk, s in zip(k_range, silhouettes) if not np.isnan(s)]
            ks, ss = zip(*valid)
            ax_sil.plot(ks, ss, "go-", markersize=7)
            ax_sil.set_xlabel("k")
            ax_sil.set_ylabel("Silhouette Score")
            ax_sil.set_title("Silhouette Scores")
            ax_sil.grid(True, alpha=0.3)
            plt.tight_layout()
            el_col2.pyplot(fig_sil)
            plt.close(fig_sil)
        else:
            el_col2.info(
                "Silhouette scores are skipped for large datasets (n > 500) "
                "to keep computation fast. Reduce the user count to enable them."
            )

    # ── PCA scatter ───────────────────────────────────────────────────────────
    st.subheader("🔵 PCA Cluster Visualization")
    projected = _pca_2d(data)
    colors_map = plt.cm.Set1(np.linspace(0, 0.8, len(unique_labels)))

    fig_pca, ax_pca = plt.subplots(figsize=(8, 5))
    for cid, color in zip(unique_labels, colors_map):
        mask = labels == cid
        lbl = f"Cluster {cid}: {personas[cid]['label']} ({personas[cid]['size']})"
        ax_pca.scatter(projected[mask, 0], projected[mask, 1],
                       c=[color], label=lbl, alpha=0.55, s=25)
    ax_pca.set_xlabel("PC1")
    ax_pca.set_ylabel("PC2")
    ax_pca.set_title("User Clusters (PCA Projection)")
    ax_pca.legend(fontsize=9)
    ax_pca.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_pca)
    plt.close(fig_pca)

    # ── Persona summary table ─────────────────────────────────────────────────
    st.subheader("📊 Persona Summary Table")

    # Use the actual feature keys from analyze_personas (snake_case)
    MEANS_KEYS = [
        "pages_per_session", "avg_session_duration", "bounce_rate",
        "purchase_frequency", "support_tickets", "login_frequency", "feature_usage_depth",
    ]

    rows = []
    for cid in unique_labels:
        p = personas[cid]
        row = {"Cluster": cid, "Persona": p["label"],
               "Size": p["size"],
               "% of Users": f"{p['size'] / len(labels) * 100:.1f}%"}
        for key in MEANS_KEYS:
            row[key] = round(p["means"].get(key, 0.0), 2)
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ── Radar chart ───────────────────────────────────────────────────────────
    st.subheader("🕸️ Feature Profiles (Radar Chart)")

    all_means = np.array([list(personas[cid]["means"].values()) for cid in unique_labels])
    col_maxs = all_means.max(axis=0)
    col_maxs[col_maxs == 0] = 1.0

    n_feat = len(FEATURE_NAMES)
    angles = np.linspace(0, 2 * np.pi, n_feat, endpoint=False).tolist()
    angles += angles[:1]

    fig_radar, ax_radar = plt.subplots(figsize=(7, 6), subplot_kw={"polar": True})
    for cid, color in zip(unique_labels, colors_map):
        means = np.array(list(personas[cid]["means"].values()))
        norm_means = means / col_maxs
        values = norm_means.tolist() + [norm_means[0]]
        ax_radar.plot(angles, values, "o-", color=color, lw=2,
                      label=f"{PERSONA_EMOJI.get(personas[cid]['label'], '👤')} "
                            f"Cluster {cid}: {personas[cid]['label']}")
        ax_radar.fill(angles, values, color=color, alpha=0.1)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(FEATURE_NAMES, fontsize=9)
    ax_radar.set_title("Normalized Feature Profiles by Persona", pad=20)
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_radar)
    plt.close(fig_radar)

    # ── Bar chart: cluster means ──────────────────────────────────────────────
    st.subheader("📊 Cluster Means per Feature")
    fig_bar, axes_bar = plt.subplots(1, len(FEATURE_NAMES), figsize=(18, 4))
    for ax_b, fname in zip(axes_bar, FEATURE_NAMES):
        feature_idx = FEATURE_NAMES.index(fname)
        vals = [list(personas[cid]["means"].values())[feature_idx] for cid in unique_labels]
        cluster_labels_bar = [f"C{cid}" for cid in unique_labels]
        ax_b.bar(cluster_labels_bar, vals,
                 color=[colors_map[i] for i in range(len(unique_labels))],
                 alpha=0.85)
        ax_b.set_title(fname, fontsize=8)
        ax_b.tick_params(axis="x", labelsize=7)
    plt.suptitle("Cluster Mean Values per Feature", fontsize=11, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig_bar)
    plt.close(fig_bar)

    # ── Persona cards ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🪪 Persona Cards")

    card_cols = st.columns(len(unique_labels))
    for col, cid in zip(card_cols, unique_labels):
        p = personas[cid]
        emoji = PERSONA_EMOJI.get(p["label"], "👤")
        pct = p["size"] / len(labels) * 100
        with col:
            with st.container(border=True):
                st.markdown(f"## {emoji} {p['label']}")
                st.markdown(f"**Cluster {cid}** · {p['size']} users ({pct:.1f}%)")
                st.divider()
                means = p["means"]
                key_map = {
                    "pages_per_session": ("📄 Pages/session", ".1f"),
                    "avg_session_duration": ("⏱ Avg session (min)", ".1f"),
                    "bounce_rate": ("↩️ Bounce rate", ".2f"),
                    "purchase_frequency": ("🛒 Purchases/month", ".1f"),
                    "login_frequency": ("🔑 Logins/month", ".1f"),
                    "feature_usage_depth": ("🔧 Feature depth", ".2f"),
                }
                for key, (label, fmt) in key_map.items():
                    val = means.get(key, 0)
                    st.markdown(f"**{label}:** {val:{fmt}}")

    with st.expander("📐 Algorithm Details"):
        st.markdown(
            r"""
            **K-Means++ Initialization**

            1. Pick the first centroid uniformly at random.
            2. For each subsequent centroid, sample a point with probability
               proportional to its squared distance from the nearest existing centroid.

            **E-M Steps** (repeated until convergence):

            - **E-step:** Assign $x_i$ to cluster $k^* = \arg\min_k \|x_i - \mu_k\|^2$
            - **M-step:** Update $\mu_k = \frac{1}{|C_k|}\sum_{x \in C_k} x$

            **Silhouette Score** for point $i$:

            $$s(i) = \frac{b(i) - a(i)}{\max(a(i),\, b(i))}$$

            where $a(i)$ is the mean intra-cluster distance and $b(i)$ is the mean
            distance to the nearest other cluster.
            """
        )

else:
    st.info("👈 Set the parameters in the sidebar and click **Discover Personas** to begin.")
