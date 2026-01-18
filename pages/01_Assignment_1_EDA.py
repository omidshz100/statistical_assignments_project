"""
Statistical Learning and Data Analysis – Assignment 1
Exploratory Data Analysis of student_habits_performance.csv
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "student_habits_performance.csv")
sns.set_style("whitegrid")


@st.cache_data
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at {path}")
    df = pd.read_csv(path)
    if "diet_quality" in df.columns:
        df["diet_quality"] = df["diet_quality"].replace("", np.nan)
    return df


@st.cache_data
def preprocess_data(df: pd.DataFrame):
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df_clean.columns if c not in numeric_cols]

    for col in cat_cols:
        df_clean[col] = df_clean[col].replace("", np.nan)

    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    for col in cat_cols:
        mode_val = df_clean[col].mode(dropna=True)
        fill_val = mode_val.iloc[0] if not mode_val.empty else "Missing"
        df_clean[col] = df_clean[col].fillna(fill_val)

    return df_clean, numeric_cols, cat_cols


@st.cache_data
def prepare_ml_matrix(df_clean: pd.DataFrame, cat_cols, numeric_cols):
    df_ml = pd.get_dummies(df_clean, columns=cat_cols, drop_first=False)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_ml)
    return X_scaled, df_ml.columns.tolist()


@st.cache_data
def run_pca(X_scaled: np.ndarray):
    pca = PCA()
    pcs = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_
    return pca, pcs, explained, explained.cumsum()


@st.cache_data
def kmeans_elbow(X_scaled: np.ndarray, k_range):
    inertia = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)
    return inertia


def missing_table(df: pd.DataFrame):
    miss = df.isna().sum()
    pct = 100 * miss / len(df)
    return pd.DataFrame({"missing_count": miss, "missing_pct": pct.round(2)})


def plot_numeric_distribution(df: pd.DataFrame, col: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(df[col], kde=True, ax=axes[0], color="#1f77b4")
    axes[0].set_title(f"Histogram: {col}")
    axes[0].set_xlabel(col)

    sns.boxplot(x=df[col], ax=axes[1], color="#ff7f0e")
    axes[1].set_title(f"Boxplot: {col}")
    axes[1].set_xlabel(col)

    plt.tight_layout()
    return fig


def plot_categorical_distribution(df: pd.DataFrame, col: str):
    counts = df[col].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, palette="viridis")
    ax.set_title(f"Category Counts: {col}")
    ax.set_xlabel("Frequency")
    ax.set_ylabel(col)
    plt.tight_layout()
    top_cat = counts.idxmax()
    return fig, top_cat


def scatter_with_trend(df: pd.DataFrame, x: str, y: str):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.regplot(data=df, x=x, y=y, ax=ax, scatter_kws={"alpha": 0.35})
    ax.set_title(f"{x} vs {y}")
    plt.tight_layout()
    return fig


def boxplot_by_category(df: pd.DataFrame, cat: str, target: str):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.boxplot(data=df, x=cat, y=target, ax=ax)
    ax.set_title(f"{target} by {cat}")
    plt.tight_layout()
    return fig


def pca_biplot(pcs, pca_model, feature_names, top_n=8):
    comp1, comp2 = pcs[:, 0], pcs[:, 1]
    loadings = pca_model.components_.T[:, :2]
    loading_strength = np.linalg.norm(loadings, axis=1)
    top_idx = np.argsort(loading_strength)[-top_n:]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(comp1, comp2, alpha=0.35, label="Observations")
    for i in top_idx:
        ax.arrow(0, 0, loadings[i, 0] * 5, loadings[i, 1] * 5,
                 color="red", alpha=0.6, head_width=0.1)
        ax.text(loadings[i, 0] * 5.5, loadings[i, 1] * 5.5,
                feature_names[i], color="red", fontsize=9)

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Biplot (top loadings shown)")
    ax.legend()
    plt.tight_layout()
    return fig


def cluster_scatter(pcs, labels):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1], hue=labels,
                    palette="tab10", ax=ax, alpha=0.6)
    ax.set_title("Clusters in PCA Space (PC1 vs PC2)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title="Cluster")
    plt.tight_layout()
    return fig


st.title("📈 EDA: Student Habits and Performance")
st.markdown("Master's level Assignment 1 — interactive exploration of habits vs exam performance.")

try:
    raw_df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error("CSV not found. Please place student_habits_performance.csv next to this script.")
    st.stop()

# Load and clean
clean_df, num_cols, cat_cols = preprocess_data(raw_df)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data Loading & Cleaning",
    "Univariate Analysis",
    "Bivariate & Multivariate",
    "PCA & Clustering",
    "Summary & Interpretation",
])

with tab1:
    st.subheader("Dataset snapshot and cleaning steps")
    st.write(f"Shape: {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")
    st.dataframe(raw_df.head(), use_container_width=True)
    st.markdown("**Data types**")
    st.dataframe(pd.DataFrame(raw_df.dtypes, columns=["dtype"]))

    st.markdown("**Missing values (before imputation)**")
    st.dataframe(missing_table(raw_df), use_container_width=True)

    st.markdown("Imputation applied: numeric → median; categorical → mode (empty diet_quality treated as missing).")
    st.markdown("**Missing values (after imputation)**")
    st.dataframe(missing_table(clean_df), use_container_width=True)

with tab2:
    st.subheader("Univariate Analysis")
    st.markdown("Numeric variables — descriptive statistics")
    st.dataframe(clean_df[num_cols].describe().T, use_container_width=True)

    st.markdown("Key numeric distributions")
    num_choice = st.selectbox("Pick a numeric variable to inspect", num_cols, index=num_cols.index("exam_score") if "exam_score" in num_cols else 0)
    st.pyplot(plot_numeric_distribution(clean_df, num_choice))

    st.caption("Skewness and outliers can be spotted from the histogram (tail) and boxplot (points beyond whiskers).")

    st.markdown("Categorical variables — frequency and mode highlight")
    cat_choice = st.selectbox("Pick a categorical variable", cat_cols)
    fig_cat, top_cat = plot_categorical_distribution(clean_df, cat_choice)
    st.pyplot(fig_cat)
    st.info(f"Most frequent category for {cat_choice}: {top_cat}")

with tab3:
    st.subheader("Bivariate & Multivariate Analysis")
    st.markdown("Correlation matrix (numeric)")
    corr = clean_df[num_cols].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
    ax_corr.set_title("Correlation Heatmap")
    plt.tight_layout()
    st.pyplot(fig_corr)

    st.markdown("Scatterplots with trend lines vs exam_score")
    pairs = [
        ("study_hours_per_day", "exam_score"),
        ("sleep_hours", "exam_score"),
        ("social_media_hours", "exam_score"),
        ("attendance_percentage", "exam_score"),
    ]
    cols = st.columns(2)
    for i, (x, y) in enumerate(pairs):
        with cols[i % 2]:
            st.pyplot(scatter_with_trend(clean_df, x, y))

    st.markdown("Exam score across categories (boxplots)")
    box_cols = ["gender", "part_time_job", "parental_education_level"]
    cols_box = st.columns(3)
    for i, cat in enumerate(box_cols):
        with cols_box[i]:
            st.pyplot(boxplot_by_category(clean_df, cat, "exam_score"))

    st.markdown("Optional: contingency table (gender × part_time_job)")
    st.dataframe(pd.crosstab(clean_df["gender"], clean_df["part_time_job"]))

with tab4:
    st.subheader("PCA & Clustering")
    X_scaled, feature_names = prepare_ml_matrix(clean_df, cat_cols, num_cols)
    pca_model, pcs, explained, cum_expl = run_pca(X_scaled)

    st.markdown("Explained variance per principal component")
    expl_df = pd.DataFrame({
        "PC": np.arange(1, len(explained) + 1),
        "Explained_Variance_Ratio": explained,
        "Cumulative": cum_expl,
    })
    st.dataframe(expl_df.head(10), use_container_width=True)
    pcs_needed = int(np.argmax(cum_expl >= 0.80) + 1)
    st.success(f"Components needed for >= 80% variance: {pcs_needed}")

    fig_ev, ax_ev = plt.subplots(figsize=(7, 4))
    ax_ev.plot(expl_df["PC"], expl_df["Explained_Variance_Ratio"], marker="o")
    ax_ev.plot(expl_df["PC"], expl_df["Cumulative"], marker="s", label="Cumulative")
    ax_ev.axhline(0.80, color="red", linestyle="--", label="80% threshold")
    ax_ev.set_xlabel("Principal Component")
    ax_ev.set_ylabel("Variance Ratio")
    ax_ev.set_title("Explained Variance")
    ax_ev.legend()
    plt.tight_layout()
    st.pyplot(fig_ev)

    st.markdown("PCA biplot (PC1 vs PC2 with top loadings)")
    st.pyplot(pca_biplot(pcs, pca_model, feature_names))

    st.markdown("K-Means elbow method (suggested K)")
    k_range = list(range(2, 9))
    inertia = kmeans_elbow(X_scaled, k_range)
    fig_elbow, ax_elbow = plt.subplots(figsize=(6, 4))
    ax_elbow.plot(k_range, inertia, marker="o")
    ax_elbow.set_xlabel("Number of clusters (K)")
    ax_elbow.set_ylabel("Inertia")
    ax_elbow.set_title("Elbow Plot")
    plt.tight_layout()
    st.pyplot(fig_elbow)

    st.markdown("Fit K=3 clusters and visualize in PCA space")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    st.pyplot(cluster_scatter(pcs, labels))
    st.caption("Clusters may reflect: high-performing, overwhelmed, and balanced students.")

with tab5:
    st.subheader("Summary & Interpretation")
    corr_exam = corr["exam_score"].drop("exam_score")
    top_pos = corr_exam.sort_values(ascending=False).head(3)
    top_neg = corr_exam.sort_values().head(3)

    bullets = [
        f"Exam score rises with study_hours_per_day; strongest positive link: {top_pos.index[0]} (r={top_pos.iloc[0]:.2f}).",
        f"Lower exam scores align with higher {top_neg.index[0]} (r={top_neg.iloc[0]:.2f}).",
        "Attendance and consistent sleep generally support better performance.",
        "Category effects: gender, part_time_job, and parental_education_level show visible differences in exam_score distributions.",
        f"PCA shows structure; about {pcs_needed} components cover 80% variance. K-Means (K=3) yields interpretable clusters.",
    ]
    for b in bullets:
        st.markdown(f"- {b}")

st.markdown("---")
st.markdown("Built with Streamlit, pandas, numpy, seaborn, matplotlib, and scikit-learn.")