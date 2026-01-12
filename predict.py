# levels_difficulty_pipeline.py
# Run with: python levels_difficulty_pipeline.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os

CSV_PATH = "/Users/hoangnguyen/Documents/py/ArrowPuzzle/100lv_1612_2.csv"  # change if needed
OUT_CSV = "levels_difficulty_1612_2.csv"

from scipy.stats import pearsonr

def plot_feature_pearson(work_df, feature_cols, target_col, title_suffix=""):
    """
    Plot Pearson correlation between each feature and target column.
    """
    corrs = []
    pvals = []

    for c in feature_cols:
        x = work_df[c].values
        y = work_df[target_col].values

        # tránh case feature hằng số
        if np.std(x) < 1e-8:
            r, p = 0.0, 1.0
        else:
            r, p = pearsonr(x, y)

        corrs.append(r)
        pvals.append(p)

    corr_df = pd.DataFrame({
        "feature": feature_cols,
        "pearson_r": corrs,
        "p_value": pvals
    }).sort_values("pearson_r")

    # ===== Plot =====
    plt.figure(figsize=(8, max(4, 0.35 * len(feature_cols))))
    colors = ["red" if r < 0 else "steelblue" for r in corr_df["pearson_r"]]

    plt.barh(
        corr_df["feature"],
        corr_df["pearson_r"],
        color=colors
    )

    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Pearson correlation (r)")
    plt.title(f"Feature correlation vs {target_col} {title_suffix}")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    return corr_df


# Load
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found at '{CSV_PATH}'. Please place it in the working directory or change CSV_PATH.")

df = pd.read_csv(CSV_PATH)

def high_precision_pca(X_scaled):
    pca_full = PCA(n_components=None)
    pca_full.fit(X_scaled)

    plt.figure(figsize=(7,4))
    plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker="o")
    plt.axhline(0.95, color="red", linestyle="--")
    plt.title("Cumulative Explained Variance")
    plt.xlabel("Number of PCA components")
    plt.ylabel("Cumulative variance")
    plt.grid(True)
    plt.show()

    return pca_full

# Expected columns (adjust if your CSV differs)
expected_cols = [
    "Level_Name","total_arrows","total_edges","max_in_degree","edges_per_arrow","dependency_density","has_cycle_bool","critical_longest_nodes","critical_edges_on_chain","num_wayBlockers","num_blackHoles","sum_wayBlocker_lockTime","avg_wayBlocker_lockTime","avg_row_blockers","avg_col_blockers","crowded_rows","crowded_cols"
]
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns in CSV: {missing}")

# Preprocess
work = df.copy()
work["has_cycle_bool"] = work["has_cycle_bool"].map({"True":1,"False":0, True:1, False:0, "1":1, "0":0}).fillna(work["has_cycle_bool"])
work["has_cycle_bool"] = work["has_cycle_bool"].astype(int)

num_cols = [c for c in expected_cols if c != "Level_Name"]
work[num_cols] = work[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

X = work[num_cols].values
feature_names = num_cols.copy()

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=min(6, X_scaled.shape[1]), random_state=42)
# pca = high_precision_pca(X_scaled)
X_pca = pca.fit_transform(X_scaled)
pc1 = X_pca[:, 0]

# Heuristic proxy to ensure PC1 direction corresponds to "harder"
proxy_weights = {
    "total_arrows": 0.05,
    "total_edges": 0.15,
    "max_in_degree": 0.18,
    "edges_per_arrow": 0.12,
    "dependency_density": 0.18,
    "has_cycle_bool": 0.20,
    "critical_nodes": 0.20,
    "critical_edges": 0.18,
    "avg_row_blockers": 0.10,
    "avg_col_blockers": 0.10,
    "crowded_rows": 0.06,
    "crowded_cols": 0.06,
    "num_wayBlockers": 0.10,
    "sum_wayBlocker_lockTime": 0.10,
    "avg_wayBlocker_lockTime": 0.10,
}
raw = work[num_cols]
raw_min = raw.min()
raw_max = raw.max()
raw_norm = (raw - raw_min) / (raw_max - raw_min + 1e-12)
proxy_vals = np.zeros(len(raw_norm))
for fname, w in proxy_weights.items():
    if fname in raw_norm.columns:
        proxy_vals += w * raw_norm[fname].values

corr = np.corrcoef(pc1, proxy_vals)[0,1]
if corr < 0:
    pc1 = -pc1
    X_pca[:,0] = pc1
    corr = np.corrcoef(pc1, proxy_vals)[0,1]

# Scale PC1 to 0-10
pc1_min, pc1_max = pc1.min(), pc1.max()
difficulty_score =10 * (pc1 - pc1_min) / (pc1_max - pc1_min + 1e-12)

work["PC1_raw"] = pc1
work["DifficultyScore_0_10"] = difficulty_score

# KMeans on first 3 PCA components
n_pca_for_kmeans = min(3, X_pca.shape[1])
X_kmeans = X_pca[:, :n_pca_for_kmeans]
kmeans = KMeans(n_clusters=5, random_state=42, n_init=20)
klabels = kmeans.fit_predict(X_kmeans)
work["kmeans_cluster"] = klabels

# Map clusters to ordered labels by mean DifficultyScore
cluster_order = work.groupby("kmeans_cluster")["DifficultyScore_0_10"].mean().sort_values().index.tolist()
label_names = ["1","2","3","4","5"]
cluster_to_label = {cid: label_names[i] for i, cid in enumerate(cluster_order)}
work["DifficultyLabel"] = work["kmeans_cluster"].map(cluster_to_label)

# Isolation Forest for outliers
iso = IsolationForest(contamination=0.05, random_state=42)
iso_pred = iso.fit_predict(X_scaled)
work["is_outlier_if_1"] = (iso_pred == -1).astype(int)

# Save results
out_cols = ["Level_Name"] + num_cols + ["PC1_raw","DifficultyScore_0_10","kmeans_cluster","DifficultyLabel","is_outlier_if_1"]
result = work[out_cols].copy()
result.to_csv(OUT_CSV, index=False)
# ================================
# PEARSON CORRELATION DIAGNOSTICS
# ================================

# 1. Correlation vs DifficultyScore (0-10)
corr_diff = plot_feature_pearson(
    work_df=work,
    feature_cols=num_cols,
    target_col="DifficultyScore_0_10",
    title_suffix="(DifficultyScore 0–10)"
)

print("\nTop correlated features with DifficultyScore:")
print(corr_diff.sort_values("pearson_r", ascending=False).head(5))

# 2. Correlation vs PC1 (raw)
corr_pc1 = plot_feature_pearson(
    work_df=work,
    feature_cols=num_cols,
    target_col="PC1_raw",
    title_suffix="(PC1 raw)"
)

print("\nTop correlated features with PC1:")
print(corr_pc1.sort_values("pearson_r", ascending=False).head(5))

# # Diagnostics prints
# print("Saved difficulty results to:", OUT_CSV)
# print("PCA explained variance ratio (first components):", np.round(pca.explained_variance_ratio_[:n_pca_for_kmeans], 4))
# print("Correlation between PC1 and heuristic proxy (should be positive):", float(corr))
# print("\nCluster -> mean difficulty (0-10):")
# print(work.groupby("DifficultyLabel")["DifficultyScore_0_10"].mean().sort_values())

# # Plots (matplotlib)
# plt.figure(figsize=(8,4))
# plt.hist(difficulty_score, bins=15)
# plt.title("Distribution of DifficultyScore (0-10) from PC1")
# plt.xlabel("DifficultyScore (0-10)")
# plt.ylabel("Count")
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(7,6))
# if X_pca.shape[1] >= 2:
#     plt.scatter(X_pca[:,0], X_pca[:,1], c=work["kmeans_cluster"])
#     plt.xlabel("PC1")
#     plt.ylabel("PC2")
#     plt.title("PC1 vs PC2 (colored by KMeans cluster)")
# else:
#     plt.scatter(X_pca[:,0], np.zeros_like(X_pca[:,0]), c=work["kmeans_cluster"])
#     plt.xlabel("PC1")
#     plt.title("PC1 (only one PCA component available)")
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8,3))
# y = np.zeros_like(pc1)
# plt.scatter(pc1, y, s=20)
# out_idx = work["is_outlier_if_1"] == 1
# plt.scatter(pc1[out_idx], y[out_idx], s=40)
# plt.title("Outliers flagged by IsolationForest on PC1 axis")
# plt.yticks([])
# plt.xlabel("PC1 (raw)")
# plt.tight_layout()
# plt.show()
