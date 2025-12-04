import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

from sklearn.manifold import TSNE

# ===============================================
# 1. Load & Preprocess
# ===============================================
INPUT_CSV = "/Users/hoangnguyen/Documents/py/ArrowPuzzle/100lv_0412.csv"
OUTPUT_CSV = "levels_difficulty_full_0512.csv"

df = pd.read_csv(INPUT_CSV)

metric_cols = [
    "total_arrows","total_edges","max_in_degree","edges_per_arrow",
    "dependency_density","has_cycle_bool","critical_nodes",
    "critical_edges","avg_row_blockers","avg_col_blockers",
    "crowded_rows","crowded_cols"
]

X = df[metric_cols].copy()

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================================
# 2. PCA (Baseline) → Difficulty Score = PC1
# ===============================================
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

df["DifficultyScore_PCA"] = X_pca[:, 0]

print("PCA variance:", pca.explained_variance_ratio_)


# ===============================================
# 3. Full Precision PCA (cumulative variance)
# ===============================================
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

pca_full = high_precision_pca(X_scaled)


# ===============================================
# 4. Find optimal number of clusters (PCA space)
# ===============================================
def find_best_k(X):
    inertias = []
    silhouettes = []
    K = range(2, 12)

    for k in K:
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    plt.figure(figsize=(12,5))

    # Elbow
    plt.subplot(1,2,1)
    plt.plot(K, inertias, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("k")
    plt.ylabel("Inertia")

    # Silhouette
    plt.subplot(1,2,2)
    plt.plot(K, silhouettes, marker="o")
    plt.title("Silhouette Score")
    plt.xlabel("k")
    plt.ylabel("Score")

    plt.show()

    best_k = K[np.argmax(silhouettes)]
    print("Best k =", best_k)
    return best_k

best_k_pca = find_best_k(X_pca)


# ===============================================
# 5. KMeans clustering (using best k)
# ===============================================
kmeans = KMeans(n_clusters=best_k_pca, random_state=42)
df["Cluster_PCA"] = kmeans.fit_predict(X_pca)

cluster_map = {
    0: "very easy",
    1: "easy",
    2: "medium",
    3: "hard",
    4: "very hard"
}

df["DifficultyLabel"] = df["Cluster_PCA"].map(
    lambda x: cluster_map.get(x, "unknown")
)


# ===============================================
# 6. Clustering trực tiếp trên RAW metrics
# ===============================================
best_k_raw = find_best_k(X_scaled)

def cluster_raw(X_scaled, k):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_scaled)
    return labels

df["Cluster_Raw"] = cluster_raw(X_scaled, best_k_raw)


# ===============================================
# 7. Isolation Forest → phát hiện level dị biệt
# ===============================================
iso = IsolationForest(contamination=0.1, random_state=42)
df["Anomaly"] = iso.fit_predict(X_scaled)
df["Anomaly"] = df["Anomaly"].map({1: "normal", -1: "anomaly"})


# ===============================================
# 8. Visualization (UMAP + t-SNE)
# ===============================================
def visualize_umap_tsne(X_scaled, labels):
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
    X_tsne = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(6,5))
    plt.scatter(X_tsne[:,0], X_tsne[:,1], c=labels, cmap="viridis")
    plt.title("t-SNE Visualization")
    plt.colorbar()
    plt.show()

visualize_umap_tsne(X_scaled, df["Cluster_PCA"])


# ===============================================
# 9. Save result
# ===============================================
df.to_csv(OUTPUT_CSV, index=False)
print("Saved difficulty results to:", OUTPUT_CSV)
