import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# ====== LOAD CSV ======
input_csv = "levels_result_1612_2.csv"
df = pd.read_csv(input_csv)

# Sử dụng 2 cột chính
df2 = df[['Estimated_Player_Time', 'Avg_Moves']].copy()

# ====== RUN KMEANS ======
kmeans = KMeans(n_clusters=5, random_state=42)
df2['cluster'] = kmeans.fit_predict(df2)

# ====== TÍNH CENTROIDS ======
centroids = kmeans.cluster_centers_
centroid_df = pd.DataFrame(centroids, columns=['time', 'moves'])
centroid_df['cluster'] = centroid_df.index

# ====== XẾP HẠNG ĐỘ KHÓ (Difficulty 1–5) ======
# Độ khó dựa trên distance từ gốc (0,0)
centroid_df['difficulty_score'] = np.sqrt(
    centroid_df['time']**2 + centroid_df['moves']**2
)

# Sort để xếp difficulty 1 → 5
centroid_df = centroid_df.sort_values('difficulty_score')
centroid_df['difficulty'] = range(1, 6)  # 1 = dễ nhất, 5 = khó nhất

# Merge difficulty vào dataframe chính
df2 = df2.merge(centroid_df[['cluster', 'difficulty']], on='cluster')
df['difficulty'] = df2['difficulty']

# ====== SAVE TO CSV ======
output_csv = "levels_result_with_difficulty_1612.csv"
df.to_csv(output_csv, index=False)
print("Saved:", output_csv)

# ====== PRINT SUMMARY ======
print("\n=== CENTROIDS & DIFFICULTY LEVELS ===")
print(centroid_df)

print("\n=== SAMPLE OUTPUT ===")
print(df[['Level_Name', 'Estimated_Player_Time', 'Avg_Moves', 'difficulty']].head())

# ====== VISUALIZATION ======
plt.figure(figsize=(7,6))
scatter = plt.scatter(
    df['Estimated_Player_Time'],
    df['Avg_Moves'],
    c=df['difficulty'],
    cmap='viridis'
)
plt.colorbar(scatter, label="Difficulty (1–5)")
plt.xlabel("Estimated Player Time")
plt.ylabel("Avg Moves")
plt.title("KMeans Clustering → Difficulty 1–5")
plt.grid(True)
plt.show()
