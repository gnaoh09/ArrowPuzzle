import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# LEVEL CURVE 1 LINE
# ===== LOAD CSV =====
# df = pd.read_csv("/Users/hoangnguyen/Documents/py/ArrowPuzzle/levels_result_with_difficulty_1612.csv")

# # Nếu DifficultyLabel là text → map sang số
# if df["DifficultyLabel"].dtype == object:
#     unique_labels = sorted(df["DifficultyLabel"].unique())
#     label_map = {label: i+1 for i, label in enumerate(unique_labels)}
#     df["DifficultyLabel_Num"] = df["DifficultyLabel"].map(label_map)
# else:
#     df["DifficultyLabel_Num"] = df["DifficultyLabel"]

# # ===== TÍNH MAE (Mean Absolute Error) =====
# mae = (df["DifficultyLabel"] - df["DifficultyLabel_Num"]).abs().mean()

# # ===== TÍNH PEARSON CORRELATION =====
# pearson_corr, p_value = pearsonr(df["DifficultyLabel"], df["DifficultyLabel_Num"])

# # ===== PRINT RESULT =====
# print("========== DIFFICULTY METRICS ==========")
# print(f"MAE (Mean Absolute Error): {mae:.4f}")
# print(f"Pearson correlation (r): {pearson_corr:.4f}")
# print(f"P-value: {p_value:.6f}")
# print("========================================")

# LEVEL CURVE 2 LINE
# # ================================
# # LOAD CSV
# # ================================
df = pd.read_csv("/Users/hoangnguyen/Documents/py/ArrowPuzzle/sheet2.csv")

# Chuyển DifficultyLabel sang số nếu là text
if df["DifficultyLabel"].dtype == object:
    unique_labels = sorted(df["DifficultyLabel"].unique())
    label_map = {label: i+1 for i, label in enumerate(unique_labels)}
    df["DifficultyLabel_Num"] = df["DifficultyLabel"].map(label_map)
else:
    df["DifficultyLabel_Num"] = df["DifficultyLabel"]

# ================================
# TÍNH MAE
# ================================
mae = (df["difficulty"] - df["DifficultyLabel_Num"]).abs().mean()

# ================================
# TÍNH PEARSON CORRELATION
# ================================
pearson_corr, p_value = pearsonr(df["difficulty"], df["DifficultyLabel_Num"])

print("\n========== DIFFICULTY METRICS ==========")
print(f"Độ lệch trung bình tuyệt đối: {mae:.4f}")
print(f"Hệ số tương quan Pearson: {pearson_corr:.4f}")
print(f"P-value: {p_value:.6f}")
print("========================================")

# ================================
# VẼ BIỂU ĐỒ SO SÁNH 2 ĐƯỜNG
# ================================
plt.figure(figsize=(18, 5))

x = range(1, len(df) + 1)

plt.plot(
    x, df["difficulty"],
    marker='o', linestyle='-', color='darkblue',
    label="Độ khó tự giải"
)

plt.plot(
    x, df["DifficultyLabel_Num"],
    marker='s', linestyle='--', color='crimson',  #crimson
    label="Độ khó cấu trúc"
)

plt.fill_between(x, df["difficulty"], df["DifficultyLabel_Num"], color='skyblue', alpha=0.15)

plt.xticks(x, df["Level_Name"], rotation=90, fontsize=7)
plt.yticks([1, 2, 3, 4, 5])
plt.xlabel("Level Name")
plt.ylabel("Difficulty Score")
plt.title("Level Curve")
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(0.5, 5.5)
plt.legend()
plt.tight_layout()

plt.show()
