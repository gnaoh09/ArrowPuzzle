import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# # LEVEL CURVE 1 LINE
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

# # LEVEL CURVE 2 LINE
# # ================================
# # LOAD CSV
# # ================================
df = pd.read_csv("D:\py\ArrowPuzzle\difficulty_plots\inference_output.csv")
df['Win Rate'] = pd.cut(df['Win Rate'], 
                        bins=[0, 20, 40, 60, 80, 100], 
                        labels=[5, 4, 3, 2, 1], 
                        include_lowest=True).astype(int)
# Chuyển DifficultyLabel sang số nếu là text
if df["predicted_stars"].dtype == object:
    unique_labels = sorted(df["predicted_stars"].unique())
    label_map = {label: i+1 for i, label in enumerate(unique_labels)}
    df["predicted_stars"] = df["predicted_stars"].map(label_map)
else:
    df["predicted_stars"] = df["predicted_stars"]

# ================================
# TÍNH MAE
# ================================
mae = (df["predicted_stars"] - df["Win Rate"]).abs().mean()

# ================================
# TÍNH PEARSON CORRELATION
# ================================
pearson_corr, p_value = pearsonr(df["predicted_stars"], df["Win Rate"])

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
    x, df["predicted_stars"],
    marker='o', linestyle='-', color='darkblue',
    label="Độ khó tự giải"
)

plt.plot(
    x, df["Win Rate"],
    marker='s', linestyle='--', color='crimson',  #crimson
    label="Độ khó thực tế"
)

plt.fill_between(x, df["predicted_stars"], df["Win Rate"], color='skyblue', alpha=0.15)

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


# import pandas as pd
# from scipy.stats import pearsonr
# import matplotlib.pyplot as plt

# # ================================
# # LOAD CSV
# # ================================
# # Thay đổi đường dẫn file của bạn ở đây
# import pandas as pd
# from scipy.stats import pearsonr
# import matplotlib.pyplot as plt
# import numpy as np

# # ================================
# # LOAD VÀ CHUẨN BỊ DỮ LIỆU
# # ================================
# def load_and_prepare_data(filepath, num_levels=50):
#     """Load và chuẩn bị dữ liệu"""
#     df = pd.read_csv(filepath)
#     df = df.head(num_levels)
    
#     # Chuyển DifficultyLabel sang số
#     if df["DifficultyLabel"].dtype == object:
#         unique_labels = sorted(df["DifficultyLabel"].unique())
#         label_map = {label: i+1 for i, label in enumerate(unique_labels)}
#         df["DifficultyLabel_Num"] = df["DifficultyLabel"].map(label_map)
#     else:
#         df["DifficultyLabel_Num"] = df["DifficultyLabel"]
    
#     # Xử lý Win Rate
#     if df["Win Rate"].dtype == object:
#         df["Win Rate"] = df["Win Rate"].str.replace('%', '').astype(float)
    
#     # Chuyển Win Rate về thang 5
#     df["Win_Rate_Score"] = pd.cut(
#         df["Win Rate"],
#         bins=[0, 20, 40, 60, 80, 100],
#         labels=[5, 4, 3, 2, 1],
#         include_lowest=True
#     ).astype(int)
    
#     return df

# # ================================
# # HÀM TÍNH METRICS
# # ================================
# def calculate_metrics(data1, data2, name1, name2):
#     """
#     Tính MAE và Pearson correlation giữa 2 series
    
#     Returns:
#     --------
#     dict: {'mae': float, 'pearson_r': float, 'p_value': float, ...}
#     """
#     # MAE (Mean Absolute Error)
#     mae = (data1 - data2).abs().mean()
    
#     # Pearson correlation
#     pearson_r, p_value = pearsonr(data1, data2)
    
#     # RMSE (Root Mean Square Error)
#     rmse = np.sqrt(((data1 - data2) ** 2).mean())
    
#     # Max deviation
#     max_dev = (data1 - data2).abs().max()
    
#     # Correlation strength
#     if abs(pearson_r) >= 0.8:
#         strength = "Rất mạnh"
#     elif abs(pearson_r) >= 0.6:
#         strength = "Mạnh"
#     elif abs(pearson_r) >= 0.4:
#         strength = "Trung bình"
#     elif abs(pearson_r) >= 0.2:
#         strength = "Yếu"
#     else:
#         strength = "Rất yếu"
    
#     return {
#         'name1': name1,
#         'name2': name2,
#         'mae': mae,
#         'rmse': rmse,
#         'max_deviation': max_dev,
#         'pearson_r': pearson_r,
#         'p_value': p_value,
#         'correlation_strength': strength
#     }

# # ================================
# # HÀM PLOT 2 ĐƯỜNG VỚI METRICS
# # ================================
# def plot_two_lines_with_metrics(df, 
#                                  line1_name, line2_name,
#                                  show_metrics=True,
#                                  show_metrics_on_plot=True,
#                                  filename=None):
#     """
#     Plot 2 đường và tính toán metrics
    
#     Parameters:
#     -----------
#     df : DataFrame
#         Dữ liệu
#     line1_name : str
#         'difficulty', 'difficulty_label', hoặc 'winrate'
#     line2_name : str
#         'difficulty', 'difficulty_label', hoặc 'winrate'
#     show_metrics : bool
#         In metrics ra console
#     show_metrics_on_plot : bool
#         Hiển thị metrics trên biểu đồ
#     filename : str
#         Tên file output (nếu None sẽ tự động tạo)
#     """
    
#     # Map line names to data
#     line_config = {
#         'difficulty': {
#             'data': df['difficulty'],
#             'name': 'Độ khó tự giải',
#             'color': 'darkblue',
#             'marker': 'o'
#         },
#         'difficulty_label': {
#             'data': df['DifficultyLabel_Num'],
#             'name': 'Độ khó cấu trúc',
#             'color': 'crimson',
#             'marker': 's'
#         },
#         'winrate': {
#             'data': df['Win_Rate_Score'],
#             'name': 'Win Rate Score',
#             'color': 'forestgreen',
#             'marker': '^'
#         }
#     }
    
#     if line1_name not in line_config or line2_name not in line_config:
#         print("❌ Tên đường không hợp lệ. Chọn: 'difficulty', 'difficulty_label', 'winrate'")
#         return None
    
#     line1 = line_config[line1_name]
#     line2 = line_config[line2_name]
    
#     # Tính metrics
#     metrics = calculate_metrics(
#         line1['data'], line2['data'],
#         line1['name'], line2['name']
#     )
    
#     # Print metrics
#     if show_metrics:
#         print("\n" + "="*70)
#         print(f"METRICS: {metrics['name1']} vs {metrics['name2']}")
#         print("="*70)
#         print(f"MAE (Mean Absolute Error):     {metrics['mae']:.4f}")
#         print(f"RMSE (Root Mean Square Error): {metrics['rmse']:.4f}")
#         print(f"Max Deviation:                 {metrics['max_deviation']:.4f}")
#         print(f"Pearson r:                     {metrics['pearson_r']:.4f}")
#         print(f"P-value:                       {metrics['p_value']:.6f}")
#         print(f"Correlation Strength:          {metrics['correlation_strength']}")
#         print("="*70)
    
#     # Create plot
#     fig, ax = plt.subplots(figsize=(20, 7))
#     x = range(1, len(df) + 1)
    
#     # Plot lines
#     ax.plot(x, line1['data'], 
#             marker=line1['marker'], linestyle='-', color=line1['color'],
#             linewidth=2.5, label=line1['name'], markersize=5)
    
#     ax.plot(x, line2['data'],
#             marker=line2['marker'], linestyle='--', color=line2['color'],
#             linewidth=2.5, label=line2['name'], markersize=5)
    
#     # Fill area
#     ax.fill_between(x, line1['data'], line2['data'], 
#                      alpha=0.2, color=line2['color'])
    
#     # Setup plot
#     ax.set_xticks(x)
#     ax.set_xticklabels(df["Level_Name"], rotation=90, fontsize=8, ha='right')
#     ax.set_yticks([1, 2, 3, 4, 5])
#     ax.set_xlabel("Level Name", fontsize=12, fontweight='bold')
#     ax.set_ylabel("Score (1-5)", fontsize=12, fontweight='bold')
#     ax.set_ylim(0.5, 5.5)
#     ax.grid(True, linestyle='--', alpha=0.4)
    
#     # Title with metrics
#     if show_metrics_on_plot:
#         title = f"{line1['name']} vs {line2['name']}\n"
#         title += f"MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f} | "
#         title += f"Pearson r: {metrics['pearson_r']:.4f} (p={metrics['p_value']:.6f})"
#         ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
#     else:
#         ax.set_title(f"{line1['name']} vs {line2['name']}", 
#                      fontsize=14, fontweight='bold')
    
#     # Legend
#     ax.legend(fontsize=11, loc='upper left')
    
#     plt.tight_layout()
    
#     # Save
#     if filename is None:
#         filename = f"plot_{line1_name}_vs_{line2_name}_with_metrics.png"
    
#     filepath = f'D:\py\ArrowPuzzle\difficulty_plots/{filename}'
#     plt.savefig(filepath, dpi=300, bbox_inches='tight')
#     print(f"✓ Đã lưu: {filename}")
#     plt.show()
    
#     return metrics

# # ================================
# # HÀM PHÂN TÍCH TẤT CẢ CÁC CẶP
# # ================================
# def analyze_all_pairs(df, save_summary=True):
#     """
#     Phân tích và plot tất cả các cặp 2 đường
#     """
#     pairs = [
#         ('difficulty', 'difficulty_label'),
#         ('difficulty', 'winrate'),
#         ('difficulty_label', 'winrate')
#     ]
    
#     all_metrics = []
    
#     print("\n" + "="*80)
#     print("PHÂN TÍCH TẤT CẢ CÁC CẶP 2 ĐƯỜNG")
#     print("="*80)
    
#     for i, (line1, line2) in enumerate(pairs, 1):
#         print(f"\n[{i}/{len(pairs)}] Analyzing {line1} vs {line2}...")
#         metrics = plot_two_lines_with_metrics(
#             df, line1, line2,
#             show_metrics=True,
#             show_metrics_on_plot=True,
#             filename=f"metrics_{line1}_vs_{line2}.png"
#         )
#         all_metrics.append(metrics)
    
#     # Tạo bảng tổng hợp
#     print("\n" + "="*80)
#     print("BẢNG TỔNG HỢP METRICS")
#     print("="*80)
    
#     metrics_df = pd.DataFrame(all_metrics)
#     print("\n" + metrics_df.to_string(index=False))
    
#     # Phân tích so sánh
#     print("\n" + "="*80)
#     print("PHÂN TÍCH SO SÁNH")
#     print("="*80)
    
#     print("\n1. ĐỘ LỆCH (MAE):")
#     print("-" * 80)
#     sorted_by_mae = sorted(all_metrics, key=lambda x: x['mae'])
#     for m in sorted_by_mae:
#         print(f"   {m['name1']:25s} vs {m['name2']:25s}: MAE = {m['mae']:.4f}")
    
#     print(f"\n   → Cặp GẦN NHẤT (MAE thấp nhất): {sorted_by_mae[0]['name1']} vs {sorted_by_mae[0]['name2']}")
#     print(f"   → Cặp XA NHẤT (MAE cao nhất): {sorted_by_mae[-1]['name1']} vs {sorted_by_mae[-1]['name2']}")
    
#     print("\n2. TƯƠNG QUAN (Pearson r):")
#     print("-" * 80)
#     sorted_by_corr = sorted(all_metrics, key=lambda x: abs(x['pearson_r']), reverse=True)
#     for m in sorted_by_corr:
#         sig = "***" if m['p_value'] < 0.001 else "**" if m['p_value'] < 0.01 else "*" if m['p_value'] < 0.05 else "ns"
#         print(f"   {m['name1']:25s} vs {m['name2']:25s}: r = {m['pearson_r']:7.4f} ({m['correlation_strength']:12s}) {sig}")
    
#     print(f"\n   → Tương quan MẠNH NHẤT: {sorted_by_corr[0]['name1']} vs {sorted_by_corr[0]['name2']}")
#     print(f"   → Tương quan YẾU NHẤT: {sorted_by_corr[-1]['name1']} vs {sorted_by_corr[-1]['name2']}")
    
#     # Tạo biểu đồ so sánh
#     create_comparison_chart(all_metrics)
    
#     # Save summary
#     if save_summary:
#         metrics_df.to_csv('D:\py\ArrowPuzzle\difficulty_plots/metrics_summary_all_pairs.csv', index=False)
#         print("\n✓ Đã lưu: metrics_summary_all_pairs.csv")
    
#     return all_metrics

# # ================================
# # HÀM TẠO BIỂU ĐỒ SO SÁNH
# # ================================
# def create_comparison_chart(all_metrics):
#     """Tạo biểu đồ so sánh các metrics"""
    
#     fig = plt.figure(figsize=(20, 12))
    
#     # Tạo grid 2x2
#     gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
#     pairs_labels = [f"{m['name1']}\nvs\n{m['name2']}" for m in all_metrics]
#     x_pos = range(len(all_metrics))
    
#     # 1. MAE comparison
#     ax1 = fig.add_subplot(gs[0, 0])
#     maes = [m['mae'] for m in all_metrics]
#     colors1 = ['skyblue', 'lightgreen', 'lightcoral']
#     bars1 = ax1.bar(x_pos, maes, color=colors1, edgecolor='black', linewidth=1.5)
#     ax1.set_xticks(x_pos)
#     ax1.set_xticklabels(pairs_labels, fontsize=9)
#     ax1.set_ylabel("MAE", fontsize=11, fontweight='bold')
#     ax1.set_title("Mean Absolute Error (MAE)", fontsize=12, fontweight='bold')
#     ax1.grid(axis='y', linestyle='--', alpha=0.4)
#     for bar, mae in zip(bars1, maes):
#         ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
#                 f'{mae:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
#     # 2. RMSE comparison
#     ax2 = fig.add_subplot(gs[0, 1])
#     rmses = [m['rmse'] for m in all_metrics]
#     bars2 = ax2.bar(x_pos, rmses, color=colors1, edgecolor='black', linewidth=1.5)
#     ax2.set_xticks(x_pos)
#     ax2.set_xticklabels(pairs_labels, fontsize=9)
#     ax2.set_ylabel("RMSE", fontsize=11, fontweight='bold')
#     ax2.set_title("Root Mean Square Error (RMSE)", fontsize=12, fontweight='bold')
#     ax2.grid(axis='y', linestyle='--', alpha=0.4)
#     for bar, rmse in zip(bars2, rmses):
#         ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
#                 f'{rmse:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
#     # 3. Pearson r comparison
#     ax3 = fig.add_subplot(gs[1, 0])
#     pearson_rs = [m['pearson_r'] for m in all_metrics]
#     colors3 = ['skyblue' if r >= 0 else 'salmon' for r in pearson_rs]
#     bars3 = ax3.bar(x_pos, pearson_rs, color=colors3, edgecolor='black', linewidth=1.5)
#     ax3.set_xticks(x_pos)
#     ax3.set_xticklabels(pairs_labels, fontsize=9)
#     ax3.set_ylabel("Pearson r", fontsize=11, fontweight='bold')
#     ax3.set_title("Hệ số tương quan Pearson", fontsize=12, fontweight='bold')
#     ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
#     ax3.grid(axis='y', linestyle='--', alpha=0.4)
#     ax3.set_ylim(-1, 1)
#     for bar, r in zip(bars3, pearson_rs):
#         height = bar.get_height()
#         va = 'bottom' if height >= 0 else 'top'
#         ax3.text(bar.get_x() + bar.get_width()/2., height,
#                 f'{r:.4f}', ha='center', va=va, fontweight='bold', fontsize=10)
    
#     # 4. Max Deviation comparison
#     ax4 = fig.add_subplot(gs[1, 1])
#     max_devs = [m['max_deviation'] for m in all_metrics]
#     bars4 = ax4.bar(x_pos, max_devs, color=colors1, edgecolor='black', linewidth=1.5)
#     ax4.set_xticks(x_pos)
#     ax4.set_xticklabels(pairs_labels, fontsize=9)
#     ax4.set_ylabel("Max Deviation", fontsize=11, fontweight='bold')
#     ax4.set_title("Độ lệch tối đa", fontsize=12, fontweight='bold')
#     ax4.grid(axis='y', linestyle='--', alpha=0.4)
#     for bar, dev in zip(bars4, max_devs):
#         ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
#                 f'{dev:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
#     plt.suptitle("So sánh Metrics cho tất cả các cặp 2 đường", 
#                  fontsize=14, fontweight='bold', y=0.995)
    
#     plt.savefig('D:\py\ArrowPuzzle\difficulty_plots/metrics_comparison_all.png', dpi=300, bbox_inches='tight')
#     print("\n✓ Đã lưu: metrics_comparison_all.png")
#     plt.show()

# # ================================
# # MAIN EXECUTION
# # ================================
# if __name__ == "__main__":
#     # Load data
#     filepath ="D:\\py\\ArrowPuzzle\\[ArrowPuzzle] Đánh giá độ khó - Sheet2.csv"
#     df = load_and_prepare_data(filepath, num_levels=50)
    
#     print("\n✓ Đã load {} levels".format(len(df)))
    
#     # Phân tích tất cả các cặp
#     all_metrics = analyze_all_pairs(df, save_summary=True)
    
#     print("\n" + "="*80)
#     print("✅ HOÀN TẤT!")
#     print("="*80)
#     print("\nĐã tạo các file:")
#     print("  1. metrics_difficulty_vs_difficulty_label.png")
#     print("  2. metrics_difficulty_vs_winrate.png")
#     print("  3. metrics_difficulty_label_vs_winrate.png")
#     print("  4. metrics_comparison_all.png")
#     print("  5. metrics_summary_all_pairs.csv")
#     print("="*80)

# import pandas as pd
# from scipy.stats import pearsonr
# import matplotlib.pyplot as plt
# import numpy as np

# # ================================
# # LOAD VÀ CHUẨN BỊ DỮ LIỆU
# # ================================
# def load_and_prepare_data(filepath, num_levels=50):
#     """Load và chuẩn bị dữ liệu"""
#     df = pd.read_csv(filepath)
#     df = df.head(num_levels)
    
#     # Chuyển DifficultyLabel sang số
#     if df["DifficultyLabel"].dtype == object:
#         unique_labels = sorted(df["DifficultyLabel"].unique())
#         label_map = {label: i+1 for i, label in enumerate(unique_labels)}
#         df["DifficultyLabel_Num"] = df["DifficultyLabel"].map(label_map)
#     else:
#         df["DifficultyLabel_Num"] = df["DifficultyLabel"]
    
#     # Xử lý Win Rate
#     if df["Win Rate"].dtype == object:
#         df["Win Rate"] = df["Win Rate"].str.replace('%', '').astype(float)
    
#     # Chuyển Win Rate về thang 5
#     df["Win_Rate_Score"] = pd.cut(
#         df["Win Rate"],
#         bins=[0, 20, 40, 60, 80, 100],
#         labels=[5, 4, 3, 2, 1],
#         include_lowest=True
#     ).astype(int)
    
#     return df

# # ================================
# # HÀM ÁP DỤNG PATTERN (QUAN TRỌNG)
# # ================================
# def apply_difficulty_pattern(df, column_name='difficulty'):
#     """
#     Áp dụng pattern: sau 1 level có độ khó > 1, thì 3 level tiếp theo có độ khó = 1
    
#     Logic:
#     - Quét từng level
#     - Khi gặp level có giá trị > 1:
#         + Giữ nguyên giá trị đó
#         + 3 level TIẾP THEO sẽ bị set = 1
#     - Sau đó tiếp tục quét và lặp lại
    
#     Parameters:
#     -----------
#     df : DataFrame
#         Dữ liệu gốc
#     column_name : str
#         Tên cột cần áp dụng pattern ('difficulty' hoặc 'DifficultyLabel_Num')
    
#     Returns:
#     --------
#     DataFrame với cột mới đã áp dụng pattern
#     """
#     df_modified = df.copy()
#     new_column_name = f"{column_name}_pattern"
    
#     # Khởi tạo cột mới với giá trị gốc
#     df_modified[new_column_name] = df_modified[column_name].copy()
    
#     # Áp dụng pattern
#     i = 0
#     while i < len(df_modified):
#         current_value = df_modified.iloc[i][column_name]
        
#         # Nếu giá trị hiện tại > 1
#         if current_value > 1:
#             # Giữ nguyên giá trị này (đã copy từ gốc)
#             # Set 3 level TIẾP THEO = 1
#             for j in range(1, 3):  # 1, 2, 3
#                 if i + j < len(df_modified):
#                     df_modified.iloc[i + j, df_modified.columns.get_loc(new_column_name)] = 1
            
#             # Nhảy sang level sau 3 level đã set
#             i += 4  # Nhảy qua level hiện tại + 3 level tiếp theo
#         else:
#             # Nếu <= 1, giữ nguyên và tiếp tục
#             i += 1
    
#     return df_modified

# # ================================
# # HÀM TÍNH METRICS
# # ================================
# def calculate_metrics(data1, data2, name1, name2):
#     """
#     Tính MAE và Pearson correlation giữa 2 series
    
#     Returns:
#     --------
#     dict: {'mae': float, 'pearson_r': float, 'p_value': float, ...}
#     """
#     # MAE (Mean Absolute Error)
#     mae = (data1 - data2).abs().mean()
    
#     # Pearson correlation
#     pearson_r, p_value = pearsonr(data1, data2)
    
#     # RMSE (Root Mean Square Error)
#     rmse = np.sqrt(((data1 - data2) ** 2).mean())
    
#     # Max deviation
#     max_dev = (data1 - data2).abs().max()
    
#     # Correlation strength
#     if abs(pearson_r) >= 0.8:
#         strength = "Rất mạnh"
#     elif abs(pearson_r) >= 0.6:
#         strength = "Mạnh"
#     elif abs(pearson_r) >= 0.4:
#         strength = "Trung bình"
#     elif abs(pearson_r) >= 0.2:
#         strength = "Yếu"
#     else:
#         strength = "Rất yếu"
    
#     return {
#         'name1': name1,
#         'name2': name2,
#         'mae': mae,
#         'rmse': rmse,
#         'max_deviation': max_dev,
#         'pearson_r': pearson_r,
#         'p_value': p_value,
#         'correlation_strength': strength
#     }

# # ================================
# # HÀM PLOT PATTERN VỚI WIN RATE
# # ================================
# def plot_pattern_vs_winrate(df, difficulty_column='difficulty', show_original=True):
#     """
#     Plot so sánh: Difficulty Pattern vs Win Rate
#     Có thể hiển thị thêm đường gốc để tham khảo
    
#     Parameters:
#     -----------
#     df : DataFrame
#         Dữ liệu đã có cột pattern
#     difficulty_column : str
#         'difficulty' hoặc 'DifficultyLabel_Num'
#     show_original : bool
#         Có hiển thị đường gốc không (mờ đi để tham khảo)
#     """
#     pattern_column = f"{difficulty_column}_pattern"
    
#     # Tính metrics giữa pattern và win rate
#     metrics = calculate_metrics(
#         df[pattern_column],
#         df['Win_Rate_Score'],
#         f'{difficulty_column} (Pattern)',
#         'Win Rate Score'
#     )
    
#     # Print metrics
#     print("\n" + "="*70)
#     print(f"METRICS: {metrics['name1']} vs {metrics['name2']}")
#     print("="*70)
#     print(f"MAE (Mean Absolute Error):     {metrics['mae']:.4f}")
#     print(f"RMSE (Root Mean Square Error): {metrics['rmse']:.4f}")
#     print(f"Max Deviation:                 {metrics['max_deviation']:.4f}")
#     print(f"Pearson r:                     {metrics['pearson_r']:.4f}")
#     print(f"P-value:                       {metrics['p_value']:.6f}")
#     print(f"Correlation Strength:          {metrics['correlation_strength']}")
#     print("="*70)
    
#     # Create plot
#     fig, ax = plt.subplots(figsize=(20, 7))
#     x = range(1, len(df) + 1)
    
#     # Plot đường gốc (mờ đi, chỉ để tham khảo)
#     if show_original:
#         ax.plot(x, df[difficulty_column], 
#                 marker='o', linestyle=':', color='gray',
#                 linewidth=1.5, label=f'{difficulty_column} (Gốc - tham khảo)', 
#                 markersize=4, alpha=0.4)
    
#     # Plot đường pattern (chính)
#     ax.plot(x, df[pattern_column],
#             marker='s', linestyle='-', color='crimson',
#             linewidth=2.5, label=f'{difficulty_column} (Pattern: 1 khó → 3 dễ)', 
#             markersize=6)
    
#     # Plot Win Rate (thực tế)
#     ax.plot(x, df['Win_Rate_Score'],
#             marker='^', linestyle='--', color='forestgreen',
#             linewidth=2.5, label='Win Rate Score (Thực tế)', markersize=6)
    
#     # Fill area giữa pattern và win rate
#     ax.fill_between(x, df[pattern_column], df['Win_Rate_Score'], 
#                      alpha=0.15, color='purple')
    
#     # Đánh dấu các điểm bị thay đổi bởi pattern
#     for i in range(len(df)):
#         if df[difficulty_column].iloc[i] != df[pattern_column].iloc[i]:
#             ax.axvline(x=i+1, color='orange', alpha=0.3, linestyle=':', linewidth=1.5)
    
#     # Setup plot
#     ax.set_xticks(x)
#     ax.set_xticklabels(df["Level_Name"], rotation=90, fontsize=8, ha='right')
#     ax.set_yticks([1, 2, 3, 4, 5])
#     ax.set_xlabel("Level Name", fontsize=12, fontweight='bold')
#     ax.set_ylabel("Score (1-5)", fontsize=12, fontweight='bold')
#     ax.set_ylim(0.5, 5.5)
#     ax.grid(True, linestyle='--', alpha=0.4)
    
#     # Title with metrics
#     title = f"So sánh: {difficulty_column} (Pattern) vs Win Rate\n"
#     title += f"MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f} | "
#     title += f"Pearson r: {metrics['pearson_r']:.4f} (p={metrics['p_value']:.6f})"
#     ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    
#     # Legend
#     ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
    
#     plt.tight_layout()
    
#     # Save
#     filename = f"FINAL_pattern_{difficulty_column}_vs_winrate.png"
#     filepath = f'D:\\py\\ArrowPuzzle\\difficulty_plots\\{filename}'
#     plt.savefig(filepath, dpi=300, bbox_inches='tight')
#     print(f"✓ Đã lưu: {filename}")
#     plt.show()
    
#     return metrics

# # ================================
# # HÀM PLOT SO SÁNH GỐC VS PATTERN
# # ================================
# def plot_original_vs_pattern(df, column_name='difficulty'):
#     """
#     So sánh đường gốc với đường đã áp dụng pattern
#     """
#     pattern_column = f"{column_name}_pattern"
    
#     # Tính metrics
#     metrics = calculate_metrics(
#         df[column_name], 
#         df[pattern_column],
#         f"{column_name} (Gốc)",
#         f"{column_name} (Pattern)"
#     )
    
#     # Print metrics
#     print("\n" + "="*70)
#     print(f"METRICS: {metrics['name1']} vs {metrics['name2']}")
#     print("="*70)
#     print(f"MAE (Mean Absolute Error):     {metrics['mae']:.4f}")
#     print(f"RMSE (Root Mean Square Error): {metrics['rmse']:.4f}")
#     print(f"Max Deviation:                 {metrics['max_deviation']:.4f}")
#     print(f"Pearson r:                     {metrics['pearson_r']:.4f}")
#     print(f"P-value:                       {metrics['p_value']:.6f}")
#     print(f"Correlation Strength:          {metrics['correlation_strength']}")
#     print("="*70)
    
#     # Create plot
#     fig, ax = plt.subplots(figsize=(20, 7))
#     x = range(1, len(df) + 1)
    
#     # Plot đường gốc
#     ax.plot(x, df[column_name], 
#             marker='o', linestyle='-', color='darkblue',
#             linewidth=2.5, label=f'{column_name} (Gốc)', markersize=5)
    
#     # Plot đường pattern
#     ax.plot(x, df[pattern_column],
#             marker='s', linestyle='--', color='crimson',
#             linewidth=2.5, label=f'{column_name} (Pattern: 1 khó → 3 dễ)', markersize=5)
    
#     # Fill area giữa 2 đường
#     ax.fill_between(x, df[column_name], df[pattern_column], 
#                      alpha=0.2, color='orange')
    
#     # Đánh dấu các điểm thay đổi
#     for i in range(len(df)):
#         if df[column_name].iloc[i] != df[pattern_column].iloc[i]:
#             ax.axvline(x=i+1, color='red', alpha=0.2, linestyle=':', linewidth=1)
#             # Thêm annotation cho điểm thay đổi
#             ax.text(i+1, df[column_name].iloc[i], f'{df[column_name].iloc[i]}→1', 
#                    fontsize=7, ha='center', va='bottom', color='red', fontweight='bold')
    
#     # Setup plot
#     ax.set_xticks(x)
#     ax.set_xticklabels(df["Level_Name"], rotation=90, fontsize=8, ha='right')
#     ax.set_yticks([1, 2, 3, 4, 5])
#     ax.set_xlabel("Level Name", fontsize=12, fontweight='bold')
#     ax.set_ylabel("Score (1-5)", fontsize=12, fontweight='bold')
#     ax.set_ylim(0.5, 5.5)
#     ax.grid(True, linestyle='--', alpha=0.4)
    
#     # Title with metrics
#     title = f"So sánh {column_name}: Gốc vs Pattern (1 khó → 3 dễ)\n"
#     title += f"MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f} | "
#     title += f"Pearson r: {metrics['pearson_r']:.4f}"
#     ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    
#     # Legend
#     ax.legend(fontsize=11, loc='upper left')
    
#     plt.tight_layout()
    
#     # Save
#     filename = f"comparison_{column_name}_original_vs_pattern.png"
#     filepath = f'D:\\py\\ArrowPuzzle\\difficulty_plots\\{filename}'
#     plt.savefig(filepath, dpi=300, bbox_inches='tight')
#     print(f"✓ Đã lưu: {filename}")
#     plt.show()
    
#     return metrics

# # ================================
# # HÀM PHÂN TÍCH TOÀN DIỆN
# # ================================
# def analyze_pattern_comprehensive(df):
#     """
#     Phân tích toàn diện pattern cho cả difficulty và difficulty_label
#     """
#     print("\n" + "="*80)
#     print("PHÂN TÍCH PATTERN: 1 KHÓ → 3 DỄ")
#     print("="*80)
    
#     # Áp dụng pattern cho cả 2 cột
#     print("\n📊 Đang áp dụng pattern...")
#     df_with_pattern = apply_difficulty_pattern(df, 'difficulty')
#     df_with_pattern = apply_difficulty_pattern(df_with_pattern, 'DifficultyLabel_Num')
    
#     # Hiển thị bảng so sánh
#     print("\n" + "="*80)
#     print("BẢNG DỮ LIỆU SO SÁNH (10 levels đầu)")
#     print("="*80)
#     display_df = df_with_pattern[['Level_Name', 'difficulty', 'difficulty_pattern', 
#                                     'DifficultyLabel_Num', 'DifficultyLabel_Num_pattern', 
#                                     'Win_Rate_Score']].head(10)
#     print(display_df.to_string(index=False))
    
#     # 1. So sánh difficulty gốc vs pattern
#     print("\n" + "="*80)
#     print("[1] DIFFICULTY: Gốc vs Pattern")
#     print("="*80)
#     metrics_diff = plot_original_vs_pattern(df_with_pattern, 'difficulty')
    
#     # 2. So sánh difficulty_label gốc vs pattern
#     print("\n" + "="*80)
#     print("[2] DIFFICULTY LABEL: Gốc vs Pattern")
#     print("="*80)
#     metrics_label = plot_original_vs_pattern(df_with_pattern, 'DifficultyLabel_Num')
    
#     # 3. Plot difficulty pattern với Win Rate
#     print("\n" + "="*80)
#     print("[3] DIFFICULTY PATTERN vs WIN RATE")
#     print("="*80)
#     metrics_diff_wr = plot_pattern_vs_winrate(df_with_pattern, 'difficulty', show_original=True)
    
#     # 4. Plot difficulty_label pattern với Win Rate
#     print("\n" + "="*80)
#     print("[4] DIFFICULTY LABEL PATTERN vs WIN RATE")
#     print("="*80)
#     metrics_label_wr = plot_pattern_vs_winrate(df_with_pattern, 'DifficultyLabel_Num', show_original=True)
    
#     # 5. Tổng hợp so sánh
#     print("\n" + "="*80)
#     print("TỔNG HỢP KẾT QUẢ")
#     print("="*80)
    
#     # So sánh với dữ liệu gốc
#     original_diff_wr_mae = (df_with_pattern['difficulty'] - df_with_pattern['Win_Rate_Score']).abs().mean()
#     original_label_wr_mae = (df_with_pattern['DifficultyLabel_Num'] - df_with_pattern['Win_Rate_Score']).abs().mean()
    
#     print("\n📈 So sánh MAE với Win Rate:")
#     print("-" * 80)
#     print(f"  Difficulty (Gốc):           {original_diff_wr_mae:.4f}")
#     print(f"  Difficulty (Pattern):       {metrics_diff_wr['mae']:.4f}")
#     change_diff = metrics_diff_wr['mae'] - original_diff_wr_mae
#     pct_diff = (change_diff / original_diff_wr_mae * 100)
#     print(f"  → Thay đổi:                 {change_diff:+.4f} ({pct_diff:+.2f}%)")
#     if change_diff < 0:
#         print(f"  ✅ Pattern CẢI THIỆN độ chính xác!")
#     else:
#         print(f"  ❌ Pattern làm GIẢM độ chính xác!")
    
#     print(f"\n  Difficulty Label (Gốc):     {original_label_wr_mae:.4f}")
#     print(f"  Difficulty Label (Pattern): {metrics_label_wr['mae']:.4f}")
#     change_label = metrics_label_wr['mae'] - original_label_wr_mae
#     pct_label = (change_label / original_label_wr_mae * 100)
#     print(f"  → Thay đổi:                 {change_label:+.4f} ({pct_label:+.2f}%)")
#     if change_label < 0:
#         print(f"  ✅ Pattern CẢI THIỆN độ chính xác!")
#     else:
#         print(f"  ❌ Pattern làm GIẢM độ chính xác!")
    
#     # So sánh tương quan
#     print("\n📊 So sánh Tương quan Pearson với Win Rate:")
#     print("-" * 80)
#     original_diff_wr_corr = pearsonr(df_with_pattern['difficulty'], df_with_pattern['Win_Rate_Score'])[0]
#     original_label_wr_corr = pearsonr(df_with_pattern['DifficultyLabel_Num'], df_with_pattern['Win_Rate_Score'])[0]
    
#     print(f"  Difficulty (Gốc):           {original_diff_wr_corr:.4f}")
#     print(f"  Difficulty (Pattern):       {metrics_diff_wr['pearson_r']:.4f}")
#     print(f"  → Thay đổi:                 {metrics_diff_wr['pearson_r'] - original_diff_wr_corr:+.4f}")
    
#     print(f"\n  Difficulty Label (Gốc):     {original_label_wr_corr:.4f}")
#     print(f"  Difficulty Label (Pattern): {metrics_label_wr['pearson_r']:.4f}")
#     print(f"  → Thay đổi:                 {metrics_label_wr['pearson_r'] - original_label_wr_corr:+.4f}")
    
#     # Số lượng level bị thay đổi
#     print("\n📝 Thống kê thay đổi:")
#     print("-" * 80)
#     diff_changed = (df_with_pattern['difficulty'] != df_with_pattern['difficulty_pattern']).sum()
#     label_changed = (df_with_pattern['DifficultyLabel_Num'] != df_with_pattern['DifficultyLabel_Num_pattern']).sum()
    
#     print(f"  Difficulty: {diff_changed}/{len(df_with_pattern)} levels bị thay đổi ({diff_changed/len(df_with_pattern)*100:.1f}%)")
#     print(f"  Difficulty Label: {label_changed}/{len(df_with_pattern)} levels bị thay đổi ({label_changed/len(df_with_pattern)*100:.1f}%)")
    
#     # Save data
#     output_file = 'D:\\py\\ArrowPuzzle\\difficulty_plots\\data_with_pattern.csv'
#     df_with_pattern.to_csv(output_file, index=False)
#     print(f"\n✓ Đã lưu dữ liệu: data_with_pattern.csv")
    
#     # Tạo summary metrics
#     summary = {
#         'Metric': ['Difficulty', 'Difficulty Label'],
#         'Original_MAE': [original_diff_wr_mae, original_label_wr_mae],
#         'Pattern_MAE': [metrics_diff_wr['mae'], metrics_label_wr['mae']],
#         'MAE_Change': [change_diff, change_label],
#         'MAE_Change_%': [pct_diff, pct_label],
#         'Original_Pearson_r': [original_diff_wr_corr, original_label_wr_corr],
#         'Pattern_Pearson_r': [metrics_diff_wr['pearson_r'], metrics_label_wr['pearson_r']],
#         'Levels_Changed': [diff_changed, label_changed]
#     }
    
#     summary_df = pd.DataFrame(summary)
#     summary_file = 'D:\\py\\ArrowPuzzle\\difficulty_plots\\pattern_summary.csv'
#     summary_df.to_csv(summary_file, index=False)
#     print(f"✓ Đã lưu summary: pattern_summary.csv")
    
#     return df_with_pattern, summary_df

# # ================================
# # MAIN EXECUTION
# # ================================
# if __name__ == "__main__":
#     # Load data
#     filepath = "D:\\py\\ArrowPuzzle\\[ArrowPuzzle] Đánh giá độ khó - Sheet2.csv"
    
#     print("="*80)
#     print("BẮT ĐẦU PHÂN TÍCH PATTERN")
#     print("="*80)
    
#     df = load_and_prepare_data(filepath, num_levels=50)
#     print(f"\n✓ Đã load {len(df)} levels")
    
#     # Chạy phân tích pattern
#     df_with_pattern, summary = analyze_pattern_comprehensive(df)
    
#     # Hiển thị summary cuối cùng
#     print("\n" + "="*80)
#     print("SUMMARY TABLE")
#     print("="*80)
#     print(summary.to_string(index=False))
    
#     print("\n" + "="*80)
#     print("✅ HOÀN TẤT!")
#     print("="*80)
#     print("\nĐã tạo các file:")
#     print("  1. comparison_difficulty_original_vs_pattern.png")
#     print("  2. comparison_DifficultyLabel_Num_original_vs_pattern.png")
#     print("  3. FINAL_pattern_difficulty_vs_winrate.png")
#     print("  4. FINAL_pattern_DifficultyLabel_Num_vs_winrate.png")
#     print("  5. data_with_pattern.csv")
#     print("  6. pattern_summary.csv")
#     print("="*80)
