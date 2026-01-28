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
# df = pd.read_csv("/Users/hoangnguyen/Documents/py/ArrowPuzzle/sheet2.csv")

# # Chuyển DifficultyLabel sang số nếu là text
# if df["DifficultyLabel"].dtype == object:
#     unique_labels = sorted(df["DifficultyLabel"].unique())
#     label_map = {label: i+1 for i, label in enumerate(unique_labels)}
#     df["DifficultyLabel_Num"] = df["DifficultyLabel"].map(label_map)
# else:
#     df["DifficultyLabel_Num"] = df["DifficultyLabel"]

# # ================================
# # TÍNH MAE
# # ================================
# mae = (df["difficulty"] - df["DifficultyLabel_Num"]).abs().mean()

# # ================================
# # TÍNH PEARSON CORRELATION
# # ================================
# pearson_corr, p_value = pearsonr(df["difficulty"], df["DifficultyLabel_Num"])

# print("\n========== DIFFICULTY METRICS ==========")
# print(f"Độ lệch trung bình tuyệt đối: {mae:.4f}")
# print(f"Hệ số tương quan Pearson: {pearson_corr:.4f}")
# print(f"P-value: {p_value:.6f}")
# print("========================================")

# # ================================
# # VẼ BIỂU ĐỒ SO SÁNH 2 ĐƯỜNG
# # ================================
# plt.figure(figsize=(18, 5))

# x = range(1, len(df) + 1)

# plt.plot(
#     x, df["difficulty"],
#     marker='o', linestyle='-', color='darkblue',
#     label="Độ khó tự giải"
# )

# plt.plot(
#     x, df["DifficultyLabel_Num"],
#     marker='s', linestyle='--', color='crimson',  #crimson
#     label="Độ khó cấu trúc"
# )

# plt.fill_between(x, df["difficulty"], df["DifficultyLabel_Num"], color='skyblue', alpha=0.15)

# plt.xticks(x, df["Level_Name"], rotation=90, fontsize=7)
# plt.yticks([1, 2, 3, 4, 5])
# plt.xlabel("Level Name")
# plt.ylabel("Difficulty Score")
# plt.title("Level Curve")
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.ylim(0.5, 5.5)
# plt.legend()
# plt.tight_layout()

# plt.show()


import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# ================================
# LOAD CSV
# ================================
# Thay đổi đường dẫn file của bạn ở đây
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np

# ================================
# LOAD VÀ CHUẨN BỊ DỮ LIỆU
# ================================
def load_and_prepare_data(filepath, num_levels=50):
    """Load và chuẩn bị dữ liệu"""
    df = pd.read_csv(filepath)
    df = df.head(num_levels)
    
    # Chuyển DifficultyLabel sang số
    if df["DifficultyLabel"].dtype == object:
        unique_labels = sorted(df["DifficultyLabel"].unique())
        label_map = {label: i+1 for i, label in enumerate(unique_labels)}
        df["DifficultyLabel_Num"] = df["DifficultyLabel"].map(label_map)
    else:
        df["DifficultyLabel_Num"] = df["DifficultyLabel"]
    
    # Xử lý Win Rate
    if df["Win Rate"].dtype == object:
        df["Win Rate"] = df["Win Rate"].str.replace('%', '').astype(float)
    
    # Chuyển Win Rate về thang 5
    df["Win_Rate_Score"] = pd.cut(
        df["Win Rate"],
        bins=[0, 20, 40, 60, 80, 100],
        labels=[5, 4, 3, 2, 1],
        include_lowest=True
    ).astype(int)
    
    return df

# ================================
# HÀM TÍNH METRICS
# ================================
def calculate_metrics(data1, data2, name1, name2):
    """
    Tính MAE và Pearson correlation giữa 2 series
    
    Returns:
    --------
    dict: {'mae': float, 'pearson_r': float, 'p_value': float, ...}
    """
    # MAE (Mean Absolute Error)
    mae = (data1 - data2).abs().mean()
    
    # Pearson correlation
    pearson_r, p_value = pearsonr(data1, data2)
    
    # RMSE (Root Mean Square Error)
    rmse = np.sqrt(((data1 - data2) ** 2).mean())
    
    # Max deviation
    max_dev = (data1 - data2).abs().max()
    
    # Correlation strength
    if abs(pearson_r) >= 0.8:
        strength = "Rất mạnh"
    elif abs(pearson_r) >= 0.6:
        strength = "Mạnh"
    elif abs(pearson_r) >= 0.4:
        strength = "Trung bình"
    elif abs(pearson_r) >= 0.2:
        strength = "Yếu"
    else:
        strength = "Rất yếu"
    
    return {
        'name1': name1,
        'name2': name2,
        'mae': mae,
        'rmse': rmse,
        'max_deviation': max_dev,
        'pearson_r': pearson_r,
        'p_value': p_value,
        'correlation_strength': strength
    }

# ================================
# HÀM PLOT 2 ĐƯỜNG VỚI METRICS
# ================================
def plot_two_lines_with_metrics(df, 
                                 line1_name, line2_name,
                                 show_metrics=True,
                                 show_metrics_on_plot=True,
                                 filename=None):
    """
    Plot 2 đường và tính toán metrics
    
    Parameters:
    -----------
    df : DataFrame
        Dữ liệu
    line1_name : str
        'difficulty', 'difficulty_label', hoặc 'winrate'
    line2_name : str
        'difficulty', 'difficulty_label', hoặc 'winrate'
    show_metrics : bool
        In metrics ra console
    show_metrics_on_plot : bool
        Hiển thị metrics trên biểu đồ
    filename : str
        Tên file output (nếu None sẽ tự động tạo)
    """
    
    # Map line names to data
    line_config = {
        'difficulty': {
            'data': df['difficulty'],
            'name': 'Độ khó tự giải',
            'color': 'darkblue',
            'marker': 'o'
        },
        'difficulty_label': {
            'data': df['DifficultyLabel_Num'],
            'name': 'Độ khó cấu trúc',
            'color': 'crimson',
            'marker': 's'
        },
        'winrate': {
            'data': df['Win_Rate_Score'],
            'name': 'Win Rate Score',
            'color': 'forestgreen',
            'marker': '^'
        }
    }
    
    if line1_name not in line_config or line2_name not in line_config:
        print("❌ Tên đường không hợp lệ. Chọn: 'difficulty', 'difficulty_label', 'winrate'")
        return None
    
    line1 = line_config[line1_name]
    line2 = line_config[line2_name]
    
    # Tính metrics
    metrics = calculate_metrics(
        line1['data'], line2['data'],
        line1['name'], line2['name']
    )
    
    # Print metrics
    if show_metrics:
        print("\n" + "="*70)
        print(f"METRICS: {metrics['name1']} vs {metrics['name2']}")
        print("="*70)
        print(f"MAE (Mean Absolute Error):     {metrics['mae']:.4f}")
        print(f"RMSE (Root Mean Square Error): {metrics['rmse']:.4f}")
        print(f"Max Deviation:                 {metrics['max_deviation']:.4f}")
        print(f"Pearson r:                     {metrics['pearson_r']:.4f}")
        print(f"P-value:                       {metrics['p_value']:.6f}")
        print(f"Correlation Strength:          {metrics['correlation_strength']}")
        print("="*70)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(20, 7))
    x = range(1, len(df) + 1)
    
    # Plot lines
    ax.plot(x, line1['data'], 
            marker=line1['marker'], linestyle='-', color=line1['color'],
            linewidth=2.5, label=line1['name'], markersize=5)
    
    ax.plot(x, line2['data'],
            marker=line2['marker'], linestyle='--', color=line2['color'],
            linewidth=2.5, label=line2['name'], markersize=5)
    
    # Fill area
    ax.fill_between(x, line1['data'], line2['data'], 
                     alpha=0.2, color=line2['color'])
    
    # Setup plot
    ax.set_xticks(x)
    ax.set_xticklabels(df["Level_Name"], rotation=90, fontsize=8, ha='right')
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Level Name", fontsize=12, fontweight='bold')
    ax.set_ylabel("Score (1-5)", fontsize=12, fontweight='bold')
    ax.set_ylim(0.5, 5.5)
    ax.grid(True, linestyle='--', alpha=0.4)
    
    # Title with metrics
    if show_metrics_on_plot:
        title = f"{line1['name']} vs {line2['name']}\n"
        title += f"MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f} | "
        title += f"Pearson r: {metrics['pearson_r']:.4f} (p={metrics['p_value']:.6f})"
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    else:
        ax.set_title(f"{line1['name']} vs {line2['name']}", 
                     fontsize=14, fontweight='bold')
    
    # Legend
    ax.legend(fontsize=11, loc='upper left')
    
    plt.tight_layout()
    
    # Save
    if filename is None:
        filename = f"plot_{line1_name}_vs_{line2_name}_with_metrics.png"
    
    filepath = f'difficulty_plots/{filename}'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Đã lưu: {filename}")
    plt.show()
    
    return metrics

# ================================
# HÀM PHÂN TÍCH TẤT CẢ CÁC CẶP
# ================================
def analyze_all_pairs(df, save_summary=True):
    """
    Phân tích và plot tất cả các cặp 2 đường
    """
    pairs = [
        ('difficulty', 'difficulty_label'),
        ('difficulty', 'winrate'),
        ('difficulty_label', 'winrate')
    ]
    
    all_metrics = []
    
    print("\n" + "="*80)
    print("PHÂN TÍCH TẤT CẢ CÁC CẶP 2 ĐƯỜNG")
    print("="*80)
    
    for i, (line1, line2) in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}] Analyzing {line1} vs {line2}...")
        metrics = plot_two_lines_with_metrics(
            df, line1, line2,
            show_metrics=True,
            show_metrics_on_plot=True,
            filename=f"metrics_{line1}_vs_{line2}.png"
        )
        all_metrics.append(metrics)
    
    # Tạo bảng tổng hợp
    print("\n" + "="*80)
    print("BẢNG TỔNG HỢP METRICS")
    print("="*80)
    
    metrics_df = pd.DataFrame(all_metrics)
    print("\n" + metrics_df.to_string(index=False))
    
    # Phân tích so sánh
    print("\n" + "="*80)
    print("PHÂN TÍCH SO SÁNH")
    print("="*80)
    
    print("\n1. ĐỘ LỆCH (MAE):")
    print("-" * 80)
    sorted_by_mae = sorted(all_metrics, key=lambda x: x['mae'])
    for m in sorted_by_mae:
        print(f"   {m['name1']:25s} vs {m['name2']:25s}: MAE = {m['mae']:.4f}")
    
    print(f"\n   → Cặp GẦN NHẤT (MAE thấp nhất): {sorted_by_mae[0]['name1']} vs {sorted_by_mae[0]['name2']}")
    print(f"   → Cặp XA NHẤT (MAE cao nhất): {sorted_by_mae[-1]['name1']} vs {sorted_by_mae[-1]['name2']}")
    
    print("\n2. TƯƠNG QUAN (Pearson r):")
    print("-" * 80)
    sorted_by_corr = sorted(all_metrics, key=lambda x: abs(x['pearson_r']), reverse=True)
    for m in sorted_by_corr:
        sig = "***" if m['p_value'] < 0.001 else "**" if m['p_value'] < 0.01 else "*" if m['p_value'] < 0.05 else "ns"
        print(f"   {m['name1']:25s} vs {m['name2']:25s}: r = {m['pearson_r']:7.4f} ({m['correlation_strength']:12s}) {sig}")
    
    print(f"\n   → Tương quan MẠNH NHẤT: {sorted_by_corr[0]['name1']} vs {sorted_by_corr[0]['name2']}")
    print(f"   → Tương quan YẾU NHẤT: {sorted_by_corr[-1]['name1']} vs {sorted_by_corr[-1]['name2']}")
    
    # Tạo biểu đồ so sánh
    create_comparison_chart(all_metrics)
    
    # Save summary
    if save_summary:
        metrics_df.to_csv('difficulty_plots/metrics_summary_all_pairs.csv', index=False)
        print("\n✓ Đã lưu: metrics_summary_all_pairs.csv")
    
    return all_metrics

# ================================
# HÀM TẠO BIỂU ĐỒ SO SÁNH
# ================================
def create_comparison_chart(all_metrics):
    """Tạo biểu đồ so sánh các metrics"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # Tạo grid 2x2
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    pairs_labels = [f"{m['name1']}\nvs\n{m['name2']}" for m in all_metrics]
    x_pos = range(len(all_metrics))
    
    # 1. MAE comparison
    ax1 = fig.add_subplot(gs[0, 0])
    maes = [m['mae'] for m in all_metrics]
    colors1 = ['skyblue', 'lightgreen', 'lightcoral']
    bars1 = ax1.bar(x_pos, maes, color=colors1, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(pairs_labels, fontsize=9)
    ax1.set_ylabel("MAE", fontsize=11, fontweight='bold')
    ax1.set_title("Mean Absolute Error (MAE)", fontsize=12, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    for bar, mae in zip(bars1, maes):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{mae:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 2. RMSE comparison
    ax2 = fig.add_subplot(gs[0, 1])
    rmses = [m['rmse'] for m in all_metrics]
    bars2 = ax2.bar(x_pos, rmses, color=colors1, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(pairs_labels, fontsize=9)
    ax2.set_ylabel("RMSE", fontsize=11, fontweight='bold')
    ax2.set_title("Root Mean Square Error (RMSE)", fontsize=12, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    for bar, rmse in zip(bars2, rmses):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{rmse:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 3. Pearson r comparison
    ax3 = fig.add_subplot(gs[1, 0])
    pearson_rs = [m['pearson_r'] for m in all_metrics]
    colors3 = ['skyblue' if r >= 0 else 'salmon' for r in pearson_rs]
    bars3 = ax3.bar(x_pos, pearson_rs, color=colors3, edgecolor='black', linewidth=1.5)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(pairs_labels, fontsize=9)
    ax3.set_ylabel("Pearson r", fontsize=11, fontweight='bold')
    ax3.set_title("Hệ số tương quan Pearson", fontsize=12, fontweight='bold')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.grid(axis='y', linestyle='--', alpha=0.4)
    ax3.set_ylim(-1, 1)
    for bar, r in zip(bars3, pearson_rs):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{r:.4f}', ha='center', va=va, fontweight='bold', fontsize=10)
    
    # 4. Max Deviation comparison
    ax4 = fig.add_subplot(gs[1, 1])
    max_devs = [m['max_deviation'] for m in all_metrics]
    bars4 = ax4.bar(x_pos, max_devs, color=colors1, edgecolor='black', linewidth=1.5)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(pairs_labels, fontsize=9)
    ax4.set_ylabel("Max Deviation", fontsize=11, fontweight='bold')
    ax4.set_title("Độ lệch tối đa", fontsize=12, fontweight='bold')
    ax4.grid(axis='y', linestyle='--', alpha=0.4)
    for bar, dev in zip(bars4, max_devs):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{dev:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.suptitle("So sánh Metrics cho tất cả các cặp 2 đường", 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig('difficulty_plots/metrics_comparison_all.png', dpi=300, bbox_inches='tight')
    print("\n✓ Đã lưu: metrics_comparison_all.png")
    plt.show()

# ================================
# MAIN EXECUTION
# ================================
if __name__ == "__main__":
    # Load data
    filepath ="/Users/hoangnguyen/Documents/py/ArrowPuzzle/[ArrowPuzzle] Đánh giá độ khó - Sheet2.csv"
    df = load_and_prepare_data(filepath, num_levels=50)
    
    print("\n✓ Đã load {} levels".format(len(df)))
    
    # Phân tích tất cả các cặp
    all_metrics = analyze_all_pairs(df, save_summary=True)
    
    print("\n" + "="*80)
    print("✅ HOÀN TẤT!")
    print("="*80)
    print("\nĐã tạo các file:")
    print("  1. metrics_difficulty_vs_difficulty_label.png")
    print("  2. metrics_difficulty_vs_winrate.png")
    print("  3. metrics_difficulty_label_vs_winrate.png")
    print("  4. metrics_comparison_all.png")
    print("  5. metrics_summary_all_pairs.csv")
    print("="*80)