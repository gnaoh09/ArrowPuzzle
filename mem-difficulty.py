import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def calculate_memory_coefficients(csv_file_path):
    """
    Tính toán các hệ số alpha, beta, gamma cho mô hình:
    Win Rate (5-1 sao) = c + α(d-1) + β(d-2) + γ(d-3)
    
    Tham số:
        csv_file_path: Đường dẫn đến file CSV chứa cột 'difficulty' và 'Win Rate'
    
    Trả về:
        dict chứa các hệ số và độ chính xác
    """
    
    # Đọc dữ liệu
    df = pd.read_csv(csv_file_path)
    
    # Xử lý Win Rate (loại bỏ dấu % nếu có)
    if df['Win Rate'].dtype == 'object':
        df['Win Rate'] = df['Win Rate'].str.rstrip('%').astype('float')
    
    # Chuyển Win Rate về thang 5 sao NGƯỢC
    # 5 sao = 0-20% (tệ), 1 sao = 80-100% (tốt)
    df['Win_Rate_5Star'] = pd.cut(df['Win Rate'], 
                                    bins=[0, 20, 40, 60, 80, 100], 
                                    labels=[5, 4, 3, 2, 1],
                                    include_lowest=True).astype(int)
    
    # Tạo features: độ khó của 3 level trước
    df['difficulty-1'] = df['difficulty'].shift(0)  # Level trước 1 bước
    df['difficulty-2'] = df['difficulty'].shift(1)  # Level trước 2 bước
    df['difficulty-3'] = df['difficulty'].shift(2)  # Level trước 3 bước
    df['difficulty-4'] = df['difficulty'].shift(3)  # Level trước 4 bước
    
    # Loại bỏ các hàng không có đủ dữ liệu (3 hàng đầu)
    df_model = df.dropna()
    
    # Chuẩn bị dữ liệu
    X = df_model[['difficulty-1', 'difficulty-2', 'difficulty-3','difficulty-4']].values
    y = df_model['Win_Rate_5Star'].values
    
    # Huấn luyện mô hình hồi quy tuyến tính với intercept
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    
    # Lấy các hệ số
    intercept = model.intercept_
    alpha, beta, gamma, delta = model.coef_
    
    # Dự đoán và đánh giá
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    results_df = df_model.copy()
    results_df.drop(columns=['difficulty-1','difficulty-2','difficulty-3','difficulty-4'], inplace=True)
    results_df.to_csv('D:\\py\\ArrowPuzzle\\mem-sheet2-cautruc-results.csv', index=False)
    # Trả về kết quả
    return {
        'intercept': intercept,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'delta': delta,
        'r2_score': r2,
        'rmse': rmse,
        'mae': mae,
        'formula': f"Win Rate (sao) = {intercept:.4f} + {alpha:.4f}×(d-1) + {beta:.4f}×(d-2) + {gamma:.4f}×(d-3)"
    }


# =====================================================================
# SỬ DỤNG
# =====================================================================

if __name__ == "__main__":
    # Đường dẫn file CSV
    csv_path = 'D:\\py\\ArrowPuzzle\\mem-sheet2-cautruc.csv'
    
    # Tính toán các hệ số
    result = calculate_memory_coefficients(csv_path)
    
    # In kết quả
    print("="*70)
    print("KẾT QUẢ TÍNH TOÁN HỆ SỐ ALPHA, BETA, GAMMA")
    print("="*70)
    print("\nQuy ước thang 5 sao:")
    print("  5 sao →   0-20%  (tệ nhất)")
    print("  4 sao →  20-40%")
    print("  3 sao →  40-60%")
    print("  2 sao →  60-80%")
    print("  1 sao →  80-100% (tốt nhất)")
    print("\n" + "-"*70)
    print("\nCÔNG THỨC:")
    print(f"  {result['formula']}")
    print("\n" + "-"*70)
    print("\nCÁC HỆ SỐ:")
    print(f"  c (intercept) = {result['intercept']:.6f}")
    print(f"  α (alpha)     = {result['alpha']:.6f}")
    print(f"  β (beta)      = {result['beta']:.6f}")
    print(f"  γ (gamma)     = {result['gamma']:.6f}")
    print(f"  δ (delta)     = {result['delta']:.6f}")
    print("\n" + "-"*70)
    print("\nĐỘ CHÍNH XÁC:")
    print(f"  R² Score = {result['r2_score']:.4f}")
    print(f"  RMSE     = {result['rmse']:.4f} sao")
    print(f"  MAE      = {result['mae']:.4f} sao")
    print("\n" + "-"*70)
    print("\nGIẢI THÍCH:")
    if result['r2_score'] >= 0.7:
        print("  ✅ Mô hình rất tốt (R² ≥ 0.7)")
    elif result['r2_score'] >= 0.5:
        print("  ✅ Mô hình khá tốt (R² ≥ 0.5)")
    elif result['r2_score'] >= 0.3:
        print("  ⚠️  Mô hình trung bình (R² ≥ 0.3)")
    else:
        print("  ❌ Mô hình yếu (R² < 0.3)")
    
    print(f"  → Mô hình giải thích được {result['r2_score']*100:.2f}% phương sai")
    print(f"  → {100 - result['r2_score']*100:.2f}% còn lại do các yếu tố khác")
    print("="*70)
    
    # Lưu kết quả ra file
    output_file = 'D:\\py\\ArrowPuzzle\\coefficients_result_cautruc.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("KẾT QUẢ TÍNH TOÁN HỆ SỐ\n")
        f.write("="*70 + "\n\n")
        f.write(f"Công thức:\n{result['formula']}\n\n")
        f.write(f"intercept (c) = {result['intercept']:.6f}\n")
        f.write(f"alpha (α)     = {result['alpha']:.6f}\n")
        f.write(f"beta (β)      = {result['beta']:.6f}\n")
        f.write(f"gamma (γ)     = {result['gamma']:.6f}\n")
        f.write(f"delta (δ)     = {result['delta']:.6f}\n\n")
        f.write(f"R² Score = {result['r2_score']:.4f}\n")
        f.write(f"RMSE     = {result['rmse']:.4f} sao\n")
        f.write(f"MAE      = {result['mae']:.4f} sao\n")
    
    
    print(f"\n✓ Đã lưu kết quả tại: {output_file}")