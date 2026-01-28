import pandas as pd
import numpy as np
import pickle
from ensemble_trainning import *

def infer_from_csv(model_path, input_csv_path, output_csv_path):
    """
    Inference using saved ensemble model on a new CSV with 'difficulty' column.
    Handles NaN in lagged features by filling with current difficulty.
    """
    # Load model
    with open(model_path, 'rb') as f:
        ensemble = pickle.load(f)
    
    # Read input data
    df = pd.read_csv(input_csv_path)
    print(f"Đã đọc {len(df)} mẫu từ {input_csv_path}")
    
    if 'difficulty' not in df.columns:
        raise ValueError("File đầu vào phải có cột 'difficulty'")
    
    # Ensure difficulty is numeric
    df['difficulty'] = pd.to_numeric(df['difficulty'], errors='coerce')
    
    # Create lagged features
    df['d'] = df['difficulty']
    df['d-1'] = df['difficulty'].shift(1)
    df['d-2'] = df['difficulty'].shift(2)
    df['d-3'] = df['difficulty'].shift(3)
    
    # Rolling window features
    df['d_avg_3'] = df['difficulty'].rolling(window=3, min_periods=1).mean()
    df['d_std_3'] = df['difficulty'].rolling(window=3, min_periods=1).std().fillna(0)
    df['d_max_3'] = df['difficulty'].rolling(window=3, min_periods=1).max()
    df['d_min_3'] = df['difficulty'].rolling(window=3, min_periods=1).min()
    df['d_change_1'] = df['difficulty'].diff().fillna(0)
    df['d_x_d1'] = df['d'] * df['d-1']
    df['d_squared'] = df['d'] ** 2
    
    # 🔥 Fill NaN in lagged features with current difficulty (reasonable assumption for start)
    for col in ['d-1', 'd-2', 'd-3', 'd_min_3', 'd_max_3', 'd_avg_3', 'd_x_d1']:
        df[col] = df[col].fillna(df['d'])
    
    # Feature columns must match training
    feature_cols = ['d', 'd-1', 'd-2', 'd-3', 
                    'd_avg_3', 'd_std_3', 'd_max_3', 'd_min_3',
                    'd_change_1', 'd_x_d1', 'd_squared']
    
    # Final check: ensure no NaN remains
    if df[feature_cols].isnull().any().any():
        print("⚠️ Cảnh báo: Vẫn còn NaN sau khi fill. Đang thay bằng 0...")
        df[feature_cols] = df[feature_cols].fillna(0)
    
    X = df[feature_cols].values
    
    # Predict
    y_pred = ensemble.predict(X)
    y_pred_rounded = np.round(y_pred).astype(int)
    
    # Add predictions to DataFrame
    df['predicted_stars'] = y_pred_rounded
    df['predicted_continuous'] = y_pred
    
    # Save result
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ Đã lưu kết quả dự đoán tại: {output_csv_path}")
    print(f"Ví dụ dự đoán đầu tiên: {y_pred_rounded[:5].tolist()}")


# =====================================================================
# Cấu hình và chạy inference
# =====================================================================

if __name__ == "__main__":
    MODEL_PATH = r"D:\py\ArrowPuzzle\difficulty_plots\ensemble_model.pkl"
    INPUT_CSV = r"D:\py\ArrowPuzzle\mem-sheet2.csv" 
    OUTPUT_CSV = r"D:\py\ArrowPuzzle\difficulty_plots\inference_output.csv"
    
    infer_from_csv(MODEL_PATH, INPUT_CSV, OUTPUT_CSV)