import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import pickle

class EnsembleModel:
    """
    Ensemble model kết hợp Ridge + Gradient Boosting + Random Forest
    """
    
    def __init__(self, weights=None):
        """
        Khởi tạo ensemble model
        
        Tham số:
            weights: Trọng số cho từng model [ridge_weight, gb_weight, rf_weight]
                    Nếu None, sẽ tự động tối ưu bằng cross-validation
        """
        # Khởi tạo các base models
        self.ridge = Ridge(alpha=1.0)
        self.gradient_boosting = GradientBoostingRegressor(
            n_estimators=100, 
            max_depth=3, 
            learning_rate=0.1, 
            random_state=42
        )
        self.random_forest = RandomForestRegressor(
            n_estimators=100, 
            max_depth=5, 
            min_samples_split=5, 
            random_state=42
        )
        
        # Trọng số mặc định (bằng nhau)
        self.weights = weights if weights is not None else [1/3, 1/3, 1/3]
        
        # Lưu feature columns
        self.feature_cols = None
        self.is_fitted = False
    
    def fit(self, X, y, feature_cols=None):
        """
        Train ensemble model
        """
        self.feature_cols = feature_cols
        
        # Train từng model
        print("Training Ridge Regression...")
        self.ridge.fit(X, y)
        
        print("Training Gradient Boosting...")
        self.gradient_boosting.fit(X, y)
        
        print("Training Random Forest...")
        self.random_forest.fit(X, y)
        
        self.is_fitted = True
        print("✓ Đã train xong cả 3 models")
    
    def predict(self, X):
        """
        Dự đoán bằng weighted average của 3 models
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train! Hãy gọi fit() trước.")
        
        # Dự đoán từ từng model
        pred_ridge = self.ridge.predict(X)
        pred_gb = self.gradient_boosting.predict(X)
        pred_rf = self.random_forest.predict(X)
        
        # Weighted average
        ensemble_pred = (
            self.weights[0] * pred_ridge +
            self.weights[1] * pred_gb +
            self.weights[2] * pred_rf
        )
        
        # Clip vào range 1-5
        ensemble_pred = np.clip(ensemble_pred, 1, 5)
        
        return ensemble_pred
    
    def get_individual_predictions(self, X):
        """
        Lấy dự đoán riêng lẻ từ từng model
        """
        return {
            'ridge': np.clip(self.ridge.predict(X), 1, 5),
            'gradient_boosting': np.clip(self.gradient_boosting.predict(X), 1, 5),
            'random_forest': np.clip(self.random_forest.predict(X), 1, 5)
        }
    
    def optimize_weights(self, X_val, y_val):
        """
        Tối ưu trọng số bằng grid search trên validation set
        """
        print("\nTối ưu hóa trọng số ensemble...")
        
        # Lấy predictions từ từng model
        preds = self.get_individual_predictions(X_val)
        
        best_score = -np.inf
        best_weights = self.weights
        
        # Grid search
        for w1 in np.linspace(0, 1, 11):
            for w2 in np.linspace(0, 1-w1, 11):
                w3 = 1 - w1 - w2
                
                # Weighted prediction
                ensemble_pred = (
                    w1 * preds['ridge'] +
                    w2 * preds['gradient_boosting'] +
                    w3 * preds['random_forest']
                )
                
                # Tính R²
                score = r2_score(y_val, ensemble_pred)
                
                if score > best_score:
                    best_score = score
                    best_weights = [w1, w2, w3]
        
        self.weights = best_weights
        print(f"✓ Trọng số tối ưu: Ridge={best_weights[0]:.3f}, GB={best_weights[1]:.3f}, RF={best_weights[2]:.3f}")
        print(f"  R² trên validation: {best_score:.4f}")
        
        return best_weights


def prepare_data(csv_file_path):
    """
    Chuẩn bị dữ liệu từ CSV
    """
    print("="*80)
    print("CHUẨN BỊ DỮ LIỆU")
    print("="*80)
    
    df = pd.read_csv(csv_file_path)
    
    print(f"\nFile đầu vào: {csv_file_path}")
    print(f"Số level: {len(df)}")
    print(f"Các cột: {df.columns.tolist()}")
    
    # Xử lý Win Rate
    if 'Win Rate' in df.columns:
        if df['Win Rate'].dtype == 'object':
            df['Win Rate'] = df['Win Rate'].str.rstrip('%').astype('float')
    
    # Chuyển sang thang 5 sao nếu chưa có
    if 'Win_Rate_5Star' not in df.columns:
        df['Win_Rate_5Star'] = pd.cut(df['Win Rate'], 
                                        bins=[0, 20, 40, 60, 80, 100], 
                                        labels=[5, 4, 3, 2, 1],
                                        include_lowest=True).astype(int)
    
    print(f"\nPhân bố Win Rate (5 sao):")
    print(df['Win_Rate_5Star'].value_counts().sort_index())
    
    # Kiểm tra xem có difficulty dự đoán ban đầu không
    has_predicted_difficulty = 'difficulty' in df.columns
    
    if has_predicted_difficulty:
        print(f"\nPhân bố Difficulty dự đoán ban đầu:")
        print(df['difficulty'].value_counts().sort_index())
    
    # Tạo features
    print("\nTạo features...")
    
    # Basic features
    if has_predicted_difficulty:
        df['d'] = df['difficulty']
        df['d-1'] = df['difficulty'].shift(1)
        df['d-2'] = df['difficulty'].shift(2)
        df['d-3'] = df['difficulty'].shift(3)
    
    # Advanced features
    if has_predicted_difficulty:
        df['d_avg_3'] = df['difficulty'].rolling(window=3, min_periods=1).mean()
        df['d_std_3'] = df['difficulty'].rolling(window=3, min_periods=1).std().fillna(0)
        df['d_max_3'] = df['difficulty'].rolling(window=3, min_periods=1).max()
        df['d_min_3'] = df['difficulty'].rolling(window=3, min_periods=1).min()
        df['d_change_1'] = df['difficulty'].diff().fillna(0)
        df['d_x_d1'] = df['d'] * df['d-1'].fillna(0)
        df['d_squared'] = df['d'] ** 2
    
    # Loại bỏ hàng thiếu dữ liệu
    df_model = df.dropna()
    
    print(f"Sau khi tạo features: {len(df_model)} samples, {df_model.shape[1]} features")
    
    return df_model


def train_ensemble_with_cv(df_model, n_splits=5):
    """
    Train ensemble với K-Fold Cross-Validation
    """
    print("\n" + "="*80)
    print(f"ENSEMBLE TRAINING VỚI {n_splits}-FOLD CROSS-VALIDATION")
    print("="*80)
    
    # Chuẩn bị features
    feature_cols = ['d', 'd-1', 'd-2', 'd-3', 
                   'd_avg_3', 'd_std_3', 'd_max_3', 'd_min_3',
                   'd_change_1', 'd_x_d1', 'd_squared']
    
    X = df_model[feature_cols].values
    y = df_model['Win_Rate_5Star'].values
    
    # K-Fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Lưu kết quả
    cv_results = {
        'ridge': [],
        'gradient_boosting': [],
        'random_forest': [],
        'ensemble_equal': [],
        'ensemble_optimized': []
    }
    
    print(f"\nBắt đầu {n_splits}-fold cross-validation...\n")
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"Fold {fold_idx}/{n_splits}:")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train ensemble
        ensemble = EnsembleModel(weights=[1/3, 1/3, 1/3])
        ensemble.fit(X_train, y_train, feature_cols=feature_cols)
        
        # Dự đoán từng model
        preds = ensemble.get_individual_predictions(X_val)
        
        # Dự đoán ensemble (equal weights)
        pred_ensemble_equal = ensemble.predict(X_val)
        
        # Optimize weights trên validation set
        ensemble.optimize_weights(X_val, y_val)
        pred_ensemble_optimized = ensemble.predict(X_val)
        
        # Tính metrics cho từng model
        for model_name, pred in preds.items():
            r2 = r2_score(y_val, pred)
            cv_results[model_name].append(r2)
        
        # Metrics cho ensemble
        r2_equal = r2_score(y_val, pred_ensemble_equal)
        r2_optimized = r2_score(y_val, pred_ensemble_optimized)
        
        cv_results['ensemble_equal'].append(r2_equal)
        cv_results['ensemble_optimized'].append(r2_optimized)
        
        print(f"  Ridge R²: {cv_results['ridge'][-1]:.4f}")
        print(f"  GB R²:    {cv_results['gradient_boosting'][-1]:.4f}")
        print(f"  RF R²:    {cv_results['random_forest'][-1]:.4f}")
        print(f"  Ensemble (equal weights) R²: {r2_equal:.4f}")
        print(f"  Ensemble (optimized) R²:     {r2_optimized:.4f}")
        print()
    
    # Tính trung bình
    print("="*80)
    print("KẾT QUẢ TRUNG BÌNH QUA CÁC FOLD")
    print("="*80)
    
    for model_name, scores in cv_results.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"{model_name:25s}: R² = {mean_score:.4f} ± {std_score:.4f}")
    
    return cv_results


def train_final_ensemble(df_model):
    """
    Train ensemble cuối cùng trên toàn bộ dữ liệu
    """
    print("\n" + "="*80)
    print("TRAINING ENSEMBLE CUỐI CÙNG (TOÀN BỘ DỮ LIỆU)")
    print("="*80)
    
    feature_cols = ['d', 'd-1', 'd-2', 'd-3', 
                   'd_avg_3', 'd_std_3', 'd_max_3', 'd_min_3',
                   'd_change_1', 'd_x_d1', 'd_squared']
    
    X = df_model[feature_cols].values
    y = df_model['Win_Rate_5Star'].values
    
    # Train ensemble với equal weights
    print("\nTrain ensemble với trọng số bằng nhau...")
    ensemble = EnsembleModel(weights=[1/3, 1/3, 1/3])
    ensemble.fit(X, y, feature_cols=feature_cols)
    
    # Dự đoán
    preds_individual = ensemble.get_individual_predictions(X)
    pred_ensemble = ensemble.predict(X)
    pred_ensemble_rounded = np.round(pred_ensemble).astype(int)
    
    # Metrics
    print("\n" + "-"*80)
    print("METRICS CHO TỪNG MODEL:")
    print("-"*80)
    
    for model_name, pred in preds_individual.items():
        pred_rounded = np.round(pred).astype(int)
        r2 = r2_score(y, pred)
        mae = mean_absolute_error(y, pred)
        accuracy = accuracy_score(y, pred_rounded)
        accuracy_1 = np.mean(np.abs(y - pred_rounded) <= 1)
        
        print(f"\n{model_name.upper()}:")
        print(f"  R²   = {r2:.4f}")
        print(f"  MAE  = {mae:.4f} sao")
        print(f"  Accuracy (exact): {accuracy:.2%}")
        print(f"  Accuracy (±1): {accuracy_1:.2%}")
    
    # Ensemble metrics
    r2 = r2_score(y, pred_ensemble)
    mae = mean_absolute_error(y, pred_ensemble)
    rmse = np.sqrt(mean_squared_error(y, pred_ensemble))
    accuracy = accuracy_score(y, pred_ensemble_rounded)
    accuracy_1 = np.mean(np.abs(y - pred_ensemble_rounded) <= 1)
    pearson_corr, p_value = pearsonr(y, pred_ensemble_rounded)
    
    print("\n" + "-"*80)
    print("ENSEMBLE (Equal Weights):")
    print("-"*80)
    print(f"  Weights: Ridge={ensemble.weights[0]:.3f}, GB={ensemble.weights[1]:.3f}, RF={ensemble.weights[2]:.3f}")
    print(f"  R²   = {r2:.4f}")
    print(f"  MAE  = {mae:.4f} sao")
    print(f"  RMSE = {rmse:.4f} sao")
    print(f"  Accuracy (exact): {accuracy:.2%}")
    print(f"  Accuracy (±1): {accuracy_1:.2%}")
    print(f"  Pearson r: {pearson_corr:.4f} (p={p_value:.6f})")
    
    # Feature importance từ Random Forest
    if hasattr(ensemble.random_forest, 'feature_importances_'):
        print("\n" + "-"*80)
        print("FEATURE IMPORTANCE (Random Forest):")
        print("-"*80)
        importances = ensemble.random_forest.feature_importances_
        for i, col in enumerate(feature_cols):
            print(f"  {col:15s}: {importances[i]:.4f}")
    
    return {
        'ensemble': ensemble,
        'feature_cols': feature_cols,
        'predictions_individual': preds_individual,
        'prediction_ensemble': pred_ensemble,
        'prediction_rounded': pred_ensemble_rounded,
        'y_true': y,
        'metrics': {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'accuracy': accuracy,
            'accuracy_1': accuracy_1,
            'pearson_corr': pearson_corr,
            'p_value': p_value
        },
        'df_model': df_model
    }


def plot_results(results):
    """
    Vẽ biểu đồ kết quả
    """
    print("\n" + "="*80)
    print("VẼ BIỂU ĐỒ KẾT QUẢ")
    print("="*80)
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    
    y_true = results['y_true']
    y_pred = results['prediction_rounded']
    preds_individual = results['predictions_individual']
    
    # ---- 1. Actual vs Predicted (Ensemble) ----
    ax1 = axes[0, 0]
    scatter = ax1.scatter(y_true, y_pred, alpha=0.6, s=100,
                         c=np.abs(y_true - y_pred), cmap='RdYlGn_r',
                         edgecolors='black', linewidth=0.5)
    ax1.plot([1, 5], [1, 5], 'r--', linewidth=2, label='Perfect')
    ax1.set_xlabel('Actual', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Predicted', fontsize=11, fontweight='bold')
    ax1.set_title(f'Ensemble: Actual vs Predicted\nR²={results["metrics"]["r2"]:.3f}', 
                  fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.5, 5.5)
    ax1.set_ylim(0.5, 5.5)
    plt.colorbar(scatter, ax=ax1)
    
    # ---- 2-4. Individual models ----
    for idx, (model_name, pred) in enumerate(preds_individual.items()):
        ax = axes[0, idx+1] if idx < 2 else axes[1, 0]
        pred_rounded = np.round(pred).astype(int)
        r2 = r2_score(y_true, pred_rounded)
        
        ax.scatter(y_true, pred_rounded, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
        ax.plot([1, 5], [1, 5], 'r--', linewidth=2)
        ax.set_xlabel('Actual', fontsize=10, fontweight='bold')
        ax.set_ylabel('Predicted', fontsize=10, fontweight='bold')
        ax.set_title(f'{model_name.title()}\nR²={r2:.3f}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, 5.5)
        ax.set_ylim(0.5, 5.5)
    
    # ---- 5. Confusion Matrix ----
    ax5 = axes[1, 1]
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])
    im = ax5.imshow(cm, cmap='Blues', aspect='auto')
    ax5.set_xticks([0, 1, 2, 3, 4])
    ax5.set_yticks([0, 1, 2, 3, 4])
    ax5.set_xticklabels([1, 2, 3, 4, 5])
    ax5.set_yticklabels([1, 2, 3, 4, 5])
    ax5.set_xlabel('Predicted', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Actual', fontsize=11, fontweight='bold')
    ax5.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    
    for i in range(5):
        for j in range(5):
            ax5.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black',
                    fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax5)
    
    # ---- 6. Time Series ----
    ax6 = axes[1, 2]
    x = range(len(y_true))
    ax6.plot(x, y_true, 'o-', linewidth=2, markersize=5, label='Actual', color='darkblue')
    ax6.plot(x, y_pred, 's--', linewidth=2, markersize=5, label='Ensemble', color='crimson', alpha=0.8)
    
    # Plot individual models
    for model_name, pred in preds_individual.items():
        pred_rounded = np.round(pred).astype(int)
        ax6.plot(x, pred_rounded, alpha=0.3, linewidth=1, linestyle=':', 
                label=model_name.split('_')[0].title())
    
    ax6.fill_between(x, y_true, y_pred, color='orange', alpha=0.2)
    ax6.set_xlabel('Level Index', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Star Rating', fontsize=11, fontweight='bold')
    ax6.set_title('Predictions over Levels', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0.5, 5.5)
    
    # ---- 7. Error Distribution ----
    ax7 = axes[2, 0]
    errors = y_true - y_pred
    ax7.hist(errors, bins=range(-5, 6), alpha=0.7, color='steelblue', edgecolor='black')
    ax7.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax7.axvline(x=np.mean(errors), color='orange', linestyle='-', linewidth=2,
                label=f'Mean={np.mean(errors):.2f}')
    ax7.set_xlabel('Error (Actual - Predicted)', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax7.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    
    # ---- 8. Model Comparison (R²) ----
    ax8 = axes[2, 1]
    models = ['Ridge', 'GB', 'RF', 'Ensemble']
    r2_scores = [
        r2_score(y_true, np.round(preds_individual['ridge']).astype(int)),
        r2_score(y_true, np.round(preds_individual['gradient_boosting']).astype(int)),
        r2_score(y_true, np.round(preds_individual['random_forest']).astype(int)),
        results['metrics']['r2']
    ]
    colors = ['steelblue', 'orange', 'green', 'crimson']
    bars = ax8.bar(models, r2_scores, color=colors, alpha=0.7, edgecolor='black')
    ax8.set_ylabel('R² Score', fontsize=11, fontweight='bold')
    ax8.set_title('Model Comparison - R²', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, r2_scores):
        ax8.text(bar.get_x() + bar.get_width()/2, val, f'{val:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ---- 9. Feature Importance ----
    ax9 = axes[2, 2]
    if hasattr(results['ensemble'].random_forest, 'feature_importances_'):
        importances = results['ensemble'].random_forest.feature_importances_
        features = results['feature_cols']
        indices = np.argsort(importances)[::-1]
        
        ax9.barh(range(len(indices)), importances[indices], color='teal', alpha=0.7, edgecolor='black')
        ax9.set_yticks(range(len(indices)))
        ax9.set_yticklabels([features[i] for i in indices], fontsize=9)
        ax9.set_xlabel('Importance', fontsize=11, fontweight='bold')
        ax9.set_title('Feature Importance (RF)', fontsize=12, fontweight='bold')
        ax9.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('D:\py\ArrowPuzzle\difficulty_plots\/ensemble_results.png', dpi=150, bbox_inches='tight')
    print("✓ Đã lưu biểu đồ tại: D:\py\ArrowPuzzle\difficulty_plots\/ensemble_results.png")


def save_results(results):
    """
    Lưu kết quả và model
    """
    print("\n" + "="*80)
    print("LƯU KẾT QUẢ")
    print("="*80)
    
    # Lưu predictions
    df_output = results['df_model'].copy()
    df_output['predicted_ensemble'] = results['prediction_rounded']
    df_output['predicted_ridge'] = np.round(results['predictions_individual']['ridge']).astype(int)
    df_output['predicted_gb'] = np.round(results['predictions_individual']['gradient_boosting']).astype(int)
    df_output['predicted_rf'] = np.round(results['predictions_individual']['random_forest']).astype(int)
    df_output['error_ensemble'] = results['y_true'] - results['prediction_rounded']
    
    output_csv = 'D:\py\ArrowPuzzle\difficulty_plots\/ensemble_predictions.csv'
    df_output.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"✓ Đã lưu predictions tại: {output_csv}")
    
    # Lưu model
    model_file = 'D:\py\ArrowPuzzle\difficulty_plots\/ensemble_model.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(results['ensemble'], f)
    print(f"✓ Đã lưu model tại: {model_file}")
    
    # Lưu báo cáo
    report_file = 'D:\py\ArrowPuzzle\difficulty_plots\/ensemble_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("BÁO CÁO ENSEMBLE MODEL\n")
        f.write("="*80 + "\n\n")
        
        f.write("ENSEMBLE: Ridge + Gradient Boosting + Random Forest\n\n")
        
        f.write("Trọng số:\n")
        f.write(f"  Ridge:             {results['ensemble'].weights[0]:.3f}\n")
        f.write(f"  Gradient Boosting: {results['ensemble'].weights[1]:.3f}\n")
        f.write(f"  Random Forest:     {results['ensemble'].weights[2]:.3f}\n\n")
        
        f.write("Metrics:\n")
        for key, val in results['metrics'].items():
            f.write(f"  {key:20s}: {val:.4f}\n")
        
        f.write("\n" + "-"*80 + "\n\n")
        
        f.write("Individual Models:\n")
        for model_name, pred in results['predictions_individual'].items():
            pred_rounded = np.round(pred).astype(int)
            r2 = r2_score(results['y_true'], pred_rounded)
            mae = mean_absolute_error(results['y_true'], pred)
            f.write(f"\n{model_name.upper()}:\n")
            f.write(f"  R²:  {r2:.4f}\n")
            f.write(f"  MAE: {mae:.4f}\n")
    
    print(f"✓ Đã lưu báo cáo tại: {report_file}")


# =====================================================================
# MAIN PROGRAM
# =====================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("ENSEMBLE TRAINING: Ridge + Gradient Boosting + Random Forest")
    print("="*80)
    
    csv_path = 'D:\py\ArrowPuzzle/mem-sheet2.csv'
    df_model = prepare_data(csv_path)
    cv_results = train_ensemble_with_cv(df_model, n_splits=2)
    results = train_final_ensemble(df_model)
    plot_results(results)
    save_results(results)
    print("\n" + "="*80)
    print("HƯỚNG DẪN SỬ DỤNG MODEL")
    print("="*80)
    
    print("\n" + "="*80)
    print("HOÀN THÀNH!")
    print("="*80)
    
    print(f"\nKết quả cuối cùng:")
    print(f"  R² = {results['metrics']['r2']:.4f}")
    print(f"  MAE = {results['metrics']['mae']:.4f} sao")
    print(f"  Accuracy (exact) = {results['metrics']['accuracy']:.2%}")
    print(f"  Accuracy (±1) = {results['metrics']['accuracy_1']:.2%}")