#!/usr/bin/env python3
"""모델링 v5: OLS + RF + XGBoost + SHAP (2019-2025, CCTV 포함)"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots_v5')
os.makedirs(PLOTS_DIR, exist_ok=True)

FEATURES = ['전용면적', '층', '건물연령', '강남구분',
            '초등학교수', '중학교수', '고등학교수',
            'CCTV수', '백화점수', '지하철역수',
            '기준금리', 'CD금리', '소비자물가지수', 'M2']
TARGET = '거래금액'

def main():
    # Load
    print("=== 데이터 로드 ===")
    df = pd.read_csv(os.path.join(DATA_DIR, 'apartment_final_v5.csv'))
    X = df[FEATURES].values
    y = df[TARGET].values
    print(f"  {len(df):,}건, {len(FEATURES)}개 독립변수")
    
    # VIF
    print("\n=== VIF ===")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    vif_df = pd.DataFrame({'변수': FEATURES})
    vif_df['VIF'] = [variance_inflation_factor(X_scaled, i) for i in range(len(FEATURES))]
    print(vif_df.sort_values('VIF', ascending=False).to_string(index=False))
    
    # Split: 70/10/20
    print("\n=== Train/Val/Test Split ===")
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=42)
    print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    
    results = {}
    
    # 1. OLS
    print("\n=== OLS ===")
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    y_pred_ols = ols.predict(X_test)
    r2_ols = r2_score(y_test, y_pred_ols)
    rmse_ols = np.sqrt(mean_squared_error(y_test, y_pred_ols))
    mae_ols = mean_absolute_error(y_test, y_pred_ols)
    print(f"  R² = {r2_ols:.4f}, RMSE = {rmse_ols:,.0f}, MAE = {mae_ols:,.0f}")
    results['OLS'] = {'R2': r2_ols, 'RMSE': float(rmse_ols), 'MAE': float(mae_ols)}
    
    # OLS coefficients
    coef_df = pd.DataFrame({'변수': FEATURES, '계수': ols.coef_})
    print(coef_df.to_string(index=False))
    
    # 2. Random Forest
    print("\n=== Random Forest ===")
    rf = RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_leaf=5,
                               n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    print(f"  R² = {r2_rf:.4f}, RMSE = {rmse_rf:,.0f}, MAE = {mae_rf:,.0f}")
    results['RF'] = {'R2': r2_rf, 'RMSE': float(rmse_rf), 'MAE': float(mae_rf)}
    
    # 3. XGBoost
    print("\n=== XGBoost ===")
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURES)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURES)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=FEATURES)
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'tree_method': 'hist',
        'seed': 42,
    }
    
    model = xgb.train(params, dtrain, num_boost_round=2000,
                      evals=[(dval, 'val')], early_stopping_rounds=50,
                      verbose_eval=100)
    
    y_pred_xgb = model.predict(dtest)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    print(f"  R² = {r2_xgb:.4f}, RMSE = {rmse_xgb:,.0f}, MAE = {mae_xgb:,.0f}")
    print(f"  Best iteration: {model.best_iteration}")
    results['XGBoost'] = {'R2': r2_xgb, 'RMSE': float(rmse_xgb), 'MAE': float(mae_xgb),
                          'best_iter': model.best_iteration}
    
    # Cross-validation
    print("\n=== XGBoost 5-fold CV ===")
    dtotal = xgb.DMatrix(X, label=y, feature_names=FEATURES)
    cv_results = xgb.cv(params, dtotal, num_boost_round=model.best_iteration,
                        nfold=5, seed=42, metrics='rmse')
    cv_rmse = cv_results['test-rmse-mean'].iloc[-1]
    cv_rmse_std = cv_results['test-rmse-std'].iloc[-1]
    # Calculate R² from CV RMSE
    y_var = np.var(y)
    cv_r2 = 1 - (cv_rmse**2) / y_var
    print(f"  CV RMSE = {cv_rmse:,.0f} ± {cv_rmse_std:,.0f}")
    print(f"  CV R² ≈ {cv_r2:.4f}")
    results['XGBoost']['CV_RMSE'] = float(cv_rmse)
    results['XGBoost']['CV_R2'] = float(cv_r2)
    
    # 4. SHAP
    print("\n=== SHAP Analysis ===")
    explainer = shap.TreeExplainer(model)
    
    # Sample for SHAP (speed)
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_test), min(5000, len(X_test)), replace=False)
    X_sample = pd.DataFrame(X_test[sample_idx], columns=FEATURES)
    shap_values = explainer.shap_values(X_sample)
    
    # Feature importance (mean |SHAP|)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    total_shap = mean_abs_shap.sum()
    shap_df = pd.DataFrame({
        '변수': FEATURES,
        'mean_abs_shap': mean_abs_shap,
        '기여도(%)': (mean_abs_shap / total_shap * 100).round(1)
    }).sort_values('mean_abs_shap', ascending=False)
    
    print(shap_df.to_string(index=False))
    results['SHAP'] = {row['변수']: {'mean_abs': float(row['mean_abs_shap']),
                                      'pct': float(row['기여도(%)'])}
                       for _, row in shap_df.iterrows()}
    
    # 5. Plots
    print("\n=== Generating Plots ===")
    
    # SHAP summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fig4_shap_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  fig4_shap_summary.png")
    
    # SHAP bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fig5_shap_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  fig5_shap_bar.png")
    
    # Top feature dependence plots
    top_features = shap_df['변수'].head(6).tolist()
    for i, feat in enumerate(top_features):
        plt.figure(figsize=(8, 6))
        feat_idx = FEATURES.index(feat)
        shap.dependence_plot(feat_idx, shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'fig{6+i}_dep_{feat}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  fig{6+i}_dep_{feat}.png")
    
    # Force plot for sample
    plt.figure(figsize=(20, 4))
    shap.force_plot(explainer.expected_value, shap_values[0], X_sample.iloc[0],
                    matplotlib=True, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fig12_force_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  fig12_force_plot.png")
    
    # Gangnam vs Non-Gangnam SHAP
    gangnam_mask = X_sample['강남구분'] == 1
    if gangnam_mask.sum() > 0:
        gangnam_shap = np.abs(shap_values[gangnam_mask.values]).mean(axis=0)
        non_gangnam_shap = np.abs(shap_values[~gangnam_mask.values]).mean(axis=0)
        
        compare_df = pd.DataFrame({
            '변수': FEATURES,
            '강남3구': gangnam_shap.round(1),
            '비강남': non_gangnam_shap.round(1)
        }).sort_values('강남3구', ascending=False)
        print("\n강남 vs 비강남 SHAP:")
        print(compare_df.to_string(index=False))
        results['gangnam_comparison'] = {
            row['변수']: {'gangnam': float(row['강남3구']), 'non_gangnam': float(row['비강남'])}
            for _, row in compare_df.iterrows()
        }
    
    # Save results
    with open(os.path.join(RESULTS_DIR, 'modeling_v5_results.json'), 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: modeling_v5_results.json")
    
    print("\n=== 모델 비교 요약 ===")
    print(f"  OLS     : R² = {r2_ols:.4f}")
    print(f"  RF      : R² = {r2_rf:.4f}")
    print(f"  XGBoost : R² = {r2_xgb:.4f} (CV R² ≈ {cv_r2:.4f})")

if __name__ == '__main__':
    main()
