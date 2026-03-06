#!/usr/bin/env python3
"""시간순 분할(Chronological Split) Robustness Check
- Train: 2019.01 ~ 2023.12
- Test: 2024.01 ~ 2025.12
- 기존 무작위 분할과 비교하여 모형의 시간적 일반화 성능 검증
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

FEATURES = ['전용면적', '층', '건물연령', '강남구분',
            '초등학교수', '중학교수', '고등학교수',
            'CCTV수', '백화점수', '지하철역수',
            '공원수', '도서관수', '학원수', '어린이집수',
            '기준금리', 'CD금리', '소비자물가지수', 'M2']
TARGET = '거래금액'

def evaluate(y_true, y_pred, label=""):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"  [{label}] R² = {r2:.4f}, RMSE = {rmse:,.0f}, MAE = {mae:,.0f}")
    return {'R2': round(r2, 4), 'RMSE': round(float(rmse), 0), 'MAE': round(float(mae), 0)}

def main():
    print("=" * 60)
    print("시간순 분할(Chronological Split) Robustness Check")
    print("=" * 60)
    
    df = pd.read_csv(os.path.join(DATA_DIR, 'apartment_final_v6_dong.csv'))
    print(f"\n전체 데이터: {len(df):,}건")
    print(f"기간: {df['거래년월'].min()} ~ {df['거래년월'].max()}")
    
    # Temporal split: Train ≤ 202312, Test ≥ 202401
    train_df = df[df['거래년월'] <= 202312]
    test_df = df[df['거래년월'] >= 202401]
    
    print(f"\n[시간순 분할]")
    print(f"  Train (2019.01~2023.12): {len(train_df):,}건")
    print(f"  Test  (2024.01~2025.12): {len(test_df):,}건")
    print(f"  Train 비율: {len(train_df)/len(df)*100:.1f}%")
    
    X_train = train_df[FEATURES].values
    y_train = train_df[TARGET].values
    X_test = test_df[FEATURES].values
    y_test = test_df[TARGET].values
    
    # Validation split from train (last 10% of train period)
    train_sorted = train_df.sort_values('거래년월')
    val_cutoff = int(len(train_sorted) * 0.9)
    X_tr = train_sorted[FEATURES].values[:val_cutoff]
    y_tr = train_sorted[TARGET].values[:val_cutoff]
    X_val = train_sorted[FEATURES].values[val_cutoff:]
    y_val = train_sorted[TARGET].values[val_cutoff:]
    print(f"  Train(학습): {len(X_tr):,}건, Val(early stop): {len(X_val):,}건")
    
    results = {}
    
    # 1. OLS
    print("\n--- OLS ---")
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    results['OLS_temporal'] = evaluate(y_test, ols.predict(X_test), "시간순")
    
    # 2. Random Forest
    print("\n--- Random Forest ---")
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5,
                               n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    results['RF_temporal'] = evaluate(y_test, rf.predict(X_test), "시간순")
    
    # 3. XGBoost
    print("\n--- XGBoost ---")
    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=FEATURES)
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
                      verbose_eval=200)
    
    y_pred = model.predict(dtest)
    results['XGB_temporal'] = evaluate(y_test, y_pred, "시간순")
    print(f"  Best iteration: {model.best_iteration}")
    
    # === 기존 무작위 분할 결과 (논문 기준) ===
    print("\n" + "=" * 60)
    print("비교: 무작위 분할 vs 시간순 분할")
    print("=" * 60)
    
    random_results = {
        'OLS': {'R2': 0.608, 'RMSE': 49426, 'MAE': 31768},
        'RF':  {'R2': 0.958, 'RMSE': 16115, 'MAE': 8063},
        'XGB': {'R2': 0.968, 'RMSE': 14221, 'MAE': 7422},
    }
    
    print(f"\n{'모형':<12} {'무작위 R²':>10} {'시간순 R²':>10} {'차이':>8} {'무작위 RMSE':>12} {'시간순 RMSE':>12}")
    print("-" * 64)
    for name, rkey in [('OLS', 'OLS_temporal'), ('RF', 'RF_temporal'), ('XGBoost', 'XGB_temporal')]:
        rr = random_results[name.replace('oost','')]
        tr = results[rkey]
        diff = tr['R2'] - rr['R2']
        print(f"{name:<12} {rr['R2']:>10.4f} {tr['R2']:>10.4f} {diff:>+8.4f} {rr['RMSE']:>12,.0f} {tr['RMSE']:>12,.0f}")
    
    # Save
    output = {
        'temporal_split': {
            'train_period': '2019.01-2023.12',
            'test_period': '2024.01-2025.12',
            'train_count': len(train_df),
            'test_count': len(test_df),
        },
        'results': results,
        'random_split_reference': random_results
    }
    
    outpath = os.path.join(RESULTS_DIR, 'temporal_split_results.json')
    with open(outpath, 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {outpath}")

if __name__ == '__main__':
    main()
