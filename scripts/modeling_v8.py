#!/usr/bin/env python3
"""
v8 모델링: 거리 기반 변수 + 시점 정합 + 교수 피드백 반영.

차이점 (v7 → v8):
  - Y: log(㎡당 가격) 유지 (v7 이미 반영)
  - X: 행정동 단순 개수 → 반경 거리 기반 + 연도별 시점 정합
  - X: 전용면적은 단위면적당 가격 산식에 들어가므로 설명변수에서 제외
  - 모형 분할: 전체 + 권역(강남/비강남) + 면적대(20평대/30평대) + 연도별 (옵션)

OLS 모형 (축소): VIF 관리를 위해 시설군당 최근접거리 1개 또는 반경 1개만.
Tree 모형 (풀셋): 거리+개수 병행.
"""
import os
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
import statsmodels.api as sm
import shap

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')
PLOTS = os.path.join(RESULTS, 'plots_v8')
os.makedirs(PLOTS, exist_ok=True)
OUT_JSON = os.path.join(RESULTS, 'modeling_v8_results.json')

# OLS용 축소 피처 (VIF 관리)
OLS_FEATURES = [
    '층', '건물연령', '강남구분',
    'subway_nearest_m', 'elem_school_nearest_m', 'middle_school_nearest_m',
    'library_nearest_m', 'park_nearest_m', 'mart_nearest_m',
    'department_nearest_m', 'academy_nearest_m', 'hospital_general_nearest_m',
    'childcare_count_1000m', 'cctv_count_500m',
    'park_within_1km', 'department_within_1km',
    '기준금리', '소비자물가지수', 'M2',
]

TREE_FEATURES = OLS_FEATURES + [
    'subway_count_1000m', 'elem_school_count_1000m', 'middle_school_count_1000m',
    'high_school_count_1000m', 'mart_count_1000m', 'department_count_2000m',
    'academy_count_1000m', 'library_count_1000m', 'hospital_count_1000m',
    'park_count_1000m', 'large_store_count_500m',
    'park_log1p_count_2km', 'department_log1p_count_2km', 'library_log1p_count_2km',
    'CD금리',
]


def metrics(y_true_log, y_pred_log):
    y_true = np.exp(y_true_log)
    y_pred = np.exp(y_pred_log)
    ape = np.abs(y_true - y_pred) / y_true
    return {
        'r2': float(r2_score(y_true_log, y_pred_log)),
        'mape': float(np.mean(ape)),
        'median_ape': float(np.median(ape)),
        'mae_log': float(mean_absolute_error(y_true_log, y_pred_log)),
        'n': int(len(y_true_log)),
    }


def fit_eval(X_tr, y_tr, X_te, y_te, features):
    out = {}
    # OLS
    ols = LinearRegression().fit(X_tr[OLS_FEATURES], y_tr)
    out['ols'] = metrics(y_te, ols.predict(X_te[OLS_FEATURES]))
    # RandomForest (sample for speed)
    rf = RandomForestRegressor(n_estimators=200, max_depth=14, n_jobs=-1, random_state=42)
    rf.fit(X_tr[features], y_tr)
    out['rf'] = metrics(y_te, rf.predict(X_te[features]))
    # XGBoost
    xgb = XGBRegressor(n_estimators=600, max_depth=8, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8,
                       n_jobs=-1, random_state=42, tree_method='hist')
    xgb.fit(X_tr[features], y_tr)
    out['xgb'] = metrics(y_te, xgb.predict(X_te[features]))
    return out, xgb


def main():
    print("=== v8 모델링 시작 ===")
    df = pd.read_csv(os.path.join(DATA_DIR, 'apartment_final_v8.csv'), low_memory=False)
    print(f"전체 거래: {len(df):,}")

    # 결측 처리: 거리 변수 결측은 극소수 예상
    all_feats = list(set(TREE_FEATURES))
    missing = df[all_feats].isna().sum()
    miss_cols = missing[missing > 0]
    if len(miss_cols):
        print(f"결측 컬럼: {miss_cols.to_dict()}")
    df = df.dropna(subset=all_feats + ['log㎡당가격'])
    print(f"결측 제거 후: {len(df):,}")

    results = {
        'config': {
            'target': 'log㎡당가격',
            'ols_features': OLS_FEATURES,
            'tree_features': TREE_FEATURES,
            'n': len(df),
        },
        'splits': {},
        'by_region': {},
        'by_size': {},
        'by_year': {},
    }

    # --- 1. 무작위 분할 ---
    print("\n--- 무작위 분할 ---")
    tr, te = train_test_split(df, test_size=0.2, random_state=42)
    out, xgb_model = fit_eval(tr, tr['log㎡당가격'].values, te, te['log㎡당가격'].values, TREE_FEATURES)
    results['splits']['random'] = out
    print(f"  OLS R²={out['ols']['r2']:.4f}  RF R²={out['rf']['r2']:.4f}  XGB R²={out['xgb']['r2']:.4f}  XGB MAPE={out['xgb']['mape']:.4f}")

    # --- 2. Group 분할 (단지 단위) ---
    print("\n--- Group 분할 (단지 단위) ---")
    gkf = GroupKFold(n_splits=5)
    groups = (df['구'] + '|' + df['법정동'] + '|' + df['아파트명']).values
    tr_idx, te_idx = next(gkf.split(df, df['log㎡당가격'], groups))
    tr, te = df.iloc[tr_idx], df.iloc[te_idx]
    out, _ = fit_eval(tr, tr['log㎡당가격'].values, te, te['log㎡당가격'].values, TREE_FEATURES)
    results['splits']['group'] = out
    print(f"  OLS R²={out['ols']['r2']:.4f}  RF R²={out['rf']['r2']:.4f}  XGB R²={out['xgb']['r2']:.4f}")

    # --- 3. 시간순 분할 (2019-2023 train, 2024-2025 test) ---
    print("\n--- 시간순 분할 ---")
    tr = df[df['거래년도'] <= 2023]; te = df[df['거래년도'] >= 2024]
    out, _ = fit_eval(tr, tr['log㎡당가격'].values, te, te['log㎡당가격'].values, TREE_FEATURES)
    results['splits']['temporal'] = out
    print(f"  OLS R²={out['ols']['r2']:.4f}  RF R²={out['rf']['r2']:.4f}  XGB R²={out['xgb']['r2']:.4f}")

    # --- 4. 권역별 (교수 피드백 제안2) ---
    print("\n--- 권역별 ---")
    for region, mask in [('강남3구', df['강남구분'] == 1), ('비강남', df['강남구분'] == 0)]:
        sub = df[mask]
        tr, te = train_test_split(sub, test_size=0.2, random_state=42)
        out, _ = fit_eval(tr, tr['log㎡당가격'].values, te, te['log㎡당가격'].values, TREE_FEATURES)
        results['by_region'][region] = out
        print(f"  {region}: n={len(sub):,} XGB R²={out['xgb']['r2']:.4f} MAPE={out['xgb']['mape']:.4f}")

    # --- 5. 면적대별 (교수 피드백: 20평대/30평대) ---
    print("\n--- 면적대별 ---")
    size_bins = [
        ('20평대 (60-85㎡)', (df['전용면적'] >= 60) & (df['전용면적'] < 85)),
        ('30평대 (85-112㎡)', (df['전용면적'] >= 85) & (df['전용면적'] < 112)),
        ('소형 (<60㎡)', df['전용면적'] < 60),
        ('대형 (≥112㎡)', df['전용면적'] >= 112),
    ]
    for name, mask in size_bins:
        sub = df[mask]
        if len(sub) < 1000:
            continue
        tr, te = train_test_split(sub, test_size=0.2, random_state=42)
        out, _ = fit_eval(tr, tr['log㎡당가격'].values, te, te['log㎡당가격'].values, TREE_FEATURES)
        results['by_size'][name] = out
        print(f"  {name}: n={len(sub):,} XGB R²={out['xgb']['r2']:.4f} MAPE={out['xgb']['mape']:.4f}")

    # --- 6. 연도별 (교수 피드백 제안1) ---
    print("\n--- 연도별 ---")
    for year in sorted(df['거래년도'].unique()):
        sub = df[df['거래년도'] == year]
        if len(sub) < 1000:
            continue
        tr, te = train_test_split(sub, test_size=0.2, random_state=42)
        out, _ = fit_eval(tr, tr['log㎡당가격'].values, te, te['log㎡당가격'].values, TREE_FEATURES)
        results['by_year'][str(year)] = out
        print(f"  {year}: n={len(sub):,} XGB R²={out['xgb']['r2']:.4f} MAPE={out['xgb']['mape']:.4f}")

    # --- 7. SHAP 상위 변수 (무작위 분할 XGB 모델 기준) ---
    print("\n--- SHAP 계산 (5,000건 샘플) ---")
    tr, te = train_test_split(df, test_size=0.2, random_state=42)
    xgb = XGBRegressor(n_estimators=600, max_depth=8, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8,
                       n_jobs=-1, random_state=42, tree_method='hist')
    xgb.fit(tr[TREE_FEATURES], tr['log㎡당가격'].values)
    sample = te.sample(min(5000, len(te)), random_state=42)
    exp = shap.TreeExplainer(xgb)
    sv = exp.shap_values(sample[TREE_FEATURES])
    shap_abs = np.abs(sv).mean(axis=0)
    shap_df = pd.DataFrame({'feature': TREE_FEATURES, 'mean_abs_shap': shap_abs})
    shap_df = shap_df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    shap_df['pct'] = 100 * shap_df['mean_abs_shap'] / shap_df['mean_abs_shap'].sum()
    shap_df.to_csv(os.path.join(RESULTS, 'shap_importance_v8.csv'), index=False)
    print(shap_df.head(15).to_string(index=False))

    results['shap_top15'] = shap_df.head(15).to_dict('records')

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n저장: {OUT_JSON}")


if __name__ == '__main__':
    main()
