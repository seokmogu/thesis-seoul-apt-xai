#!/usr/bin/env python3
"""
연도 × 권역 교차 분석 (v8).

7개년(2019-2025) × 3권역(전체/강남3구/비강남) = 21 서브모델.
각 서브모델: XGB R², MAPE, SHAP Top 10
전용면적은 종속변수 log(거래금액/전용면적) 산식에 사용되므로 설명변수에서 제외한다.

출력:
  results/v8_year_region_results.json
  results/v8_year_region_summary.csv
  results/v8_year_region_shap_top.csv
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import shap

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')

# v8 최종 피처셋 — 거리 + 시점 정합
DISTANCE_FEATS = [
    'subway_nearest_m', 'subway_count_1000m',
    'elem_school_nearest_m', 'elem_school_count_1000m',
    'middle_school_nearest_m',
    'library_nearest_m',
    'park_nearest_m', 'park_within_1km',
    'mart_nearest_m', 'mart_count_1000m',
    'department_nearest_m', 'department_within_1km',
    'academy_nearest_m', 'academy_count_1000m',
    'childcare_count_1000m',
    'hospital_general_nearest_m', 'hospital_count_1000m',
    'cctv_count_500m',
]
COMMON = ['층', '건물연령', '강남구분',
          '기준금리', 'CD금리', '소비자물가지수', 'M2']
FEATURES = COMMON + DISTANCE_FEATS


def metrics(y_true_log, y_pred_log):
    y_true = np.exp(y_true_log)
    y_pred = np.exp(y_pred_log)
    ape = np.abs(y_true - y_pred) / y_true
    return {
        'r2': float(r2_score(y_true_log, y_pred_log)),
        'mape': float(np.mean(ape)),
        'median_ape': float(np.median(ape)),
        'n': int(len(y_true_log)),
    }


def fit_xgb(X_tr, y_tr, X_te, y_te):
    xgb = XGBRegressor(n_estimators=400, max_depth=8, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8,
                       n_jobs=-1, random_state=42, tree_method='hist')
    xgb.fit(X_tr, y_tr)
    return xgb, metrics(y_te, xgb.predict(X_te))


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, 'apartment_final_v8.csv'), low_memory=False)
    df = df.dropna(subset=FEATURES + ['log㎡당가격'])
    print(f"데이터: {len(df):,}")

    results = {}
    summary = []
    shap_rows = []

    regions = {
        '전체': df,
        '강남3구': df[df['강남구분'] == 1],
        '비강남': df[df['강남구분'] == 0],
    }

    for region, rdf in regions.items():
        for year in sorted(rdf['거래년도'].unique()):
            sub = rdf[rdf['거래년도'] == year]
            if len(sub) < 500:
                print(f"  skip {region}-{year}: n={len(sub)} too small")
                continue
            tr, te = train_test_split(sub, test_size=0.2, random_state=42)
            xgb, m = fit_xgb(tr[FEATURES], tr['log㎡당가격'].values,
                             te[FEATURES], te['log㎡당가격'].values)
            results.setdefault(region, {})[str(year)] = m
            summary.append({'region': region, 'year': int(year),
                            'n': len(sub), 'r2': m['r2'],
                            'mape': m['mape'], 'median_ape': m['median_ape']})
            # SHAP
            sample = te.sample(min(2000, len(te)), random_state=42)
            exp = shap.TreeExplainer(xgb)
            sv = exp.shap_values(sample[FEATURES])
            abs_mean = np.abs(sv).mean(axis=0)
            pct = 100 * abs_mean / abs_mean.sum()
            order = np.argsort(abs_mean)[::-1][:10]
            for rank, idx in enumerate(order, 1):
                shap_rows.append({'region': region, 'year': int(year),
                                  'rank': rank, 'feature': FEATURES[idx],
                                  'mean_abs_shap': float(abs_mean[idx]),
                                  'pct': float(pct[idx])})
            print(f"  {region}-{year}: n={len(sub):,} R²={m['r2']:.4f} MAPE={m['mape']:.4f} "
                  f"Top3={[FEATURES[i] for i in order[:3]]}")

    # 저장
    with open(os.path.join(RESULTS, 'v8_year_region_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    pd.DataFrame(summary).to_csv(os.path.join(RESULTS, 'v8_year_region_summary.csv'), index=False)
    pd.DataFrame(shap_rows).to_csv(os.path.join(RESULTS, 'v8_year_region_shap_top.csv'), index=False)
    print("\n저장: v8_year_region_summary.csv, v8_year_region_shap_top.csv")

    # 피벗 요약
    print("\n=== R² 피벗 (연도 × 권역) ===")
    piv = pd.DataFrame(summary).pivot(index='year', columns='region', values='r2')
    print(piv.round(4).to_string())
    print("\n=== MAPE 피벗 ===")
    piv2 = pd.DataFrame(summary).pivot(index='year', columns='region', values='mape')
    print(piv2.round(4).to_string())


if __name__ == '__main__':
    main()
