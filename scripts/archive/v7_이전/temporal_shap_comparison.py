#!/usr/bin/env python3
"""시기별 SHAP Top 5 변화 분석
2019~2021 (유동성 장세) vs 2022~2023 (금리 인상기) vs 2024~2025 (금리 인하 전환)
"""
import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

FEATURES = ['전용면적', '층', '건물연령', '강남구분',
            '초등학교수', '중학교수', '고등학교수',
            'CCTV수', '백화점수', '지하철역수',
            '공원수', '도서관수', '학원수', '어린이집수',
            '기준금리', 'CD금리', '소비자물가지수', 'M2']
TARGET = '거래금액'

XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 8, 'learning_rate': 0.1,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'min_child_weight': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
    'tree_method': 'hist', 'seed': 42,
}

PERIODS = {
    '2019-2021 (유동성 장세)': (201901, 202112),
    '2022-2023 (금리 인상기)': (202201, 202312),
    '2024-2025 (금리 인하 전환)': (202401, 202512),
}

def analyze_period(df_period, label):
    print(f"\n{'='*50}\n[{label}] n={len(df_period):,}\n{'='*50}")
    X = df_period[FEATURES].values
    y = df_period[TARGET].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    Xtr2, Xv, ytr2, yv = train_test_split(Xtr, ytr, test_size=0.125, random_state=42)

    dtr = xgb.DMatrix(Xtr2, label=ytr2, feature_names=FEATURES)
    dv = xgb.DMatrix(Xv, label=yv, feature_names=FEATURES)
    dte = xgb.DMatrix(Xte, label=yte, feature_names=FEATURES)
    model = xgb.train(XGB_PARAMS, dtr, num_boost_round=2000,
                      evals=[(dv, 'val')], early_stopping_rounds=50, verbose_eval=0)

    from sklearn.metrics import r2_score, mean_absolute_percentage_error
    pred = model.predict(dte)
    r2 = r2_score(yte, pred)
    mape = mean_absolute_percentage_error(yte, pred) * 100
    print(f"  R²={r2:.4f}, MAPE={mape:.2f}%")

    idx = np.random.RandomState(42).choice(len(Xte), size=min(5000, len(Xte)), replace=False)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(xgb.DMatrix(Xte[idx], feature_names=FEATURES))
    mean_abs = np.abs(sv).mean(axis=0)
    total = mean_abs.sum()
    rank = sorted(zip(FEATURES, mean_abs), key=lambda t: -t[1])

    res = {
        'n': int(len(df_period)),
        'R2': round(r2, 4),
        'MAPE': round(mape, 2),
        'SHAP_top10': [{'rank': i+1, 'feature': f, 'mean_abs_shap': round(float(v), 2),
                        'pct': round(float(v/total*100), 1)} for i, (f, v) in enumerate(rank[:10])]
    }
    print(f"  Top5: {[f'{f}({v/total*100:.1f}%)' for f,v in rank[:5]]}")
    return res

def main():
    df = pd.read_csv(os.path.join(DATA_DIR, 'apartment_final_v6_dong.csv'))
    print(f"전체: {len(df):,}건")

    results = {}
    for label, (start, end) in PERIODS.items():
        sub = df[(df['거래년월'] >= start) & (df['거래년월'] <= end)]
        results[label] = analyze_period(sub, label)

    # 비교표
    print(f"\n{'='*70}\n시기별 SHAP Top 5 비교\n{'='*70}")
    periods = list(PERIODS.keys())
    print(f"{'순위':<4}", end="")
    for p in periods:
        print(f"  {p[:15]:<20}", end="")
    print()
    for rank_i in range(5):
        print(f"{rank_i+1:<4}", end="")
        for p in periods:
            item = results[p]['SHAP_top10'][rank_i]
            print(f"  {item['feature']}({item['pct']}%){'':<5}", end="")
        print()

    path = os.path.join(RESULTS_DIR, 'temporal_shap_comparison.json')
    with open(path, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {path}")

if __name__ == '__main__':
    main()
