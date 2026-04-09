#!/usr/bin/env python3
"""강남/비강남 별도 모형 + 강남구분 더미 제거.
기존 통합 모형 SHAP의 재집계 방식이 아닌, 실제로 지역별 XGB를 각각 적합하여
SHAP 순위 구조가 지역별로 어떻게 다른지 직접 비교한다.
"""
import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb
import shap

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

# 강남구분 더미 제외
FEATURES = ['전용면적', '층', '건물연령',
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

def ev(y, p):
    return {
        'R2': round(r2_score(y, p), 4),
        'RMSE': round(float(np.sqrt(mean_squared_error(y, p))), 0),
        'MAPE': round(mean_absolute_percentage_error(y, p) * 100, 2),
    }

def fit_region(df_region, label):
    print(f"\n{'='*60}\n[{label}] n={len(df_region):,}\n{'='*60}")
    X = df_region[FEATURES].values
    y = df_region[TARGET].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    Xtr2, Xv, ytr2, yv = train_test_split(Xtr, ytr, test_size=0.125, random_state=42)

    res = {'n': int(len(df_region))}
    ols = LinearRegression().fit(Xtr, ytr)
    res['OLS'] = ev(yte, ols.predict(Xte))
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5,
                               n_jobs=-1, random_state=42).fit(Xtr, ytr)
    res['RF'] = ev(yte, rf.predict(Xte))
    dtr = xgb.DMatrix(Xtr2, label=ytr2, feature_names=FEATURES)
    dv = xgb.DMatrix(Xv, label=yv, feature_names=FEATURES)
    dte = xgb.DMatrix(Xte, label=yte, feature_names=FEATURES)
    model = xgb.train(XGB_PARAMS, dtr, num_boost_round=2000,
                      evals=[(dv, 'val')], early_stopping_rounds=50, verbose_eval=0)
    res['XGB'] = ev(yte, model.predict(dte))
    print(f"  OLS R²={res['OLS']['R2']}  RF R²={res['RF']['R2']}  XGB R²={res['XGB']['R2']}")
    print(f"  XGB MAPE={res['XGB']['MAPE']}%")

    # SHAP on test sample
    idx = np.random.RandomState(42).choice(len(Xte), size=min(5000, len(Xte)), replace=False)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(xgb.DMatrix(Xte[idx], feature_names=FEATURES))
    mean_abs = np.abs(sv).mean(axis=0)
    rank = sorted(zip(FEATURES, mean_abs), key=lambda t: -t[1])
    res['SHAP'] = [{'feature': f, 'mean_abs_shap': round(float(v), 2)} for f, v in rank]
    print(f"  SHAP Top5: {[f for f,_ in rank[:5]]}")
    return res

def main():
    df = pd.read_csv(os.path.join(DATA_DIR, 'apartment_final_v6_dong.csv'))
    # 강남구분 컬럼으로 분할 (강남 3구 등 기존 정의 사용)
    gn = df[df['강남구분'] == 1].copy()
    non = df[df['강남구분'] == 0].copy()
    print(f"강남: {len(gn):,}  비강남: {len(non):,}")

    out = {
        'note': '강남구분 더미 제거, 강남/비강남 별도 XGB 적합',
        'features': FEATURES,
        '강남': fit_region(gn, '강남'),
        '비강남': fit_region(non, '비강남'),
    }

    # SHAP 순위 비교표
    print(f"\n{'='*70}\nSHAP 순위 비교 (강남 vs 비강남, 별도 모형)\n{'='*70}")
    gn_rank = {r['feature']: (i+1, r['mean_abs_shap']) for i, r in enumerate(out['강남']['SHAP'])}
    non_rank = {r['feature']: (i+1, r['mean_abs_shap']) for i, r in enumerate(out['비강남']['SHAP'])}
    print(f"{'변수':<12}{'강남순위':>8}{'강남SHAP':>14}{'비강남순위':>10}{'비강남SHAP':>14}{'배율(강/비)':>14}")
    for f in FEATURES:
        gr, gv = gn_rank[f]; nr, nv = non_rank[f]
        ratio = gv / nv if nv > 0 else float('inf')
        print(f"{f:<12}{gr:>8}{gv:>14,.0f}{nr:>10}{nv:>14,.0f}{ratio:>14.2f}")

    path = os.path.join(RESULTS_DIR, 'region_separate_results.json')
    with open(path, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {path}")

if __name__ == '__main__':
    main()
