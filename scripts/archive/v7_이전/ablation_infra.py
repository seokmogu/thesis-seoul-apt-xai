#!/usr/bin/env python3
"""Ablation study: 학원수/어린이집수 제거 시 R²·SHAP 변화
구 단위 균등 배분된 두 변수를 빼고 XGBoost/RF를 재적합하여
(a) 예측 성능 변화, (b) 다른 변수 SHAP 순위 재편을 확인한다.
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

FEATURES_FULL = ['전용면적', '층', '건물연령', '강남구분',
                 '초등학교수', '중학교수', '고등학교수',
                 'CCTV수', '백화점수', '지하철역수',
                 '공원수', '도서관수', '학원수', '어린이집수',
                 '기준금리', 'CD금리', '소비자물가지수', 'M2']
DROP = ['학원수', '어린이집수']
FEATURES_ABL = [f for f in FEATURES_FULL if f not in DROP]
TARGET = '거래금액'

XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 8, 'learning_rate': 0.1,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'min_child_weight': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
    'tree_method': 'hist', 'seed': 42,
}

def eval_all(y, p):
    return {
        'R2': round(r2_score(y, p), 4),
        'RMSE': round(float(np.sqrt(mean_squared_error(y, p))), 0),
        'MAPE': round(mean_absolute_percentage_error(y, p) * 100, 2),
    }

def run(features, label, df):
    print(f"\n{'='*60}\n[{label}] n_features={len(features)}\n{'='*60}")
    X = df[features].values
    y = df[TARGET].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    Xtr2, Xv, ytr2, yv = train_test_split(Xtr, ytr, test_size=0.125, random_state=42)

    res = {}
    ols = LinearRegression().fit(Xtr, ytr)
    res['OLS'] = eval_all(yte, ols.predict(Xte))
    print(f"  OLS : R²={res['OLS']['R2']}  RMSE={res['OLS']['RMSE']:.0f}  MAPE={res['OLS']['MAPE']}%")

    rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5,
                               n_jobs=-1, random_state=42).fit(Xtr, ytr)
    res['RF'] = eval_all(yte, rf.predict(Xte))
    print(f"  RF  : R²={res['RF']['R2']}  RMSE={res['RF']['RMSE']:.0f}  MAPE={res['RF']['MAPE']}%")

    dtr = xgb.DMatrix(Xtr2, label=ytr2, feature_names=features)
    dv = xgb.DMatrix(Xv, label=yv, feature_names=features)
    dte = xgb.DMatrix(Xte, label=yte, feature_names=features)
    model = xgb.train(XGB_PARAMS, dtr, num_boost_round=2000,
                      evals=[(dv, 'val')], early_stopping_rounds=50, verbose_eval=0)
    res['XGB'] = eval_all(yte, model.predict(dte))
    print(f"  XGB : R²={res['XGB']['R2']}  RMSE={res['XGB']['RMSE']:.0f}  MAPE={res['XGB']['MAPE']}%")

    # SHAP on 5000 sample
    sample_idx = np.random.RandomState(42).choice(len(Xte), size=min(5000, len(Xte)), replace=False)
    X_sample = Xte[sample_idx]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(xgb.DMatrix(X_sample, feature_names=features))
    mean_abs = np.abs(sv).mean(axis=0)
    shap_rank = sorted(zip(features, mean_abs), key=lambda t: -t[1])
    res['SHAP_top10'] = [{'feature': f, 'mean_abs_shap': round(float(v), 2)} for f, v in shap_rank[:10]]
    print(f"  SHAP Top5: {[f for f,_ in shap_rank[:5]]}")
    return res

def main():
    df = pd.read_csv(os.path.join(DATA_DIR, 'apartment_final_v6_dong.csv'))
    print(f"전체: {len(df):,}건")

    full = run(FEATURES_FULL, 'FULL (18 features)', df)
    abl = run(FEATURES_ABL, 'ABLATION (without 학원수/어린이집수)', df)

    print(f"\n{'='*60}\n성능 변화\n{'='*60}")
    print(f"{'모형':<8}{'FULL R²':>12}{'ABL R²':>12}{'ΔR²':>10}{'FULL MAPE':>12}{'ABL MAPE':>12}")
    for m in ['OLS', 'RF', 'XGB']:
        d = abl[m]['R2'] - full[m]['R2']
        print(f"{m:<8}{full[m]['R2']:>12.4f}{abl[m]['R2']:>12.4f}{d:>+10.4f}{full[m]['MAPE']:>11.2f}%{abl[m]['MAPE']:>11.2f}%")

    out = {
        'dropped': DROP,
        'full': full,
        'ablation': abl,
        'delta_R2': {m: round(abl[m]['R2'] - full[m]['R2'], 4) for m in ['OLS','RF','XGB']},
    }
    path = os.path.join(RESULTS_DIR, 'ablation_infra_results.json')
    with open(path, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {path}")

if __name__ == '__main__':
    main()
