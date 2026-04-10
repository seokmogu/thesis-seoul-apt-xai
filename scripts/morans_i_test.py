#!/usr/bin/env python3
"""OLS 잔차의 Moran's I 공간 자기상관 검정
행정동 단위 평균 잔차를 구하고, 인접 행정동 기반 공간 가중 행렬로 Moran's I 산출.
"""
import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from libpysal.weights import Queen, KNN
from esda.moran import Moran

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

FEATURES = ['전용면적', '층', '건물연령', '강남구분',
            '초등학교수', '중학교수', '고등학교수',
            'CCTV수', '백화점수', '지하철역수',
            '공원수', '도서관수', '학원수', '어린이집수',
            '기준금리', 'CD금리', '소비자물가지수', 'M2']
TARGET = '거래금액'

def main():
    df = pd.read_csv(os.path.join(DATA_DIR, 'apartment_final_v6_dong.csv'))
    print(f"전체: {len(df):,}건, 행정동: {df['행정동'].nunique()}개")

    # OLS 적합
    X = df[FEATURES].values
    y = df[TARGET].values
    ols = LinearRegression().fit(X, y)
    df['residual'] = y - ols.predict(X)

    # 행정동별 평균 잔차
    dong_resid = df.groupby(['구', '행정동'])['residual'].mean().reset_index()
    dong_resid.columns = ['구', '행정동', 'mean_residual']
    print(f"행정동별 평균 잔차: {len(dong_resid)}개")

    # 같은 구 내 행정동을 인접으로 정의 (GeoJSON 없이 구 기반 인접성)
    # 구별 행정동 목록으로 가중 행렬 생성
    dongs = dong_resid['행정동'].tolist()
    gus = dong_resid['구'].tolist()
    n = len(dongs)

    # KNN 기반 (평균 잔차 유사성이 아닌 인덱스 기반) - 구 내 인접
    # 같은 구 내 행정동끼리 연결하는 가중 행렬
    neighbors = {}
    for i in range(n):
        neighbors[i] = [j for j in range(n) if gus[j] == gus[i] and j != i]

    from libpysal.weights import W
    w = W(neighbors)
    w.transform = 'r'  # row-standardize

    residuals = dong_resid['mean_residual'].values

    # Moran's I
    mi = Moran(residuals, w)
    print(f"\nMoran's I = {mi.I:.4f}")
    print(f"Expected I = {mi.EI:.4f}")
    print(f"Z-score = {mi.z_norm:.4f}")
    print(f"p-value = {mi.p_norm:.6f}")
    print(f"Variance = {mi.VI_norm:.6f}")

    interpretation = "양의 공간 자기상관" if mi.I > 0 and mi.p_norm < 0.05 else \
                     "음의 공간 자기상관" if mi.I < 0 and mi.p_norm < 0.05 else \
                     "공간 자기상관 없음"
    print(f"해석: {interpretation}")

    # XGBoost 잔차로도 검정
    import xgboost as xgb
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    Xtr2, Xv, ytr2, yv = train_test_split(Xtr, ytr, test_size=0.125, random_state=42)
    params = {'objective': 'reg:squarederror', 'max_depth': 8, 'learning_rate': 0.1,
              'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 5,
              'reg_alpha': 0.1, 'reg_lambda': 1.0, 'tree_method': 'hist', 'seed': 42}
    dtr = xgb.DMatrix(Xtr2, label=ytr2, feature_names=FEATURES)
    dv = xgb.DMatrix(Xv, label=yv, feature_names=FEATURES)
    model = xgb.train(params, dtr, num_boost_round=2000,
                      evals=[(dv, 'val')], early_stopping_rounds=50, verbose_eval=0)

    # 전체 데이터에 대한 XGB 잔차
    dall = xgb.DMatrix(X, feature_names=FEATURES)
    df['xgb_residual'] = y - model.predict(dall)
    dong_xgb = df.groupby(['구', '행정동'])['xgb_residual'].mean().reset_index()
    xgb_resid = dong_xgb['xgb_residual'].values

    mi_xgb = Moran(xgb_resid, w)
    print(f"\n[XGBoost 잔차]")
    print(f"Moran's I = {mi_xgb.I:.4f}")
    print(f"Z-score = {mi_xgb.z_norm:.4f}")
    print(f"p-value = {mi_xgb.p_norm:.6f}")

    result = {
        'method': 'Moran\'s I (구 내 행정동 인접 가중 행렬, row-standardized)',
        'n_dongs': int(n),
        'OLS': {
            'Morans_I': round(mi.I, 4),
            'Expected_I': round(mi.EI, 4),
            'Z_score': round(mi.z_norm, 4),
            'p_value': round(mi.p_norm, 6),
            'interpretation': interpretation,
        },
        'XGBoost': {
            'Morans_I': round(mi_xgb.I, 4),
            'Expected_I': round(mi_xgb.EI, 4),
            'Z_score': round(mi_xgb.z_norm, 4),
            'p_value': round(mi_xgb.p_norm, 6),
            'interpretation': "양의 공간 자기상관" if mi_xgb.I > 0 and mi_xgb.p_norm < 0.05 else "공간 자기상관 없음",
        }
    }

    path = os.path.join(RESULTS_DIR, 'morans_i_results.json')
    with open(path, 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {path}")

if __name__ == '__main__':
    main()
