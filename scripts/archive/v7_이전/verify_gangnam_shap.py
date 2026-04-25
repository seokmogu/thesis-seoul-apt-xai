#!/usr/bin/env python3
"""강남 지역 별도 모델의 17개 변수 전체 SHAP 순위 검증."""
import os, json, warnings
warnings.filterwarnings('ignore')
os.environ.setdefault('DYLD_LIBRARY_PATH', '/opt/homebrew/opt/libomp/lib')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data', 'apartment_final_v6_dong.csv')

FEATURES_17 = ['전용면적', '층', '건물연령',
               '초등학교수', '중학교수', '고등학교수',
               'CCTV수', '백화점수', '지하철역수',
               '공원수', '도서관수', '학원수', '어린이집수',
               '기준금리', 'CD금리', '소비자물가지수', 'M2']

XGB_PARAMS = dict(objective='reg:squarederror', max_depth=8, learning_rate=0.1,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  reg_alpha=0.1, reg_lambda=1.0, tree_method='hist', seed=42)

df = pd.read_csv(DATA)
if 'log㎡당가격' not in df.columns:
    df['log㎡당가격'] = np.log(df['거래금액'] / df['전용면적'])

gn = df[df['강남구분'] == 1].copy()
print(f"강남 3구 n={len(gn):,}")

X = gn[FEATURES_17].values
y = gn['log㎡당가격'].values
Xtv, Xt, ytv, yt = train_test_split(X, y, test_size=0.2, random_state=42)
Xtr, Xv, ytr, yv = train_test_split(Xtv, ytv, test_size=0.125, random_state=42)

dtr = xgb.DMatrix(Xtr, label=ytr)
dv = xgb.DMatrix(Xv, label=yv)
dt = xgb.DMatrix(Xt, label=yt)
model = xgb.train(XGB_PARAMS, dtr, num_boost_round=2000,
                  evals=[(dv, 'val')], early_stopping_rounds=50, verbose_eval=0)

# SHAP 전체 17개
samp_size = min(3000, len(Xt))
idx = np.random.default_rng(42).choice(len(Xt), samp_size, replace=False)
Xsamp = Xt[idx]
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(Xsamp)
mean_abs = np.abs(sv).mean(axis=0)
total = mean_abs.sum()
rank = sorted(zip(FEATURES_17, mean_abs), key=lambda x: -x[1])

print("\n=== 강남 17개 전체 SHAP 순위 ===")
for i, (feat, val) in enumerate(rank, 1):
    pct = val / total * 100
    print(f"  {i:2d}. {feat:12s}  |SHAP|={val:.4f}  비중={pct:.2f}%")

# 어린이집수 특정 추출
for feat, val in rank:
    if feat == '어린이집수':
        eorini_rank = [i for i, (f, _) in enumerate(rank, 1) if f == '어린이집수'][0]
        eorini_pct = val / total * 100
        eorini_ratio = val / rank[0][1]
        print(f"\n어린이집수: {eorini_rank}위 / 17, 비중 {eorini_pct:.2f}%, 1위({rank[0][0]}) 대비 {eorini_ratio:.3f}배")

# JSON으로도 저장
out = {
    'full_shap_rank': [{'rank': i, '변수': f, 'mean_abs': float(v), 'pct': float(v/total*100)}
                       for i, (f, v) in enumerate(rank, 1)],
    'n': len(gn),
    'best_iter': int(model.best_iteration)
}
with open(os.path.join(BASE, 'results', 'gangnam_full_shap_v7.json'), 'w') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\n저장: results/gangnam_full_shap_v7.json")
