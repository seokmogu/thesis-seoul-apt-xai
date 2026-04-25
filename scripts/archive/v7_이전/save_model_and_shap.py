#!/usr/bin/env python3
"""XGBoost 모델 학습 + 저장 + SHAP 값 계산 + 저장"""
import os, sys, warnings, pickle
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

FEATURES = ['전용면적', '층', '건물연령', '강남구분',
            '초등학교수', '중학교수', '고등학교수',
            'CCTV수', '백화점수', '지하철역수',
            '공원수', '도서관수', '학원수', '어린이집수',
            '기준금리', 'CD금리', '소비자물가지수', 'M2']
TARGET = '거래금액'

print("1. 데이터 로드...")
df = pd.read_csv(os.path.join(DATA_DIR, 'apartment_final_v6_dong.csv'))
print(f"   {len(df):,}건 로드")

X = df[FEATURES]
y = df[TARGET]

# 동일한 분할 (random_state=42, 70/10/20)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42)
print(f"   Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

print("2. XGBoost 학습...")
model = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=0.1,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
    eval_metric='rmse'
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
print(f"   Best iteration: {model.best_iteration}")

# 모델 저장
model_path = os.path.join(RESULTS_DIR, 'xgb_model_v6.json')
model.save_model(model_path)
print(f"   모델 저장: {model_path}")

print("3. SHAP 값 계산 (5,000건)...")
np.random.seed(42)
sample_idx = np.random.choice(len(X_test), size=min(5000, len(X_test)), replace=False)
X_sample = X_test.iloc[sample_idx].copy()

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_sample)

# SHAP 저장
shap_save = {
    'shap_values': shap_values,
    'X_sample': X_sample,
    'feature_cols': FEATURES
}
shap_path = os.path.join(RESULTS_DIR, 'shap_values_v6.pkl')
with open(shap_path, 'wb') as f:
    pickle.dump(shap_save, f)
print(f"   SHAP 저장: {shap_path}")

# 간단한 검증
from sklearn.metrics import r2_score, mean_squared_error
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\n✅ 완료! R²={r2:.4f}, RMSE={rmse:.0f}")
print(f"   (기존 결과: R²=0.968, RMSE=14,221)")
