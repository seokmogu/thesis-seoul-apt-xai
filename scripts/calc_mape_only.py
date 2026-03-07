#!/usr/bin/env python3
"""Calculate MAPE for both random and temporal splits using original paper parameters."""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

FEATURES = ['전용면적', '층', '건물연령', '강남구분',
            '초등학교수', '중학교수', '고등학교수',
            'CCTV수', '백화점수', '지하철역수',
            '공원수', '도서관수', '학원수', '어린이집수',
            '기준금리', 'CD금리', '소비자물가지수', 'M2']
TARGET = '거래금액'

xgb_params = {
    'objective': 'reg:squarederror',
    'max_depth': 8, 'learning_rate': 0.05,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'min_child_weight': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
    'tree_method': 'hist', 'seed': 42,
}

def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return r2, rmse, mape

df = pd.read_csv(os.path.join(DATA_DIR, 'apartment_final_v6_dong.csv'))
X_all = df[FEATURES].values
y_all = df[TARGET].values

# ===== Random split (original paper) =====
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

ols = LinearRegression().fit(X_tr, y_tr)
r2, rmse, mape = metrics(y_te, ols.predict(X_te))
print(f"[무작위] OLS      R²={r2:.4f}  RMSE={rmse:,.0f}  MAPE={mape:.2f}%")

rf = RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf.fit(X_tr, y_tr)
r2, rmse, mape = metrics(y_te, rf.predict(X_te))
print(f"[무작위] RF       R²={r2:.4f}  RMSE={rmse:,.0f}  MAPE={mape:.2f}%")

X_tr2, X_val, y_tr2, y_val = train_test_split(X_tr, y_tr, test_size=0.1, random_state=42)
dtrain = xgb.DMatrix(X_tr2, label=y_tr2, feature_names=FEATURES)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURES)
dtest = xgb.DMatrix(X_te, label=y_te, feature_names=FEATURES)
model = xgb.train(xgb_params, dtrain, num_boost_round=2000,
                  evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=0)
r2, rmse, mape = metrics(y_te, model.predict(dtest))
print(f"[무작위] XGBoost  R²={r2:.4f}  RMSE={rmse:,.0f}  MAPE={mape:.2f}%")

# ===== Temporal split =====
train_df = df[df['거래년월'] <= 202312]
test_df = df[df['거래년월'] >= 202401]
X_train_t = train_df[FEATURES].values; y_train_t = train_df[TARGET].values
X_test_t = test_df[FEATURES].values; y_test_t = test_df[TARGET].values

ols_t = LinearRegression().fit(X_train_t, y_train_t)
r2, rmse, mape = metrics(y_test_t, ols_t.predict(X_test_t))
print(f"\n[시간순] OLS      R²={r2:.4f}  RMSE={rmse:,.0f}  MAPE={mape:.2f}%")

rf_t = RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf_t.fit(X_train_t, y_train_t)
r2, rmse, mape = metrics(y_test_t, rf_t.predict(X_test_t))
print(f"[시간순] RF       R²={r2:.4f}  RMSE={rmse:,.0f}  MAPE={mape:.2f}%")

train_sorted = train_df.sort_values('거래년월')
val_cutoff = int(len(train_sorted) * 0.9)
X_tr_t = train_sorted[FEATURES].values[:val_cutoff]; y_tr_t = train_sorted[TARGET].values[:val_cutoff]
X_val_t = train_sorted[FEATURES].values[val_cutoff:]; y_val_t = train_sorted[TARGET].values[val_cutoff:]
dtrain_t = xgb.DMatrix(X_tr_t, label=y_tr_t, feature_names=FEATURES)
dval_t = xgb.DMatrix(X_val_t, label=y_val_t, feature_names=FEATURES)
dtest_t = xgb.DMatrix(X_test_t, label=y_test_t, feature_names=FEATURES)
model_t = xgb.train(xgb_params, dtrain_t, num_boost_round=2000,
                    evals=[(dval_t, 'val')], early_stopping_rounds=50, verbose_eval=0)
r2, rmse, mape = metrics(y_test_t, model_t.predict(dtest_t))
print(f"[시간순] XGBoost  R²={r2:.4f}  RMSE={rmse:,.0f}  MAPE={mape:.2f}%")
