#!/usr/bin/env python3
"""GroupKFold / Group train-test split: 같은 단지가 train/test에 동시 등장하지 않도록 분할.
단지 키 = 법정동 + 아파트명
"""
import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb

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

def ev(y, p, label):
    r = {'R2': round(r2_score(y, p), 4),
         'RMSE': round(float(np.sqrt(mean_squared_error(y, p))), 0),
         'MAPE': round(mean_absolute_percentage_error(y, p) * 100, 2)}
    print(f"  [{label}] R²={r['R2']}  RMSE={r['RMSE']:.0f}  MAPE={r['MAPE']}%")
    return r

def main():
    df = pd.read_csv(os.path.join(DATA_DIR, 'apartment_final_v6_dong.csv'))
    df['_group'] = df['법정동'].astype(str) + '|' + df['아파트명'].astype(str)
    print(f"전체: {len(df):,}건, 단지수: {df['_group'].nunique():,}")

    X = df[FEATURES].values
    y = df[TARGET].values
    groups = df['_group'].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    print(f"Train: {len(tr_idx):,} ({df.iloc[tr_idx]['_group'].nunique():,} 단지)")
    print(f"Test : {len(te_idx):,} ({df.iloc[te_idx]['_group'].nunique():,} 단지)")

    # group overlap check
    tr_g = set(df.iloc[tr_idx]['_group'])
    te_g = set(df.iloc[te_idx]['_group'])
    print(f"Overlap 단지: {len(tr_g & te_g)}")

    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xte, yte = X[te_idx], y[te_idx]
    # val split from train (random, not group — for early stopping only)
    rng = np.random.RandomState(42)
    val_mask = rng.rand(len(Xtr)) < 0.1
    Xtr2, Xv = Xtr[~val_mask], Xtr[val_mask]
    ytr2, yv = ytr[~val_mask], ytr[val_mask]

    res = {}
    print("\n--- OLS ---")
    ols = LinearRegression().fit(Xtr, ytr)
    res['OLS'] = ev(yte, ols.predict(Xte), 'group')

    print("\n--- RF ---")
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5,
                               n_jobs=-1, random_state=42).fit(Xtr, ytr)
    res['RF'] = ev(yte, rf.predict(Xte), 'group')

    print("\n--- XGBoost ---")
    dtr = xgb.DMatrix(Xtr2, label=ytr2, feature_names=FEATURES)
    dv = xgb.DMatrix(Xv, label=yv, feature_names=FEATURES)
    dte = xgb.DMatrix(Xte, label=yte, feature_names=FEATURES)
    m = xgb.train(XGB_PARAMS, dtr, num_boost_round=2000,
                  evals=[(dv, 'val')], early_stopping_rounds=50, verbose_eval=0)
    res['XGB'] = ev(yte, m.predict(dte), 'group')
    print(f"  best_iter={m.best_iteration}")

    out = {
        'split': 'GroupShuffleSplit by 법정동+아파트명, test_size=0.2, seed=42',
        'n_train': int(len(tr_idx)),
        'n_test': int(len(te_idx)),
        'n_train_groups': int(df.iloc[tr_idx]['_group'].nunique()),
        'n_test_groups': int(df.iloc[te_idx]['_group'].nunique()),
        'results': res,
    }
    path = os.path.join(RESULTS_DIR, 'group_split_results.json')
    with open(path, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {path}")

if __name__ == '__main__':
    main()
