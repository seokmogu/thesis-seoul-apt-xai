#!/usr/bin/env python3
"""모델링 v7: Y=log(㎡당 가격), 교수 피드백 반영 (단위면적 정규화)

- 주 모형: log(거래금액 / 전용면적) ~ 18 features
- 전용면적은 독립변수로 유지 (Option B)
- 3개 분할: random / group / chronological
- 지역 별도 모형: 전체 / 강남3구 / 비강남 (교수 피드백: 연도별 → 지역별 전환)
- 시기별 3구간 (보조)
- SHAP은 log 스케일 + 역변환 해석
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
os.environ.setdefault('DYLD_LIBRARY_PATH', '/opt/homebrew/opt/libomp/lib')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'data')
RESULTS_DIR = os.path.join(BASE, 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots_v7_m2price')
OUT_JSON = os.path.join(RESULTS_DIR, 'modeling_v7_m2price_results.json')
os.makedirs(PLOTS_DIR, exist_ok=True)

FEATURES = ['전용면적', '층', '건물연령', '강남구분',
            '초등학교수', '중학교수', '고등학교수',
            'CCTV수', '백화점수', '지하철역수',
            '공원수', '도서관수', '학원수', '어린이집수',
            '기준금리', 'CD금리', '소비자물가지수', 'M2']

XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 8,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'tree_method': 'hist',
    'seed': 42,
}

def log_metrics(y_true_log, y_pred_log):
    """log 스케일 + 원 스케일(㎡당 만원) 성능 지표."""
    r2 = r2_score(y_true_log, y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    mae_log = mean_absolute_error(y_true_log, y_pred_log)
    y_true = np.exp(y_true_log)
    y_pred = np.exp(y_pred_log)
    rmse_orig = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_orig = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    median_ape = np.median(np.abs((y_true - y_pred) / y_true)) * 100
    return {
        'R2_log': float(r2),
        'RMSE_log': float(rmse_log),
        'MAE_log': float(mae_log),
        'RMSE_원단위_만원per㎡': float(rmse_orig),
        'MAE_원단위_만원per㎡': float(mae_orig),
        'MAPE_%': float(mape),
        'Median_APE_%': float(median_ape),
    }

def fit_ols(X_train, y_train, X_test, y_test):
    m = LinearRegression()
    m.fit(X_train, y_train)
    return m, log_metrics(y_test, m.predict(X_test))

def fit_rf(X_train, y_train, X_test, y_test):
    m = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5,
                              n_jobs=-1, random_state=42)
    m.fit(X_train, y_train)
    return m, log_metrics(y_test, m.predict(X_test))

def fit_xgb(X_train, y_train, X_val, y_val, X_test, y_test, features=None):
    features = features or FEATURES
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)
    m = xgb.train(XGB_PARAMS, dtrain, num_boost_round=2000,
                  evals=[(dval, 'val')], early_stopping_rounds=50,
                  verbose_eval=0)
    met = log_metrics(y_test, m.predict(dtest))
    met['best_iter'] = int(m.best_iteration)
    return m, met

def shap_analysis(model, X_test, features=FEATURES, sample_size=5000):
    explainer = shap.TreeExplainer(model)
    np.random.seed(42)
    idx = np.random.choice(len(X_test), min(sample_size, len(X_test)), replace=False)
    X_sample = pd.DataFrame(X_test[idx] if isinstance(X_test, np.ndarray) else X_test.iloc[idx].values,
                            columns=features)
    shap_vals = explainer.shap_values(X_sample)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    total = mean_abs.sum()
    df = pd.DataFrame({'변수': features,
                       'mean_abs_shap_log': mean_abs,
                       '기여도_%': (mean_abs/total*100).round(2)}
                     ).sort_values('mean_abs_shap_log', ascending=False).reset_index(drop=True)
    return df, shap_vals, X_sample, explainer

def run_split(name, X_train, X_val, X_test, y_train, y_val, y_test):
    print(f"\n=== {name} split: train={len(X_train):,}, val={len(X_val):,}, test={len(X_test):,} ===")
    out = {}
    _, m_ols = fit_ols(X_train, y_train, X_test, y_test)
    print(f"  OLS    R²_log={m_ols['R2_log']:.4f}  Median_APE={m_ols['Median_APE_%']:.2f}%")
    out['OLS'] = m_ols
    _, m_rf = fit_rf(X_train, y_train, X_test, y_test)
    print(f"  RF     R²_log={m_rf['R2_log']:.4f}  Median_APE={m_rf['Median_APE_%']:.2f}%")
    out['RF'] = m_rf
    xgb_model, m_xgb = fit_xgb(X_train, y_train, X_val, y_val, X_test, y_test)
    print(f"  XGB    R²_log={m_xgb['R2_log']:.4f}  Median_APE={m_xgb['Median_APE_%']:.2f}%  best_iter={m_xgb['best_iter']}")
    out['XGB'] = m_xgb
    return out, xgb_model

def main():
    results = {'config': {'target': 'log(거래금액/전용면적)', 'features': FEATURES,
                          'n_features': len(FEATURES), 'xgb_params': XGB_PARAMS}}

    print("=== 데이터 로드 ===")
    df = pd.read_csv(os.path.join(DATA_DIR, 'apartment_final_v6_dong.csv'))
    df['㎡당가격'] = df['거래금액'] / df['전용면적']
    df['log㎡당가격'] = np.log(df['㎡당가격'])
    print(f"  총 {len(df):,}건")
    print(f"  ㎡당 가격: mean={df['㎡당가격'].mean():.1f}, median={df['㎡당가격'].median():.1f}, "
          f"min={df['㎡당가격'].min():.1f}, max={df['㎡당가격'].max():.1f} (만원/㎡)")
    print(f"  log 변환 후: mean={df['log㎡당가격'].mean():.4f}, std={df['log㎡당가격'].std():.4f}")

    # 기술통계
    desc = df[FEATURES + ['거래금액', '㎡당가격']].describe().T[['mean','std','min','25%','50%','75%','max']]
    desc.to_csv(os.path.join(RESULTS_DIR, 'descriptive_stats_v7.csv'))
    results['descriptive_stats'] = {v: desc.loc[v].to_dict() for v in desc.index}
    print(f"  기술통계 저장: descriptive_stats_v7.csv")

    X_all = df[FEATURES].values
    y_all = df['log㎡당가격'].values
    groups = (df['법정동'].astype(str) + '__' + df['아파트명'].astype(str)).values
    ym = df['거래년월'].values

    # === VIF ===
    print("\n=== VIF ===")
    Xs = StandardScaler().fit_transform(X_all)
    vif = [variance_inflation_factor(Xs, i) for i in range(len(FEATURES))]
    vif_df = pd.DataFrame({'변수': FEATURES, 'VIF': np.round(vif, 2)}
                          ).sort_values('VIF', ascending=False).reset_index(drop=True)
    print(vif_df.to_string(index=False))
    vif_df.to_csv(os.path.join(RESULTS_DIR, 'vif_v7.csv'), index=False)
    results['VIF'] = {r['변수']: float(r['VIF']) for _, r in vif_df.iterrows()}

    # === 1. Random split (70/10/20) ===
    Xtv, Xt, ytv, yt = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    Xtr, Xv, ytr, yv = train_test_split(Xtv, ytv, test_size=0.125, random_state=42)
    split_random, xgb_main = run_split('Random', Xtr, Xv, Xt, ytr, yv, yt)
    results['split_random'] = split_random

    # === 2. Group split (legal-dong + apt name) ===
    print(f"\n=== Group split 준비: unique groups = {len(np.unique(groups)):,} ===")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    (train_idx, test_idx), = gss.split(X_all, y_all, groups=groups)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=42)
    (tr_idx, v_idx), = gss2.split(X_all[train_idx], y_all[train_idx], groups=groups[train_idx])
    real_tr = train_idx[tr_idx]; real_v = train_idx[v_idx]
    split_group, _ = run_split('Group', X_all[real_tr], X_all[real_v], X_all[test_idx],
                                y_all[real_tr], y_all[real_v], y_all[test_idx])
    results['split_group'] = split_group

    # === 3. Chronological split (train < 2024, test 2024~2025) ===
    print("\n=== Chronological split: train < 202401, val=202401~6, test=202407~ ===")
    tr_mask = ym < 202401
    v_mask = (ym >= 202401) & (ym < 202407)
    t_mask = ym >= 202407
    split_time, _ = run_split('Chronological', X_all[tr_mask], X_all[v_mask], X_all[t_mask],
                               y_all[tr_mask], y_all[v_mask], y_all[t_mask])
    results['split_chronological'] = split_time

    # === 4. SHAP (random split 기준) ===
    print("\n=== SHAP Analysis (random split XGB) ===")
    shap_df, shap_vals, X_samp, explainer = shap_analysis(xgb_main, Xt, FEATURES)
    print(shap_df.to_string(index=False))
    shap_df.to_csv(os.path.join(RESULTS_DIR, 'shap_importance_v7.csv'), index=False)
    results['SHAP_global'] = {r['변수']: {'mean_abs': float(r['mean_abs_shap_log']),
                                          'pct': float(r['기여도_%'])}
                              for _, r in shap_df.iterrows()}

    # === 5. Plots ===
    print("\n=== Plots ===")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_samp, show=False)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, 'fig4_shap_summary.png'), dpi=200, bbox_inches='tight'); plt.close()
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_samp, plot_type='bar', show=False)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, 'fig5_shap_bar.png'), dpi=200, bbox_inches='tight'); plt.close()
    for i, feat in enumerate(shap_df['변수'].head(6).tolist()):
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(FEATURES.index(feat), shap_vals, X_samp, show=False)
        plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, f'fig{6+i}_dep_{feat}.png'), dpi=200, bbox_inches='tight'); plt.close()
    print(f"  plots saved to {PLOTS_DIR}")

    # === 6. 지역 별도 모형 (전체/강남/비강남) — 교수 피드백 핵심 ===
    print("\n=== 지역 별도 모형 (강남구분 제거, 17 features) ===")
    FEATURES_17 = [f for f in FEATURES if f != '강남구분']
    for region_name, mask in [('전체', np.ones(len(df), dtype=bool)),
                              ('강남3구', df['강남구분'] == 1),
                              ('비강남', df['강남구분'] == 0)]:
        print(f"\n  [{region_name}] n={mask.sum():,}")
        Xr = df.loc[mask, FEATURES_17].values
        yr = df.loc[mask, 'log㎡당가격'].values
        Xrtv, Xrt, yrtv, yrt = train_test_split(Xr, yr, test_size=0.2, random_state=42)
        Xrtr, Xrv, yrtr, yrv = train_test_split(Xrtv, yrtv, test_size=0.125, random_state=42)
        _, m_ols = fit_ols(Xrtr, yrtr, Xrt, yrt)
        _, m_rf = fit_rf(Xrtr, yrtr, Xrt, yrt)
        xgb_r, m_xgb = fit_xgb(Xrtr, yrtr, Xrv, yrv, Xrt, yrt, features=FEATURES_17)
        print(f"    OLS R²={m_ols['R2_log']:.4f}  RF R²={m_rf['R2_log']:.4f}  XGB R²={m_xgb['R2_log']:.4f} (MedAPE {m_xgb['Median_APE_%']:.2f}%)")
        # SHAP
        sdf, svals, Xsamp_r, _ = shap_analysis(xgb_r, Xrt, FEATURES_17, sample_size=min(3000, len(Xrt)))
        print(f"    SHAP Top5: {sdf['변수'].head(5).tolist()}")
        results[f'regional_{region_name}'] = {
            'n': int(mask.sum()),
            'OLS': m_ols, 'RF': m_rf, 'XGB': m_xgb,
            'SHAP_top': sdf.head(10).to_dict(orient='records')
        }

    # === 7. 시기별 3구간 (보조) ===
    print("\n=== 시기별 3구간 XGB ===")
    periods = [('2019_2021_유동성', (df['거래년도'] >= 2019) & (df['거래년도'] <= 2021)),
               ('2022_2023_금리인상', (df['거래년도'] >= 2022) & (df['거래년도'] <= 2023)),
               ('2024_2025_금리인하', (df['거래년도'] >= 2024) & (df['거래년도'] <= 2025))]
    for pname, pmask in periods:
        n = pmask.sum()
        if n < 1000:
            continue
        print(f"\n  [{pname}] n={n:,}")
        Xp = df.loc[pmask, FEATURES].values
        yp = df.loc[pmask, 'log㎡당가격'].values
        Xptv, Xpt, yptv, ypt = train_test_split(Xp, yp, test_size=0.2, random_state=42)
        Xptr, Xpv, yptr, ypv = train_test_split(Xptv, yptv, test_size=0.125, random_state=42)
        xgb_p, m_p = fit_xgb(Xptr, yptr, Xpv, ypv, Xpt, ypt)
        sdf, _, _, _ = shap_analysis(xgb_p, Xpt, FEATURES, sample_size=min(3000, len(Xpt)))
        print(f"    XGB R²={m_p['R2_log']:.4f}  Top5: {sdf['변수'].head(5).tolist()}")
        results[f'period_{pname}'] = {
            'n': int(n), 'XGB': m_p,
            'SHAP_top': sdf.head(10).to_dict(orient='records')
        }

    # === 저장 ===
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n=== 저장 완료: {OUT_JSON} ===")
    print("\n요약:")
    print(f"  주 분석 (random split, XGB): R²_log={results['split_random']['XGB']['R2_log']:.4f}, "
          f"Median_APE={results['split_random']['XGB']['Median_APE_%']:.2f}%")
    print(f"  Group split XGB:    R²_log={results['split_group']['XGB']['R2_log']:.4f}")
    print(f"  Chronological XGB:  R²_log={results['split_chronological']['XGB']['R2_log']:.4f}")

if __name__ == '__main__':
    main()
