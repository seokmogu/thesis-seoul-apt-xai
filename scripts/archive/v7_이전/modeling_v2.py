#!/usr/bin/env python3
"""모델링 v2: 논문 리뷰 피드백 반영
- OLS 비표준화 계수 + t값/p값/표준오차
- VIF 계산
- 상관관계 행렬
- 기술통계량 완성
- GridSearchCV 적용
- Validation set 분리 (data leakage 해결)
- SHAP 시각화 생성
"""
import os, warnings, sys
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

# statsmodels for OLS with t-values
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
PLOT_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

def load_and_prepare():
    """v1과 동일한 데이터 준비"""
    df = pd.read_csv(os.path.join(DATA_DIR, 'apartment_final.csv'))
    
    schools = pd.read_csv(os.path.join(DATA_DIR, 'seoul_schools.csv'))
    if '도로명주소' in schools.columns:
        schools['구'] = schools['도로명주소'].str.extract(r'서울특별시\s+(\S+구)')
        school_count = schools.groupby('구').size().rename('학교수')
        if '학교종류' in schools.columns:
            elem = schools[schools['학교종류'].str.contains('초등', na=False)].groupby('구').size().rename('초등학교수')
            middle = schools[schools['학교종류'].str.contains('중학', na=False)].groupby('구').size().rename('중학교수')
            high = schools[schools['학교종류'].str.contains('고등', na=False)].groupby('구').size().rename('고등학교수')
            school_detail = pd.concat([school_count, elem, middle, high], axis=1).fillna(0)
        else:
            school_detail = school_count.to_frame()
        df = df.merge(school_detail, on='구', how='left')
    
    parks = pd.read_csv(os.path.join(DATA_DIR, 'seoul_parks.csv'))
    addr_cols = [c for c in parks.columns if 'ADDR' in c.upper() or '주소' in c]
    if addr_cols:
        addr_col = addr_cols[0]
        parks['구'] = parks[addr_col].str.extract(r'(\S+구)')
        park_count = parks.groupby('구').size().rename('공원수')
        df = df.merge(park_count, on='구', how='left')
    
    stores = pd.read_csv(os.path.join(DATA_DIR, 'seoul_large_stores.csv'))
    if 'RDNWHLADDR' in stores.columns:
        stores['구'] = stores['RDNWHLADDR'].str.extract(r'서울특별시\s+(\S+구)')
        if 'UPTAENM' in stores.columns:
            dept_count = stores[stores['UPTAENM'] == '백화점'].groupby('구').size().rename('백화점수')
            df = df.merge(dept_count, on='구', how='left')
    
    fill_cols = ['학교수', '초등학교수', '중학교수', '고등학교수', '공원수', '백화점수']
    for c in fill_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    
    df = df.dropna(subset=['구'])
    return df

def select_features(df):
    feature_candidates = [
        '전용면적', '층', '건물연령',
        '강남구분', '학교수', '초등학교수', '중학교수', '고등학교수', '공원수', '백화점수',
        '기준금리', 'CD금리', '소비자물가지수', 'M2',
    ]
    features = [f for f in feature_candidates if f in df.columns]
    target = '거래금액'
    subset = df[features + [target]].dropna()
    X = subset[features]
    y = subset[target]
    return X, y, features

def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"\n  {name}: R²={r2:.4f}, RMSE={rmse:,.0f}, MAE={mae:,.0f}, MAPE={mape:.2f}%")
    return {'model': name, 'R2': round(r2, 4), 'RMSE': round(rmse, 0), 'MAE': round(mae, 0), 'MAPE': round(mape, 2)}

def main():
    print("=" * 60)
    print("모델링 v2 — 논문 리뷰 피드백 반영")
    print("=" * 60)
    
    df = load_and_prepare()
    X, y, features = select_features(df)
    print(f"데이터: {len(X):,}건, 변수 {len(features)}개")
    
    # ━━━ 0. 기술통계량 ━━━
    print("\n[0] 기술통계량 생성")
    desc = X.copy()
    desc['거래금액'] = y.values
    desc_stats = desc.describe().T
    desc_stats['median'] = desc.median()
    desc_stats = desc_stats[['count', 'mean', 'std', 'min', '25%', 'median', '75%', 'max']]
    desc_stats.columns = ['N', '평균', '표준편차', '최솟값', '1사분위', '중앙값', '3사분위', '최댓값']
    desc_stats = desc_stats.round(2)
    desc_stats.to_csv(os.path.join(RESULTS_DIR, 'descriptive_stats.csv'), encoding='utf-8-sig')
    print(desc_stats.to_string())
    
    # ━━━ 0b. 상관관계 행렬 ━━━
    print("\n[0b] 상관관계 행렬 생성")
    corr = desc.corr().round(3)
    corr.to_csv(os.path.join(RESULTS_DIR, 'correlation_matrix.csv'), encoding='utf-8-sig')
    print("  저장 완료: correlation_matrix.csv")
    
    # ━━━ 0c. VIF ━━━
    print("\n[0c] VIF (다중공선성 진단)")
    X_vif = sm.add_constant(X)
    vif_data = pd.DataFrame({
        '변수': features,
        'VIF': [variance_inflation_factor(X_vif.values, i+1) for i in range(len(features))]
    }).sort_values('VIF', ascending=False)
    vif_data.to_csv(os.path.join(RESULTS_DIR, 'vif.csv'), index=False, encoding='utf-8-sig')
    print(vif_data.to_string(index=False))
    
    # ━━━ Train / Validation / Test Split ━━━
    # 8:1:1로 분리 — early stopping에 validation set 사용 (data leakage 방지)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.125, random_state=42)
    # 최종: train 70%, val 10%, test 20%
    print(f"\n  Train: {len(X_train):,} / Val: {len(X_val):,} / Test: {len(X_test):,}")
    
    results = []
    
    # ━━━ 1. OLS (비표준화 계수 + statsmodels) ━━━
    print("\n[1] OLS 다중회귀 (비표준화 계수, statsmodels)")
    X_ols = sm.add_constant(X_train_full)  # OLS는 train+val 전체 사용
    X_ols_test = sm.add_constant(X_test)
    
    ols_model = sm.OLS(y_train_full, X_ols).fit()
    y_pred_ols = ols_model.predict(X_ols_test)
    
    res_ols = evaluate_model("OLS", y_test.values, y_pred_ols.values)
    results.append(res_ols)
    
    # OLS 상세 결과 저장
    ols_summary = pd.DataFrame({
        '변수': ['(상수)'] + features,
        '비표준화 회귀계수': ols_model.params.values.round(3),
        '표준오차': ols_model.bse.values.round(3),
        't값': ols_model.tvalues.values.round(3),
        'p값': ols_model.pvalues.values.round(4),
    })
    ols_summary.to_csv(os.path.join(RESULTS_DIR, 'ols_detailed.csv'), index=False, encoding='utf-8-sig')
    print("\n  OLS 회귀계수 (비표준화):")
    print(ols_summary.to_string(index=False))
    print(f"\n  Adj. R² = {ols_model.rsquared_adj:.4f}")
    print(f"  F-stat = {ols_model.fvalue:.1f} (p={ols_model.f_pvalue:.2e})")
    
    # 표준화 계수도 별도 저장 (참고용)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)
    ols_std = LinearRegression().fit(X_train_scaled, y_train_full)
    std_coefs = pd.DataFrame({
        '변수': features,
        '표준화 회귀계수': ols_std.coef_.round(3)
    }).sort_values('표준화 회귀계수', key=abs, ascending=False)
    std_coefs.to_csv(os.path.join(RESULTS_DIR, 'ols_standardized.csv'), index=False, encoding='utf-8-sig')
    
    # ━━━ 2. Random Forest (GridSearchCV) ━━━
    print("\n[2] Random Forest (GridSearchCV)")
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_leaf': [5, 10],
    }
    rf_base = RandomForestRegressor(n_jobs=-1, random_state=42)
    rf_cv = GridSearchCV(rf_base, rf_param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0)
    rf_cv.fit(X_train_full, y_train_full)
    
    print(f"  Best params: {rf_cv.best_params_}")
    print(f"  Best CV R²: {rf_cv.best_score_:.4f}")
    
    rf = rf_cv.best_estimator_
    y_pred_rf = rf.predict(X_test)
    res_rf = evaluate_model("Random Forest", y_test.values, y_pred_rf)
    results.append(res_rf)
    
    # RF CV results 저장
    rf_cv_results = pd.DataFrame({
        '파라미터': [str(p) for p in rf_cv.cv_results_['params']],
        'CV R² 평균': rf_cv.cv_results_['mean_test_score'].round(4),
        'CV R² 표준편차': rf_cv.cv_results_['std_test_score'].round(4),
    }).sort_values('CV R² 평균', ascending=False)
    rf_cv_results.to_csv(os.path.join(RESULTS_DIR, 'rf_gridsearch.csv'), index=False, encoding='utf-8-sig')
    
    # ━━━ 3. XGBoost (GridSearchCV + proper validation) ━━━
    print("\n[3] XGBoost (GridSearchCV + validation set)")
    try:
        from xgboost import XGBRegressor
    except ImportError:
        os.system("pip install --break-system-packages xgboost -q")
        from xgboost import XGBRegressor
    
    # Stage 1: GridSearchCV for structure params
    xgb_param_grid = {
        'max_depth': [6, 8, 10],
        'learning_rate': [0.03, 0.05, 0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
    }
    xgb_base = XGBRegressor(
        n_estimators=300, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1,
    )
    xgb_cv = GridSearchCV(xgb_base, xgb_param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0)
    xgb_cv.fit(X_train_full, y_train_full)
    
    print(f"  Best params: {xgb_cv.best_params_}")
    print(f"  Best CV R²: {xgb_cv.best_score_:.4f}")
    
    # Stage 2: Refit with early stopping using VALIDATION set (not test set)
    best_params = xgb_cv.best_params_
    xgb_final = XGBRegressor(
        n_estimators=1000,  # high, rely on early stopping
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    xgb_final.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"  Best iteration: {xgb_final.best_iteration}")
    
    y_pred_xgb = xgb_final.predict(X_test)
    res_xgb = evaluate_model("XGBoost", y_test.values, y_pred_xgb)
    results.append(res_xgb)
    
    # XGB CV results 저장
    xgb_cv_results = pd.DataFrame({
        '파라미터': [str(p) for p in xgb_cv.cv_results_['params']],
        'CV R² 평균': xgb_cv.cv_results_['mean_test_score'].round(4),
        'CV R² 표준편차': xgb_cv.cv_results_['std_test_score'].round(4),
    }).sort_values('CV R² 평균', ascending=False)
    xgb_cv_results.to_csv(os.path.join(RESULTS_DIR, 'xgb_gridsearch.csv'), index=False, encoding='utf-8-sig')
    
    # 5-Fold CV score for final model
    cv_scores = cross_val_score(
        XGBRegressor(**{**best_params, 'n_estimators': xgb_final.best_iteration,
                        'min_child_weight': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
                        'random_state': 42, 'n_jobs': -1}),
        X_train_full, y_train_full, cv=5, scoring='r2'
    )
    print(f"  5-Fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # ━━━ 4. SHAP 분석 ━━━
    print("\n[4] SHAP 분석")
    try:
        import shap
    except ImportError:
        os.system("pip install --break-system-packages shap -q")
        import shap
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'DejaVu Sans'  # fallback, Korean may not render
    
    explainer = shap.TreeExplainer(xgb_final)
    
    sample_size = min(5000, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=42)
    shap_values = explainer.shap_values(X_sample)
    
    shap_importance = pd.Series(np.abs(shap_values).mean(axis=0), index=features).sort_values(ascending=False)
    print("\n  SHAP 변수 중요도:")
    for feat, imp in shap_importance.items():
        print(f"    {feat:15s}: {imp:>10,.1f}")
    
    # SHAP 저장
    shap_df = pd.DataFrame(shap_values, columns=features)
    shap_df.to_csv(os.path.join(RESULTS_DIR, 'shap_values.csv'), index=False, encoding='utf-8-sig')
    
    # ━━━ SHAP 시각화 ━━━
    print("\n  SHAP 시각화 생성...")
    
    # 그림 4: Summary Plot (dot)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=features, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'fig4_shap_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    ✅ fig4_shap_summary.png")
    
    # 그림 5: Bar Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=features, plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'fig5_shap_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    ✅ fig5_shap_bar.png")
    
    # 그림 6: Dependence plot - 전용면적
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.dependence_plot('전용면적', shap_values, X_sample, feature_names=features, show=False, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'fig6_dep_area.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    ✅ fig6_dep_area.png")
    
    # 그림 7: Dependence plot - 건물연령
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.dependence_plot('건물연령', shap_values, X_sample, feature_names=features, show=False, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'fig7_dep_age.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    ✅ fig7_dep_age.png")
    
    # 그림 8: Dependence plot - M2
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.dependence_plot('M2', shap_values, X_sample, feature_names=features, show=False, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'fig8_dep_m2.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    ✅ fig8_dep_m2.png")
    
    # 그림 9: Force plot (단일 예측 사례) — HTML로 저장
    idx = 0
    shap.force_plot(explainer.expected_value, shap_values[idx], X_sample.iloc[idx],
                    feature_names=features, matplotlib=True, show=False)
    plt.savefig(os.path.join(PLOT_DIR, 'fig9_force_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    ✅ fig9_force_plot.png")
    
    # ━━━ 5. 강남/비강남 비교 ━━━
    print("\n[5] 강남 vs 비강남 비교")
    
    region_results = []
    for label, mask_val in [("강남3구", 1), ("비강남", 0)]:
        mask = X_test['강남구분'] == mask_val
        if mask.sum() == 0:
            continue
        y_sub = y_test[mask].values
        y_pred_sub = y_pred_xgb[mask]
        
        r2_sub = r2_score(y_sub, y_pred_sub)
        rmse_sub = np.sqrt(mean_squared_error(y_sub, y_pred_sub))
        mae_sub = mean_absolute_error(y_sub, y_pred_sub)
        
        region_results.append({
            '지역': label, '건수': int(mask.sum()),
            '평균실거래가(만원)': round(y_sub.mean(), 0),
            '표준편차(만원)': round(y_sub.std(), 0),
            'R2': round(r2_sub, 4),
            'RMSE(만원)': round(rmse_sub, 0),
            'MAE(만원)': round(mae_sub, 0),
        })
        print(f"  {label}: 건수={mask.sum():,}, 평균={y_sub.mean():,.0f}만원, R²={r2_sub:.4f}")
    
    # 전체도 추가
    region_results.append({
        '지역': '전체', '건수': len(y_test),
        '평균실거래가(만원)': round(y_test.mean(), 0),
        '표준편차(만원)': round(y_test.std(), 0),
        'R2': res_xgb['R2'], 'RMSE(만원)': res_xgb['RMSE'], 'MAE(만원)': res_xgb['MAE'],
    })
    pd.DataFrame(region_results).to_csv(os.path.join(RESULTS_DIR, 'region_comparison.csv'), index=False, encoding='utf-8-sig')
    
    # 강남/비강남 SHAP 비교
    gangnam_mask = X_sample['강남구분'] == 1
    shap_gangnam = np.abs(shap_values[gangnam_mask.values]).mean(axis=0)
    shap_non = np.abs(shap_values[~gangnam_mask.values]).mean(axis=0)
    
    shap_region = pd.DataFrame({
        '변수': features,
        '강남3구 SHAP': shap_gangnam.round(1),
        '비강남 SHAP': shap_non.round(1),
        '차이': (shap_gangnam - shap_non).round(1),
    }).sort_values('강남3구 SHAP', ascending=False)
    shap_region.to_csv(os.path.join(RESULTS_DIR, 'shap_region_comparison.csv'), index=False, encoding='utf-8-sig')
    print("\n  강남/비강남 SHAP 비교:")
    print(shap_region.to_string(index=False))
    
    # ━━━ 결과 종합 ━━━
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False, encoding='utf-8-sig')
    
    # 변수 중요도 통합
    fi_rf = pd.Series(rf.feature_importances_, index=features)
    fi_xgb = pd.Series(xgb_final.feature_importances_, index=features)
    ols_coefs = dict(zip(features, ols_model.params.values[1:]))  # skip const
    ols_pvals = dict(zip(features, ols_model.pvalues.values[1:]))
    ols_tvals = dict(zip(features, ols_model.tvalues.values[1:]))
    ols_ses = dict(zip(features, ols_model.bse.values[1:]))
    
    importance_df = pd.DataFrame({
        'variable': features,
        'OLS_coef': [ols_coefs.get(f, 0) for f in features],
        'OLS_se': [ols_ses.get(f, 0) for f in features],
        'OLS_tval': [ols_tvals.get(f, 0) for f in features],
        'OLS_pval': [ols_pvals.get(f, 0) for f in features],
        'RF_importance': [fi_rf.get(f, 0) for f in features],
        'XGB_importance': [fi_xgb.get(f, 0) for f in features],
        'SHAP_importance': [shap_importance.get(f, 0) for f in features],
    }).sort_values('SHAP_importance', ascending=False)
    importance_df.to_csv(os.path.join(RESULTS_DIR, 'feature_importance.csv'), index=False, encoding='utf-8-sig')
    
    # 하이퍼파라미터 기록
    hyperparams = {
        'RF': {**rf_cv.best_params_, 'n_jobs': -1, 'random_state': 42},
        'XGBoost_GridSearch': {**xgb_cv.best_params_, 'n_estimators': 300, 'min_child_weight': 5,
                               'reg_alpha': 0.1, 'reg_lambda': 1.0},
        'XGBoost_Final': {**best_params, 'n_estimators': xgb_final.best_iteration,
                          'min_child_weight': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
                          'early_stopping_rounds': 50},
        'XGBoost_CV_R2': f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
        'split': 'train 70% / val 10% / test 20%',
        'SHAP_sample_size': sample_size,
        'total_data': len(X),
        'train_data': len(X_train),
        'val_data': len(X_val),
        'test_data': len(X_test),
    }
    with open(os.path.join(RESULTS_DIR, 'hyperparameters.json'), 'w', encoding='utf-8') as f:
        json.dump(hyperparams, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ v2 모델링 완료!")
    print("=" * 60)
    print(f"  결과: {RESULTS_DIR}/")
    print(f"  시각화: {PLOT_DIR}/")
    print(f"  새로 생성된 파일:")
    for f in ['descriptive_stats.csv', 'correlation_matrix.csv', 'vif.csv',
              'ols_detailed.csv', 'ols_standardized.csv', 'rf_gridsearch.csv',
              'xgb_gridsearch.csv', 'region_comparison.csv', 'shap_region_comparison.csv',
              'hyperparameters.json']:
        print(f"    - {f}")
    print(f"  시각화:")
    for f in ['fig4_shap_summary.png', 'fig5_shap_bar.png', 'fig6_dep_area.png',
              'fig7_dep_age.png', 'fig8_dep_m2.png', 'fig9_force_plot.png']:
        print(f"    - plots/{f}")

if __name__ == '__main__':
    main()
