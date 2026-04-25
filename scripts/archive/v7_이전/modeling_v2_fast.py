#!/usr/bin/env python3
"""모델링 v2 (fast) — GridSearchCV 경량화 + SHAP 시각화
이전 실행에서 OLS/VIF/기술통계/상관행렬은 완료됨.
여기서는 RF GridSearch, XGBoost GridSearch, SHAP 시각화만 수행.
"""
import os, warnings, json
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
PLOT_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

def load_and_prepare():
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
        parks['구'] = parks[addr_cols[0]].str.extract(r'(\S+구)')
        park_count = parks.groupby('구').size().rename('공원수')
        df = df.merge(park_count, on='구', how='left')
    stores = pd.read_csv(os.path.join(DATA_DIR, 'seoul_large_stores.csv'))
    if 'RDNWHLADDR' in stores.columns:
        stores['구'] = stores['RDNWHLADDR'].str.extract(r'서울특별시\s+(\S+구)')
        if 'UPTAENM' in stores.columns:
            dept_count = stores[stores['UPTAENM'] == '백화점'].groupby('구').size().rename('백화점수')
            df = df.merge(dept_count, on='구', how='left')
    for c in ['학교수','초등학교수','중학교수','고등학교수','공원수','백화점수']:
        if c in df.columns: df[c] = df[c].fillna(0)
    return df.dropna(subset=['구'])

def main():
    print("모델링 v2 (fast) 시작")
    df = load_and_prepare()
    
    features = ['전용면적','층','건물연령','강남구분','학교수','초등학교수','중학교수','고등학교수','공원수','백화점수','기준금리','CD금리','소비자물가지수','M2']
    features = [f for f in features if f in df.columns]
    target = '거래금액'
    subset = df[features + [target]].dropna()
    X, y = subset[features], subset[target]
    print(f"데이터: {len(X):,}건")
    
    # 샘플링으로 GridSearchCV 가속 (5만건으로)
    sample_idx = np.random.RandomState(42).choice(len(X), size=min(50000, len(X)), replace=False)
    X_gs, y_gs = X.iloc[sample_idx], y.iloc[sample_idx]
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.125, random_state=42)
    print(f"Train: {len(X_train):,} / Val: {len(X_val):,} / Test: {len(X_test):,}")
    
    results = []
    
    # ━━━ OLS (이전 결과 재사용) ━━━
    import statsmodels.api as sm
    X_ols = sm.add_constant(X_train_full)
    X_ols_test = sm.add_constant(X_test)
    ols_model = sm.OLS(y_train_full, X_ols).fit()
    y_pred_ols = ols_model.predict(X_ols_test)
    r2 = r2_score(y_test, y_pred_ols)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_ols))
    mae = mean_absolute_error(y_test, y_pred_ols)
    mape = np.mean(np.abs((y_test - y_pred_ols) / y_test)) * 100
    results.append({'model': 'OLS', 'R2': round(r2,4), 'RMSE': round(rmse,0), 'MAE': round(mae,0), 'MAPE': round(mape,2)})
    print(f"OLS: R²={r2:.4f}")
    
    # ━━━ RF GridSearchCV (5만건 샘플) ━━━
    print("\n[RF] GridSearchCV on 50k sample...")
    rf_grid = GridSearchCV(
        RandomForestRegressor(n_jobs=-1, random_state=42),
        {'n_estimators': [200], 'max_depth': [10, 15], 'min_samples_leaf': [5, 10]},
        cv=3, scoring='r2', n_jobs=-1
    )
    rf_grid.fit(X_gs, y_gs)
    print(f"  Best: {rf_grid.best_params_}, CV R²={rf_grid.best_score_:.4f}")
    
    # 전체 데이터로 재학습
    rf = RandomForestRegressor(**rf_grid.best_params_, n_jobs=-1, random_state=42)
    rf.fit(X_train_full, y_train_full)
    y_pred_rf = rf.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    mape_rf = np.mean(np.abs((y_test - y_pred_rf) / y_test)) * 100
    results.append({'model': 'Random Forest', 'R2': round(r2_rf,4), 'RMSE': round(rmse_rf,0), 'MAE': round(mae_rf,0), 'MAPE': round(mape_rf,2)})
    print(f"  RF (full): R²={r2_rf:.4f}, RMSE={rmse_rf:,.0f}")
    
    pd.DataFrame(rf_grid.cv_results_)[['params','mean_test_score','std_test_score']].to_csv(
        os.path.join(RESULTS_DIR, 'rf_gridsearch.csv'), index=False, encoding='utf-8-sig')
    
    # ━━━ XGBoost GridSearchCV (5만건) + early stopping (val set) ━━━
    print("\n[XGB] GridSearchCV on 50k sample...")
    from xgboost import XGBRegressor
    
    xgb_grid = GridSearchCV(
        XGBRegressor(n_estimators=300, min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1),
        {'max_depth': [6, 8], 'learning_rate': [0.05, 0.1], 'subsample': [0.8], 'colsample_bytree': [0.8]},
        cv=3, scoring='r2', n_jobs=-1
    )
    xgb_grid.fit(X_gs, y_gs)
    print(f"  Best: {xgb_grid.best_params_}, CV R²={xgb_grid.best_score_:.4f}")
    
    bp = xgb_grid.best_params_
    xgb_final = XGBRegressor(
        n_estimators=1000, max_depth=bp['max_depth'], learning_rate=bp['learning_rate'],
        subsample=bp['subsample'], colsample_bytree=bp['colsample_bytree'],
        min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, early_stopping_rounds=50
    )
    xgb_final.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"  Best iteration: {xgb_final.best_iteration}")
    
    y_pred_xgb = xgb_final.predict(X_test)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    mape_xgb = np.mean(np.abs((y_test - y_pred_xgb) / y_test)) * 100
    results.append({'model': 'XGBoost', 'R2': round(r2_xgb,4), 'RMSE': round(rmse_xgb,0), 'MAE': round(mae_xgb,0), 'MAPE': round(mape_xgb,2)})
    print(f"  XGB (full): R²={r2_xgb:.4f}, RMSE={rmse_xgb:,.0f}")
    
    pd.DataFrame(xgb_grid.cv_results_)[['params','mean_test_score','std_test_score']].to_csv(
        os.path.join(RESULTS_DIR, 'xgb_gridsearch.csv'), index=False, encoding='utf-8-sig')
    
    # 5-Fold CV
    cv_scores = cross_val_score(
        XGBRegressor(n_estimators=xgb_final.best_iteration, max_depth=bp['max_depth'],
                     learning_rate=bp['learning_rate'], subsample=bp['subsample'],
                     colsample_bytree=bp['colsample_bytree'], min_child_weight=5,
                     reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1),
        X_gs, y_gs, cv=5, scoring='r2'
    )
    print(f"  5-Fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # ━━━ SHAP ━━━
    print("\n[SHAP] 분석 + 시각화...")
    import shap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    explainer = shap.TreeExplainer(xgb_final)
    sample_size = min(5000, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=42)
    shap_values = explainer.shap_values(X_sample)
    
    shap_importance = pd.Series(np.abs(shap_values).mean(axis=0), index=features).sort_values(ascending=False)
    print("  SHAP importance:")
    for f, v in shap_importance.items():
        print(f"    {f:15s}: {v:>10,.1f}")
    
    shap_df = pd.DataFrame(shap_values, columns=features)
    shap_df.to_csv(os.path.join(RESULTS_DIR, 'shap_values.csv'), index=False, encoding='utf-8-sig')
    
    # 시각화
    print("  시각화 생성...")
    
    shap.summary_plot(shap_values, X_sample, feature_names=features, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'fig4_shap_summary.png'), dpi=150, bbox_inches='tight')
    plt.close('all')
    print("    ✅ fig4")
    
    shap.summary_plot(shap_values, X_sample, feature_names=features, plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'fig5_shap_bar.png'), dpi=150, bbox_inches='tight')
    plt.close('all')
    print("    ✅ fig5")
    
    for idx, (var, fname) in enumerate([('전용면적','fig6_dep_area'), ('건물연령','fig7_dep_age'), ('M2','fig8_dep_m2')]):
        fig, ax = plt.subplots(figsize=(8,6))
        shap.dependence_plot(var, shap_values, X_sample, feature_names=features, show=False, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f'{fname}.png'), dpi=150, bbox_inches='tight')
        plt.close('all')
        print(f"    ✅ {fname}")
    
    shap.force_plot(explainer.expected_value, shap_values[0], X_sample.iloc[0], feature_names=features, matplotlib=True, show=False)
    plt.savefig(os.path.join(PLOT_DIR, 'fig9_force_plot.png'), dpi=150, bbox_inches='tight')
    plt.close('all')
    print("    ✅ fig9")
    
    # ━━━ 강남/비강남 ━━━
    print("\n[강남/비강남] 비교...")
    region_results = []
    for label, val in [("강남3구", 1), ("비강남", 0)]:
        mask = X_test['강남구분'] == val
        ys, yp = y_test[mask].values, y_pred_xgb[mask]
        region_results.append({
            '지역': label, '건수': int(mask.sum()),
            '평균실거래가(만원)': round(ys.mean()), '표준편차(만원)': round(ys.std()),
            'R2': round(r2_score(ys, yp), 4),
            'RMSE(만원)': round(np.sqrt(mean_squared_error(ys, yp))),
            'MAE(만원)': round(mean_absolute_error(ys, yp)),
        })
        print(f"  {label}: N={mask.sum():,}, 평균={ys.mean():,.0f}만원, R²={r2_score(ys,yp):.4f}")
    region_results.append({'지역':'전체','건수':len(y_test),'평균실거래가(만원)':round(y_test.mean()),'표준편차(만원)':round(y_test.std()),'R2':round(r2_xgb,4),'RMSE(만원)':round(rmse_xgb),'MAE(만원)':round(mae_xgb)})
    pd.DataFrame(region_results).to_csv(os.path.join(RESULTS_DIR, 'region_comparison.csv'), index=False, encoding='utf-8-sig')
    
    gangnam_mask = X_sample['강남구분'] == 1
    shap_g = np.abs(shap_values[gangnam_mask.values]).mean(axis=0)
    shap_n = np.abs(shap_values[~gangnam_mask.values]).mean(axis=0)
    shap_region = pd.DataFrame({'변수': features, '강남3구': shap_g.round(1), '비강남': shap_n.round(1), '차이': (shap_g-shap_n).round(1)})
    shap_region.to_csv(os.path.join(RESULTS_DIR, 'shap_region_comparison.csv'), index=False, encoding='utf-8-sig')
    
    # ━━━ 저장 ━━━
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False, encoding='utf-8-sig')
    
    fi_rf = pd.Series(rf.feature_importances_, index=features)
    fi_xgb = pd.Series(xgb_final.feature_importances_, index=features)
    ols_coefs = dict(zip(features, ols_model.params.values[1:]))
    ols_pvals = dict(zip(features, ols_model.pvalues.values[1:]))
    ols_tvals = dict(zip(features, ols_model.tvalues.values[1:]))
    ols_ses = dict(zip(features, ols_model.bse.values[1:]))
    
    importance_df = pd.DataFrame({
        'variable': features,
        'OLS_coef': [round(ols_coefs.get(f,0), 3) for f in features],
        'OLS_se': [round(ols_ses.get(f,0), 3) for f in features],
        'OLS_tval': [round(ols_tvals.get(f,0), 3) for f in features],
        'OLS_pval': [round(ols_pvals.get(f,0), 6) for f in features],
        'RF_importance': [round(fi_rf.get(f,0), 6) for f in features],
        'XGB_importance': [round(fi_xgb.get(f,0), 6) for f in features],
        'SHAP_importance': [round(shap_importance.get(f,0), 3) for f in features],
    }).sort_values('SHAP_importance', ascending=False)
    importance_df.to_csv(os.path.join(RESULTS_DIR, 'feature_importance.csv'), index=False, encoding='utf-8-sig')
    
    hyperparams = {
        'RF_best': rf_grid.best_params_,
        'XGB_GridSearch_best': xgb_grid.best_params_,
        'XGB_final_n_estimators': xgb_final.best_iteration,
        'XGB_5fold_CV_R2': f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
        'split': 'train 70% / val 10% / test 20%',
        'GridSearchCV_sample': len(X_gs),
        'SHAP_sample': sample_size,
        'total': len(X), 'train': len(X_train), 'val': len(X_val), 'test': len(X_test),
    }
    with open(os.path.join(RESULTS_DIR, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparams, f, ensure_ascii=False, indent=2)
    
    print("\n✅ 완료!")
    for f in os.listdir(PLOT_DIR):
        print(f"  plots/{f}")

if __name__ == '__main__':
    main()
