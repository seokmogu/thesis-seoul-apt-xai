#!/usr/bin/env python3
"""모델링 v3 — Gemini 3.1 Pro 리뷰 반영
핵심 변경: 학교수(총합) 변수 제거 (구조적 선형결합 해소)
"""
import os, warnings, json
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

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
        df = df.merge(parks.groupby('구').size().rename('공원수'), on='구', how='left')
    stores = pd.read_csv(os.path.join(DATA_DIR, 'seoul_large_stores.csv'))
    if 'RDNWHLADDR' in stores.columns:
        stores['구'] = stores['RDNWHLADDR'].str.extract(r'서울특별시\s+(\S+구)')
        if 'UPTAENM' in stores.columns:
            df = df.merge(stores[stores['UPTAENM']=='백화점'].groupby('구').size().rename('백화점수'), on='구', how='left')
    for c in ['학교수','초등학교수','중학교수','고등학교수','공원수','백화점수']:
        if c in df.columns: df[c] = df[c].fillna(0)
    return df.dropna(subset=['구'])

def evaluate(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"  {name}: R²={r2:.4f}, RMSE={rmse:,.0f}, MAE={mae:,.0f}, MAPE={mape:.2f}%")
    return {'model': name, 'R2': round(r2,4), 'RMSE': round(rmse,0), 'MAE': round(mae,0), 'MAPE': round(mape,2)}

def main():
    print("=" * 60)
    print("모델링 v3 — 학교수(총합) 제거")
    print("=" * 60)
    
    df = load_and_prepare()
    
    # ★ 핵심 변경: 학교수(총합) 제거 → 13개 변수
    features = ['전용면적','층','건물연령','강남구분',
                '초등학교수','중학교수','고등학교수',  # 학교수 제거!
                '공원수','백화점수',
                '기준금리','CD금리','소비자물가지수','M2']
    features = [f for f in features if f in df.columns]
    target = '거래금액'
    subset = df[features + [target]].dropna()
    X, y = subset[features], subset[target]
    print(f"데이터: {len(X):,}건, 변수 {len(features)}개")
    print(f"변수: {features}")
    
    # 기술통계
    desc = X.copy(); desc['거래금액'] = y.values
    desc_stats = desc.describe().T
    desc_stats['median'] = desc.median()
    desc_stats = desc_stats[['count','mean','std','min','25%','median','75%','max']]
    desc_stats.columns = ['N','평균','표준편차','최솟값','1사분위','중앙값','3사분위','최댓값']
    desc_stats.round(2).to_csv(os.path.join(RESULTS_DIR, 'descriptive_stats.csv'), encoding='utf-8-sig')
    
    # 상관행렬
    desc.corr().round(3).to_csv(os.path.join(RESULTS_DIR, 'correlation_matrix.csv'), encoding='utf-8-sig')
    
    # VIF
    X_vif = sm.add_constant(X)
    vif_data = pd.DataFrame({
        '변수': features,
        'VIF': [variance_inflation_factor(X_vif.values, i+1) for i in range(len(features))]
    }).sort_values('VIF', ascending=False)
    vif_data.to_csv(os.path.join(RESULTS_DIR, 'vif.csv'), index=False, encoding='utf-8-sig')
    print("\nVIF:")
    print(vif_data.to_string(index=False))
    
    # Split
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.125, random_state=42)
    print(f"\nTrain: {len(X_train):,} / Val: {len(X_val):,} / Test: {len(X_test):,}")
    
    results = []
    
    # OLS
    print("\n[OLS]")
    X_ols = sm.add_constant(X_train_full)
    ols = sm.OLS(y_train_full, X_ols).fit()
    y_pred_ols = ols.predict(sm.add_constant(X_test))
    results.append(evaluate("OLS", y_test.values, y_pred_ols.values))
    
    ols_detail = pd.DataFrame({
        '변수': ['(상수)'] + features,
        '비표준화 회귀계수': ols.params.values.round(3),
        '표준오차': ols.bse.values.round(3),
        't값': ols.tvalues.values.round(3),
        'p값': ols.pvalues.values.round(4),
    })
    ols_detail.to_csv(os.path.join(RESULTS_DIR, 'ols_detailed.csv'), index=False, encoding='utf-8-sig')
    print(ols_detail.to_string(index=False))
    print(f"  Adj R² = {ols.rsquared_adj:.4f}, F = {ols.fvalue:.1f}")
    
    # RF GridSearchCV
    print("\n[RF]")
    X_gs = X.sample(50000, random_state=42)
    y_gs = y.loc[X_gs.index]
    rf_grid = GridSearchCV(
        RandomForestRegressor(n_jobs=-1, random_state=42),
        {'n_estimators':[200], 'max_depth':[10,15], 'min_samples_leaf':[5,10]},
        cv=3, scoring='r2', n_jobs=-1
    )
    rf_grid.fit(X_gs, y_gs)
    print(f"  Best: {rf_grid.best_params_}, CV R²={rf_grid.best_score_:.4f}")
    rf = RandomForestRegressor(**rf_grid.best_params_, n_jobs=-1, random_state=42)
    rf.fit(X_train_full, y_train_full)
    y_pred_rf = rf.predict(X_test)
    results.append(evaluate("Random Forest", y_test.values, y_pred_rf))
    
    # XGBoost
    print("\n[XGB]")
    from xgboost import XGBRegressor
    xgb_grid = GridSearchCV(
        XGBRegressor(n_estimators=300, min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1),
        {'max_depth':[6,8], 'learning_rate':[0.05,0.1], 'subsample':[0.8], 'colsample_bytree':[0.8]},
        cv=3, scoring='r2', n_jobs=-1
    )
    xgb_grid.fit(X_gs, y_gs)
    bp = xgb_grid.best_params_
    print(f"  Best: {bp}, CV R²={xgb_grid.best_score_:.4f}")
    
    xgb = XGBRegressor(n_estimators=1000, max_depth=bp['max_depth'], learning_rate=bp['learning_rate'],
                        subsample=bp['subsample'], colsample_bytree=bp['colsample_bytree'],
                        min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
                        random_state=42, n_jobs=-1, early_stopping_rounds=50)
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"  Best iteration: {xgb.best_iteration}")
    y_pred_xgb = xgb.predict(X_test)
    results.append(evaluate("XGBoost", y_test.values, y_pred_xgb))
    
    cv_scores = cross_val_score(
        XGBRegressor(n_estimators=xgb.best_iteration, **bp, min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1),
        X_gs, y_gs, cv=5, scoring='r2')
    print(f"  5-Fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # SHAP
    print("\n[SHAP]")
    import shap, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    
    explainer = shap.TreeExplainer(xgb)
    sample_size = min(5000, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=42)
    shap_values = explainer.shap_values(X_sample)
    
    shap_imp = pd.Series(np.abs(shap_values).mean(axis=0), index=features).sort_values(ascending=False)
    print("  SHAP importance:")
    for f, v in shap_imp.items(): print(f"    {f:15s}: {v:>10,.1f}")
    
    pd.DataFrame(shap_values, columns=features).to_csv(os.path.join(RESULTS_DIR, 'shap_values.csv'), index=False, encoding='utf-8-sig')
    
    # Plots
    for plot_func, fname in [
        (lambda: shap.summary_plot(shap_values, X_sample, feature_names=features, show=False), 'fig4_shap_summary'),
        (lambda: shap.summary_plot(shap_values, X_sample, feature_names=features, plot_type='bar', show=False), 'fig5_shap_bar'),
    ]:
        plot_func(); plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f'{fname}.png'), dpi=150, bbox_inches='tight'); plt.close('all')
        print(f"    ✅ {fname}")
    
    for var, fname in [('전용면적','fig6_dep_area'),('건물연령','fig7_dep_age'),('M2','fig8_dep_m2')]:
        fig,ax=plt.subplots(figsize=(8,6))
        shap.dependence_plot(var, shap_values, X_sample, feature_names=features, show=False, ax=ax)
        plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, f'{fname}.png'), dpi=150, bbox_inches='tight'); plt.close('all')
        print(f"    ✅ {fname}")
    
    shap.force_plot(explainer.expected_value, shap_values[0], X_sample.iloc[0], feature_names=features, matplotlib=True, show=False)
    plt.savefig(os.path.join(PLOT_DIR, 'fig9_force_plot.png'), dpi=150, bbox_inches='tight'); plt.close('all')
    print("    ✅ fig9")
    
    # Region comparison
    print("\n[강남/비강남]")
    region_results = []
    for label, val in [("강남3구",1),("비강남",0)]:
        mask = X_test['강남구분']==val
        ys, yp = y_test[mask].values, y_pred_xgb[mask]
        region_results.append({'지역':label,'건수':int(mask.sum()),'평균실거래가(만원)':round(ys.mean()),
                               '표준편차(만원)':round(ys.std()),'R2':round(r2_score(ys,yp),4),
                               'RMSE(만원)':round(np.sqrt(mean_squared_error(ys,yp))),'MAE(만원)':round(mean_absolute_error(ys,yp))})
        print(f"  {label}: N={mask.sum():,}, R²={r2_score(ys,yp):.4f}")
    region_results.append({'지역':'전체','건수':len(y_test),'평균실거래가(만원)':round(y_test.mean()),
                           '표준편차(만원)':round(y_test.std()),'R2':results[-1]['R2'],'RMSE(만원)':results[-1]['RMSE'],'MAE(만원)':results[-1]['MAE']})
    pd.DataFrame(region_results).to_csv(os.path.join(RESULTS_DIR, 'region_comparison.csv'), index=False, encoding='utf-8-sig')
    
    gm = X_sample['강남구분']==1
    sg, sn = np.abs(shap_values[gm.values]).mean(axis=0), np.abs(shap_values[~gm.values]).mean(axis=0)
    pd.DataFrame({'변수':features,'강남3구':sg.round(1),'비강남':sn.round(1),'차이':(sg-sn).round(1)}).to_csv(
        os.path.join(RESULTS_DIR, 'shap_region_comparison.csv'), index=False, encoding='utf-8-sig')
    
    # Save all
    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False, encoding='utf-8-sig')
    
    fi_rf = pd.Series(rf.feature_importances_, index=features)
    fi_xgb = pd.Series(xgb.feature_importances_, index=features)
    pd.DataFrame({
        'variable': features,
        'OLS_coef': [round(ols.params.iloc[i+1],3) for i in range(len(features))],
        'OLS_se': [round(ols.bse.iloc[i+1],3) for i in range(len(features))],
        'OLS_tval': [round(ols.tvalues.iloc[i+1],3) for i in range(len(features))],
        'OLS_pval': [round(ols.pvalues.iloc[i+1],6) for i in range(len(features))],
        'RF_importance': [round(fi_rf[f],6) for f in features],
        'XGB_importance': [round(fi_xgb[f],6) for f in features],
        'SHAP_importance': [round(shap_imp[f],3) for f in features],
    }).sort_values('SHAP_importance', ascending=False).to_csv(
        os.path.join(RESULTS_DIR, 'feature_importance.csv'), index=False, encoding='utf-8-sig')
    
    json.dump({
        'version': 'v3',
        'change': '학교수(총합) 변수 제거 — 구조적 선형결합 해소',
        'features': features,
        'n_features': len(features),
        'RF_best': rf_grid.best_params_,
        'XGB_best': bp, 'XGB_n_estimators': xgb.best_iteration,
        'XGB_5fold_CV': f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
        'split': 'train 70%/val 10%/test 20%',
        'total': len(X), 'train': len(X_train), 'val': len(X_val), 'test': len(X_test),
    }, open(os.path.join(RESULTS_DIR, 'hyperparameters.json'), 'w'), ensure_ascii=False, indent=2)
    
    print("\n✅ v3 완료!")

if __name__ == '__main__':
    main()
