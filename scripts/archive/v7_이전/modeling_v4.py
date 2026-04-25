"""
Modeling v4: 14 independent variables (지하철역수 추가)
OLS, Random Forest, XGBoost + SHAP analysis
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings, os, json
warnings.filterwarnings('ignore')

# Font setup
font_path = '/home/nexus/.fonts/NotoSansKR-Regular.otf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Noto Sans KR'
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = '/home/nexus/thesis-seoul-apt-xai/results'
PLOT_DIR = os.path.join(OUT_DIR, 'plots_v4')
os.makedirs(PLOT_DIR, exist_ok=True)

# ===================== DATA =====================
print("=" * 60)
print("MODELING v4: 14 Variables (+ 지하철역수)")
print("=" * 60)

df = pd.read_csv('/home/nexus/thesis-seoul-apt-xai/data/apartment_final_v2.csv')

# Load environment variables (schools, parks, dept stores)
# These should already be in a separate file or we reconstruct
# Check if we need to add school/park/store data
print(f"Data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# We need: 초등학교수, 중학교수, 고등학교수, 공원수, 백화점수
# These were in apartment_final.csv in v3 but seem missing now
# Let's check the original preprocessing
env_cols = ['초등학교수', '중학교수', '고등학교수', '공원수', '백화점수']
missing_cols = [c for c in env_cols if c not in df.columns]
if missing_cols:
    print(f"Missing env columns: {missing_cols}")
    # Load from the original preprocessed data or env data
    # Check if there's environment data
    env_files = [f for f in os.listdir('/home/nexus/thesis-seoul-apt-xai/data/') if 'env' in f.lower() or 'school' in f.lower() or 'preprocess' in f.lower()]
    print(f"Available env files: {env_files}")
    
    # Try loading from preprocessed data
    try:
        df_orig = pd.read_csv('/home/nexus/thesis-seoul-apt-xai/data/apartment_preprocessed.csv')
        for c in missing_cols:
            if c in df_orig.columns:
                # Merge by 구
                mapping = df_orig.groupby('구')[c].first().reset_index()
                df = df.merge(mapping, on='구', how='left')
                print(f"  Added {c} from preprocessed data")
    except:
        print("No preprocessed file found, checking other sources...")

# Define features
FEATURES = ['전용면적', '층', '건물연령', '강남구분', 
            '초등학교수', '중학교수', '고등학교수', '공원수', '백화점수',
            '기준금리', 'CD금리', '소비자물가지수', 'M2', '지하철역수']
TARGET = '거래금액'

# Check all features exist
available = [f for f in FEATURES if f in df.columns]
missing = [f for f in FEATURES if f not in df.columns]
print(f"\nAvailable features: {len(available)}/{len(FEATURES)}")
if missing:
    print(f"MISSING: {missing}")
    print("Attempting to load environment data...")
    
    # Load individual env datasets
    data_dir = '/home/nexus/thesis-seoul-apt-xai/data/'
    
    # Schools
    try:
        schools = pd.read_csv(data_dir + 'seoul_schools.csv')
        print(f"Schools data: {schools.shape}, cols: {list(schools.columns)[:10]}")
    except:
        print("No schools CSV")
    
    # Parks  
    try:
        parks = pd.read_csv(data_dir + 'seoul_parks.csv')
        print(f"Parks data: {parks.shape}")
    except:
        print("No parks CSV")
        
    # Stores
    try:
        stores = pd.read_csv(data_dir + 'seoul_stores.csv')
        print(f"Stores data: {stores.shape}")
    except:
        print("No stores CSV")

print(f"\nFinal check - all features in df: {all(f in df.columns for f in FEATURES)}")

if not all(f in df.columns for f in FEATURES):
    print("\n⚠️ Some features missing. Need to reconstruct from raw data.")
    print("Available columns:", list(df.columns))
    exit(1)

X = df[FEATURES].copy()
y = df[TARGET].copy()

print(f"\nX shape: {X.shape}, y shape: {y.shape}")
print(f"Features: {FEATURES}")

# ===================== SPLIT =====================
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.125, random_state=42)
# 0.125 of 0.8 = 0.1 of total -> 70/10/20

print(f"\nTrain: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# ===================== OLS =====================
print("\n" + "=" * 40)
print("1. OLS Regression")
print("=" * 40)

ols = LinearRegression()
ols.fit(X_train_full, y_train_full)
y_pred_ols = ols.predict(X_test)

r2_ols = r2_score(y_test, y_pred_ols)
rmse_ols = np.sqrt(mean_squared_error(y_test, y_pred_ols))
mae_ols = mean_absolute_error(y_test, y_pred_ols)
mape_ols = mean_absolute_percentage_error(y_test, y_pred_ols) * 100

print(f"R²: {r2_ols:.3f}, RMSE: {rmse_ols:.0f}, MAE: {mae_ols:.0f}, MAPE: {mape_ols:.2f}%")

# OLS coefficients
print("\nOLS Coefficients:")
for feat, coef in zip(FEATURES, ols.coef_):
    print(f"  {feat}: {coef:.3f}")
print(f"  Intercept: {ols.intercept_:.3f}")

# VIF
from numpy.linalg import inv
X_vif = X_train_full.values
corr = np.corrcoef(X_vif.T)
try:
    vif_values = np.diag(inv(corr))
    print("\nVIF:")
    for feat, v in sorted(zip(FEATURES, vif_values), key=lambda x: -x[1]):
        print(f"  {feat}: {v:.2f}")
except:
    print("VIF calculation failed (singular matrix)")

# ===================== RANDOM FOREST =====================
print("\n" + "=" * 40)
print("2. Random Forest")
print("=" * 40)

# GridSearch on subsample
np.random.seed(42)
idx_sub = np.random.choice(len(X_train_full), 50000, replace=False)
X_sub = X_train_full.iloc[idx_sub]
y_sub = y_train_full.iloc[idx_sub]

rf_params = {
    'max_depth': [10, 15],
    'min_samples_leaf': [5, 10],
    'n_estimators': [200]
}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1), 
                       rf_params, cv=3, scoring='r2', n_jobs=-1, verbose=0)
rf_grid.fit(X_sub, y_sub)
print(f"Best RF params: {rf_grid.best_params_}")

rf = RandomForestRegressor(**rf_grid.best_params_, random_state=42, n_jobs=-1)
rf.fit(X_train_full, y_train_full)
y_pred_rf = rf.predict(X_test)

r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf) * 100

print(f"R²: {r2_rf:.3f}, RMSE: {rmse_rf:.0f}, MAE: {mae_rf:.0f}, MAPE: {mape_rf:.2f}%")

# ===================== XGBOOST =====================
print("\n" + "=" * 40)
print("3. XGBoost")
print("=" * 40)

xgb_params = {
    'learning_rate': [0.05, 0.1],
    'max_depth': [6, 8],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}
xgb_grid = GridSearchCV(xgb.XGBRegressor(random_state=42, n_estimators=500, n_jobs=-1, tree_method='hist'),
                        xgb_params, cv=3, scoring='r2', n_jobs=-1, verbose=0)
xgb_grid.fit(X_sub, y_sub)
print(f"Best XGB params: {xgb_grid.best_params_}")

xgb_model = xgb.XGBRegressor(
    **xgb_grid.best_params_,
    n_estimators=2000,
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
    early_stopping_rounds=50
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print(f"Best iteration: {xgb_model.best_iteration}")

y_pred_xgb = xgb_model.predict(X_test)

r2_xgb = r2_score(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mape_xgb = mean_absolute_percentage_error(y_test, y_pred_xgb) * 100

print(f"R²: {r2_xgb:.3f}, RMSE: {rmse_xgb:.0f}, MAE: {mae_xgb:.0f}, MAPE: {mape_xgb:.2f}%")

# Cross-validation
cv_scores = cross_val_score(xgb.XGBRegressor(**xgb_grid.best_params_, n_estimators=xgb_model.best_iteration,
                                               random_state=42, n_jobs=-1, tree_method='hist'),
                            X_train_full, y_train_full, cv=5, scoring='r2', n_jobs=-1)
print(f"5-Fold CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ===================== COMPARISON TABLE =====================
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(f"{'Model':<20} {'R²':>8} {'RMSE':>12} {'MAE':>12} {'MAPE':>8}")
print("-" * 60)
print(f"{'OLS':<20} {r2_ols:>8.3f} {rmse_ols:>12,.0f} {mae_ols:>12,.0f} {mape_ols:>7.2f}%")
print(f"{'Random Forest':<20} {r2_rf:>8.3f} {rmse_rf:>12,.0f} {mae_rf:>12,.0f} {mape_rf:>7.2f}%")
print(f"{'XGBoost':<20} {r2_xgb:>8.3f} {rmse_xgb:>12,.0f} {mae_xgb:>12,.0f} {mape_xgb:>7.2f}%")

# ===================== SHAP =====================
print("\n" + "=" * 40)
print("4. SHAP Analysis")
print("=" * 40)

explainer = shap.TreeExplainer(xgb_model)

# Sample 5000 for SHAP
np.random.seed(42)
shap_idx = np.random.choice(len(X_test), 5000, replace=False)
X_shap = X_test.iloc[shap_idx]
shap_values = explainer.shap_values(X_shap)

# Global importance
mean_shap = np.abs(shap_values).mean(axis=0)
importance = pd.DataFrame({'feature': FEATURES, 'mean_abs_shap': mean_shap})
importance = importance.sort_values('mean_abs_shap', ascending=False)
importance['pct'] = importance['mean_abs_shap'] / importance['mean_abs_shap'].sum() * 100

print("\nSHAP Feature Importance (Global):")
for _, row in importance.iterrows():
    print(f"  {row['feature']:<15} {row['mean_abs_shap']:>12,.0f} ({row['pct']:.1f}%)")

# ===================== SHAP PLOTS =====================
print("\nGenerating SHAP plots...")

# Fig4: Summary plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_shap, show=False, max_display=14)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'fig4_shap_summary.png'), dpi=300, bbox_inches='tight')
plt.close()

# Fig5: Bar plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_shap, plot_type='bar', show=False, max_display=14)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'fig5_shap_bar.png'), dpi=300, bbox_inches='tight')
plt.close()

# Fig6: Dependence - 전용면적
plt.figure(figsize=(8, 6))
shap.dependence_plot('전용면적', shap_values, X_shap, show=False)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'fig6_dep_area.png'), dpi=300, bbox_inches='tight')
plt.close()

# Fig7: Dependence - 건물연령
plt.figure(figsize=(8, 6))
shap.dependence_plot('건물연령', shap_values, X_shap, show=False)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'fig7_dep_age.png'), dpi=300, bbox_inches='tight')
plt.close()

# Fig8: Dependence - M2
plt.figure(figsize=(8, 6))
shap.dependence_plot('M2', shap_values, X_shap, show=False)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'fig8_dep_m2.png'), dpi=300, bbox_inches='tight')
plt.close()

# Fig_new: Dependence - 지하철역수
plt.figure(figsize=(8, 6))
shap.dependence_plot('지하철역수', shap_values, X_shap, show=False)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'fig_dep_subway.png'), dpi=300, bbox_inches='tight')
plt.close()

# Fig9: Waterfall (first sample)
plt.figure(figsize=(10, 8))
shap.plots.waterfall(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value,
                                       data=X_shap.iloc[0], feature_names=FEATURES), show=False)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'fig9_waterfall.png'), dpi=300, bbox_inches='tight')
plt.close()

# ===================== GANGNAM vs NON-GANGNAM =====================
print("\n" + "=" * 40)
print("5. Gangnam vs Non-Gangnam")
print("=" * 40)

gangnam_mask = X_test['강남구분'] == 1
X_gn = X_test[gangnam_mask]
X_ngn = X_test[~gangnam_mask]
y_gn = y_test[gangnam_mask]
y_ngn = y_test[~gangnam_mask]

# Performance by region
for name, Xr, yr in [("강남3구", X_gn, y_gn), ("비강남", X_ngn, y_ngn)]:
    yp = xgb_model.predict(Xr)
    r2 = r2_score(yr, yp)
    rmse = np.sqrt(mean_squared_error(yr, yp))
    mae = mean_absolute_error(yr, yp)
    print(f"{name}: N={len(yr):,}, R²={r2:.3f}, RMSE={rmse:,.0f}, MAE={mae:,.0f}, Mean={yr.mean():,.0f}")

# SHAP by region
shap_gn_idx = np.random.choice(len(X_gn), min(2000, len(X_gn)), replace=False)
shap_ngn_idx = np.random.choice(len(X_ngn), min(3000, len(X_ngn)), replace=False)

sv_gn = explainer.shap_values(X_gn.iloc[shap_gn_idx])
sv_ngn = explainer.shap_values(X_ngn.iloc[shap_ngn_idx])

print("\nSHAP by Region:")
print(f"{'Feature':<15} {'강남3구':>12} {'비강남':>12} {'차이':>12}")
print("-" * 55)
for i, feat in enumerate(FEATURES):
    gn_val = np.abs(sv_gn[:, i]).mean()
    ngn_val = np.abs(sv_ngn[:, i]).mean()
    print(f"{feat:<15} {gn_val:>12,.0f} {ngn_val:>12,.0f} {gn_val - ngn_val:>12,.0f}")

# ===================== SAVE RESULTS =====================
results = {
    'model_comparison': {
        'OLS': {'R2': round(r2_ols, 3), 'RMSE': round(rmse_ols), 'MAE': round(mae_ols), 'MAPE': round(mape_ols, 2)},
        'RF': {'R2': round(r2_rf, 3), 'RMSE': round(rmse_rf), 'MAE': round(mae_rf), 'MAPE': round(mape_rf, 2)},
        'XGBoost': {'R2': round(r2_xgb, 3), 'RMSE': round(rmse_xgb), 'MAE': round(mae_xgb), 'MAPE': round(mape_xgb, 2)},
    },
    'xgb_cv_r2': f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}",
    'xgb_best_iteration': xgb_model.best_iteration,
    'xgb_best_params': xgb_grid.best_params_,
    'rf_best_params': rf_grid.best_params_,
    'features': FEATURES,
    'n_features': len(FEATURES),
    'shap_importance': importance[['feature', 'mean_abs_shap', 'pct']].to_dict('records'),
}

with open(os.path.join(OUT_DIR, 'modeling_v4_results.json'), 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)

print(f"\n✅ All results saved to {OUT_DIR}/")
print(f"✅ Plots saved to {PLOT_DIR}/")
print("\nDONE!")
