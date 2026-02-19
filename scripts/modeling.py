#!/usr/bin/env python3
"""모델링: OLS → Random Forest → XGBoost → SHAP 분석"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_and_prepare():
    """데이터 로드 및 모델링용 변수 준비"""
    print("=" * 60)
    print("데이터 로드 및 변수 준비")
    print("=" * 60)
    
    df = pd.read_csv(os.path.join(DATA_DIR, 'apartment_final.csv'))
    
    # 구 단위 시설 수 집계 추가
    # 학교
    schools = pd.read_csv(os.path.join(DATA_DIR, 'seoul_schools.csv'))
    if '도로명주소' in schools.columns:
        schools['구'] = schools['도로명주소'].str.extract(r'서울특별시\s+(\S+구)')
        school_count = schools.groupby('구').size().rename('학교수')
        # 학교 종류별
        if '학교종류' in schools.columns:
            elem = schools[schools['학교종류'].str.contains('초등', na=False)].groupby('구').size().rename('초등학교수')
            middle = schools[schools['학교종류'].str.contains('중학', na=False)].groupby('구').size().rename('중학교수')
            high = schools[schools['학교종류'].str.contains('고등', na=False)].groupby('구').size().rename('고등학교수')
            school_detail = pd.concat([school_count, elem, middle, high], axis=1).fillna(0)
        else:
            school_detail = school_count.to_frame()
        df = df.merge(school_detail, on='구', how='left')
    
    # 지하철역 수 (구 단위 — 역 이름에서 추출 어려우므로 전체 서울 평균 사용)
    subway = pd.read_csv(os.path.join(DATA_DIR, 'seoul_subway_stations.csv'))
    # 노선 수를 변수로 활용
    
    # 공원 수 — 구 정보 추출
    parks = pd.read_csv(os.path.join(DATA_DIR, 'seoul_parks.csv'))
    if any('ADDR' in c or '주소' in c or 'P_ADDR' in c for c in parks.columns):
        addr_col = [c for c in parks.columns if 'ADDR' in c.upper() or '주소' in c][0] if [c for c in parks.columns if 'ADDR' in c.upper() or '주소' in c] else None
        if addr_col:
            parks['구'] = parks[addr_col].str.extract(r'(\S+구)')
            park_count = parks.groupby('구').size().rename('공원수')
            df = df.merge(park_count, on='구', how='left')
    
    # 백화점/대형점포 수
    stores = pd.read_csv(os.path.join(DATA_DIR, 'seoul_large_stores.csv'))
    if 'RDNWHLADDR' in stores.columns:
        stores['구'] = stores['RDNWHLADDR'].str.extract(r'서울특별시\s+(\S+구)')
        # 백화점만
        if 'UPTAENM' in stores.columns:
            dept_count = stores[stores['UPTAENM'] == '백화점'].groupby('구').size().rename('백화점수')
            df = df.merge(dept_count, on='구', how='left')
    
    # 결측치 처리
    fill_cols = ['학교수', '초등학교수', '중학교수', '고등학교수', '공원수', '백화점수']
    for c in fill_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    
    # 구 결측 제거
    df = df.dropna(subset=['구'])
    
    print(f"최종 데이터: {len(df):,}건")
    print(f"컬럼: {list(df.columns)}")
    
    return df

def select_features(df):
    """모델링용 독립변수 선택"""
    # 독립변수 후보
    feature_candidates = [
        # 물리적 특성
        '전용면적', '층', '건물연령',
        # 입지/환경 (구 단위)
        '강남구분', '학교수', '초등학교수', '중학교수', '고등학교수', '공원수', '백화점수',
        # 거시경제
        '기준금리', 'CD금리', '소비자물가지수', 'M2',
    ]
    
    features = [f for f in feature_candidates if f in df.columns]
    target = '거래금액'
    
    # 결측치 있는 행 제거
    subset = df[features + [target]].dropna()
    
    X = subset[features]
    y = subset[target]
    
    print(f"\n독립변수 ({len(features)}개): {features}")
    print(f"종속변수: {target}")
    print(f"분석 데이터: {len(subset):,}건")
    
    return X, y, features

def evaluate_model(name, y_true, y_pred):
    """모델 평가 지표"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"\n{'='*40}")
    print(f"📊 {name} 결과")
    print(f"{'='*40}")
    print(f"  R²:    {r2:.4f}")
    print(f"  RMSE:  {rmse:,.0f} (만원)")
    print(f"  MAE:   {mae:,.0f} (만원)")
    print(f"  MAPE:  {mape:.2f}%")
    
    return {'model': name, 'R2': round(r2, 4), 'RMSE': round(rmse, 0), 'MAE': round(mae, 0), 'MAPE': round(mape, 2)}

def main():
    # 데이터 준비
    df = load_and_prepare()
    X, y, features = select_features(df)
    
    # Train/Test Split (8:2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTrain: {len(X_train):,}건 / Test: {len(X_test):,}건")
    
    results = []
    
    # ─── 1. OLS (다중회귀) ───
    print("\n" + "━" * 60)
    print("1️⃣  OLS 다중회귀분석")
    print("━" * 60)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    ols = LinearRegression()
    ols.fit(X_train_scaled, y_train)
    y_pred_ols = ols.predict(X_test_scaled)
    
    res_ols = evaluate_model("OLS 다중회귀", y_test.values, y_pred_ols)
    results.append(res_ols)
    
    # OLS 계수
    print("\n  📌 OLS 회귀계수 (표준화):")
    coefs = pd.Series(ols.coef_, index=features).sort_values(key=abs, ascending=False)
    for feat, coef in coefs.items():
        print(f"    {feat:15s}: {coef:>10,.1f}")
    
    # ─── 2. Random Forest ───
    print("\n" + "━" * 60)
    print("2️⃣  Random Forest")
    print("━" * 60)
    
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=10, 
                                n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    res_rf = evaluate_model("Random Forest", y_test.values, y_pred_rf)
    results.append(res_rf)
    
    # RF Feature Importance
    print("\n  📌 RF 변수 중요도:")
    fi_rf = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    for feat, imp in fi_rf.items():
        bar = '█' * int(imp * 50)
        print(f"    {feat:15s}: {imp:.4f} {bar}")
    
    # ─── 3. XGBoost ───
    print("\n" + "━" * 60)
    print("3️⃣  XGBoost")
    print("━" * 60)
    
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("  ⚠️ xgboost 미설치, 설치 중...")
        os.system("pip install --break-system-packages xgboost -q")
        from xgboost import XGBRegressor
    
    xgb = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred_xgb = xgb.predict(X_test)
    
    res_xgb = evaluate_model("XGBoost", y_test.values, y_pred_xgb)
    results.append(res_xgb)
    
    # XGB Feature Importance
    print("\n  📌 XGBoost 변수 중요도:")
    fi_xgb = pd.Series(xgb.feature_importances_, index=features).sort_values(ascending=False)
    for feat, imp in fi_xgb.items():
        bar = '█' * int(imp * 50)
        print(f"    {feat:15s}: {imp:.4f} {bar}")
    
    # ─── 4. SHAP 분석 ───
    print("\n" + "━" * 60)
    print("4️⃣  SHAP 분석 (XGBoost)")
    print("━" * 60)
    
    try:
        import shap
    except ImportError:
        print("  ⚠️ shap 미설치, 설치 중...")
        os.system("pip install --break-system-packages shap -q")
        import shap
    
    explainer = shap.TreeExplainer(xgb)
    
    # 샘플링 (전체 데이터가 크면 시간이 오래 걸림)
    sample_size = min(5000, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=42)
    shap_values = explainer.shap_values(X_sample)
    
    # SHAP 평균 절대값
    print(f"\n  📌 SHAP 변수 중요도 (평균 |SHAP|, 샘플 {sample_size}건):")
    shap_importance = pd.Series(np.abs(shap_values).mean(axis=0), index=features).sort_values(ascending=False)
    for feat, imp in shap_importance.items():
        bar = '█' * int(imp / shap_importance.max() * 30)
        print(f"    {feat:15s}: {imp:>10,.1f} {bar}")
    
    # SHAP 결과 저장
    shap_df = pd.DataFrame(shap_values, columns=features)
    shap_df.to_csv(os.path.join(RESULTS_DIR, 'shap_values.csv'), index=False, encoding='utf-8-sig')
    
    # ─── 5. 강남 vs 비강남 비교 ───
    print("\n" + "━" * 60)
    print("5️⃣  강남 vs 비강남 비교분석")
    print("━" * 60)
    
    for label, mask_val in [("강남3구 (강남/서초/송파)", 1), ("비강남", 0)]:
        mask = X_test['강남구분'] == mask_val
        if mask.sum() == 0:
            continue
        y_sub = y_test[mask].values
        y_pred_sub = y_pred_xgb[mask]
        
        r2_sub = r2_score(y_sub, y_pred_sub)
        rmse_sub = np.sqrt(mean_squared_error(y_sub, y_pred_sub))
        print(f"\n  {label}:")
        print(f"    건수: {mask.sum():,}")
        print(f"    평균 실거래가: {y_sub.mean():,.0f}만원")
        print(f"    R²: {r2_sub:.4f}")
        print(f"    RMSE: {rmse_sub:,.0f}만원")
    
    # 강남/비강남 SHAP 비교
    gangnam_mask = X_sample['강남구분'] == 1
    if gangnam_mask.sum() > 0:
        print(f"\n  📌 SHAP 변수 중요도 비교:")
        print(f"  {'변수':15s} | {'강남3구':>10s} | {'비강남':>10s} | {'차이':>10s}")
        print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
        
        shap_gangnam = np.abs(shap_values[gangnam_mask.values]).mean(axis=0)
        shap_non = np.abs(shap_values[~gangnam_mask.values]).mean(axis=0)
        
        for i, feat in enumerate(features):
            diff = shap_gangnam[i] - shap_non[i]
            print(f"  {feat:15s} | {shap_gangnam[i]:>10,.1f} | {shap_non[i]:>10,.1f} | {diff:>+10,.1f}")
    
    # ─── 결과 종합 ───
    print("\n" + "━" * 60)
    print("📊 모델 성능 비교 종합")
    print("━" * 60)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    results_df.to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False, encoding='utf-8-sig')
    
    # 변수 중요도 저장
    importance_df = pd.DataFrame({
        'variable': features,
        'OLS_coef': [coefs.get(f, 0) for f in features],
        'RF_importance': [fi_rf.get(f, 0) for f in features],
        'XGB_importance': [fi_xgb.get(f, 0) for f in features],
        'SHAP_importance': [shap_importance.get(f, 0) for f in features],
    }).sort_values('SHAP_importance', ascending=False)
    importance_df.to_csv(os.path.join(RESULTS_DIR, 'feature_importance.csv'), index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 결과 저장 완료:")
    print(f"  - {RESULTS_DIR}/model_comparison.csv")
    print(f"  - {RESULTS_DIR}/feature_importance.csv")
    print(f"  - {RESULTS_DIR}/shap_values.csv")

if __name__ == '__main__':
    main()
