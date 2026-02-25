"""v5 행정동 데이터로 OLS 회귀계수 테이블 생성"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'apartment_final_v5_dong.csv')
df = pd.read_csv(DATA)

features = ['전용면적', '층', '건물연령', '강남구분', '초등학교수', '중학교수', '고등학교수',
            'CCTV수', '백화점수', '지하철역수', '기준금리', 'CD금리', '소비자물가지수', 'M2']
target = '거래금액'

X = df[features].copy()
y = df[target].copy()

# Drop NaN
mask = X.notna().all(axis=1) & y.notna()
X = X[mask]
y = y[mask]

X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()

# Save summary
print(model.summary())

# Save coefficients table
coef_df = pd.DataFrame({
    '변수': ['(상수)'] + features,
    '비표준화 회귀계수': model.params.values,
    '표준오차': model.bse.values,
    't값': model.tvalues.values,
    'p값': model.pvalues.values
})
coef_df.to_csv(os.path.join(os.path.dirname(__file__), '..', 'results', 'ols_detailed_v5_dong.csv'), index=False)
print("\nSaved to results/ols_detailed_v5_dong.csv")
print(f"\nR² = {model.rsquared:.6f}")
print(f"Adj R² = {model.rsquared_adj:.6f}")
print(f"F = {model.fvalue:.1f}, p = {model.f_pvalue}")
print(f"N = {len(y)}")
