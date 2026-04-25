#!/usr/bin/env python3
"""
v7 vs v8 어블레이션: 변수셋·시점 처리별 비교.

비교할 피처셋:
  A. 행정동만 (v7 원본): legacy_counts (초/중/고/어린이집/학원/공원/도서관/백화점/CCTV/지하철역수)
  B. 거리만 (시점 무관): v8 거리변수, 연도 스냅샷 없음 (모든 거래에 2026 스냅샷 적용)
  C. 거리+시점: v8 전체 (연도별 시설 active 필터)
  D. 거리+시점+행정동: C + legacy 보조 (robustness)

공통 변수: 전용면적·층·건물연령·강남구분·거시경제(기준금리/CD/CPI/M2)

분할: Random / Temporal(≤2023 train, ≥2024 test) / Group(단지 단위 5-fold)

출력:
  results/ablation_v7_v8.json
  results/ablation_v7_v8.csv (요약표)
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import r2_score

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')

COMMON = ['전용면적', '층', '건물연령', '강남구분',
          '기준금리', 'CD금리', '소비자물가지수', 'M2']

# A: 행정동 단순 개수 (v7 원본 + 레거시 유지)
LEGACY = ['초등학교수_legacy', '중학교수_legacy', '고등학교수_legacy',
          'CCTV수_legacy', '백화점수_legacy', '지하철역수_legacy',
          '공원수_legacy', '도서관수_legacy', '학원수_legacy', '어린이집수_legacy']

# B/C: 거리 변수 (학술 문헌 관행 기반 축약; 시설별 최근접 + 1km 개수 선택)
DISTANCE = [
    'subway_nearest_m', 'subway_count_1000m',
    'elem_school_nearest_m', 'elem_school_count_1000m',
    'middle_school_nearest_m',
    'library_nearest_m',
    'park_nearest_m', 'park_within_1km',
    'mart_nearest_m', 'mart_count_1000m',
    'department_nearest_m', 'department_within_1km',
    'academy_nearest_m', 'academy_count_1000m',
    'childcare_count_1000m',
    'hospital_general_nearest_m', 'hospital_count_1000m',
    'cctv_count_500m',
]


def metrics(y_true, y_pred):
    ape = np.abs(np.exp(y_true) - np.exp(y_pred)) / np.exp(y_true)
    return {
        'r2': float(r2_score(y_true, y_pred)),
        'mape': float(np.mean(ape)),
        'median_ape': float(np.median(ape)),
        'n': int(len(y_true)),
    }


def eval_split(df, features, split_kind):
    y = df['log㎡당가격'].values
    if split_kind == 'random':
        tr, te = train_test_split(df, test_size=0.2, random_state=42)
    elif split_kind == 'temporal':
        tr = df[df['거래년도'] <= 2023]; te = df[df['거래년도'] >= 2024]
    elif split_kind == 'group':
        groups = (df['구'] + '|' + df['법정동'] + '|' + df['아파트명']).values
        gkf = GroupKFold(n_splits=5)
        tr_i, te_i = next(gkf.split(df, y, groups))
        tr = df.iloc[tr_i]; te = df.iloc[te_i]

    out = {'n_train': len(tr), 'n_test': len(te)}
    xgb = XGBRegressor(n_estimators=400, max_depth=8, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8,
                       n_jobs=-1, random_state=42, tree_method='hist')
    xgb.fit(tr[features], tr['log㎡당가격'].values)
    out['xgb'] = metrics(te['log㎡당가격'].values, xgb.predict(te[features]))
    ols = LinearRegression().fit(tr[features], tr['log㎡당가격'].values)
    out['ols'] = metrics(te['log㎡당가격'].values, ols.predict(te[features]))
    return out


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, 'apartment_final_v8.csv'), low_memory=False)
    # A에는 레거시 컬럼이 필요하니 결측 제거
    df = df.dropna(subset=LEGACY + DISTANCE + COMMON + ['log㎡당가격'])
    print(f"완전 레코드: {len(df):,}")

    # v7 스냅샷 에뮬레이션 위한 C_snap (C: v8 전체) — 현재 v8 이미 시점 정합
    # B (거리만·시점무관)는 apartment_distance_features.csv (2026 스냅샷)를 이용
    # v8 데이터엔 연도별 거리만 있으니, B용 2026 스냅샷 거리를 따로 결합
    snap = pd.read_csv(os.path.join(DATA_DIR, 'apartment_distance_features.csv'))
    snap = snap.rename(columns={'gu': '구', 'bjd': '법정동', 'apt_name_raw': '아파트명'})
    keep_snap_cols = ['구', '법정동', '아파트명'] + [c for c in snap.columns if c.endswith('_m') or c.endswith('_1km') or c.endswith('_500m') or c.endswith('_1000m') or c.endswith('_2000m') or c.endswith('_2km')]
    snap = snap[keep_snap_cols].drop_duplicates(subset=['구', '법정동', '아파트명'])
    # 이름 중복 방지
    snap = snap.rename(columns={c: f'{c}__snap' for c in snap.columns if c not in ['구', '법정동', '아파트명']})
    df_b = df.merge(snap, on=['구', '법정동', '아파트명'], how='left')
    DISTANCE_SNAP = [f'{c}__snap' for c in DISTANCE]
    df_b = df_b.dropna(subset=DISTANCE_SNAP)
    print(f"B용 스냅샷 결합 후: {len(df_b):,}")

    scenarios = {
        'A_행정동만_v7': (df, COMMON + LEGACY),
        'B_거리만_시점무관(2026스냅샷)': (df_b, COMMON + DISTANCE_SNAP),
        'C_거리+시점정합(v8)': (df, COMMON + DISTANCE),
        'D_거리+시점+행정동': (df, COMMON + DISTANCE + LEGACY),
    }

    results = {}
    rows = []
    for name, (d, feats) in scenarios.items():
        print(f"\n=== {name} (n={len(d):,}, features={len(feats)}) ===")
        results[name] = {'n': len(d), 'n_features': len(feats)}
        for split in ['random', 'temporal', 'group']:
            try:
                r = eval_split(d, feats, split)
                results[name][split] = r
                print(f"  {split:>8}: XGB R²={r['xgb']['r2']:.4f} MAPE={r['xgb']['mape']:.4f}  OLS R²={r['ols']['r2']:.4f}")
                rows.append({'scenario': name, 'split': split,
                             'xgb_r2': r['xgb']['r2'], 'xgb_mape': r['xgb']['mape'],
                             'ols_r2': r['ols']['r2']})
            except Exception as e:
                print(f"  {split} 실패: {e}")

    with open(os.path.join(RESULTS, 'ablation_v7_v8.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'ablation_v7_v8.csv'), index=False)
    print(f"\n저장: results/ablation_v7_v8.json, .csv")

    # 비교표 출력
    print("\n=== 요약 (XGB R²) ===")
    piv = pd.DataFrame(rows).pivot(index='scenario', columns='split', values='xgb_r2')
    print(piv.round(4).to_string())


if __name__ == '__main__':
    main()
