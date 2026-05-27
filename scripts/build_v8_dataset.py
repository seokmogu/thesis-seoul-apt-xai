#!/usr/bin/env python3
"""
v8 데이터셋: 거래 레코드 × 연도별 거리 변수 조인.

입력:
  data/apartment_final_v6_dong.csv (거래 391,826행, 기존 인프라 집계 컬럼 포함)
  data/apartment_distance_features_yearly.csv (60,207행 = 8,601 단지 × 7년)

조인 키: (구, 법정동, 아파트명) × 거래년도

검증:
  - 조인 누락률 < 1% 목표
  - 주요 거리 변수의 결측률 점검

출력:
  data/apartment_final_v8.csv
"""
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def main():
    trades = pd.read_csv(os.path.join(DATA_DIR, 'apartment_final_v6_dong.csv'), low_memory=False)
    dist = pd.read_csv(os.path.join(DATA_DIR, 'apartment_distance_features_yearly.csv'), low_memory=False)
    dist = dist.rename(columns={'gu': '구', 'bjd': '법정동', 'apt_name_raw': '아파트명'})
    print(f"거래: {len(trades):,}, 연도별 거리변수: {len(dist):,}")

    # 기존 행정동 경계 집계 컬럼은 기준선 비교용으로만 보존하고 이름 충돌을 피한다.
    admin_count_cols = ['초등학교수', '중학교수', '고등학교수', 'CCTV수', '백화점수',
                        '지하철역수', '공원수', '도서관수', '학원수', '어린이집수']
    rename = {c: f'{c}_admin_count' for c in admin_count_cols if c in trades.columns}
    trades = trades.rename(columns=rename)
    print(f"행정동 집계 기준선 컬럼 rename: {len(rename)}개")

    merged = trades.merge(
        dist, on=['구', '법정동', '아파트명', '거래년도'], how='left', validate='many_to_one')

    dropped = merged['lat'].isna().sum() if 'lat' in merged.columns else len(merged)
    print(f"조인 결측: {dropped:,} ({100*dropped/len(merged):.2f}%)")

    if dropped > 0:
        # 누락 샘플 분석
        miss = merged[merged['lat'].isna()]
        print(f"누락 거래 연도별: {miss['거래년도'].value_counts().sort_index().to_dict()}")
        print("누락 거래 상위 단지 5건:")
        print(miss.groupby(['구', '법정동', '아파트명']).size().sort_values(ascending=False).head(5).to_string())

    # 핵심 파생: log(거래금액/전용면적)
    merged['㎡당가격'] = merged['거래금액'] / merged['전용면적']
    import numpy as np
    merged['log㎡당가격'] = np.log(merged['㎡당가격'])

    out = os.path.join(DATA_DIR, 'apartment_final_v8.csv')
    merged.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"\n저장: {out}")
    print(f"최종 shape: {merged.shape}")

    # 주요 거리 변수 요약 (조인 성공건만)
    ok = merged.dropna(subset=['lat'])
    key_cols = ['subway_nearest_m', 'elem_school_nearest_m', 'library_nearest_m',
                'department_nearest_m', 'mart_nearest_m', 'park_nearest_m',
                'academy_nearest_m', 'hospital_general_nearest_m']
    print("\n주요 최근접거리(m) 중앙값:")
    for c in key_cols:
        if c in ok.columns:
            print(f"  {c}: {ok[c].median():.0f}")


if __name__ == '__main__':
    main()
