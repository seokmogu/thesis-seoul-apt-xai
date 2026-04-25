#!/usr/bin/env python3
"""전처리 v5: 2019-2025 데이터 + CCTV(공원 대체) + 구 단위 변수"""
import os, sys
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def main():
    # 1. 실거래 데이터 로드
    print("=== 1. 실거래 데이터 ===")
    df = pd.read_csv(os.path.join(DATA_DIR, 'apartment_trades_v2.csv'))
    print(f"  원본: {len(df):,}건, {df['거래년월'].min()}~{df['거래년월'].max()}")
    
    # 기본 전처리
    df['거래금액'] = pd.to_numeric(df['거래금액'], errors='coerce')
    df['건축년도'] = pd.to_numeric(df['건축년도'], errors='coerce')
    df['전용면적'] = pd.to_numeric(df['전용면적'], errors='coerce')
    df['층'] = pd.to_numeric(df['층'], errors='coerce')
    df = df.dropna(subset=['거래금액', '건축년도', '전용면적', '층'])
    
    # 건물연령
    df['건물연령'] = df['거래년도'] - df['건축년도']
    
    # 이상치 제거
    df = df[df['거래금액'] > 0]
    df = df[df['전용면적'] > 10]
    df = df[df['전용면적'] < 300]
    df = df[df['층'] > 0]
    df = df[df['건물연령'] >= 0]
    df = df[df['건물연령'] <= 80]
    
    # 평당가격 (만원/평)
    df['평당가격'] = df['거래금액'] / (df['전용면적'] / 3.3058)
    
    # 강남구분
    gangnam_gu = ['강남구', '서초구', '송파구']
    df['강남구분'] = df['구'].apply(lambda x: 1 if x in gangnam_gu else 0)
    
    print(f"  전처리 후: {len(df):,}건")
    
    # 2. 거시경제 변수
    print("\n=== 2. 거시경제 변수 ===")
    ecos = pd.read_csv(os.path.join(DATA_DIR, 'ecos_macro.csv'))
    ecos.rename(columns={'년월': '거래년월'}, inplace=True)
    ecos['거래년월'] = ecos['거래년월'].astype(str)
    df['거래년월'] = df['거래년월'].astype(str)
    
    # 2025년 거시 데이터 확인
    ecos_max = ecos['거래년월'].max()
    print(f"  거시데이터 범위: {ecos['거래년월'].min()}~{ecos_max}")
    
    if int(ecos_max) < 202512:
        print(f"  ⚠️ 거시데이터가 {ecos_max}까지만 있음 — 마지막 값으로 forward fill")
        last_row = ecos.iloc[-1].copy()
        missing_months = [m for m in df['거래년월'].unique() if m > ecos_max]
        new_rows = []
        for m in sorted(missing_months):
            row = last_row.copy()
            row['거래년월'] = m
            new_rows.append(row)
        if new_rows:
            ecos = pd.concat([ecos, pd.DataFrame(new_rows)], ignore_index=True)
            print(f"  {len(new_rows)}개월 forward fill 완료")
    
    df = df.merge(ecos, on='거래년월', how='left')
    macro_cols = ['기준금리', 'CD금리', '소비자물가지수', 'M2']
    missing_macro = df[macro_cols].isnull().sum()
    print(f"  거시 변수 결측: {dict(missing_macro[missing_macro > 0])}")
    
    # 3. 구별 환경 변수
    print("\n=== 3. 구별 환경 변수 ===")
    
    # 학교 (구 단위 집계)
    schools = pd.read_csv(os.path.join(DATA_DIR, 'schools_raw.csv'))
    school_gu = schools[schools['school_type'].isin(['초등학교', '중학교', '고등학교'])]\
        .groupby(['gu', 'school_type']).size().unstack(fill_value=0).reset_index()
    school_gu.columns = ['구', '고등학교수', '중학교수', '초등학교수']
    print(f"  학교 구별 집계: {len(school_gu)}개 구")
    
    # 지하철
    subway = pd.read_csv(os.path.join(DATA_DIR, 'subway_per_gu.csv'))
    print(f"  지하철: {len(subway)}개 구")
    
    # CCTV (공원 대체)
    cctv = pd.read_csv(os.path.join(DATA_DIR, 'cctv_per_gu.csv'))
    print(f"  CCTV: {len(cctv)}개 구")
    
    # 백화점
    env = pd.read_csv(os.path.join(DATA_DIR, 'env_per_gu.csv'))
    if '백화점수' in env.columns:
        dept = env[['구', '백화점수']]
    else:
        dept = env[['구'] + [c for c in env.columns if '백화점' in c]]
    print(f"  백화점: {len(dept)}개 구")
    
    # 머지
    df = df.merge(school_gu, on='구', how='left')
    df = df.merge(subway, on='구', how='left')
    df = df.merge(cctv, on='구', how='left')
    df = df.merge(dept, on='구', how='left')
    
    # 4. 최종 정리
    print("\n=== 4. 최종 정리 ===")
    
    target = '거래금액'
    features = ['전용면적', '층', '건물연령', '강남구분',
                '초등학교수', '중학교수', '고등학교수',
                'CCTV수', '백화점수', '지하철역수',
                '기준금리', 'CD금리', '소비자물가지수', 'M2']
    
    keep_cols = [target, '건축년도', '전용면적', '층', '법정동', '도로명', '아파트명',
                 '거래년월', '거래년도', '건물연령', '구', '강남구분', '평당가격'] + \
                [c for c in features if c not in ['전용면적', '층', '건물연령', '강남구분']]
    
    # Drop rows with missing features
    before = len(df)
    df = df.dropna(subset=features)
    print(f"  결측 제거: {before:,} → {len(df):,} ({before - len(df)} 제거)")
    
    # 저장
    out_path = os.path.join(DATA_DIR, 'apartment_final_v5.csv')
    df[keep_cols].to_csv(out_path, index=False, encoding='utf-8-sig')
    
    print(f"\n=== 완료 ===")
    print(f"  파일: apartment_final_v5.csv")
    print(f"  건수: {len(df):,}")
    print(f"  기간: {df['거래년월'].min()} ~ {df['거래년월'].max()}")
    print(f"  구: {df['구'].nunique()}개")
    print(f"  법정동: {df['법정동'].nunique()}개")
    print(f"  독립변수 ({len(features)}개): {features}")
    print(f"\n기술통계:")
    print(df[features + [target]].describe().round(2).to_string())

if __name__ == '__main__':
    main()
