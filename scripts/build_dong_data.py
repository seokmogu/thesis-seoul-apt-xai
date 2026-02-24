#!/usr/bin/env python3
"""행정동 단위 전체 데이터 구축 파이프라인"""
import json, re, os, sys
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_api_keys

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')

def build_bjd_hjd_mapping():
    """법정동→행정동 매핑 테이블"""
    with open(f'{DATA}/seoul_hjd_boundary.geojson') as f:
        hjd_data = json.load(f)
    
    hjd_by_gu = defaultdict(list)
    for feat in hjd_data['features']:
        parts = feat['properties']['adm_nm'].split()
        if len(parts) >= 3:
            hjd_by_gu[parts[1]].append(parts[2])
    
    with open(f'{DATA}/bjd_hjd_nominatim.json') as f:
        nominatim_map = json.load(f)
    
    def auto_map(gu, bjd, hjd_list):
        if not isinstance(bjd, str):
            return None
        if bjd in hjd_list:
            return bjd
        base = bjd.replace('동', '')
        matches = [h for h in hjd_list if h.startswith(base) and h.endswith('동')]
        if matches:
            return matches[0]
        ga_match = re.match(r'(.+동)(\d+가)$', bjd)
        if ga_match:
            dong_part = ga_match.group(1).replace('동', '')
            target = f'{dong_part}{ga_match.group(2)}동'
            if target in hjd_list:
                return target
            matches = [h for h in hjd_list if h.startswith(dong_part)]
            if matches:
                return matches[0]
        return None
    
    return hjd_by_gu, nominatim_map, auto_map

def map_to_hjd(gu, bjd, hjd_by_gu, nominatim_map, auto_map_fn):
    if not isinstance(bjd, str) or not isinstance(gu, str):
        return None
    result = auto_map_fn(gu, bjd, hjd_by_gu.get(gu, []))
    if not result:
        result = nominatim_map.get(f'{gu}|{bjd}')
    return result if result else None

def main():
    hjd_by_gu, nominatim_map, auto_map_fn = build_bjd_hjd_mapping()
    
    # Load 행정동 경계
    gdf_hjd = gpd.read_file(f'{DATA}/seoul_hjd_boundary.geojson').to_crs('EPSG:4326')
    gdf_hjd['구'] = gdf_hjd['adm_nm'].apply(lambda x: x.split()[1] if len(x.split()) >= 3 else '')
    gdf_hjd['행정동'] = gdf_hjd['adm_nm'].apply(lambda x: x.split()[2] if len(x.split()) >= 3 else '')
    
    # === 1. 실거래 데이터 ===
    print("=== 1. 실거래 데이터 행정동 매핑 ===")
    trade = pd.read_csv(f'{DATA}/apartment_trades_v2.csv')
    trade['행정동'] = trade.apply(lambda r: map_to_hjd(r['구'], r['법정동'], hjd_by_gu, nominatim_map, auto_map_fn), axis=1)
    mapped = trade.dropna(subset=['행정동'])
    print(f"  매핑: {len(mapped):,}/{len(trade):,} ({len(mapped)/len(trade)*100:.1f}%)")
    
    # 전처리
    mapped = mapped.copy()
    mapped['거래금액'] = pd.to_numeric(mapped['거래금액'], errors='coerce')
    mapped['건축년도'] = pd.to_numeric(mapped['건축년도'], errors='coerce')
    mapped['전용면적'] = pd.to_numeric(mapped['전용면적'], errors='coerce')
    mapped['층'] = pd.to_numeric(mapped['층'], errors='coerce')
    mapped = mapped.dropna(subset=['거래금액', '건축년도', '전용면적', '층'])
    mapped['건물연령'] = mapped['거래년도'] - mapped['건축년도']
    mapped = mapped[(mapped['거래금액'] > 0) & (mapped['전용면적'] > 10) & 
                     (mapped['전용면적'] < 300) & (mapped['층'] > 0) & 
                     (mapped['건물연령'] >= 0) & (mapped['건물연령'] <= 80)]
    mapped['강남구분'] = mapped['구'].apply(lambda x: 1 if x in ['강남구', '서초구', '송파구'] else 0)
    print(f"  전처리 후: {len(mapped):,}건")
    
    # === 2. CCTV → 행정동 ===
    print("\n=== 2. CCTV spatial join ===")
    with open(f'{DATA}/cctv_raw.json') as f:
        cctv = json.load(f)
    gdf_cctv = gpd.GeoDataFrame(
        cctv, geometry=[Point(float(c['WGSXPT']), float(c['WGSYPT'])) for c in cctv], crs='EPSG:4326')
    cctv_joined = gpd.sjoin(gdf_cctv, gdf_hjd[['geometry', '구', '행정동']], how='left', predicate='within')
    cctv_per_hjd = cctv_joined.groupby(['구', '행정동']).size().reset_index(name='CCTV수')
    print(f"  {len(cctv_per_hjd)}개 행정동")
    
    # === 3. 지하철 → 행정동 ===
    print("\n=== 3. 지하철 spatial join ===")
    with open(f'{DATA}/subway_stations_api.json') as f:
        subway = json.load(f)
    gdf_sub = gpd.GeoDataFrame(
        subway, geometry=[Point(float(s['LOT']), float(s['LAT'])) for s in subway], crs='EPSG:4326')
    sub_joined = gpd.sjoin(gdf_sub, gdf_hjd[['geometry', '구', '행정동']], how='inner', predicate='within')
    sub_per_hjd = sub_joined.groupby(['구', '행정동']).size().reset_index(name='지하철역수')
    print(f"  {len(sub_per_hjd)}개 행정동 (나머지 0)")
    
    # === 4. 학교 → 행정동 ===
    print("\n=== 4. 학교 행정동 매핑 ===")
    schools = pd.read_csv(f'{DATA}/schools_raw.csv')
    schools = schools[schools['school_type'].isin(['초등학교', '중학교', '고등학교'])].copy()
    schools['행정동'] = schools.apply(
        lambda r: map_to_hjd(r['gu'], r['dong'], hjd_by_gu, nominatim_map, auto_map_fn) if pd.notna(r.get('dong')) and pd.notna(r.get('gu')) else None,
        axis=1)
    school_mapped = schools.dropna(subset=['행정동'])
    print(f"  매핑: {len(school_mapped)}/{len(schools)} ({len(school_mapped)/len(schools)*100:.1f}%)")
    
    school_pivot = school_mapped.groupby(['gu', '행정동', 'school_type']).size().unstack(fill_value=0).reset_index()
    cols = school_pivot.columns.tolist()
    cols[0] = '구'
    school_pivot.columns = cols
    # Rename school type columns
    rename = {}
    for c in school_pivot.columns:
        if c in ['초등학교', '중학교', '고등학교']:
            rename[c] = f'{c}수'
    school_pivot = school_pivot.rename(columns=rename)
    print(f"  {len(school_pivot)}개 행정동")
    
    # === 5. 백화점 (구 단위 유지) ===
    print("\n=== 5. 백화점 (구 단위) ===")
    env = pd.read_csv(f'{DATA}/env_per_gu.csv')
    dept = env[['구', '백화점수']]
    
    # === 6. 거시경제 ===
    print("\n=== 6. 거시경제 ===")
    ecos = pd.read_csv(f'{DATA}/ecos_macro.csv')
    ecos.rename(columns={'년월': '거래년월'}, inplace=True)
    ecos['거래년월'] = ecos['거래년월'].astype(str)
    mapped['거래년월'] = mapped['거래년월'].astype(str)
    
    # === 7. 전체 머지 ===
    print("\n=== 7. 전체 병합 ===")
    df = mapped.merge(ecos, on='거래년월', how='left')
    df = df.merge(school_pivot, on=['구', '행정동'], how='left')
    df = df.merge(cctv_per_hjd, on=['구', '행정동'], how='left')
    df = df.merge(sub_per_hjd, on=['구', '행정동'], how='left')
    df = df.merge(dept, on='구', how='left')
    
    # Fill NaN for infra (행정동에 학교/지하철 없으면 0)
    infra_cols = ['초등학교수', '중학교수', '고등학교수', 'CCTV수', '지하철역수', '백화점수']
    for c in infra_cols:
        if c in df.columns:
            na = df[c].isna().sum()
            if na > 0:
                print(f"  {c} 결측: {na:,} → 0 채움")
            df[c] = df[c].fillna(0)
    
    # 거시 결측 확인
    macro_cols = ['기준금리', 'CD금리', '소비자물가지수', 'M2']
    macro_na = df[macro_cols].isna().sum()
    print(f"  거시 결측: {dict(macro_na[macro_na > 0])}")
    df = df.dropna(subset=macro_cols)
    
    features = ['전용면적', '층', '건물연령', '강남구분',
                '초등학교수', '중학교수', '고등학교수',
                'CCTV수', '백화점수', '지하철역수',
                '기준금리', 'CD금리', '소비자물가지수', 'M2']
    
    print(f"\n=== 최종 결과 ===")
    print(f"  건수: {len(df):,}")
    print(f"  기간: {df['거래년월'].min()} ~ {df['거래년월'].max()}")
    print(f"  행정동: {df['행정동'].nunique()}개")
    print(f"  변수 ({len(features)}): {features}")
    
    # 저장
    keep = ['거래금액', '건축년도', '전용면적', '층', '법정동', '행정동', '아파트명',
            '거래년월', '거래년도', '건물연령', '구', '강남구분'] + \
           [c for c in features if c not in ['전용면적', '층', '건물연령', '강남구분']]
    
    df[keep].to_csv(f'{DATA}/apartment_final_v5_dong.csv', index=False, encoding='utf-8-sig')
    print(f"\n  Saved: apartment_final_v5_dong.csv")
    
    print(f"\n기술통계:")
    print(df[features + ['거래금액']].describe().round(2).to_string())

if __name__ == '__main__':
    main()
