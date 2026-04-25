#!/usr/bin/env python3
"""수집된 추가 변수를 행정동 단위로 집계 (v2: 구+동 매핑 수정)"""
import os, sys
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

sys.path.insert(0, os.path.dirname(__file__))
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def main():
    print("=" * 60)
    print("추가 변수 행정동 단위 집계 (v2)")
    print("=" * 60)

    # 기존 데이터에서 구-행정동 매핑
    final = pd.read_csv(os.path.join(DATA_DIR, 'apartment_final_v5_dong.csv'))
    gu_dong = final[['구', '행정동']].drop_duplicates()
    dong_per_gu = gu_dong.groupby('구')['행정동'].count().reset_index(name='동수')
    print(f"기존 데이터: {len(gu_dong)}개 구-행정동 조합")

    # 행정동 경계 로드 (공원/도서관 spatial join용)
    hjd_gdf = gpd.read_file(os.path.join(DATA_DIR, 'seoul_hjd_boundary.geojson')).to_crs('EPSG:4326')
    hjd_gdf['행정동'] = hjd_gdf['temp'].str.split(' ').str[-1]
    hjd_gdf['구'] = hjd_gdf['sggnm']

    # === 1. 공원수 (좌표 spatial join) ===
    print("\n=== 1. 공원수 ===")
    parks = pd.read_csv(os.path.join(DATA_DIR, 'seoul_parks.csv'))
    parks_valid = parks.dropna(subset=['YCRD', 'XCRD'])
    parks_valid = parks_valid[parks_valid['YCRD'].astype(float) > 33]
    geometry = [Point(float(x), float(y)) for x, y in zip(parks_valid['XCRD'], parks_valid['YCRD'])]
    parks_gdf = gpd.GeoDataFrame(parks_valid, geometry=geometry, crs='EPSG:4326')
    joined = gpd.sjoin(parks_gdf, hjd_gdf[['구', '행정동', 'geometry']], how='inner', predicate='within')
    park_counts = joined.groupby(['구', '행정동']).size().reset_index(name='공원수')
    print(f"  공원수: {len(park_counts)}개 동 ({len(joined)}건 매칭)")

    # === 2. 도서관수 (좌표 spatial join, XCNTS=위도, YDNTS=경도) ===
    print("\n=== 2. 도서관수 ===")
    libs = pd.read_csv(os.path.join(DATA_DIR, 'seoul_libraries.csv'))
    libs_valid = libs.dropna(subset=['XCNTS', 'YDNTS'])
    geometry = [Point(float(lon), float(lat)) for lat, lon in zip(libs_valid['XCNTS'], libs_valid['YDNTS'])]
    libs_gdf = gpd.GeoDataFrame(libs_valid, geometry=geometry, crs='EPSG:4326')
    joined = gpd.sjoin(libs_gdf, hjd_gdf[['구', '행정동', 'geometry']], how='inner', predicate='within')
    lib_counts = joined.groupby(['구', '행정동']).size().reset_index(name='도서관수')
    print(f"  도서관수: {len(lib_counts)}개 동 ({len(joined)}건 매칭)")

    # === 3. 학원수 (구 단위 → 행정동 균등배분) ===
    print("\n=== 3. 학원수 ===")
    academies = pd.read_csv(os.path.join(DATA_DIR, 'seoul_academies.csv'))
    if 'ADMST_ZONE_NM' in academies.columns:
        academies['구'] = academies['ADMST_ZONE_NM'].str.strip()
    gu_acad = academies.groupby('구').size().reset_index(name='학원수_구')
    gu_acad = gu_acad.merge(dong_per_gu, on='구', how='left')
    gu_acad['학원수_avg'] = (gu_acad['학원수_구'] / gu_acad['동수']).round(0).astype(int)

    acad_per_dong = gu_dong.merge(gu_acad[['구', '학원수_avg']], on='구', how='left')
    acad_per_dong = acad_per_dong.rename(columns={'학원수_avg': '학원수'})
    print(f"  학원수: 구별 배분 완료")
    print(f"  강남구 학원수/동: {gu_acad[gu_acad['구']=='강남구']['학원수_avg'].values}")

    # === 4. 어린이집수 (구 단위 → 행정동 균등배분) ===
    print("\n=== 4. 어린이집수 ===")
    childcare = pd.read_csv(os.path.join(DATA_DIR, 'seoul_childcare.csv'))
    childcare = childcare[childcare['CRSTATUSNAME'] == '정상']
    gu_cc = childcare.groupby('SIGUNNAME').size().reset_index(name='어린이집수_구')
    gu_cc = gu_cc.rename(columns={'SIGUNNAME': '구'})
    gu_cc = gu_cc.merge(dong_per_gu, on='구', how='left')
    gu_cc['어린이집수_avg'] = (gu_cc['어린이집수_구'] / gu_cc['동수']).round(0).astype(int)

    cc_per_dong = gu_dong.merge(gu_cc[['구', '어린이집수_avg']], on='구', how='left')
    cc_per_dong = cc_per_dong.rename(columns={'어린이집수_avg': '어린이집수'})
    print(f"  어린이집수: 구별 배분 완료")

    # === 합치기 ===
    result = gu_dong.copy()
    result = result.merge(park_counts, on=['구', '행정동'], how='left')
    result = result.merge(lib_counts, on=['구', '행정동'], how='left')
    result = result.merge(acad_per_dong[['구', '행정동', '학원수']], on=['구', '행정동'], how='left')
    result = result.merge(cc_per_dong[['구', '행정동', '어린이집수']], on=['구', '행정동'], how='left')

    for col in ['공원수', '도서관수', '학원수', '어린이집수']:
        result[col] = result[col].fillna(0).astype(int)

    path = os.path.join(DATA_DIR, 'additional_vars_per_dong.csv')
    result.to_csv(path, index=False, encoding='utf-8-sig')

    print(f"\n{'='*60}")
    print(f"✅ 최종 저장: {path}")
    print(f"{'='*60}")
    print(f"\n{result.describe().to_string()}")
    print(f"\n구별 학원수 Top 5:")
    print(gu_acad.nlargest(5, '학원수_구')[['구', '학원수_구', '동수', '학원수_avg']].to_string(index=False))


if __name__ == '__main__':
    main()
