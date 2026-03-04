#!/usr/bin/env python3
"""수집된 추가 변수를 행정동 단위로 집계"""
import os, sys, re
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

sys.path.insert(0, os.path.dirname(__file__))
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def load_hjd_boundary():
    """행정동 경계 로드"""
    path = os.path.join(DATA_DIR, 'seoul_hjd_boundary.geojson')
    gdf = gpd.read_file(path).to_crs('EPSG:4326')
    # 행정동 이름 컬럼 찾기
    for c in gdf.columns:
        if 'adm_nm' in c.lower() or 'ADM_NM' in c:
            gdf = gdf.rename(columns={c: 'dong_name'})
            break
        elif 'name' in c.lower() and c != 'geometry':
            gdf = gdf.rename(columns={c: 'dong_name'})
            break
    print(f"  행정동 경계: {len(gdf)}개 동, 컬럼: {list(gdf.columns)[:5]}")
    return gdf


def spatial_join_count(points_df, lat_col, lon_col, dong_gdf, label):
    """좌표 기반 데이터를 행정동에 spatial join 후 카운트"""
    valid = points_df.dropna(subset=[lat_col, lon_col]).copy()
    valid[lat_col] = valid[lat_col].astype(float)
    valid[lon_col] = valid[lon_col].astype(float)
    valid = valid[(valid[lat_col] > 33) & (valid[lat_col] < 39)]  # 한국 위도 범위
    valid = valid[(valid[lon_col] > 124) & (valid[lon_col] < 132)]  # 한국 경도 범위

    geometry = [Point(lon, lat) for lon, lat in zip(valid[lon_col], valid[lat_col])]
    pts_gdf = gpd.GeoDataFrame(valid, geometry=geometry, crs='EPSG:4326')

    joined = gpd.sjoin(pts_gdf, dong_gdf, how='inner', predicate='within')

    dong_col = 'dong_name' if 'dong_name' in joined.columns else joined.columns[0]
    counts = joined.groupby(dong_col).size().reset_index(name=label)
    print(f"  {label}: {len(counts)}개 동 집계 ({len(valid)}건 중 {len(joined)}건 매칭)")
    return counts


def aggregate_academies_by_address(dong_gdf):
    """학원을 주소에서 구 추출 → 구-행정동 기반 집계 (구 단위 fallback)"""
    print("\n=== 학원수 집계 ===")
    df = pd.read_csv(os.path.join(DATA_DIR, 'seoul_academies.csv'))

    # ADMST_ZONE_NM에서 구 이름 추출 (예: "강남구" 등)
    if 'ADMST_ZONE_NM' in df.columns:
        df['구'] = df['ADMST_ZONE_NM'].str.extract(r'(서울특별시\s+)?(\S+구)', expand=True)[1]
    elif 'FA_RDNMA' in df.columns:
        df['구'] = df['FA_RDNMA'].str.extract(r'서울특별시\s+(\S+구)')[0]

    # 도로명주소에서 동 추출 시도
    if 'FA_RDNMA' in df.columns:
        # 주소에서 동 이름 추출 (어려움 - 도로명이라 법정동이 아님)
        pass

    # 구 단위 집계 후 행정동에 배분
    gu_counts = df.groupby('구').size().reset_index(name='학원수_구')
    print(f"  구별 학원수: {gu_counts.to_string(index=False)[:200]}")

    # 행정동별 집계: 각 구의 학원수를 구 내 행정동 수로 균등 배분 (근사)
    dong_col = 'dong_name' if 'dong_name' in dong_gdf.columns else dong_gdf.columns[0]

    # 행정동에서 구 추출
    dong_gu = dong_gdf[[dong_col]].copy()
    dong_gu['구'] = dong_gu[dong_col].apply(lambda x: extract_gu_from_dong(str(x)))

    dong_gu = dong_gu.merge(gu_counts, on='구', how='left')
    dong_counts_per_gu = dong_gu.groupby('구')[dong_col].count().reset_index(name='동수')
    dong_gu = dong_gu.merge(dong_counts_per_gu, on='구', how='left')
    dong_gu['학원수'] = (dong_gu['학원수_구'] / dong_gu['동수']).round(0).astype(int)

    result = dong_gu[[dong_col, '학원수']].rename(columns={dong_col: 'dong_name'})
    print(f"  학원수: {len(result)}개 동 집계")
    return result


def extract_gu_from_dong(dong_name):
    """행정동 이름에서 구 추출"""
    # 행정동 이름에 구 정보가 없을 수 있음 → 매핑 테이블 필요
    # 일단 기존 데이터에서 구-동 매핑 활용
    return None  # placeholder


def aggregate_all():
    """전체 집계"""
    print("=" * 60)
    print("추가 변수 행정동 단위 집계")
    print("=" * 60)

    dong_gdf = load_hjd_boundary()
    dong_col = 'dong_name' if 'dong_name' in dong_gdf.columns else dong_gdf.columns[0]

    results = pd.DataFrame({dong_col: dong_gdf[dong_col].unique()})

    # 1. 공원수 (좌표 기반)
    print("\n=== 공원수 ===")
    parks = pd.read_csv(os.path.join(DATA_DIR, 'seoul_parks.csv'))
    park_counts = spatial_join_count(parks, 'YCRD', 'XCRD', dong_gdf, '공원수')
    results = results.merge(park_counts, left_on=dong_col, right_on=park_counts.columns[0], how='left')
    if park_counts.columns[0] != dong_col and park_counts.columns[0] in results.columns:
        results = results.drop(columns=[park_counts.columns[0]])

    # 2. 도서관 (좌표 기반 - XCNTS=위도, YDNTS=경도)
    print("\n=== 도서관수 ===")
    libs = pd.read_csv(os.path.join(DATA_DIR, 'seoul_libraries.csv'))
    lib_counts = spatial_join_count(libs, 'XCNTS', 'YDNTS', dong_gdf, '도서관수')
    results = results.merge(lib_counts, left_on=dong_col, right_on=lib_counts.columns[0], how='left')
    if lib_counts.columns[0] != dong_col and lib_counts.columns[0] in results.columns:
        results = results.drop(columns=[lib_counts.columns[0]])

    # 3. 학원수 (구 단위 → 행정동 배분)
    print("\n=== 학원수 ===")
    academies = pd.read_csv(os.path.join(DATA_DIR, 'seoul_academies.csv'))
    if 'ADMST_ZONE_NM' in academies.columns:
        academies['구'] = academies['ADMST_ZONE_NM'].str.strip()
    elif 'FA_RDNMA' in academies.columns:
        academies['구'] = academies['FA_RDNMA'].str.extract(r'서울특별시\s+(\S+구)')[0]

    gu_academy = academies.groupby('구').size().reset_index(name='학원수_구')

    # 기존 데이터에서 구-행정동 매핑 가져오기
    final_data = pd.read_csv(os.path.join(DATA_DIR, 'apartment_final_v5_dong.csv'))
    gu_dong_map = final_data[['구', '행정동']].drop_duplicates()
    dong_per_gu = gu_dong_map.groupby('구')['행정동'].count().reset_index(name='동수')

    gu_academy = gu_academy.merge(dong_per_gu, on='구', how='left')
    gu_academy['학원수_per_dong'] = (gu_academy['학원수_구'] / gu_academy['동수']).round(0).astype(int)

    # 행정동별 배분
    dong_academy = gu_dong_map.merge(gu_academy[['구', '학원수_per_dong']], on='구', how='left')
    dong_academy = dong_academy.rename(columns={'행정동': dong_col, '학원수_per_dong': '학원수'})
    dong_academy = dong_academy[[dong_col, '학원수']].drop_duplicates()
    results = results.merge(dong_academy, on=dong_col, how='left')
    print(f"  학원수: {len(dong_academy)}개 동")

    # 4. 어린이집 (구 단위 → 행정동 배분)
    print("\n=== 어린이집수 ===")
    childcare = pd.read_csv(os.path.join(DATA_DIR, 'seoul_childcare.csv'))
    childcare = childcare[childcare['CRSTATUSNAME'] == '정상']  # 정상 운영만
    gu_childcare = childcare.groupby('SIGUNNAME').size().reset_index(name='어린이집수_구')
    gu_childcare = gu_childcare.rename(columns={'SIGUNNAME': '구'})
    gu_childcare = gu_childcare.merge(dong_per_gu, on='구', how='left')
    gu_childcare['어린이집수_per_dong'] = (gu_childcare['어린이집수_구'] / gu_childcare['동수']).round(0).astype(int)

    dong_childcare = gu_dong_map.merge(gu_childcare[['구', '어린이집수_per_dong']], on='구', how='left')
    dong_childcare = dong_childcare.rename(columns={'행정동': dong_col, '어린이집수_per_dong': '어린이집수'})
    dong_childcare = dong_childcare[[dong_col, '어린이집수']].drop_duplicates()
    results = results.merge(dong_childcare, on=dong_col, how='left')
    print(f"  어린이집수: {len(dong_childcare)}개 동")

    # 결측값 0으로 채우기
    for col in ['공원수', '도서관수', '학원수', '어린이집수']:
        if col in results.columns:
            results[col] = results[col].fillna(0).astype(int)

    # 저장
    path = os.path.join(DATA_DIR, 'additional_vars_per_dong.csv')
    results.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 최종 저장: {path}")
    print(results.describe().to_string())

    return results


if __name__ == '__main__':
    aggregate_all()
