#!/usr/bin/env python3
"""추가 변수 수집: 학원수, 공원수, 도서관수, 어린이집수 → 행정동 단위 집계"""
import os, sys, json, time
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_api_keys, request_with_retry

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
keys = load_api_keys()


def collect_academies():
    """NEIS API에서 서울 학원 정보 수집 (행정동 매핑은 주소 기반)"""
    print("\n=== 1. 학원수 수집 (NEIS) ===")
    api_key = keys['NEIS_API_KEY']
    all_rows = []
    page = 1
    page_size = 1000

    while True:
        r = request_with_retry(
            "https://open.neis.go.kr/hub/acaInsTiInfo",
            params={
                'KEY': api_key, 'Type': 'json',
                'pIndex': page, 'pSize': page_size,
                'ATPT_OFCDC_SC_CODE': 'B10'  # 서울
            }
        )
        data = r.json()
        if 'acaInsTiInfo' not in data:
            break

        rows = data['acaInsTiInfo'][1]['row']
        all_rows.extend(rows)
        total = data['acaInsTiInfo'][0]['head'][0]['list_total_count']
        print(f"  Page {page}: {len(all_rows)}/{total}")

        if len(all_rows) >= total:
            break
        page += 1
        time.sleep(0.3)

    df = pd.DataFrame(all_rows)
    path = os.path.join(DATA_DIR, 'seoul_academies.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  ✅ 학원 {len(df)}건 저장: {path}")
    return df


def collect_seoul_api(service, desc, page_size=1000):
    """서울 열린데이터 API 페이지네이션 수집"""
    api_key = keys['SEOUL_API_KEY']
    all_rows = []
    start = 1

    while True:
        end = start + page_size - 1
        url = f"http://openapi.seoul.go.kr:8088/{api_key}/json/{service}/{start}/{end}/"
        r = request_with_retry(url)
        data = r.json()
        key = list(data.keys())[0]

        if isinstance(data[key], dict) and 'row' in data[key]:
            rows = data[key]['row']
            all_rows.extend(rows)
            total = data[key].get('list_total_count', len(all_rows))
            print(f"  {desc}: {len(all_rows)}/{total}")
            if len(all_rows) >= total:
                break
        else:
            print(f"  {desc}: API 응답 이상 - {data[key]}")
            break

        start += page_size
        time.sleep(0.3)

    return all_rows


def collect_libraries():
    """서울시 공공도서관 수집"""
    print("\n=== 2. 도서관수 수집 ===")
    rows = collect_seoul_api('SeoulPublicLibraryInfo', '도서관')
    df = pd.DataFrame(rows)
    path = os.path.join(DATA_DIR, 'seoul_libraries.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  ✅ 도서관 {len(df)}건 저장")
    return df


def collect_childcare():
    """서울시 어린이집 수집"""
    print("\n=== 3. 어린이집수 수집 ===")
    rows = collect_seoul_api('ChildCareInfo', '어린이집')
    df = pd.DataFrame(rows)
    path = os.path.join(DATA_DIR, 'seoul_childcare.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  ✅ 어린이집 {len(df)}건 저장")
    return df


def aggregate_parks_by_dong():
    """이미 수집된 공원 데이터를 행정동별 집계"""
    print("\n=== 4. 공원수 행정동 집계 ===")
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        print("  ❌ geopandas 필요")
        return None

    parks_path = os.path.join(DATA_DIR, 'seoul_parks.csv')
    hjd_path = os.path.join(DATA_DIR, 'seoul_hjd_boundary.geojson')

    if not os.path.exists(parks_path):
        print("  ❌ seoul_parks.csv 없음")
        return None

    parks = pd.read_csv(parks_path)
    print(f"  공원 데이터: {len(parks)}건")

    # 좌표 컬럼 확인
    lat_col = None
    lon_col = None
    for col in parks.columns:
        if 'XCRD' == col or 'XCRD_G' == col:
            # 서울 열린데이터에서 X=경도, Y=위도인 경우가 있음
            pass
        if col in ['YCRD', 'YCRD_G', 'lat', 'LAT', 'latitude']:
            lat_col = col
        if col in ['XCRD', 'XCRD_G', 'lon', 'LON', 'longitude']:
            lon_col = col

    # 서울 열린데이터: XCRD=경도, YCRD=위도 (일반적으로)
    # 하지만 실제로는 반대인 경우도 있으니 확인
    if lat_col and lon_col:
        sample_x = parks[lon_col].dropna().iloc[0] if len(parks[lon_col].dropna()) > 0 else 0
        sample_y = parks[lat_col].dropna().iloc[0] if len(parks[lat_col].dropna()) > 0 else 0
        print(f"  좌표 샘플: {lon_col}={sample_x}, {lat_col}={sample_y}")

        # 서울 좌표 범위: 위도 37.4~37.7, 경도 126.7~127.2
        if 126 < sample_x < 128:
            # X가 경도
            lon_col, lat_col = lon_col, lat_col
        elif 126 < sample_y < 128:
            # Y가 경도 (swap)
            lon_col, lat_col = lat_col, lon_col
            print(f"  좌표 swap: lon={lon_col}, lat={lat_col}")

    if not lat_col or not lon_col:
        print(f"  ❌ 좌표 컬럼 못찾음. 컬럼: {list(parks.columns)}")
        return None

    parks_valid = parks.dropna(subset=[lat_col, lon_col])
    parks_valid = parks_valid[parks_valid[lat_col].astype(float) > 0]

    if os.path.exists(hjd_path):
        dong_gdf = gpd.read_file(hjd_path)
        geometry = [Point(float(x), float(y)) for x, y in
                    zip(parks_valid[lon_col], parks_valid[lat_col])]
        parks_gdf = gpd.GeoDataFrame(parks_valid, geometry=geometry, crs='EPSG:4326')
        dong_gdf = dong_gdf.to_crs('EPSG:4326')

        joined = gpd.sjoin(parks_gdf, dong_gdf, how='left', predicate='within')
        # 행정동 이름 컬럼 찾기
        dong_col = None
        for c in dong_gdf.columns:
            if '동' in str(c).lower() or 'adm' in str(c).lower() or 'name' in str(c).lower():
                dong_col = c
                break
        if dong_col is None:
            dong_col = dong_gdf.columns[0]

        park_counts = joined.groupby(dong_col).size().reset_index(name='공원수')
        path = os.path.join(DATA_DIR, 'parks_per_dong.csv')
        park_counts.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"  ✅ 공원수 {len(park_counts)}개 동 집계 완료")
        return park_counts
    else:
        print("  ❌ 행정동 경계 파일 없음")
        return None


def main():
    print("=" * 60)
    print("추가 변수 수집 시작")
    print("=" * 60)

    # 1. 학원
    collect_academies()

    # 2. 도서관
    collect_libraries()

    # 3. 어린이집
    collect_childcare()

    # 4. 공원 (행정동 집계)
    aggregate_parks_by_dong()

    print("\n" + "=" * 60)
    print("수집 완료!")
    print("=" * 60)


if __name__ == '__main__':
    main()
