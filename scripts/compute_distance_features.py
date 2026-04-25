#!/usr/bin/env python3
"""
아파트 단지 × 시설군 거리 변수 계산.

교수 피드백 "도서관은 도보 몇 km까지? 백화점 및 기타 변수들의 조건?"에 대해
두 축 지표를 병행 산출:
  1) nearest_dist_m : 단지 → 가장 가까운 시설까지 직선거리 (haversine, 미터)
  2) count_500m / count_1km / count_2km : 반경별 시설 개수
도보권(500m, 1km)과 차량권(2km) 둘 다 산출해 회귀에서 실증적으로 고를 수 있게 한다.

이 단계에서는 "시점 필터 없이" 전체 시설을 대상으로 계산하며,
연도별 시점 필터링(Task #4)은 후단 모듈에서 수행한다.

입력:
  data/apartment_coords.csv (단지 7,999개 좌표)
  data/schools_coords.csv (학교 1,415)
  data/academies_coords.csv (학원, 완료시점 기준)
  data/seoul_childcare.csv (어린이집 LA/LO)
  data/seoul_parks.csv (공원 XCRD/YCRD)
  data/seoul_libraries.csv (도서관 XCNTS=lat/YDNTS=lng)
  data/subway_stations_api.json (지하철 LAT/LOT)
  data/seoul_marts_kakao.csv (대형마트 x/y)
  data/seoul_department_stores_kakao.csv (백화점 x/y)
  data/cctv_raw.json (CCTV WGSXPT/WGSYPT)

출력:
  data/apartment_distance_features.csv
    컬럼: gu, bjd, apt_name_raw, lat, lng,
          {facility}_nearest_m, {facility}_count_500m, {facility}_count_1km, {facility}_count_2km
"""
import os
import sys
import json
import math
import time
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
EARTH_M = 6_371_000.0  # meters
RADII_M = [500, 1000, 2000]


def load_facility(name: str, path: str, lat_col: str, lng_col: str,
                  sub_type: str = None, type_col: str = None) -> np.ndarray:
    """시설 좌표를 (N, 2) ndarray(위도, 경도 라디안)으로 반환."""
    if path.endswith('.json'):
        with open(path) as f:
            rows = json.load(f)
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(path, low_memory=False)

    if type_col and sub_type:
        df = df[df[type_col] == sub_type]

    lat = pd.to_numeric(df[lat_col], errors='coerce')
    lng = pd.to_numeric(df[lng_col], errors='coerce')
    mask = lat.notna() & lng.notna() & lat.between(37.3, 37.75) & lng.between(126.7, 127.3)
    coords = np.radians(np.column_stack([lat[mask].values, lng[mask].values]))
    print(f"  {name}: {len(coords):,} 지점 로드")
    return coords


def compute_features(apt_coords: np.ndarray, facility_coords: np.ndarray):
    """각 아파트에 대해 최근접거리와 반경별 개수를 반환."""
    if len(facility_coords) == 0:
        n = len(apt_coords)
        return (np.full(n, np.nan),
                {f'count_{r}m': np.zeros(n, dtype=int) for r in RADII_M})
    tree = BallTree(facility_coords, metric='haversine')
    # 최근접거리
    dist_rad, _ = tree.query(apt_coords, k=1)
    nearest_m = (dist_rad.flatten() * EARTH_M)
    # 반경별 카운트
    counts = {}
    for r in RADII_M:
        ind = tree.query_radius(apt_coords, r=r / EARTH_M)
        counts[f'count_{r}m'] = np.array([len(x) for x in ind], dtype=int)
    return nearest_m, counts


def main():
    # 아파트 좌표 로드
    apt = pd.read_csv(os.path.join(DATA_DIR, 'apartment_coords.csv'))
    apt = apt.dropna(subset=['lat', 'lng']).reset_index(drop=True)
    apt_coords = np.radians(apt[['lat', 'lng']].values)
    print(f"아파트 단지: {len(apt):,}")
    out = apt[['gu', 'bjd', 'apt_name_raw', 'lat', 'lng']].copy()

    # 시설 세트 (codex 리뷰 반영: cctv는 count만, 병원 추가)
    specs = [
        ('subway', 'data/subway_stations_api.json', 'LAT', 'LOT', None, None),
        ('elem_school', 'data/schools_coords.csv', 'lat', 'lng', '초등학교', 'school_type'),
        ('middle_school', 'data/schools_coords.csv', 'lat', 'lng', '중학교', 'school_type'),
        ('high_school', 'data/schools_coords.csv', 'lat', 'lng', '고등학교', 'school_type'),
        ('childcare', 'data/seoul_childcare.csv', 'LA', 'LO', None, None),
        ('park', 'data/seoul_parks.csv', 'YCRD', 'XCRD', None, None),
        ('library', 'data/seoul_libraries.csv', 'XCNTS', 'YDNTS', None, None),  # XCNTS=lat, YDNTS=lng
        ('mart', 'data/seoul_marts_kakao.csv', 'y', 'x', None, None),
        ('department', 'data/seoul_department_stores_kakao.csv', 'y', 'x', None, None),
        ('cctv', 'data/cctv_raw.json', 'WGSYPT', 'WGSXPT', None, None),
        ('hospital', 'data/seoul_hospitals_kakao.csv', 'y', 'x', None, None),
        ('hospital_general', 'data/seoul_hospitals_kakao.csv', 'y', 'x', '종합병원', '종별'),
        ('academy', 'data/academies_coords.csv', 'lat', 'lng', None, None),
    ]
    # codex 리뷰 반영: 최근접거리를 제외할 시설군 (변별력 낮음 또는 밀도 지표가 더 적절)
    nearest_drop = {'cctv'}
    # codex 리뷰 반영: 0이 많아 파생변수 필요한 희소 시설군
    sparse_facilities = {'park', 'department', 'library'}

    for name, rel, lat_c, lng_c, sub, tcol in specs:
        print(f"\n=== {name} ===")
        t0 = time.time()
        path = os.path.join(os.path.dirname(__file__), '..', rel)
        try:
            coords = load_facility(name, path, lat_c, lng_c, sub, tcol)
        except Exception as e:
            print(f"  스킵 ({e})")
            continue
        nearest_m, counts = compute_features(apt_coords, coords)
        if name not in nearest_drop:
            out[f'{name}_nearest_m'] = np.round(nearest_m, 1)
        for k, v in counts.items():
            out[f'{name}_{k}'] = v
        # 희소 시설군 보조 변수: within_1km 더미, log1p(count_2km)
        if name in sparse_facilities:
            out[f'{name}_within_1km'] = (counts['count_1000m'] > 0).astype(int)
            out[f'{name}_log1p_count_2km'] = np.log1p(counts['count_2000m'])
        # 요약
        finite = nearest_m[np.isfinite(nearest_m)]
        if len(finite) and name not in nearest_drop:
            print(f"  nearest(m): median={np.median(finite):.0f}, p90={np.percentile(finite,90):.0f}")
        for k, v in counts.items():
            print(f"  {k}: median={int(np.median(v))}, p90={int(np.percentile(v,90))}, max={int(v.max())}")
        print(f"  elapsed {time.time()-t0:.1f}s")

    out_path = os.path.join(DATA_DIR, 'apartment_distance_features.csv')
    out.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n저장: {out_path} ({len(out):,} 행, {len(out.columns)} 컬럼)")


if __name__ == '__main__':
    main()
