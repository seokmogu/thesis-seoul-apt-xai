#!/usr/bin/env python3
"""
연도별 시설 스냅샷 기반 거리 변수 계산.

각 거래 시점에 "실제 존재/영업중"이었던 시설만 대상으로 거리·개수 산출.
교수 피드백 "물리적 조건이 시간에 따라 변동되는 건 고려 못한 점" 대응.

스냅샷 기준 날짜: 각 연도(Y) 말일 (Y-12-31)
  - 개업일 ≤ Y-12-31 이고 (폐업일 > Y-12-31 or 폐업일 없음) 인 시설만 포함

시설별 시간 필드:
  schools         founded (YYYYMMDD int), 폐교일 없음 → 개교만 필터
  academies       ESTBL_YMD + REG_STTUS_NM("개원"만) + CAA_BEGIN_YMD/CAA_END_YMD(휴원)
  childcare       CRCNFMDT + CRSTATUSNAME("정상") + CRPAUSEBEGINDT/CRPAUSEENDDT
  parks           OPEN_YMD (YYYY.M.D)
  large_stores    APVPERMYMD + TRDSTATENM("영업/정상") + DCBYMD
  subway          기본값 + subway_new_openings_2019_2025.csv
시설별 시간 필드 없음 (스냅샷 그대로):
  library, mart, department, cctv, hospital, hospital_general

출력:
  data/apartment_distance_features_yearly.csv
    (단지 × 연도) = 8,601 × 7 = 60,207행
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
EARTH_M = 6_371_000.0
RADII = [500, 1000, 2000]
YEARS = list(range(2019, 2026))  # 2019~2025


def parse_date_safe(s, fmt=None):
    return pd.to_datetime(s, errors='coerce', format=fmt)


def load_schools():
    df = pd.read_csv(os.path.join(DATA_DIR, 'schools_coords.csv'))
    df = df.dropna(subset=['lat', 'lng']).copy()
    df['open_dt'] = pd.to_datetime(df['founded'].astype(str).str.zfill(8),
                                    errors='coerce', format='%Y%m%d')
    df['close_dt'] = pd.NaT
    return df


def load_academies():
    df = pd.read_csv(os.path.join(DATA_DIR, 'academies_coords.csv'))
    df = df.dropna(subset=['lat', 'lng']).copy()
    df['open_dt'] = pd.to_datetime(df['ESTBL_YMD'].astype(str).str.zfill(8),
                                    errors='coerce', format='%Y%m%d')
    # 폐업: REG_STTUS_NM이 '개원'이 아닌 경우는 수집 시점까지 존재했을 수 있으므로,
    # 일단 '개원'만 영구 유지, 나머지는 수집 시점(2026-02) 기준 이전으로 가정
    # 보수적 처리: 개원 이외면 2026-01-01 이후 부존재로 간주
    df['close_dt'] = pd.NaT
    non_active = df['REG_STTUS_NM'].fillna('').ne('개원')
    df.loc[non_active, 'close_dt'] = pd.Timestamp('2026-01-01')
    return df


def load_childcare():
    df = pd.read_csv(os.path.join(DATA_DIR, 'seoul_childcare.csv'))
    df = df.dropna(subset=['LA', 'LO']).rename(columns={'LA': 'lat', 'LO': 'lng'}).copy()
    df['open_dt'] = parse_date_safe(df['CRCNFMDT'])
    df['close_dt'] = parse_date_safe(df['CRABLDT']) if 'CRABLDT' in df.columns else pd.NaT
    return df


def load_parks():
    df = pd.read_csv(os.path.join(DATA_DIR, 'seoul_parks.csv'))
    df = df.rename(columns={'YCRD': 'lat', 'XCRD': 'lng'})
    df = df.dropna(subset=['lat', 'lng']).copy()
    # OPEN_YMD format "1968.9.10"
    df['open_dt'] = pd.to_datetime(df['OPEN_YMD'].astype(str).str.replace('.', '-', regex=False),
                                    errors='coerce')
    df['close_dt'] = pd.NaT
    return df


def load_large_stores():
    df = pd.read_csv(os.path.join(DATA_DIR, 'seoul_large_stores_v2.csv'), low_memory=False)
    df = df.dropna(subset=['lat', 'lng']).copy()
    df['open_dt'] = parse_date_safe(df['APVPERMYMD'])
    df['close_dt'] = parse_date_safe(df['DCBYMD'])
    return df


def load_static(file, lat_col, lng_col, fmt='csv'):
    if fmt == 'json':
        with open(os.path.join(DATA_DIR, file)) as f:
            df = pd.DataFrame(json.load(f))
    else:
        df = pd.read_csv(os.path.join(DATA_DIR, file), low_memory=False)
    df = df.rename(columns={lat_col: 'lat', lng_col: 'lng'})
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lng'] = pd.to_numeric(df['lng'], errors='coerce')
    df = df.dropna(subset=['lat', 'lng']).copy()
    df['open_dt'] = pd.NaT
    df['close_dt'] = pd.NaT
    return df


def load_subway_with_opening():
    df = pd.DataFrame(json.load(open(os.path.join(DATA_DIR, 'subway_stations_api.json'))))
    df = df.rename(columns={'LAT': 'lat', 'LOT': 'lng'})
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lng'] = pd.to_numeric(df['lng'], errors='coerce')
    df = df.dropna(subset=['lat', 'lng']).copy()
    df['open_dt'] = pd.NaT  # 기본: 분석기간 이전 개통으로 가정
    df['close_dt'] = pd.NaT

    # 2019~2025 신규 역 덮어쓰기
    new_df = pd.read_csv(os.path.join(DATA_DIR, 'subway_new_openings_2019_2025.csv'))
    new_df['open_dt'] = pd.to_datetime(new_df['opening_date'], errors='coerce')
    # is_new_station=True인 것만 별도 행으로 append (좌표는 Kakao keyword로 대충 찾거나 bjd 대표)
    # 단순화: 신규역은 고유좌표 부재이므로 bjd 대표좌표 fallback. 여기선 해당 구/bjd의 성공단지 평균 사용
    apt = pd.read_csv(os.path.join(DATA_DIR, 'apartment_coords.csv'))
    bjd_ctr = apt.groupby(['gu', 'bjd']).agg(lat=('lat', 'mean'), lng=('lng', 'mean')).reset_index()
    extra = new_df[new_df['is_new_station'] == True].merge(
        bjd_ctr, left_on=['gu', 'bjd'], right_on=['gu', 'bjd'], how='left')
    extra = extra.dropna(subset=['lat', 'lng'])
    extra = extra[['lat', 'lng', 'open_dt']].assign(close_dt=pd.NaT)
    return pd.concat([df, extra], ignore_index=True)


def coords_radians(df, active_mask):
    sub = df.loc[active_mask, ['lat', 'lng']]
    return np.radians(sub.values) if len(sub) else np.empty((0, 2))


def active_at(df, cutoff: pd.Timestamp) -> pd.Series:
    opened = df['open_dt'].isna() | (df['open_dt'] <= cutoff)
    closed = df['close_dt'].notna() & (df['close_dt'] <= cutoff)
    return opened & ~closed


def features(apt_rad, facility_rad):
    n = len(apt_rad)
    if len(facility_rad) == 0:
        return np.full(n, np.nan), {f'count_{r}m': np.zeros(n, dtype=int) for r in RADII}
    tree = BallTree(facility_rad, metric='haversine')
    d, _ = tree.query(apt_rad, k=1)
    nearest = d.flatten() * EARTH_M
    counts = {f'count_{r}m': np.array([len(x) for x in tree.query_radius(apt_rad, r=r / EARTH_M)],
                                       dtype=int) for r in RADII}
    return nearest, counts


def main():
    apt = pd.read_csv(os.path.join(DATA_DIR, 'apartment_coords.csv'))
    apt = apt.dropna(subset=['lat', 'lng']).reset_index(drop=True)
    apt_rad = np.radians(apt[['lat', 'lng']].values)
    print(f"아파트: {len(apt):,}")

    # 시간가변 시설
    dynamic = {
        'subway': load_subway_with_opening(),
        'elem_school': load_schools().query('school_type == "초등학교"'),
        'middle_school': load_schools().query('school_type == "중학교"'),
        'high_school': load_schools().query('school_type == "고등학교"'),
        'childcare': load_childcare(),
        'park': load_parks(),
        'academy': load_academies(),
        'large_store': load_large_stores(),  # 커피숍·편의점 등 근린시설
    }
    # 시간불변 (스냅샷)
    static = {
        'library': load_static('seoul_libraries.csv', 'XCNTS', 'YDNTS'),
        'mart': load_static('seoul_marts_kakao.csv', 'y', 'x'),
        'department': load_static('seoul_department_stores_kakao.csv', 'y', 'x'),
        'cctv': load_static('cctv_raw.json', 'WGSYPT', 'WGSXPT', fmt='json'),
        'hospital': load_static('seoul_hospitals_kakao.csv', 'y', 'x'),
    }
    hospital_general = load_static('seoul_hospitals_kakao.csv', 'y', 'x')
    hg_full = pd.read_csv(os.path.join(DATA_DIR, 'seoul_hospitals_kakao.csv'))
    general_names = set(hg_full[hg_full.get('종별') == '종합병원']['id'].astype(str))

    nearest_drop = {'cctv'}
    sparse_facilities = {'park', 'department', 'library'}

    rows = []
    for year in YEARS:
        cutoff = pd.Timestamp(f'{year}-12-31')
        print(f"\n=== {year}년 스냅샷 (cutoff={cutoff.date()}) ===")
        year_out = apt[['gu', 'bjd', 'apt_name_raw', 'lat', 'lng']].copy()
        year_out['거래년도'] = year

        for name, df in dynamic.items():
            mask = active_at(df, cutoff)
            coords = coords_radians(df, mask)
            print(f"  {name}: active={mask.sum():,}/{len(df):,}")
            nearest, counts = features(apt_rad, coords)
            if name not in nearest_drop:
                year_out[f'{name}_nearest_m'] = np.round(nearest, 1)
            for k, v in counts.items():
                year_out[f'{name}_{k}'] = v
            if name in sparse_facilities:
                year_out[f'{name}_within_1km'] = (counts['count_1000m'] > 0).astype(int)
                year_out[f'{name}_log1p_count_2km'] = np.log1p(counts['count_2000m'])

        # 정적 시설은 한 번만 계산 (연도 무관) — 첫 해에만 계산 후 재사용
        for name, df in static.items():
            coords = np.radians(df[['lat', 'lng']].values)
            nearest, counts = features(apt_rad, coords)
            if name not in nearest_drop:
                year_out[f'{name}_nearest_m'] = np.round(nearest, 1)
            for k, v in counts.items():
                year_out[f'{name}_{k}'] = v
            if name in sparse_facilities:
                year_out[f'{name}_within_1km'] = (counts['count_1000m'] > 0).astype(int)
                year_out[f'{name}_log1p_count_2km'] = np.log1p(counts['count_2000m'])

        # 종합병원 (정적 필터)
        if general_names:
            hg = hg_full[hg_full['종별'] == '종합병원'][['y', 'x']].rename(columns={'y': 'lat', 'x': 'lng'})
            hg['lat'] = pd.to_numeric(hg['lat'], errors='coerce')
            hg['lng'] = pd.to_numeric(hg['lng'], errors='coerce')
            hg = hg.dropna()
            coords = np.radians(hg.values)
            nearest, counts = features(apt_rad, coords)
            year_out['hospital_general_nearest_m'] = np.round(nearest, 1)
            for k, v in counts.items():
                year_out[f'hospital_general_{k}'] = v

        rows.append(year_out)

    final = pd.concat(rows, ignore_index=True)
    out = os.path.join(DATA_DIR, 'apartment_distance_features_yearly.csv')
    final.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"\n저장: {out} ({len(final):,} 행, {len(final.columns)} 컬럼)")
    # 연도별 대표 시설 수 변화
    print("\n연도별 active 시설수 요약:")
    for name in ['academy', 'childcare', 'large_store', 'subway']:
        cnts = [active_at(dynamic[name], pd.Timestamp(f'{y}-12-31')).sum() for y in YEARS]
        print(f"  {name}: {dict(zip(YEARS, cnts))}")


if __name__ == '__main__':
    main()
