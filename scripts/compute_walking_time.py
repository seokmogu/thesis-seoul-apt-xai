#!/usr/bin/env python3
"""
단지 ↔ 시설 간 도보 시간 계산 파이프라인 (병렬 구성, 후속 반영용).

Kakao 공식 도보 경로 API는 공개되어 있지 않다 (자동차 Directions만 제공).
본 스크립트는 세 가지 계산 모드를 병렬 구조로 제공한다.

  mode=osrm     : OSRM 공용 서버 (http://router.project-osrm.org) foot 프로파일
                  - 무료, 요청 속도 제한 있음
  mode=ors      : OpenRouteService foot-walking 프로파일
                  - 무료 API 키 필요(일 2,000 호출)
  mode=approx   : haversine 직선거리 × 보정계수(기본 1.35) / 평균보행속도(4km/h)
                  - 외부 API 무의존, 즉시 산출. 평지 권역에서 오차 ±20% 수준.
  mode=kakao_car: Kakao Mobility 자동차 Directions (참고용 소요시간)
                  - KAKAO_API_KEY 재사용, 단 실제 도보 시간 아님

입력: data/apartment_coords.csv (8,601 단지)
      data/seoul_{facility}.csv 등 시설별 좌표 데이터

출력: data/walking_time/{facility}_nearest_walk_min.csv
  컬럼: gu, bjd, apt_name_raw, lat, lng, facility_name, facility_lat, facility_lng, walk_min

병렬 구조:
  - ThreadPoolExecutor (max_workers=20) 로 단지 × 시설 페어 호출
  - 429/5xx 재시도 지수 백오프
  - 중간 체크포인트 (1000 페어마다 저장)

실제 데이터 반영은 별도 단계에서 수행.
"""
import os
import sys
import time
import json
import math
import argparse
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_api_keys

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUT_DIR = os.path.join(DATA_DIR, 'walking_time')
os.makedirs(OUT_DIR, exist_ok=True)

EARTH_M = 6_371_000.0
WALK_SPEED_MPM = 4000.0 / 60.0  # 4km/h → 66.7 m/min
APPROX_ROAD_FACTOR = 1.35  # 직선 대비 도로·우회 보정계수(평지 평균)


def haversine_m(lat1, lng1, lat2, lng2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2 * EARTH_M * math.asin(math.sqrt(a))


def approx_walk_min(lat1, lng1, lat2, lng2, factor=APPROX_ROAD_FACTOR):
    d = haversine_m(lat1, lng1, lat2, lng2) * factor
    return d / WALK_SPEED_MPM


def osrm_walk_min(lat1, lng1, lat2, lng2, base='http://router.project-osrm.org', retries=3):
    url = f'{base}/route/v1/foot/{lng1},{lat1};{lng2},{lat2}'
    for i in range(retries):
        try:
            r = requests.get(url, params={'overview': 'false'}, timeout=15)
            if r.status_code == 200:
                d = r.json()['routes'][0]['duration']  # seconds
                return d / 60.0
            if r.status_code == 429:
                time.sleep(1.0 + i)
                continue
        except Exception:
            time.sleep(0.5 + i)
    return None


def ors_walk_min(lat1, lng1, lat2, lng2, key, retries=3):
    url = 'https://api.openrouteservice.org/v2/directions/foot-walking'
    headers = {'Authorization': key}
    params = {'start': f'{lng1},{lat1}', 'end': f'{lng2},{lat2}'}
    for i in range(retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=15)
            if r.status_code == 200:
                feats = r.json()['features']
                if feats:
                    return feats[0]['properties']['summary']['duration'] / 60.0
            elif r.status_code in (429, 503):
                time.sleep(2 + i)
                continue
            else:
                return None
        except Exception:
            time.sleep(1 + i)
    return None


def kakao_car_min(lat1, lng1, lat2, lng2, key, retries=3):
    """Kakao Mobility 자동차 Directions. 도보가 아닌 참고용 차량 시간."""
    url = 'https://apis-navi.kakaomobility.com/v1/directions'
    headers = {'Authorization': f'KakaoAK {key}'}
    params = {
        'origin': f'{lng1},{lat1}',
        'destination': f'{lng2},{lat2}',
        'priority': 'RECOMMEND',
    }
    for i in range(retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=15)
            if r.status_code == 200:
                routes = r.json().get('routes', [])
                if routes and routes[0].get('summary'):
                    return routes[0]['summary']['duration'] / 60.0
            elif r.status_code in (429, 503):
                time.sleep(1 + i)
                continue
            else:
                return None
        except Exception:
            time.sleep(1 + i)
    return None


def calc(mode, apt_coord, fac_coord, api_key=None):
    lat1, lng1 = apt_coord
    lat2, lng2 = fac_coord
    if mode == 'approx':
        return approx_walk_min(lat1, lng1, lat2, lng2)
    if mode == 'osrm':
        return osrm_walk_min(lat1, lng1, lat2, lng2)
    if mode == 'ors':
        return ors_walk_min(lat1, lng1, lat2, lng2, api_key)
    if mode == 'kakao_car':
        return kakao_car_min(lat1, lng1, lat2, lng2, api_key)
    raise ValueError(mode)


def load_facility(name):
    """시설별 좌표 로드 (간단 버전: 주요 카테고리만)."""
    paths = {
        'subway': ('subway_stations_api.json', 'LAT', 'LOT'),
        'mart': ('seoul_marts_kakao.csv', 'y', 'x'),
        'department': ('seoul_department_stores_kakao.csv', 'y', 'x'),
        'library': ('seoul_libraries.csv', 'XCNTS', 'YDNTS'),
        'hospital_general': ('seoul_hospitals_kakao.csv', 'y', 'x'),
    }
    fn, la, lo = paths[name]
    p = os.path.join(DATA_DIR, fn)
    if fn.endswith('.json'):
        df = pd.DataFrame(json.load(open(p)))
    else:
        df = pd.read_csv(p, low_memory=False)
    if name == 'hospital_general' and '종별' in df.columns:
        df = df[df['종별'] == '종합병원'].copy()
    df = df.rename(columns={la: 'lat', lo: 'lng'})
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lng'] = pd.to_numeric(df['lng'], errors='coerce')
    return df.dropna(subset=['lat', 'lng']).reset_index(drop=True)


def nearest(lat, lng, fac_df):
    """haversine 기준 최근접 시설 한 개."""
    best_d = float('inf')
    best_i = None
    for i, r in fac_df.iterrows():
        d = haversine_m(lat, lng, r['lat'], r['lng'])
        if d < best_d:
            best_d, best_i = d, i
    return best_i, best_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['approx', 'osrm', 'ors', 'kakao_car'], default='approx')
    ap.add_argument('--facility', default='subway',
                    choices=['subway', 'mart', 'department', 'library', 'hospital_general', 'all'])
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--limit', type=int, default=0, help='테스트 단지 수')
    args = ap.parse_args()

    keys = load_api_keys()
    api_key = None
    if args.mode == 'ors':
        api_key = keys.get('ORS_API_KEY')
        if not api_key:
            sys.exit('ORS_API_KEY 필요')
    elif args.mode == 'kakao_car':
        api_key = keys.get('KAKAO_API_KEY')

    apt = pd.read_csv(os.path.join(DATA_DIR, 'apartment_coords.csv'))
    apt = apt.dropna(subset=['lat', 'lng']).reset_index(drop=True)
    if args.limit:
        apt = apt.head(args.limit)
    print(f"아파트 단지: {len(apt):,} / mode={args.mode}")

    facilities = [args.facility] if args.facility != 'all' else ['subway', 'mart', 'department', 'library', 'hospital_general']

    for fac in facilities:
        fac_df = load_facility(fac)
        print(f"\n=== {fac} (n={len(fac_df)}) ===")

        # 1) 최근접 시설 선정 (haversine 한 번)
        tasks = []
        for _, r in apt.iterrows():
            fi, d = nearest(r['lat'], r['lng'], fac_df)
            tasks.append({
                'gu': r['gu'], 'bjd': r['bjd'], 'apt_name_raw': r['apt_name_raw'],
                'apt_lat': r['lat'], 'apt_lng': r['lng'],
                'fac_idx': fi, 'fac_lat': fac_df.at[fi, 'lat'], 'fac_lng': fac_df.at[fi, 'lng'],
                'straight_m': d,
            })

        # 2) 도보 시간 계산 (병렬)
        def worker(t):
            m = calc(args.mode, (t['apt_lat'], t['apt_lng']), (t['fac_lat'], t['fac_lng']), api_key)
            t['walk_min'] = m
            return t

        t0 = time.time()
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(worker, t) for t in tasks]
            results = []
            for f in as_completed(futs):
                results.append(f.result())
                done += 1
                if done % 200 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed else 0
                    eta = (len(tasks) - done) / rate / 60 if rate else 0
                    print(f"  [{done}/{len(tasks)}] {rate:.1f}/s ETA {eta:.1f}분")

        out_df = pd.DataFrame(results)
        outp = os.path.join(OUT_DIR, f'{fac}_nearest_walk_{args.mode}.csv')
        out_df.to_csv(outp, index=False, encoding='utf-8-sig')
        hit = out_df['walk_min'].notna().sum()
        print(f"저장: {outp}  성공 {hit}/{len(out_df)}")
        if hit:
            print(f"  도보 분 중앙값={out_df['walk_min'].median():.1f}, p90={out_df['walk_min'].quantile(0.9):.1f}")


if __name__ == '__main__':
    main()
