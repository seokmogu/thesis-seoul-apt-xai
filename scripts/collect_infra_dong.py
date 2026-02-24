#!/usr/bin/env python3
"""동 단위 인프라 변수 수집: 지하철역, 버스정류장, 병원 등
좌표 기반 데이터는 행정동 경계 GeoJSON으로 spatial join"""
import os, sys, json, time
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_api_keys, request_with_retry

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_dong_boundaries():
    """서울시 행정동 경계 로드 — 없으면 법정동 경계 다운로드"""
    # 법정동 경계 파일 확인
    bjd_path = os.path.join(DATA_DIR, 'seoul_bjd_boundary.geojson')
    if os.path.exists(bjd_path):
        print("기존 법정동 경계 파일 사용")
        return gpd.read_file(bjd_path)
    
    # 구 경계만 있으면 법정동 경계가 필요
    print("법정동 경계 파일이 필요합니다. 다운로드 시도...")
    # SGIS 또는 다른 소스에서 가져와야 함
    return None

def subway_per_dong():
    """지하철역 동 단위 집계 (좌표 → 법정동 매핑)"""
    print("\n=== 지하철역 동 단위 매핑 ===")
    
    with open(os.path.join(DATA_DIR, 'subway_stations_api.json')) as f:
        stations = json.load(f)
    
    # 서울 범위 필터
    seoul_stations = []
    for s in stations:
        lat, lon = float(s['LAT']), float(s['LOT'])
        if 37.4 <= lat <= 37.7 and 126.7 <= lon <= 127.2:
            seoul_stations.append(s)
    
    print(f"서울 범위 지하철역: {len(seoul_stations)}개")
    
    # GeoDataFrame
    gdf_stations = gpd.GeoDataFrame(
        seoul_stations,
        geometry=[Point(float(s['LOT']), float(s['LAT'])) for s in seoul_stations],
        crs='EPSG:4326'
    )
    
    return gdf_stations

def collect_bus_stops():
    """서울시 버스정류장 좌표 수집"""
    print("\n=== 버스정류장 수집 ===")
    keys = load_api_keys()
    api_key = keys['SEOUL_API_KEY']
    
    url = f"http://openapi.seoul.go.kr:8088/{api_key}/json/busStopLocationXY/1/1000/"
    r = request_with_retry(url)
    data = r.json()
    
    # Check API name
    key = list(data.keys())[0]
    total = data[key]['list_total_count']
    print(f"버스정류장 총: {total}개")
    
    all_stops = data[key]['row']
    
    # Paginate
    page = 1001
    while len(all_stops) < total:
        end = min(page + 999, total)
        url = f"http://openapi.seoul.go.kr:8088/{api_key}/json/busStopLocationXY/{page}/{end}/"
        r = request_with_retry(url)
        d = r.json()
        rows = d[list(d.keys())[0]]['row']
        all_stops.extend(rows)
        page += 1000
        print(f"  {len(all_stops)}/{total}")
        time.sleep(0.2)
    
    print(f"수집 완료: {len(all_stops)}개")
    return all_stops

def collect_hospitals():
    """서울시 병원 데이터 수집"""
    print("\n=== 병원/의료시설 수집 ===")
    keys = load_api_keys()
    api_key = keys['SEOUL_API_KEY']
    
    # 서울시 병원 인허가 정보
    url = f"http://openapi.seoul.go.kr:8088/{api_key}/json/LOCALDATA_020301/1/1000/"
    try:
        r = request_with_retry(url)
        data = r.json()
        key = list(data.keys())[0]
        if 'list_total_count' in data[key]:
            total = data[key]['list_total_count']
            print(f"병원 총: {total}개")
            return data[key].get('row', []), total
        else:
            print(f"API 응답: {data}")
            return [], 0
    except Exception as e:
        print(f"병원 API 에러: {e}")
        return [], 0

def main():
    # 1. 지하철역 (이미 좌표 있음)
    gdf_subway = subway_per_dong()
    
    # 2. 버스정류장
    try:
        bus_stops = collect_bus_stops()
        with open(os.path.join(DATA_DIR, 'bus_stops_raw.json'), 'w') as f:
            json.dump(bus_stops, f, ensure_ascii=False, indent=2)
        print(f"버스정류장 저장: bus_stops_raw.json")
    except Exception as e:
        print(f"버스정류장 수집 실패: {e}")
        bus_stops = []
    
    # 3. 병원
    try:
        hospitals, h_total = collect_hospitals()
        print(f"병원 샘플: {hospitals[0] if hospitals else 'none'}")
    except Exception as e:
        print(f"병원 수집 실패: {e}")
    
    print("\n=== 요약 ===")
    print(f"지하철역: {len(gdf_subway)}개 (좌표 보유)")
    print(f"버스정류장: {len(bus_stops)}개")
    print(f"\n다음 단계: 법정동 경계 GeoJSON으로 spatial join 필요")

if __name__ == '__main__':
    main()
