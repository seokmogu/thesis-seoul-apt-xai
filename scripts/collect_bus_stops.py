#!/usr/bin/env python3
"""서울시 버스정류소 위치 (busStopLocationXY) 전수 수집."""
import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_api_keys, request_with_retry

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
SERVICE = 'busStopLocationXY'
PAGE = 1000


def main():
    keys = load_api_keys()
    key = keys['SEOUL_API_KEY']
    base = f'http://openapi.seoul.go.kr:8088/{key}/json/{SERVICE}'
    r = request_with_retry(f'{base}/1/1/')
    total = r.json()[SERVICE]['list_total_count']
    print(f"전체 {total}")
    rows = []
    start = 1
    while start <= total:
        end = min(start + PAGE - 1, total)
        r = request_with_retry(f'{base}/{start}/{end}/')
        batch = r.json().get(SERVICE, {}).get('row', [])
        rows.extend(batch)
        print(f"  {start}-{end} 누적 {len(rows)}/{total}")
        start = end + 1
        time.sleep(0.1)
    df = pd.DataFrame(rows)
    print(f"\n수집: {len(df):,} 행, 컬럼: {list(df.columns)}")
    # 좌표 컬럼 확인
    if 'XCRD' in df.columns:
        df['xn'] = pd.to_numeric(df['XCRD'], errors='coerce')
        df['yn'] = pd.to_numeric(df['YCRD'], errors='coerce')
        if df['yn'].dropna().between(37, 38).sum() > 100:
            print(f"  YCRD가 위도 — OK (WGS84)")
        else:
            print(f"  YCRD 범위 확인 필요")
    path = os.path.join(DATA_DIR, 'seoul_bus_stops.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"저장: {path}")


if __name__ == '__main__':
    main()
