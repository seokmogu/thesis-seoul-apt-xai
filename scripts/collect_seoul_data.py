#!/usr/bin/env python3
"""서울열린데이터광장 - 지하철역, 공원, 대형마트 위치 수집"""
import os, json
import pandas as pd
from utils import load_api_keys, request_with_retry

BASE_URL = "http://openapi.seoul.go.kr:8088/{key}/json/{service}/{start}/{end}"

SERVICES = {
    'subway_stations': {
        'service': 'SearchSTNBySubwayLineInfo',
        'desc': '지하철역 위치',
    },
    'parks': {
        'service': 'SearchParkInfoService',
        'desc': '공원 위치',
    },
    'marts': {
        'service': 'TbgisBuildingAll',  # 대형마트
        'desc': '대형마트 위치',
    },
}

# Alternative mart service names to try
MART_ALTERNATIVES = ['LargeMarketInfo', 'TbgisBigmartW']

def fetch_service(key, service_name, desc, max_rows=1000):
    """Fetch all rows from a Seoul Open Data service."""
    print(f"\nCollecting {desc} ({service_name})...")
    
    all_rows = []
    start = 1
    batch = 1000
    
    while True:
        end = start + batch - 1
        url = BASE_URL.format(key=key, service=service_name, start=start, end=end)
        try:
            r = request_with_retry(url)
            data = r.json()
            
            # Check for error
            if 'RESULT' in data:
                code = data['RESULT'].get('CODE', '')
                msg = data['RESULT'].get('MESSAGE', '')
                if code != 'INFO-000':
                    print(f"  API response: {code} - {msg}")
                    return None
            
            # Find the data key (service name is usually the key)
            data_key = None
            for k in data:
                if k != 'RESULT':
                    data_key = k
                    break
            
            if not data_key:
                print(f"  No data key found")
                return None
            
            service_data = data[data_key]
            total = int(service_data.get('list_total_count', 0))
            rows = service_data.get('row', [])
            all_rows.extend(rows)
            
            print(f"  Fetched {len(all_rows)}/{total}")
            
            if len(all_rows) >= total:
                break
            start = end + 1
            
        except Exception as e:
            print(f"  Error: {e}")
            break
    
    return all_rows

def collect():
    keys = load_api_keys()
    api_key = keys['SEOUL_API_KEY']
    out_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    # 1. Subway stations
    rows = fetch_service(api_key, 'SearchSTNBySubwayLineInfo', '지하철역 위치')
    if rows:
        df = pd.DataFrame(rows)
        path = os.path.join(out_dir, 'seoul_subway_stations.csv')
        df.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"  Saved {len(df)} rows to {path}")
    
    # 2. Parks
    rows = fetch_service(api_key, 'SearchParkInfoService', '공원 위치')
    if rows:
        df = pd.DataFrame(rows)
        path = os.path.join(out_dir, 'seoul_parks.csv')
        df.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"  Saved {len(df)} rows to {path}")
    
    # 3. Large marts - try multiple service names
    mart_services = ['LargeMarketInfo', 'TbgisBigmartW', 'ListLargeRetailStoreService']
    rows = None
    for svc in mart_services:
        rows = fetch_service(api_key, svc, f'대형마트 ({svc})')
        if rows:
            break
    
    if rows:
        df = pd.DataFrame(rows)
        path = os.path.join(out_dir, 'seoul_marts.csv')
        df.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"  Saved {len(df)} rows to {path}")
    else:
        print("  Could not find working mart service. You may need to check Seoul Open Data portal manually.")

if __name__ == '__main__':
    collect()
