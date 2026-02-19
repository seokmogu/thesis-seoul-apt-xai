#!/usr/bin/env python3
"""나이스 교육정보 API - 서울 학교 정보 수집"""
import os
import pandas as pd
from utils import load_api_keys, request_with_retry

URL = "https://open.neis.go.kr/hub/schoolInfo"

def collect():
    keys = load_api_keys()
    api_key = keys['NEIS_API_KEY']
    out_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    all_rows = []
    page = 1
    page_size = 1000
    
    print("Collecting Seoul school info...")
    
    while True:
        params = {
            'KEY': api_key,
            'Type': 'json',
            'pIndex': page,
            'pSize': page_size,
            'ATPT_OFCDC_SC_CODE': 'B10',  # 서울
        }
        
        try:
            r = request_with_retry(URL, params=params)
            data = r.json()
            
            if 'schoolInfo' not in data:
                result = data.get('RESULT', {})
                code = result.get('CODE', '')
                if code == 'INFO-200':  # no more data
                    break
                print(f"  Error: {code} - {result.get('MESSAGE', '')}")
                break
            
            info = data['schoolInfo']
            # First element is head, second is row
            rows = info[1]['row']
            all_rows.extend(rows)
            
            total = info[0]['head'][0]['list_total_count']
            print(f"  Page {page}: {len(all_rows)}/{total}")
            
            if len(all_rows) >= total:
                break
            page += 1
            
        except Exception as e:
            print(f"  Error on page {page}: {e}")
            break
    
    if all_rows:
        df = pd.DataFrame(all_rows)
        # Select relevant columns
        cols = {
            'SCHUL_NM': '학교명',
            'ORG_RDNMA': '도로명주소',
            'SCHUL_KND_SC_NM': '학교종류',
            'ORG_TELNO': '전화번호',
            'HMPG_ADRES': '홈페이지',
            'FOND_SC_NM': '설립구분',
        }
        available = {k: v for k, v in cols.items() if k in df.columns}
        result = df[list(available.keys())].rename(columns=available)
        
        path = os.path.join(out_dir, 'seoul_schools.csv')
        result.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"Saved {len(result)} rows to {path}")
        print("Note: NEIS API does not provide lat/lng. Geocoding from addresses will be needed separately.")

if __name__ == '__main__':
    collect()
