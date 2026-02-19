#!/usr/bin/env python3
"""한국은행 ECOS API - 거시경제 지표 수집"""
import os
import pandas as pd
from utils import load_api_keys, request_with_retry

BASE_URL = "https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/1/1000/{stat_code}/{period}/{start}/{end}/{item_code}"

INDICATORS = [
    {'name': '기준금리', 'stat_code': '722Y001', 'item_code': '0101000', 'period': 'M'},
    {'name': 'CD금리', 'stat_code': '721Y001', 'item_code': '2010000', 'period': 'M'},
    {'name': '소비자물가지수', 'stat_code': '901Y009', 'item_code': '0', 'period': 'M'},
    {'name': 'M2', 'stat_code': '161Y008', 'item_code': 'BBGA00', 'period': 'M'},
]

def collect():
    keys = load_api_keys()
    api_key = keys['ECOS_API_KEY']
    out_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    all_dfs = []
    
    for ind in INDICATORS:
        name = ind['name']
        print(f"Collecting {name}...")
        
        url = BASE_URL.format(
            key=api_key,
            stat_code=ind['stat_code'],
            period=ind['period'],
            start='201901',
            end='202412',
            item_code=ind['item_code'],
        )
        
        try:
            r = request_with_retry(url)
            data = r.json()
            
            if 'StatisticSearch' not in data:
                result = data.get('RESULT', {})
                print(f"  Error: {result.get('CODE', '?')} - {result.get('MESSAGE', '?')}")
                continue
            
            rows = data['StatisticSearch']['row']
            df = pd.DataFrame(rows)
            df = df[['TIME', 'DATA_VALUE']].rename(columns={
                'TIME': '년월',
                'DATA_VALUE': name,
            })
            all_dfs.append(df)
            print(f"  Got {len(df)} rows")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    if all_dfs:
        result = all_dfs[0]
        for df in all_dfs[1:]:
            result = result.merge(df, on='년월', how='outer')
        result = result.sort_values('년월').reset_index(drop=True)
        
        path = os.path.join(out_dir, 'ecos_macro.csv')
        result.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"\nSaved {len(result)} rows to {path}")

if __name__ == '__main__':
    collect()
