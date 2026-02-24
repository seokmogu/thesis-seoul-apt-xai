#!/usr/bin/env python3
"""나이스 API: 서울 학교 전체 수집 → 동 단위 집계"""
import os, sys, re, json
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_api_keys, request_with_retry

def collect_all_schools():
    keys = load_api_keys()
    api_key = keys['NEIS_API_KEY']
    url = 'https://open.neis.go.kr/hub/schoolInfo'
    
    all_schools = []
    page = 1
    while True:
        params = {
            'KEY': api_key,
            'Type': 'json',
            'pIndex': page,
            'pSize': 1000,
            'ATPT_OFCDC_SC_CODE': 'B10',  # 서울
        }
        r = request_with_retry(url, params=params, max_retries=3)
        data = r.json()
        
        if 'schoolInfo' not in data:
            break
        
        rows = data['schoolInfo'][1]['row']
        all_schools.extend(rows)
        total = data['schoolInfo'][0]['head'][0]['list_total_count']
        print(f"  Page {page}: {len(rows)} schools (total: {len(all_schools)}/{total})")
        
        if len(all_schools) >= total:
            break
        page += 1
    
    print(f"\n총 {len(all_schools)}개 학교 수집")
    return all_schools

def extract_gu_dong(address, address_detail):
    """도로명주소에서 구와 동 추출"""
    gu = dong = None
    
    # ORG_RDNMA: "서울특별시 송파구 송이로 42"
    gu_match = re.search(r'서울특별시\s+(\S+구)', address or '')
    if gu_match:
        gu = gu_match.group(1)
    
    # ORG_RDNDA: "(송파동/가락고등학교)" or "/ 경복초등학교 (능동)"
    dong_match = re.search(r'\(([^/\)]+동)', address_detail or '')
    if not dong_match:
        dong_match = re.search(r'([^\s/\(]+동)', address_detail or '')
    if dong_match:
        dong = dong_match.group(1).strip()
    
    return gu, dong

def main():
    schools = collect_all_schools()
    
    records = []
    for s in schools:
        gu, dong = extract_gu_dong(s.get('ORG_RDNMA', ''), s.get('ORG_RDNDA', ''))
        records.append({
            'school_name': s['SCHUL_NM'],
            'school_type': s['SCHUL_KND_SC_NM'],
            'gu': gu,
            'dong': dong,
            'address': s.get('ORG_RDNMA', ''),
            'address_detail': s.get('ORG_RDNDA', ''),
            'founded': s.get('FOND_YMD', ''),
        })
    
    df = pd.DataFrame(records)
    
    # Stats
    print(f"\n구 추출 성공: {df['gu'].notna().sum()}/{len(df)}")
    print(f"동 추출 성공: {df['dong'].notna().sum()}/{len(df)}")
    print(f"\n학교 유형별:")
    print(df['school_type'].value_counts())
    
    # 동 단위 집계
    dong_counts = df.groupby(['gu', 'dong', 'school_type']).size().unstack(fill_value=0)
    dong_counts.columns = [f'{c}수' for c in dong_counts.columns]
    dong_counts = dong_counts.reset_index()
    
    # Save raw + aggregated
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    df.to_csv(os.path.join(out_dir, 'schools_raw.csv'), index=False, encoding='utf-8-sig')
    dong_counts.to_csv(os.path.join(out_dir, 'schools_per_dong.csv'), index=False, encoding='utf-8-sig')
    
    print(f"\nSaved: schools_raw.csv ({len(df)} rows), schools_per_dong.csv ({len(dong_counts)} rows)")
    print(f"\nSample dong counts:")
    print(dong_counts.head(10))

if __name__ == '__main__':
    main()
