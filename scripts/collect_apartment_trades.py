#!/usr/bin/env python3
"""국토교통부 아파트매매 실거래가 수집"""
import os, sys, time
import xml.etree.ElementTree as ET
import pandas as pd
from utils import load_api_keys, request_with_retry

DISTRICT_CODES = [
    11110, 11140, 11170, 11200, 11215, 11230, 11260, 11290, 11305,
    11320, 11350, 11380, 11410, 11440, 11470, 11500, 11530, 11545,
    11560, 11590, 11620, 11650, 11680, 11710, 11740
]

URL = "https://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev"

COLUMNS = ['거래금액', '건축년도', '전용면적', '층', '법정동', '도로명', '아파트명', '년', '월']

# API field mapping (XML tag names from the API response)
FIELD_MAP = {
    'dealAmount': '거래금액',
    'buildYear': '건축년도',
    'excluUseAr': '전용면적',
    'floor': '층',
    'umdNm': '법정동',
    'roadNm': '도로명',
    'aptNm': '아파트명',
    'dealYear': '년',
    'dealMonth': '월',
}

def parse_xml_items(text):
    """Parse XML response and return list of dicts."""
    root = ET.fromstring(text)
    result_code = root.findtext('.//resultCode', '')
    if result_code not in ('00', '000', '0'):
        return None, 0, result_code
    total_count = int(root.findtext('.//totalCount', '0'))
    items = []
    for item_el in root.iter('item'):
        row = {}
        for api_field, col_name in FIELD_MAP.items():
            row[col_name] = (item_el.findtext(api_field) or '').strip()
        items.append(row)
    return items, total_count, result_code

def collect():
    keys = load_api_keys()
    api_key = keys['DATA_GO_KR_KEY_DECODED']
    
    all_rows = []
    months = [f"{y}{m:02d}" for y in range(2019, 2025) for m in range(1, 13)]
    
    total = len(DISTRICT_CODES) * len(months)
    done = 0
    
    for code in DISTRICT_CODES:
        for month in months:
            done += 1
            params = {
                'serviceKey': api_key,
                'LAWD_CD': code,
                'DEAL_YMD': month,
                'pageNo': 1,
                'numOfRows': 1000,
            }
            try:
                r = request_with_retry(URL, params=params)
                items, total_count, rc = parse_xml_items(r.text)
                
                if items is None:
                    if done % 100 == 0:
                        print(f"  [{done}/{total}] {code} {month}: code={rc}")
                    continue
                
                all_rows.extend(items)
                
                # Handle pagination
                fetched = len(items)
                page = 1
                while fetched < total_count:
                    page += 1
                    params['pageNo'] = page
                    r = request_with_retry(URL, params=params)
                    more_items, _, _ = parse_xml_items(r.text)
                    if not more_items:
                        break
                    all_rows.extend(more_items)
                    fetched += len(more_items)
                    time.sleep(0.1)
                
                if done % 50 == 0:
                    print(f"  [{done}/{total}] {code} {month}: {total_count} rows (total collected: {len(all_rows)})")
                    
            except Exception as e:
                print(f"  [{done}/{total}] {code} {month}: ERROR {e}")
            
            time.sleep(0.05)
    
    df = pd.DataFrame(all_rows)
    if not df.empty:
        df['거래금액'] = df['거래금액'].astype(str).str.strip().str.replace(',', '')
        df['거래년월'] = df['년'].astype(str) + df['월'].astype(str).str.zfill(2)
        df = df.drop(columns=['년', '월'], errors='ignore')
    
    out = os.path.join(os.path.dirname(__file__), 'data', 'apartment_trades.csv')
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"Saved {len(df)} rows to {out}")

if __name__ == '__main__':
    collect()
