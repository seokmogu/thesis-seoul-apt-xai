#!/usr/bin/env python3
"""아파트 실거래가 수집 v2 — 2019.01~2025.12, 동 단위 보존"""
import os, sys, time, json
import xml.etree.ElementTree as ET
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_api_keys, request_with_retry

DISTRICT_CODES = [
    11110, 11140, 11170, 11200, 11215, 11230, 11260, 11290, 11305,
    11320, 11350, 11380, 11410, 11440, 11470, 11500, 11530, 11545,
    11560, 11590, 11620, 11650, 11680, 11710, 11740
]

# 구코드 → 구이름 매핑
GU_MAP = {
    11110: '종로구', 11140: '중구', 11170: '용산구', 11200: '성동구', 11215: '광진구',
    11230: '동대문구', 11260: '중랑구', 11290: '성북구', 11305: '강북구', 11320: '도봉구',
    11350: '노원구', 11380: '은평구', 11410: '서대문구', 11440: '마포구', 11470: '양천구',
    11500: '강서구', 11530: '구로구', 11545: '금천구', 11560: '영등포구', 11590: '동작구',
    11620: '관악구', 11650: '서초구', 11680: '강남구', 11710: '송파구', 11740: '강동구'
}

URL = "https://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev"

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
    'dealDay': '일',
    'sggCd': '구코드',
    'umdCd': '동코드',
    'jibun': '지번',
    'bonbun': '본번',
    'bubun': '부번',
}

def parse_xml_items(text):
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
    # 2019.01 ~ 2025.12
    months = [f"{y}{m:02d}" for y in range(2019, 2026) for m in range(1, 13)]

    total = len(DISTRICT_CODES) * len(months)
    done = 0
    errors = 0
    
    sys.stdout.flush()

    for code in DISTRICT_CODES:
        gu_name = GU_MAP[code]
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
                r = request_with_retry(URL, params=params, max_retries=5)
                items, total_count, rc = parse_xml_items(r.text)

                if items is None:
                    if rc not in ('00', '000', '0'):
                        errors += 1
                    continue

                # Add gu info
                for item in items:
                    item['구'] = gu_name
                all_rows.extend(items)

                # Pagination
                fetched = len(items)
                page = 1
                while fetched < total_count:
                    page += 1
                    params['pageNo'] = page
                    r = request_with_retry(URL, params=params, max_retries=5)
                    more_items, _, _ = parse_xml_items(r.text)
                    if not more_items:
                        break
                    for item in more_items:
                        item['구'] = gu_name
                    all_rows.extend(more_items)
                    fetched += len(more_items)
                    time.sleep(0.1)

                if done % 25 == 0:
                    print(f"  [{done}/{total}] {gu_name} {month}: {total_count}건 (누적: {len(all_rows):,})", flush=True)

            except Exception as e:
                errors += 1
                print(f"  [{done}/{total}] {gu_name} {month}: ERROR {e}")

            time.sleep(0.05)

    print(f"\n수집 완료: {len(all_rows):,}건 (에러: {errors}건)")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df['거래금액'] = df['거래금액'].astype(str).str.strip().str.replace(',', '').astype(int)
        df['건축년도'] = pd.to_numeric(df['건축년도'], errors='coerce')
        df['전용면적'] = pd.to_numeric(df['전용면적'], errors='coerce')
        df['층'] = pd.to_numeric(df['층'], errors='coerce')
        df['거래년월'] = df['년'].astype(str) + df['월'].astype(str).str.zfill(2)
        df['거래년도'] = df['년'].astype(int)
        df = df.drop(columns=['년', '월', '일'], errors='ignore')

    out = os.path.join(os.path.dirname(__file__), '..', 'data', 'apartment_trades_v2.csv')
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"Saved to {out}")

if __name__ == '__main__':
    collect()
