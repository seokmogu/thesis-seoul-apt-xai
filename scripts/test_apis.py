#!/usr/bin/env python3
"""Test each API with a minimal request"""
import json
from utils import load_api_keys, request_with_retry

keys = load_api_keys()

results = {}

# 1. 국토교통부 아파트매매
print("=== 1. 국토교통부 아파트매매 실거래가 ===")
try:
    import xml.etree.ElementTree as ET
    r = request_with_retry(
        "https://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev",
        params={'serviceKey': keys['DATA_GO_KR_KEY_DECODED'], 'LAWD_CD': 11680, 'DEAL_YMD': 202401, 'pageNo': 1, 'numOfRows': 5}
    )
    root = ET.fromstring(r.text)
    rc = root.findtext('.//resultCode','')
    total = root.findtext('.//totalCount','0')
    print(f"  resultCode: {rc}, totalCount: {total}")
    results['apartment'] = 'OK' if rc in ('00','000') else f"FAIL: {rc}"
except Exception as e:
    results['apartment'] = f'FAIL: {e}'
    print(f"  {e}")

# 2. 서울열린데이터광장 - 지하철
print("\n=== 2. 서울열린데이터광장 (지하철역) ===")
try:
    url = f"http://openapi.seoul.go.kr:8088/{keys['SEOUL_API_KEY']}/json/SearchSTNBySubwayLineInfo/1/3"
    r = request_with_retry(url)
    data = r.json()
    if 'SearchSTNBySubwayLineInfo' in data:
        total = data['SearchSTNBySubwayLineInfo']['list_total_count']
        print(f"  OK, total stations: {total}")
        results['seoul_subway'] = 'OK'
    else:
        print(f"  {data.get('RESULT', data)}")
        results['seoul_subway'] = f"FAIL: {data.get('RESULT', {}).get('MESSAGE', '')}"
except Exception as e:
    results['seoul_subway'] = f'FAIL: {e}'
    print(f"  {e}")

# 2b. 서울 공원
print("\n=== 2b. 서울열린데이터광장 (공원) ===")
try:
    url = f"http://openapi.seoul.go.kr:8088/{keys['SEOUL_API_KEY']}/json/SearchParkInfoService/1/3"
    r = request_with_retry(url)
    data = r.json()
    if 'SearchParkInfoService' in data:
        total = data['SearchParkInfoService']['list_total_count']
        print(f"  OK, total parks: {total}")
        results['seoul_parks'] = 'OK'
    else:
        print(f"  {data.get('RESULT', data)}")
        results['seoul_parks'] = f"FAIL: {data.get('RESULT', {}).get('MESSAGE', '')}"
except Exception as e:
    results['seoul_parks'] = f'FAIL: {e}'
    print(f"  {e}")

# 3. ECOS
print("\n=== 3. 한국은행 ECOS ===")
try:
    url = f"https://ecos.bok.or.kr/api/StatisticSearch/{keys['ECOS_API_KEY']}/json/kr/1/5/722Y001/M/201901/201903/0101000"
    r = request_with_retry(url)
    data = r.json()
    if 'StatisticSearch' in data:
        rows = data['StatisticSearch']['row']
        print(f"  OK, got {len(rows)} rows. Sample: {rows[0]['TIME']}={rows[0]['DATA_VALUE']}")
        results['ecos'] = 'OK'
    else:
        print(f"  {data.get('RESULT', data)}")
        results['ecos'] = f"FAIL: {data.get('RESULT', {}).get('MESSAGE', '')}"
except Exception as e:
    results['ecos'] = f'FAIL: {e}'
    print(f"  {e}")

# 4. NEIS 학교정보
print("\n=== 4. 나이스 학교정보 ===")
try:
    r = request_with_retry(
        "https://open.neis.go.kr/hub/schoolInfo",
        params={'KEY': keys['NEIS_API_KEY'], 'Type': 'json', 'pIndex': 1, 'pSize': 3, 'ATPT_OFCDC_SC_CODE': 'B10'}
    )
    data = r.json()
    if 'schoolInfo' in data:
        total = data['schoolInfo'][0]['head'][0]['list_total_count']
        print(f"  OK, total schools: {total}")
        results['neis'] = 'OK'
    else:
        print(f"  {data.get('RESULT', data)}")
        results['neis'] = f"FAIL: {data.get('RESULT', {}).get('MESSAGE', '')}"
except Exception as e:
    results['neis'] = f'FAIL: {e}'
    print(f"  {e}")

print("\n=== SUMMARY ===")
for k, v in results.items():
    status = "✅" if v == 'OK' else "❌"
    print(f"  {status} {k}: {v}")
