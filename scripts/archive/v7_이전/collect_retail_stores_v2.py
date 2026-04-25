#!/usr/bin/env python3
"""
대규모점포 인허가 (LOCALDATA_020301) 재수집 — 물리적 점포 단위.

LOCALDATA_072405(휴게음식점 등)는 법인/점포 단위 5천여 건이고,
LOCALDATA_020301(대규모점포 인허가)는 건물 단위 2천여 건으로
"물리적 백화점/마트 개수"에 가깝다.

필드 보존:
  APVPERMYMD (인허가일), TRDSTATENM (영업상태), DCBYMD (폐업일)
  UPTAENM (업태: 백화점/대형마트/SSM 등), BPLCNM (상호), X/Y (TM좌표)
"""
import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_api_keys, request_with_retry

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
SERVICE = 'LOCALDATA_020301'
PAGE_SIZE = 1000


def fetch_all(api_key: str) -> list:
    base = f'http://openapi.seoul.go.kr:8088/{api_key}/json/{SERVICE}'
    r = request_with_retry(f'{base}/1/1/')
    total = r.json()[SERVICE]['list_total_count']
    print(f"  전체 건수: {total:,}")
    rows = []
    start = 1
    while start <= total:
        end = min(start + PAGE_SIZE - 1, total)
        r = request_with_retry(f'{base}/{start}/{end}/')
        batch = r.json().get(SERVICE, {}).get('row', [])
        if not batch:
            print(f"  {start}-{end} 빈응답 종료")
            break
        rows.extend(batch)
        print(f"  {start:>5}-{end:<5} 누적 {len(rows):>5,}/{total:,}")
        start = end + 1
        time.sleep(0.1)
    return rows


def main():
    keys = load_api_keys()
    rows = fetch_all(keys['SEOUL_API_KEY'])
    df = pd.DataFrame(rows)
    print(f"\n수집 완료: {len(df):,} 행, {len(df.columns)} 컬럼")
    print(f"컬럼: {list(df.columns)}")

    for col in ('APVPERMYMD', 'DCBYMD', 'CLGSTDT', 'CLGENDDT', 'ROPNYMD', 'APVCANCELYMD'):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({'': pd.NA})

    out = os.path.join(DATA_DIR, 'seoul_retail_stores_v2.csv')
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"저장: {out}")

    if 'UPTAENM' in df.columns:
        print("\n업태별 분포:")
        print(df['UPTAENM'].value_counts().head(20).to_string())

    if 'TRDSTATENM' in df.columns:
        print("\n영업상태별:")
        print(df['TRDSTATENM'].value_counts().to_string())

    # 연도별 영업중 추이
    if 'APVPERMYMD' in df.columns and 'DCBYMD' in df.columns and 'UPTAENM' in df.columns:
        df['개업년'] = pd.to_datetime(df['APVPERMYMD'], errors='coerce').dt.year
        df['폐업년'] = pd.to_datetime(df['DCBYMD'], errors='coerce').dt.year
        print("\n연도별 영업중 점포 수 (업태별):")
        utypes = df['UPTAENM'].value_counts().head(5).index.tolist()
        for yr in [2019, 2022, 2025]:
            active = df[(df['개업년'] <= yr) & (df['폐업년'].isna() | (df['폐업년'] > yr))]
            print(f"  {yr}년말 영업중 총 {len(active)}")
            for u in utypes:
                cnt = (active['UPTAENM'] == u).sum()
                print(f"    {u}: {cnt}")


if __name__ == '__main__':
    main()
