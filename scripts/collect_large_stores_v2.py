#!/usr/bin/env python3
"""
대규모점포(LOCALDATA_072405) 재수집 — 인허가일/폐업일/영업상태 시간필드 포함.

기존 `seoul_large_stores.csv`는 9개 컬럼만 저장되어 연도별 영업여부 복원 불가.
본 스크립트는 서울열린데이터광장 LOCALDATA_072405 전체 컬럼을 보존한다.

출력:
  data/seoul_large_stores_v2.csv (전체, UPTAENM 무관)
  data/seoul_department_stores_v2.csv (UPTAENM ∈ {백화점} 필터)

핵심 필드:
  APVPERMYMD : 인허가일자 (개업일)
  TRDSTATENM : 영업상태명 (영업/폐업 등)
  DCBYMD     : 폐업일자
  UPTAENM    : 업태명
  X, Y       : TM 좌표 (EPSG:5179)

TM좌표는 별도 스크립트에서 WGS84로 변환해 거리 계산에 사용한다.
"""
import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_api_keys, request_with_retry

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
SERVICE = 'LOCALDATA_072405'
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
        data = r.json().get(SERVICE, {})
        batch = data.get('row', [])
        if not batch:
            print(f"  {start}-{end} 빈응답 종료")
            break
        rows.extend(batch)
        print(f"  {start:>6}-{end:<6} 누적 {len(rows):>6,}/{total:,}")
        start = end + 1
        time.sleep(0.1)
    return rows


def main():
    keys = load_api_keys()
    rows = fetch_all(keys['SEOUL_API_KEY'])
    df = pd.DataFrame(rows)
    print(f"\n수집 완료: {len(df):,} 행, {len(df.columns)} 컬럼")

    # 날짜 공백 정제
    for col in ('APVPERMYMD', 'DCBYMD', 'CLGSTDT', 'CLGENDDT', 'ROPNYMD', 'APVCANCELYMD'):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({'': pd.NA})

    # 전체 저장
    out_all = os.path.join(DATA_DIR, 'seoul_large_stores_v2.csv')
    df.to_csv(out_all, index=False, encoding='utf-8-sig')
    print(f"전체 저장: {out_all}")

    # 업태별 분포
    if 'UPTAENM' in df.columns:
        print("\n업태별 상위 15:")
        print(df['UPTAENM'].value_counts().head(15).to_string())

        dept = df[df['UPTAENM'] == '백화점'].copy()
        out_dept = os.path.join(DATA_DIR, 'seoul_department_stores_v2.csv')
        dept.to_csv(out_dept, index=False, encoding='utf-8-sig')
        print(f"\n백화점 {len(dept)}행 → {out_dept}")

        # 연도별 영업중 추이 미리보기
        if len(dept):
            dept['개업년'] = pd.to_datetime(dept['APVPERMYMD'], errors='coerce').dt.year
            dept['폐업년'] = pd.to_datetime(dept['DCBYMD'], errors='coerce').dt.year
            print("\n백화점 개업년 분포:")
            print(dept['개업년'].value_counts().sort_index().tail(20).to_string())
            # 특정 연도 영업중 수 계산
            for yr in [2019, 2022, 2025]:
                active = ((dept['개업년'] <= yr) &
                          (dept['폐업년'].isna() | (dept['폐업년'] > yr))).sum()
                print(f"  {yr}년말 영업중: {active}")


if __name__ == '__main__':
    main()
