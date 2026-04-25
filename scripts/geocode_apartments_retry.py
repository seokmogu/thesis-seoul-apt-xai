#!/usr/bin/env python3
"""
지오코딩 실패 602단지 2단계 재시도.

Stage 1: 쿼리 변형 재시도
  - 특수문자/로마숫자(Ⅰ/Ⅱ) 제거
  - 영문 어미 제거 (LEEPS, PARK 등)
  - "빌라/타운/힐/팰리스" 등 접미사 변형
  - address 검색으로 폴백

Stage 2: 법정동 중심좌표 fallback
  - 성공 단지들의 법정동 평균 좌표를 bjd 센터로 활용
  - 실패 단지에 해당 bjd 중심좌표 부여 (match_level=0, fallback=True)

최종 apartment_coords.csv 에 합쳐 저장.
"""
import os
import re
import sys
import time
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_api_keys

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
KEYWORD = 'https://dapi.kakao.com/v2/local/search/keyword.json'
ADDR = 'https://dapi.kakao.com/v2/local/search/address.json'

ROMAN_MAP = str.maketrans({'Ⅰ': '1', 'Ⅱ': '2', 'Ⅲ': '3', 'Ⅳ': '4', 'Ⅴ': '5',
                            'ⅰ': '1', 'ⅱ': '2', 'ⅲ': '3'})
SUFFIX_RE = re.compile(r'(빌라|타운|힐|팰리스|아파트|맨션|오피스텔|하우스)\b.*$')
ENG_RE = re.compile(r'[A-Za-z]+')


def variations(name: str):
    """단지명 쿼리 변형 후보 생성."""
    base = (name or '').strip()
    if not base:
        return []
    v = [base]
    # 로마숫자 → 아라비아
    t = base.translate(ROMAN_MAP)
    if t != base: v.append(t)
    # 괄호 제거
    t = re.sub(r'\s*\([^)]*\)', '', base).strip()
    if t and t != base: v.append(t)
    # 영문 제거
    t = ENG_RE.sub('', base).strip()
    if t and t != base and len(t) >= 2: v.append(t)
    # 접미어 제거
    t = SUFFIX_RE.sub('', base).strip()
    if t and t != base and len(t) >= 2: v.append(t)
    # 공백 제거
    t = re.sub(r'\s+', '', base)
    if t != base: v.append(t)
    seen, out = set(), []
    for x in v:
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out


def pick_best(docs, gu, bjd):
    best, score = None, -1
    for d in docs:
        t = (d.get('address_name', '') or '') + ' ' + (d.get('road_address_name', '') or '')
        if '서울' not in t:
            continue
        s = 1
        if gu and gu in t:
            s = 2
            if bjd and bjd in t:
                s = 3
        if s > score:
            score = s; best = d
    return best, score


def geocode_retry(session, headers, gu, bjd, name):
    for v in variations(name):
        for q in [f"{v} {gu} {bjd}", f"{v} {bjd}", f"{v} {gu}"]:
            try:
                r = session.get(KEYWORD, headers=headers, params={'query': q, 'size': 15}, timeout=10)
                if r.status_code != 200:
                    continue
                docs = r.json().get('documents', [])
                best, score = pick_best(docs, gu, bjd)
                if best and score >= 2:
                    return best, score, q, v
            except Exception:
                continue
    # 마지막: 주소 직접 검색 (동 수준)
    try:
        r = session.get(ADDR, headers=headers, params={'query': f'서울 {gu} {bjd}', 'size': 1}, timeout=10)
        if r.status_code == 200:
            docs = r.json().get('documents', [])
            if docs:
                d = docs[0]
                return {'place_name': None, 'address_name': d['address_name'],
                        'road_address_name': None, 'x': d['x'], 'y': d['y']}, 1, f'서울 {gu} {bjd}', 'addr'
    except Exception:
        pass
    return None, 0, None, None


def main():
    keys = load_api_keys()
    headers = {'Authorization': f"KakaoAK {keys['KAKAO_API_KEY']}"}
    session = requests.Session()

    ok = pd.read_csv(os.path.join(DATA_DIR, 'apartment_coords.csv'))
    failed = pd.read_csv(os.path.join(DATA_DIR, 'apartment_coords_failed.csv'))
    print(f"성공 {len(ok)}, 실패 {len(failed)}")

    # 법정동 중심좌표 (성공 단지 기반)
    bjd_ctr = ok.groupby(['gu', 'bjd']).agg(lat=('lat', 'mean'), lng=('lng', 'mean')).reset_index()
    print(f"법정동 중심좌표 확보: {len(bjd_ctr)} bjd")

    new_rows = []
    still_failed = []
    t0 = time.time()
    for i, rec in failed.iterrows():
        gu, bjd, name = rec['gu'], rec['bjd'], rec['apt_name_raw']
        best, score, q, variant = geocode_retry(session, headers, gu, bjd, name)
        if best and score >= 1:
            new_rows.append({
                'gu': gu, 'bjd': bjd, 'apt_name_raw': name,
                'apt_name_clean': variant or name,
                'matched_name': best.get('place_name'),
                'matched_addr': best.get('address_name'),
                'matched_road': best.get('road_address_name'),
                'lat': float(best['y']), 'lng': float(best['x']),
                'match_level': score, 'query': q,
            })
        else:
            still_failed.append({'gu': gu, 'bjd': bjd, 'apt_name_raw': name})
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(failed)}] hit={len(new_rows)} miss={len(still_failed)}")

    print(f"\n변형 재시도: 추가 {len(new_rows)}건, 남은 실패 {len(still_failed)}건")

    # 남은 실패는 bjd 중심좌표 fallback
    fallback_rows = []
    for r in still_failed:
        ctr = bjd_ctr[(bjd_ctr['gu'] == r['gu']) & (bjd_ctr['bjd'] == r['bjd'])]
        if len(ctr):
            fallback_rows.append({
                'gu': r['gu'], 'bjd': r['bjd'], 'apt_name_raw': r['apt_name_raw'],
                'apt_name_clean': r['apt_name_raw'],
                'matched_name': None, 'matched_addr': f"{r['gu']} {r['bjd']} 중심(fallback)",
                'matched_road': None,
                'lat': float(ctr['lat'].iloc[0]), 'lng': float(ctr['lng'].iloc[0]),
                'match_level': 0, 'query': 'bjd_centroid_fallback',
            })
    print(f"bjd 중심좌표 fallback: {len(fallback_rows)}건")

    # 합치기
    extra = pd.DataFrame(new_rows + fallback_rows)
    full = pd.concat([ok, extra], ignore_index=True)
    out = os.path.join(DATA_DIR, 'apartment_coords.csv')
    full.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"\n최종 apartment_coords.csv: {len(full)}건 ({len(ok)} + {len(extra)})")
    print(f"match_level 분포: {full['match_level'].value_counts().sort_index().to_dict()}")

    # 잔여 완전 실패 기록
    really_failed = [r for r in still_failed
                     if not any((f['gu']==r['gu'] and f['bjd']==r['bjd'] and f['apt_name_raw']==r['apt_name_raw'])
                                for f in fallback_rows)]
    pd.DataFrame(really_failed).to_csv(os.path.join(DATA_DIR, 'apartment_coords_failed.csv'),
                                        index=False, encoding='utf-8-sig')
    print(f"최종 실패(fallback도 불가): {len(really_failed)}")


if __name__ == '__main__':
    main()
