#!/usr/bin/env python3
"""
아파트 단지 지오코딩 (Kakao Local API).

입력: data/apartment_final_v6_dong.csv (구, 법정동, 아파트명)
출력: data/apartment_coords.csv (gu, bjd, apt_name_raw, apt_name_clean, matched_name, matched_road, lat, lng, match_level)
     data/apartment_coords_failed.csv (미매칭 단지)

매칭 전략 (점수 높은 순 채택):
  3 = 구+법정동 모두 일치
  2 = 구만 일치
  1 = 같은 시 (서울) 내 어떤 결과라도 있음
  0 = 결과 없음 (실패)
"""
import os
import re
import sys
import time
import csv
import json
import argparse
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_api_keys

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
SRC_CSV = os.path.join(DATA_DIR, 'apartment_final_v6_dong.csv')
OUT_CSV = os.path.join(DATA_DIR, 'apartment_coords.csv')
FAIL_CSV = os.path.join(DATA_DIR, 'apartment_coords_failed.csv')

KAKAO_KEYWORD = 'https://dapi.kakao.com/v2/local/search/keyword.json'

PAREN_RE = re.compile(r'\([^)]*\)')
TRAIL_DONG_RE = re.compile(r'\s*\d+동(~\d+동)?\s*$')


def clean_apt_name(name: str) -> str:
    if not isinstance(name, str):
        return ''
    s = PAREN_RE.sub('', name)
    s = TRAIL_DONG_RE.sub('', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def pick_best(docs, gu: str, bjd: str):
    best = None
    best_score = -1
    for d in docs:
        addr = d.get('address_name', '') or ''
        road = d.get('road_address_name', '') or ''
        text = addr + ' ' + road
        if '서울' not in text:
            continue
        score = 1
        if gu and gu in text:
            score = 2
            if bjd and bjd in text:
                score = 3
        if score > best_score:
            best_score = score
            best = d
    return best, best_score


def geocode_one(session: requests.Session, headers: dict, gu: str, bjd: str, name: str):
    name_clean = clean_apt_name(name)
    queries = [
        f"{name_clean} {gu} {bjd}",
        f"{name_clean} {gu}",
        f"{name_clean} {bjd}",
    ]
    for q in queries:
        try:
            r = session.get(KAKAO_KEYWORD, headers=headers, params={'query': q, 'size': 15}, timeout=10)
            if r.status_code == 429:
                time.sleep(1.0)
                r = session.get(KAKAO_KEYWORD, headers=headers, params={'query': q, 'size': 15}, timeout=10)
            if r.status_code != 200:
                continue
            docs = r.json().get('documents', [])
            best, score = pick_best(docs, gu, bjd)
            if best and score >= 2:
                return best, score, name_clean, q
        except Exception as e:
            print(f"    err query={q!r}: {e}")
            continue
    # last resort: accept any Seoul result from the first query
    try:
        r = session.get(KAKAO_KEYWORD, headers=headers, params={'query': queries[0], 'size': 15}, timeout=10)
        if r.status_code == 200:
            docs = r.json().get('documents', [])
            best, score = pick_best(docs, gu, bjd)
            if best:
                return best, score, name_clean, queries[0]
    except Exception:
        pass
    return None, 0, name_clean, queries[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='테스트용 단지 수 제한 (0=전체)')
    ap.add_argument('--resume', action='store_true', help='기존 출력에 없는 단지만 다시 시도')
    ap.add_argument('--sleep', type=float, default=0.05, help='요청 간 지연 (초)')
    args = ap.parse_args()

    keys = load_api_keys()
    api_key = keys['KAKAO_API_KEY']
    headers = {'Authorization': f'KakaoAK {api_key}'}
    session = requests.Session()

    df = pd.read_csv(SRC_CSV, usecols=['구', '법정동', '아파트명'])
    uniq = df.drop_duplicates(subset=['구', '법정동', '아파트명']).reset_index(drop=True)
    print(f"유니크 단지: {len(uniq)}")

    done = set()
    rows_existing = []
    if args.resume and os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV)
        for _, r in prev.iterrows():
            done.add((r['gu'], r['bjd'], r['apt_name_raw']))
        rows_existing = prev.to_dict('records')
        print(f"이미 처리된 단지: {len(done)} (resume)")

    if args.limit:
        uniq = uniq.head(args.limit)

    out_rows = list(rows_existing)
    fail_rows = []
    t0 = time.time()
    for i, rec in uniq.iterrows():
        gu, bjd, name = rec['구'], rec['법정동'], rec['아파트명']
        key = (gu, bjd, name)
        if key in done:
            continue
        best, score, name_clean, q = geocode_one(session, headers, gu, bjd, name)
        if best is None:
            fail_rows.append({'gu': gu, 'bjd': bjd, 'apt_name_raw': name,
                              'apt_name_clean': name_clean, 'query': q})
        else:
            out_rows.append({
                'gu': gu, 'bjd': bjd,
                'apt_name_raw': name,
                'apt_name_clean': name_clean,
                'matched_name': best.get('place_name'),
                'matched_addr': best.get('address_name'),
                'matched_road': best.get('road_address_name'),
                'lat': float(best['y']),
                'lng': float(best['x']),
                'match_level': score,
                'query': q,
            })
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(uniq)}] hit={len(out_rows)-len(rows_existing)} miss={len(fail_rows)} "
                  f"elapsed={elapsed:.0f}s rate={((i+1)/elapsed):.1f}/s")
            pd.DataFrame(out_rows).to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
            pd.DataFrame(fail_rows).to_csv(FAIL_CSV, index=False, encoding='utf-8-sig')
        time.sleep(args.sleep)

    pd.DataFrame(out_rows).to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    pd.DataFrame(fail_rows).to_csv(FAIL_CSV, index=False, encoding='utf-8-sig')
    print(f"\n완료: 성공 {len(out_rows)}, 실패 {len(fail_rows)}")
    print(f"  → {OUT_CSV}")
    print(f"  → {FAIL_CSV}")


if __name__ == '__main__':
    main()
