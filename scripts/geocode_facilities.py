#!/usr/bin/env python3
"""
주소만 있는 시설(학교·학원) 지오코딩.

Kakao address search API 사용. 학교는 `address`(도로명) 우선, 학원은
`FA_RDNMA` 도로명주소 사용. 주소가 비면 `SITEWHLADDR`/`ORG_RDNDA` fallback.

출력:
  data/schools_coords.csv (학교 + lat/lng/FOND_YMD)
  data/academies_coords.csv (학원 + lat/lng + 시간필드)
"""
import os
import sys
import time
import argparse
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_api_keys

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
ADDR_URL = 'https://dapi.kakao.com/v2/local/search/address.json'
KEYWORD_URL = 'https://dapi.kakao.com/v2/local/search/keyword.json'


def geocode_address(session, headers, addr: str):
    if not isinstance(addr, str) or not addr.strip():
        return None
    r = session.get(ADDR_URL, headers=headers, params={'query': addr.strip(), 'size': 1}, timeout=10)
    if r.status_code == 429:
        time.sleep(1); r = session.get(ADDR_URL, headers=headers, params={'query': addr.strip(), 'size': 1}, timeout=10)
    if r.status_code != 200:
        return None
    docs = r.json().get('documents', [])
    if docs:
        d = docs[0]
        return float(d['y']), float(d['x'])
    return None


def geocode_keyword(session, headers, q: str):
    r = session.get(KEYWORD_URL, headers=headers, params={'query': q, 'size': 1}, timeout=10)
    if r.status_code != 200:
        return None
    docs = r.json().get('documents', [])
    if docs and '서울' in docs[0].get('address_name', ''):
        return float(docs[0]['y']), float(docs[0]['x'])
    return None


def run_schools(headers):
    src = os.path.join(DATA_DIR, 'schools_raw.csv')
    out = os.path.join(DATA_DIR, 'schools_coords.csv')
    df = pd.read_csv(src)
    print(f"학교 {len(df)}건 지오코딩 시작")
    session = requests.Session()
    lats, lngs, sources = [], [], []
    for i, r in df.iterrows():
        addr = r.get('address', '')
        hit = geocode_address(session, headers, addr)
        src_ = 'addr'
        if not hit:
            hit = geocode_keyword(session, headers, f"{r['school_name']} {r.get('gu','')}")
            src_ = 'keyword'
        if hit:
            lats.append(hit[0]); lngs.append(hit[1]); sources.append(src_)
        else:
            lats.append(None); lngs.append(None); sources.append('fail')
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(df)}] hits={sum(1 for v in lats if v)}")
    df['lat'] = lats; df['lng'] = lngs; df['geocode_source'] = sources
    df.to_csv(out, index=False, encoding='utf-8-sig')
    hit = df['lat'].notna().sum()
    print(f"학교 완료: {hit}/{len(df)} ({100*hit/len(df):.1f}%) → {out}")


def run_academies(headers):
    src = os.path.join(DATA_DIR, 'seoul_academies.csv')
    out = os.path.join(DATA_DIR, 'academies_coords.csv')
    df = pd.read_csv(src)
    print(f"학원 {len(df)}건 지오코딩 시작")
    session = requests.Session()
    lats, lngs, sources = [], [], []
    t0 = time.time()
    for i, r in df.iterrows():
        addr = str(r.get('FA_RDNMA', '') or '').strip()
        hit = geocode_address(session, headers, addr) if addr else None
        src_ = 'rdnma'
        if not hit:
            hit = geocode_keyword(session, headers, f"{r['ACA_NM']} {r.get('ADMST_ZONE_NM','')}")
            src_ = 'keyword'
        if hit:
            lats.append(hit[0]); lngs.append(hit[1]); sources.append(src_)
        else:
            lats.append(None); lngs.append(None); sources.append('fail')
        if (i + 1) % 1000 == 0:
            hit_cnt = sum(1 for v in lats if v)
            rate = (i + 1) / (time.time() - t0)
            eta = (len(df) - i - 1) / rate / 60
            print(f"  [{i+1}/{len(df)}] hits={hit_cnt} rate={rate:.0f}/s ETA={eta:.1f}min")
            # 중간 저장
            df.iloc[:i+1].assign(lat=lats[:i+1], lng=lngs[:i+1], geocode_source=sources[:i+1]).to_csv(out, index=False, encoding='utf-8-sig')
    df['lat'] = lats; df['lng'] = lngs; df['geocode_source'] = sources
    df.to_csv(out, index=False, encoding='utf-8-sig')
    hit = df['lat'].notna().sum()
    print(f"학원 완료: {hit}/{len(df)} ({100*hit/len(df):.1f}%) → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', choices=['schools', 'academies', 'both'], default='both')
    args = ap.parse_args()
    keys = load_api_keys()
    headers = {'Authorization': f"KakaoAK {keys['KAKAO_API_KEY']}"}
    if args.target in ('schools', 'both'):
        run_schools(headers)
    if args.target in ('academies', 'both'):
        run_academies(headers)


if __name__ == '__main__':
    main()
