#!/usr/bin/env python3
"""
Kakao Place API로 서울 전역 대형마트/백화점 수집.

카테고리 기반:
  MT1 = 대형마트 (이마트, 홈플러스, 롯데마트, 하나로마트 등)
  백화점 = 카테고리 코드 부재 → 키워드 검색 후 category_name에 '> 백화점 >' 포함된 것만 채택

구별로 중심좌표+반경 5km 기준 카테고리 검색을 수행(Kakao는 1쿼리당 최대 45페이지=675건 제한)
중복은 Kakao place `id`로 제거.
"""
import os
import sys
import time
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_api_keys

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# 서울 25개 구 중심좌표 (대략값; 실제 검색 범위는 반경 7km로 여유롭게)
GU_CENTERS = {
    '종로구': (126.9793, 37.5730), '중구': (126.9974, 37.5641),
    '용산구': (126.9902, 37.5326), '성동구': (127.0369, 37.5633),
    '광진구': (127.0824, 37.5385), '동대문구': (127.0397, 37.5744),
    '중랑구': (127.0929, 37.6063), '성북구': (127.0167, 37.5894),
    '강북구': (127.0257, 37.6396), '도봉구': (127.0472, 37.6688),
    '노원구': (127.0568, 37.6543), '은평구': (126.9290, 37.6027),
    '서대문구': (126.9368, 37.5791), '마포구': (126.9085, 37.5663),
    '양천구': (126.8667, 37.5170), '강서구': (126.8495, 37.5509),
    '구로구': (126.8874, 37.4954), '금천구': (126.9015, 37.4565),
    '영등포구': (126.8962, 37.5264), '동작구': (126.9395, 37.5124),
    '관악구': (126.9514, 37.4784), '서초구': (127.0326, 37.4836),
    '강남구': (127.0473, 37.5172), '송파구': (127.1058, 37.5145),
    '강동구': (127.1238, 37.5301),
}

KEYWORD_URL = 'https://dapi.kakao.com/v2/local/search/keyword.json'
CATEGORY_URL = 'https://dapi.kakao.com/v2/local/search/category.json'


def paged(url, headers, params):
    """Kakao 페이지네이션 (최대 45페이지)."""
    out = []
    for page in range(1, 46):
        p = dict(params, page=page, size=15)
        r = requests.get(url, headers=headers, params=p, timeout=10)
        if r.status_code != 200:
            break
        d = r.json()
        docs = d.get('documents', [])
        out.extend(docs)
        if d.get('meta', {}).get('is_end') or not docs:
            break
        time.sleep(0.05)
    return out


def collect_category(headers, code: str) -> pd.DataFrame:
    """카테고리 코드 기준 서울 전 구 수집."""
    items = {}
    for gu, (x, y) in GU_CENTERS.items():
        docs = paged(CATEGORY_URL, headers,
                     {'category_group_code': code, 'x': x, 'y': y, 'radius': 7000})
        print(f"  [{code}] {gu}: {len(docs)}")
        for d in docs:
            if '서울' not in d.get('address_name', ''):
                continue
            items[d['id']] = d
    return pd.DataFrame(items.values())


def collect_department_stores(headers) -> pd.DataFrame:
    """백화점은 키워드 검색 + category_name 필터."""
    items = {}
    brands = ['백화점', '롯데백화점', '신세계백화점', '현대백화점', '갤러리아', 'AK플라자',
              '현대시티아울렛', '타임스퀘어', '롯데프리미엄아울렛', '현대프리미엄아울렛']
    # 서울 전역 bbox
    rect = '126.7643,37.4283,127.1843,37.7017'
    for q in brands:
        docs = paged(KEYWORD_URL, headers, {'query': q, 'rect': rect})
        hit = 0
        for d in docs:
            cat = d.get('category_name', '') or ''
            if '> 백화점 >' not in cat and '> 아울렛' not in cat:
                continue
            if '서울' not in d.get('address_name', ''):
                continue
            items[d['id']] = d
            hit += 1
        print(f"  키워드 '{q}': {len(docs)} 중 {hit}건 채택, 누적 유니크 {len(items)}")
    return pd.DataFrame(items.values())


def main():
    keys = load_api_keys()
    headers = {'Authorization': f"KakaoAK {keys['KAKAO_API_KEY']}"}

    print("=== 대형마트(MT1) ===")
    mart = collect_category(headers, 'MT1')
    path = os.path.join(DATA_DIR, 'seoul_marts_kakao.csv')
    mart.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"대형마트 저장: {len(mart)} rows → {path}")

    print("\n=== 백화점 ===")
    dept = collect_department_stores(headers)
    path = os.path.join(DATA_DIR, 'seoul_department_stores_kakao.csv')
    dept.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"백화점 저장: {len(dept)} rows → {path}")

    if len(dept):
        print("\n백화점 샘플:")
        for _, r in dept.head(15).iterrows():
            print(f"  {r.get('place_name')} | {r.get('category_name','')[:40]} | {r.get('address_name','')}")

    # 브랜드별 요약
    if 'category_name' in dept.columns:
        print("\n백화점 브랜드 분포:")
        dept['브랜드'] = dept['category_name'].fillna('').apply(
            lambda s: s.split('>')[-1].strip() if '>' in s else '미분류')
        print(dept['브랜드'].value_counts().to_string())


if __name__ == '__main__':
    main()
