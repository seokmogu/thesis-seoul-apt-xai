#!/usr/bin/env python3
"""Kakao Place API로 서울 전역 병원(HP8) 수집.

category_group_code=HP8 로 25구 중심 반경 7km 카테고리 검색. 종별(종합병원/병원/의원/치과 등) 구분은 category_name으로.
"""
import os
import sys
import time
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_api_keys
from collect_kakao_shopping import GU_CENTERS, paged, CATEGORY_URL

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def main():
    keys = load_api_keys()
    headers = {'Authorization': f"KakaoAK {keys['KAKAO_API_KEY']}"}
    items = {}
    for gu, (x, y) in GU_CENTERS.items():
        docs = paged(CATEGORY_URL, headers,
                     {'category_group_code': 'HP8', 'x': x, 'y': y, 'radius': 7000})
        print(f"  {gu}: {len(docs)}")
        for d in docs:
            if '서울' not in (d.get('address_name') or ''):
                continue
            items[d['id']] = d
    df = pd.DataFrame(items.values())
    print(f"\n서울 병원 유니크: {len(df)}")

    # 종별 분류 (category_name 말단)
    if 'category_name' in df.columns:
        df['종별'] = df['category_name'].fillna('').apply(
            lambda s: s.split('>')[-1].strip() if '>' in s else '미분류')
        print("\n종별 분포:")
        print(df['종별'].value_counts().head(15).to_string())

    path = os.path.join(DATA_DIR, 'seoul_hospitals_kakao.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"\n저장: {path}")


if __name__ == '__main__':
    main()
