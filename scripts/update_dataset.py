#!/usr/bin/env python3
"""기존 최종 데이터에 추가 변수(공원수, 도서관수, 학원수, 어린이집수) merge"""
import os, sys
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def main():
    print("=== 데이터셋 업데이트 ===")

    # 기존 데이터 로드
    final = pd.read_csv(os.path.join(DATA_DIR, 'apartment_final_v5_dong.csv'))
    print(f"기존 데이터: {len(final)}건, {len(final.columns)}개 변수")
    print(f"기존 변수: {list(final.columns)}")

    # 추가 변수 로드
    additional = pd.read_csv(os.path.join(DATA_DIR, 'additional_vars_per_dong.csv'))
    print(f"\n추가 변수: {list(additional.columns)}")

    # Merge
    merged = final.merge(additional, on=['구', '행정동'], how='left')

    # 결측 0으로
    for col in ['공원수', '도서관수', '학원수', '어린이집수']:
        merged[col] = merged[col].fillna(0).astype(int)

    print(f"\n업데이트된 데이터: {len(merged)}건, {len(merged.columns)}개 변수")
    print(f"변수 목록: {list(merged.columns)}")

    # 추가 변수 기술통계
    print(f"\n추가 변수 기술통계:")
    print(merged[['공원수', '도서관수', '학원수', '어린이집수']].describe().to_string())

    # 저장
    out_path = os.path.join(DATA_DIR, 'apartment_final_v6_dong.csv')
    merged.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 저장: {out_path}")

    return merged


if __name__ == '__main__':
    main()
