#!/usr/bin/env python3
"""
seoul_large_stores_v2.csv의 TM좌표(EPSG:5179)를 WGS84로 변환.

EPSG:5179 = Korea 2000 / Unified CS (국가기본도 체계)
"""
import os
import pandas as pd
from pyproj import Transformer

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
SRC = os.path.join(DATA_DIR, 'seoul_large_stores_v2.csv')
OUT = os.path.join(DATA_DIR, 'seoul_large_stores_v2.csv')  # 제자리 업데이트

# 서울시 LOCALDATA는 EPSG:2097 (Bessel 1841 + TM Central 127°, 중부원점)
tr = Transformer.from_crs('EPSG:2097', 'EPSG:4326', always_xy=True)

df = pd.read_csv(SRC, dtype={'X': str, 'Y': str})
print(f"원본: {len(df):,} 행")

df['X'] = df['X'].astype(str).str.strip()
df['Y'] = df['Y'].astype(str).str.strip()
x = pd.to_numeric(df['X'], errors='coerce')
y = pd.to_numeric(df['Y'], errors='coerce')
mask = x.notna() & y.notna()
print(f"유효 TM좌표: {mask.sum():,} ({100*mask.mean():.1f}%)")

lng, lat = tr.transform(x[mask].values, y[mask].values)
df['lng'] = pd.NA
df['lat'] = pd.NA
df.loc[mask, 'lng'] = lng
df.loc[mask, 'lat'] = lat

# 서울 범위 sanity check (위도 37.4~37.7, 경도 126.7~127.2)
valid = (df['lat'].between(37.3, 37.75) & df['lng'].between(126.7, 127.3))
print(f"서울 bbox 내: {valid.sum():,} ({100*valid.mean():.1f}%)")

df.to_csv(OUT, index=False, encoding='utf-8-sig')
print(f"저장: {OUT}")
i0 = mask.idxmax()
print(f"샘플: X={x.loc[i0]:.1f} Y={y.loc[i0]:.1f} → lat={df.loc[i0,'lat']:.6f} lng={df.loc[i0,'lng']:.6f}")
