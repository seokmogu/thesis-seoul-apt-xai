#!/usr/bin/env python3
"""Recheck administrative-dong coverage from apartment coordinates.

This script intentionally avoids geopandas/shapely so the coverage audit can run
in the existing thesis environment. It assigns each apartment coordinate to the
Seoul administrative-dong GeoJSON by point-in-polygon, then compares that result
with the legacy 법정동->행정동 name column in the final modeling CSV.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def iter_rings(geometry):
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if gtype == "Polygon":
        for ring in coords:
            yield ring
    elif gtype == "MultiPolygon":
        for poly in coords:
            for ring in poly:
                yield ring


def ring_contains(lng: float, lat: float, ring) -> bool:
    inside = False
    j = len(ring) - 1
    for i, (xi, yi) in enumerate(ring):
        xj, yj = ring[j]
        crosses = ((yi > lat) != (yj > lat)) and (
            lng < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-15) + xi
        )
        if crosses:
            inside = not inside
        j = i
    return inside


def geom_contains(lng: float, lat: float, geometry) -> bool:
    rings = list(iter_rings(geometry))
    if not rings:
        return False
    if not ring_contains(lng, lat, rings[0]):
        return False
    return not any(ring_contains(lng, lat, hole) for hole in rings[1:])


def load_boundaries():
    obj = json.loads((DATA / "seoul_hjd_boundary.geojson").read_text())
    rows = []
    for feature in obj["features"]:
        prop = feature["properties"]
        rings = list(iter_rings(feature["geometry"]))
        flat = [pt for ring in rings for pt in ring]
        lngs = [p[0] for p in flat]
        lats = [p[1] for p in flat]
        adm_nm = prop["adm_nm"]
        rows.append(
            {
                "gu": prop.get("sggnm") or adm_nm.split()[1],
                "dong": adm_nm.split()[-1],
                "adm_nm": adm_nm,
                "geometry": feature["geometry"],
                "bbox": (min(lngs), min(lats), max(lngs), max(lats)),
            }
        )
    return rows


def assign_point(lng: float, lat: float, boundaries):
    for b in boundaries:
        min_lng, min_lat, max_lng, max_lat = b["bbox"]
        if not (min_lng <= lng <= max_lng and min_lat <= lat <= max_lat):
            continue
        if geom_contains(lng, lat, b["geometry"]):
            return b["gu"], b["dong"], b["adm_nm"]
    return None, None, None


def main():
    RESULTS.mkdir(exist_ok=True)
    boundaries = load_boundaries()

    coords = pd.read_csv(DATA / "apartment_coords.csv")
    coords[["구_spatial", "행정동_spatial", "adm_nm"]] = coords.apply(
        lambda r: pd.Series(assign_point(float(r["lng"]), float(r["lat"]), boundaries)),
        axis=1,
    )

    final = pd.read_csv(DATA / "apartment_final_v8.csv")
    apt_spatial = coords[
        ["gu", "bjd", "apt_name_raw", "구_spatial", "행정동_spatial", "adm_nm"]
    ].drop_duplicates()
    merged = final.merge(
        apt_spatial,
        left_on=["구", "법정동", "아파트명"],
        right_on=["gu", "bjd", "apt_name_raw"],
        how="left",
    )
    matched = merged.dropna(subset=["구_spatial", "행정동_spatial"])

    spatial = (
        matched.groupby(["구_spatial", "행정동_spatial"])
        .agg(거래건수=("아파트명", "size"), 단지수=("아파트명", "nunique"), 법정동수=("법정동", "nunique"))
        .reset_index()
        .rename(columns={"구_spatial": "구", "행정동_spatial": "행정동"})
        .sort_values(["구", "행정동"])
    )
    spatial.to_csv(RESULTS / "admin_dong_spatial_coverage_20260527.csv", index=False, encoding="utf-8-sig")

    current_vs_spatial = (
        matched[["구", "법정동", "아파트명", "행정동", "구_spatial", "행정동_spatial", "adm_nm"]]
        .drop_duplicates()
        .sort_values(["구", "법정동", "아파트명"])
    )
    current_vs_spatial.to_csv(
        RESULTS / "admin_dong_current_vs_spatial_20260527.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_pairs = pd.DataFrame(
        [{"구": b["gu"], "행정동": b["dong"], "adm_nm": b["adm_nm"]} for b in boundaries]
    )
    official_by_gu = all_pairs.groupby("구").size().rename("전체행정동")
    spatial_by_gu = spatial.groupby("구").size().rename("좌표기준_분석행정동")
    current_by_gu = final[["구", "행정동"]].drop_duplicates().groupby("구").size().rename("기존매핑_분석행정동")
    by_gu = (
        pd.concat([official_by_gu, spatial_by_gu, current_by_gu], axis=1)
        .fillna(0)
        .astype(int)
    )
    by_gu["좌표기준_제외"] = by_gu["전체행정동"] - by_gu["좌표기준_분석행정동"]
    by_gu.to_csv(RESULTS / "admin_dong_coverage_by_gu_20260527.csv", encoding="utf-8-sig")

    spatial_pairs = set(zip(spatial["구"], spatial["행정동"]))
    missing = [
        {"구": row["구"], "행정동": row["행정동"], "adm_nm": row["adm_nm"]}
        for _, row in all_pairs.iterrows()
        if (row["구"], row["행정동"]) not in spatial_pairs
    ]
    summary = {
        "official_admin_dongs": int(len(all_pairs)),
        "spatial_admin_dongs_in_sample": int(len(spatial_pairs)),
        "legacy_name_admin_dongs": int(final[["구", "행정동"]].drop_duplicates().shape[0]),
        "transactions_in_final": int(len(final)),
        "transactions_with_spatial_match": int(len(matched)),
        "apartments_with_spatial_match": int(
            matched[["구", "법정동", "아파트명"]].drop_duplicates().shape[0]
        ),
        "missing_official_admin_dongs": missing,
    }
    (RESULTS / "admin_dong_coverage_20260527.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
