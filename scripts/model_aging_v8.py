#!/usr/bin/env python3
"""
v8 모델 노후화 보조 실험.

과거 연도에 학습한 XGBoost 모형을 이후 연도 거래에 그대로 적용하여,
시장 국면 변화에 따른 미래연도 예측 안정성을 점검한다.

출력:
  results/model_aging_v8.csv
  results/model_aging_v8.json
  results/model_aging_v8_summary.md
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT, "data", "apartment_final_v8.csv")
RESULTS = os.path.join(ROOT, "results")
OUT_CSV = os.path.join(RESULTS, "model_aging_v8.csv")
OUT_JSON = os.path.join(RESULTS, "model_aging_v8.json")
OUT_MD = os.path.join(RESULTS, "model_aging_v8_summary.md")

OLS_FEATURES = [
    "층", "건물연령", "강남구분",
    "subway_nearest_m", "elem_school_nearest_m", "middle_school_nearest_m",
    "library_nearest_m", "park_nearest_m", "mart_nearest_m",
    "department_nearest_m", "academy_nearest_m", "hospital_general_nearest_m",
    "childcare_count_1000m", "cctv_count_500m",
    "park_within_1km", "department_within_1km",
    "기준금리", "소비자물가지수", "M2",
]

TREE_FEATURES = OLS_FEATURES + [
    "subway_count_1000m", "elem_school_count_1000m", "middle_school_count_1000m",
    "high_school_count_1000m", "mart_count_1000m", "department_count_2000m",
    "academy_count_1000m", "library_count_1000m", "hospital_count_1000m",
    "park_count_1000m", "large_store_count_500m",
    "park_log1p_count_2km", "department_log1p_count_2km", "library_log1p_count_2km",
    "CD금리",
]

SCENARIOS = [
    {"scenario": "train_2019", "train_start": 2019, "train_end": 2019},
    {"scenario": "train_2019_2020", "train_start": 2019, "train_end": 2020},
    {"scenario": "train_2019_2021", "train_start": 2019, "train_end": 2021},
]


def metrics(y_true_log, y_pred_log):
    y_true = np.exp(y_true_log)
    y_pred = np.exp(y_pred_log)
    ape = np.abs(y_true - y_pred) / y_true
    return {
        "r2": float(r2_score(y_true_log, y_pred_log)),
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "mae_log": float(mean_absolute_error(y_true_log, y_pred_log)),
        "mean_actual_price_per_m2": float(np.mean(y_true)),
        "mean_pred_price_per_m2": float(np.mean(y_pred)),
        "mean_pct_bias": float((np.mean(y_pred) - np.mean(y_true)) / np.mean(y_true)),
        "n": int(len(y_true_log)),
    }


def fit_xgb(train_df):
    model = XGBRegressor(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
    )
    model.fit(train_df[TREE_FEATURES], train_df["log㎡당가격"].values)
    return model


def format_pct(value):
    return f"{100 * value:.1f}%"


def write_summary(rows, metadata):
    df = pd.DataFrame(rows)
    yearly = df[df["test_scope"] == "year"].copy()
    future_all = df[df["test_scope"] == "future_all"].copy()

    lines = [
        "# v8 모델 노후화 보조 실험 결과",
        "",
        "## 실험 설계",
        "",
        "- 데이터: `data/apartment_final_v8.csv`",
        "- 종속변수: `log㎡당가격`",
        "- 모형: XGBoost, 반복 하위분석용 설정(`n_estimators=400`, `max_depth=8`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`)",
        "- 목적: 과거 연도 학습 모형을 이후 연도에 그대로 적용했을 때 미래연도 예측 안정성이 유지되는지 확인",
        "",
        "## 미래 전체 테스트 요약",
        "",
        "| 학습 기간 | 테스트 기간 | train n | test n | R² | MAPE | Median APE | 평균 편향 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in future_all.iterrows():
        lines.append(
            f"| {row['train_label']} | {row['test_label']} | "
            f"{int(row['train_n']):,} | {int(row['test_n']):,} | "
            f"{row['r2']:.3f} | {format_pct(row['mape'])} | "
            f"{format_pct(row['median_ape'])} | {format_pct(row['mean_pct_bias'])} |"
        )

    lines.extend([
        "",
        "## 연도별 테스트 결과",
        "",
        "| 학습 기간 | 테스트 연도 | test n | R² | MAPE | Median APE | 평균 편향 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])

    for _, row in yearly.iterrows():
        lines.append(
            f"| {row['train_label']} | {int(row['test_year'])} | "
            f"{int(row['test_n']):,} | {row['r2']:.3f} | "
            f"{format_pct(row['mape'])} | {format_pct(row['median_ape'])} | "
            f"{format_pct(row['mean_pct_bias'])} |"
        )

    lines.extend([
        "",
        "## 1차 판정",
        "",
        "- 이 파일은 논문 본문 반영 전 검토용 산출물이다.",
        "- R²와 MAPE가 기존 연도별 동일연도 하위모형보다 크게 악화되면, 성능 과시가 아니라 모델 노후화와 AVM 재검증 필요성의 근거로 해석한다.",
        "- 평균 편향이 큰 경우에는 시장 국면의 가격 수준 이동을 과거 모형이 따라가지 못한 것으로 볼 수 있으나, 변수 생성 오류나 결측 처리 차이가 없는지 먼저 확인해야 한다.",
        "",
        "## 검증 메타데이터",
        "",
        "```json",
        json.dumps(metadata, ensure_ascii=False, indent=2),
        "```",
        "",
    ])

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    print("=== v8 모델 노후화 보조 실험 시작 ===")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    before = len(df)
    df = df.dropna(subset=TREE_FEATURES + ["log㎡당가격", "거래년도"])
    df["거래년도"] = df["거래년도"].astype(int)
    years = sorted(df["거래년도"].unique())
    print(f"데이터: {before:,} -> 결측 제거 후 {len(df):,}; years={years}")

    rows = []
    for scenario in SCENARIOS:
        train_start = scenario["train_start"]
        train_end = scenario["train_end"]
        train_df = df[(df["거래년도"] >= train_start) & (df["거래년도"] <= train_end)]
        test_years = [y for y in years if y > train_end]
        if train_df.empty or not test_years:
            continue

        train_label = str(train_start) if train_start == train_end else f"{train_start}-{train_end}"
        print(f"\n--- {train_label} 학습: n={len(train_df):,} ---")
        model = fit_xgb(train_df)

        future_df = df[df["거래년도"].isin(test_years)]
        pred = model.predict(future_df[TREE_FEATURES])
        m = metrics(future_df["log㎡당가격"].values, pred)
        rows.append({
            **scenario,
            "train_label": train_label,
            "train_n": len(train_df),
            "test_scope": "future_all",
            "test_year": None,
            "test_label": f"{min(test_years)}-{max(test_years)}",
            "test_n": len(future_df),
            **m,
        })
        print(
            f"  future all {min(test_years)}-{max(test_years)}: "
            f"R²={m['r2']:.4f}, MAPE={m['mape']:.4f}, bias={m['mean_pct_bias']:.4f}"
        )

        for test_year in test_years:
            test_df = df[df["거래년도"] == test_year]
            pred = model.predict(test_df[TREE_FEATURES])
            m = metrics(test_df["log㎡당가격"].values, pred)
            rows.append({
                **scenario,
                "train_label": train_label,
                "train_n": len(train_df),
                "test_scope": "year",
                "test_year": int(test_year),
                "test_label": str(test_year),
                "test_n": len(test_df),
                **m,
            })
            print(
                f"  {test_year}: n={len(test_df):,}, R²={m['r2']:.4f}, "
                f"MAPE={m['mape']:.4f}, bias={m['mean_pct_bias']:.4f}"
            )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)

    metadata = {
        "data_path": os.path.relpath(DATA_PATH, ROOT),
        "n_after_dropna": int(len(df)),
        "years": [int(y) for y in years],
        "target": "log㎡당가격",
        "features": TREE_FEATURES,
        "scenarios": SCENARIOS,
        "xgboost": {
            "n_estimators": 400,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "tree_method": "hist",
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "rows": rows}, f, ensure_ascii=False, indent=2)
    write_summary(rows, metadata)

    print(f"\n저장: {OUT_CSV}")
    print(f"저장: {OUT_JSON}")
    print(f"저장: {OUT_MD}")


if __name__ == "__main__":
    main()
