# v8 모델 노후화 보조 실험 결과

## 실험 설계

- 데이터: `data/apartment_final_v8.csv`
- 종속변수: `log㎡당가격`
- 모형: XGBoost, 반복 하위분석용 설정(`n_estimators=400`, `max_depth=8`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`)
- 목적: 과거 연도 학습 모형을 이후 연도에 그대로 적용했을 때 미래연도 예측 안정성이 유지되는지 확인

## 미래 전체 테스트 요약

| 학습 기간 | 테스트 기간 | train n | test n | R² | MAPE | Median APE | 평균 편향 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2019 | 2020-2025 | 74,896 | 316,930 | 0.377 | 25.4% | 24.5% | -27.8% |
| 2019-2020 | 2021-2025 | 158,812 | 233,014 | 0.598 | 20.1% | 17.5% | -19.7% |
| 2019-2021 | 2022-2025 | 202,191 | 189,635 | 0.712 | 20.0% | 16.2% | -4.6% |

## 연도별 테스트 결과

| 학습 기간 | 테스트 연도 | test n | R² | MAPE | Median APE | 평균 편향 |
|---|---:|---:|---:|---:|---:|---:|
| 2019 | 2020 | 83,916 | 0.768 | 16.0% | 14.1% | -11.8% |
| 2019 | 2021 | 43,379 | 0.256 | 27.6% | 28.8% | -27.5% |
| 2019 | 2022 | 12,788 | 0.331 | 27.2% | 27.7% | -27.2% |
| 2019 | 2023 | 35,565 | 0.409 | 24.4% | 24.1% | -25.4% |
| 2019 | 2024 | 57,710 | 0.266 | 27.5% | 27.6% | -30.8% |
| 2019 | 2025 | 83,572 | -0.075 | 32.4% | 32.8% | -37.6% |
| 2019-2020 | 2021 | 43,379 | 0.759 | 15.9% | 14.0% | -12.0% |
| 2019-2020 | 2022 | 12,788 | 0.669 | 19.9% | 18.3% | -16.1% |
| 2019-2020 | 2023 | 35,565 | 0.738 | 16.3% | 13.5% | -12.0% |
| 2019-2020 | 2024 | 57,710 | 0.643 | 19.4% | 17.2% | -18.9% |
| 2019-2020 | 2025 | 83,572 | 0.413 | 24.3% | 22.7% | -26.7% |
| 2019-2021 | 2022 | 12,788 | 0.802 | 16.5% | 12.2% | -0.6% |
| 2019-2021 | 2023 | 35,565 | 0.740 | 19.7% | 15.5% | 6.8% |
| 2019-2021 | 2024 | 57,710 | 0.751 | 19.0% | 15.0% | -1.6% |
| 2019-2021 | 2025 | 83,572 | 0.652 | 21.5% | 18.2% | -11.2% |

## 1차 판정

- 이 파일은 논문 본문 반영 전 검토용 산출물이다.
- R²와 MAPE가 기존 연도별 동일연도 하위모형보다 크게 악화되면, 성능 과시가 아니라 모델 노후화와 AVM 재검증 필요성의 근거로 해석한다.
- 평균 편향이 큰 경우에는 시장 국면의 가격 수준 이동을 과거 모형이 따라가지 못한 것으로 볼 수 있으나, 변수 생성 오류나 결측 처리 차이가 없는지 먼저 확인해야 한다.

## 검증 메타데이터

```json
{
  "data_path": "data/apartment_final_v8.csv",
  "n_after_dropna": 391826,
  "years": [
    2019,
    2020,
    2021,
    2022,
    2023,
    2024,
    2025
  ],
  "target": "log㎡당가격",
  "features": [
    "층",
    "건물연령",
    "강남구분",
    "subway_nearest_m",
    "elem_school_nearest_m",
    "middle_school_nearest_m",
    "library_nearest_m",
    "park_nearest_m",
    "mart_nearest_m",
    "department_nearest_m",
    "academy_nearest_m",
    "hospital_general_nearest_m",
    "childcare_count_1000m",
    "cctv_count_500m",
    "park_within_1km",
    "department_within_1km",
    "기준금리",
    "소비자물가지수",
    "M2",
    "subway_count_1000m",
    "elem_school_count_1000m",
    "middle_school_count_1000m",
    "high_school_count_1000m",
    "mart_count_1000m",
    "department_count_2000m",
    "academy_count_1000m",
    "library_count_1000m",
    "hospital_count_1000m",
    "park_count_1000m",
    "large_store_count_500m",
    "park_log1p_count_2km",
    "department_log1p_count_2km",
    "library_log1p_count_2km",
    "CD금리"
  ],
  "scenarios": [
    {
      "scenario": "train_2019",
      "train_start": 2019,
      "train_end": 2019
    },
    {
      "scenario": "train_2019_2020",
      "train_start": 2019,
      "train_end": 2020
    },
    {
      "scenario": "train_2019_2021",
      "train_start": 2019,
      "train_end": 2021
    }
  ],
  "xgboost": {
    "n_estimators": 400,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "tree_method": "hist"
  }
}
```
