# 연구설계서 (Research Proposal)

## 논문 제목 (안)

**국문:** XGBoost와 SHAP을 활용한 서울시 아파트 매매가격 결정요인 분석
**영문:** Analysis of Determinants of Apartment Sale Prices in Seoul Using XGBoost and SHAP

### 부제 대안
- "설명가능한 인공지능(XAI) 접근을 중심으로"
- "머신러닝 기반 예측 및 해석을 중심으로"

---

## 1. 연구 배경 및 필요성

### 1.1 연구 배경
- 서울시 아파트는 국민 자산의 핵심 (가구 평균 자산 중 부동산 73%)
- 가격 변동이 경제·사회적 파급력이 크나, 결정요인이 복잡하고 비선형적
- 최근 머신러닝(ML) 기반 가격 예측 연구 활발 (LSTM, XGBoost 등)
- **그러나 기존 연구는 "예측 정확도"에만 집중 → "왜 이 가격인가?"를 설명하지 못함**

### 1.2 연구의 필요성
- 정책 수립을 위해서는 예측뿐 아니라 **결정요인의 영향력 해석**이 필수
- 해외에서는 XAI(설명가능 AI)를 활용한 부동산 연구가 활발 (Neves 2024, Dou 2023, Kee 2025)
- 국내에서는 XAI를 부동산에 적용한 연구가 **사실상 전무**
- 한양대 부동산융합대학원 조민지(2023)의 LSTM/GRU 연구를 확장하여 **해석력을 보완**

### 1.3 연구 목적
1. 서울시 아파트 매매가격에 대한 머신러닝 예측 모형 구축
2. SHAP(SHapley Additive exPlanations)을 활용하여 가격 결정요인의 영향력 분석
3. 지역별(강남/비강남 등) 결정요인 차이 비교
4. 부동산 정책 수립을 위한 시사점 도출

---

## 2. 이론적 배경 및 선행연구

### 2.1 부동산 가격 결정요인 이론
- **헤도닉 가격모형 (Hedonic Price Model):** Lancaster(1966), Rosen(1974)
  - 부동산 가격 = 내재적 특성들의 가치 합
  - 전통적으로 다중회귀분석 사용
- **한계:** 비선형 관계, 변수 간 상호작용 포착 어려움

### 2.2 머신러닝 기반 가격 예측 선행연구

| 연구 | 방법론 | 대상 | 핵심 결과 |
|---|---|---|---|
| 조민지(2023) 한양대 | VECM, LSTM, GRU | 서울 아파트 | 딥러닝 > 시계열 예측력 |
| 이학만(2025) | LSTM | 부산 아파트 | 정확도 0.98 |
| 진수정(2024) 서울대 | SRGCNN | 서울 아파트 | 공간모형 + 딥러닝 |
| 김선현(2022) | 다중회귀 | 대구 아파트 | 6개 결정요인 도출 |
| Jin & Xu(2025) | ML 앙상블 | 중국 주택 | 가격지수 예측 (68회 인용) |
| Choy & Ho(2023) | RF 등 | 서베이 | ML 부동산 연구 리뷰 (74회 인용) |

### 2.3 설명가능한 AI (XAI) 선행연구

| 연구 | 방법론 | 대상 | 핵심 결과 |
|---|---|---|---|
| Neves et al.(2024) | XAI + SHAP | 리스본 | 오픈데이터 + 스마트시티 (48회 인용) |
| Dou et al.(2023) | XAI + SHAP | 상하이 57,842건 | 근린환경 영향 분석 (51회 인용) |
| Acharya et al.(2024) | XAI + Fairness | 일반 | 공정성 결합 (62회 인용) |
| Tchuente(2024) | SHAP + PDP | 프랑스 | 자동감정평가 AVM (12회 인용) |
| Kee & Ho(2025) | XGBoost + SHAP | 홍콩 | beeswarm plot 활용 |
| Krämer et al.(2023) | ALE plots | 독일 | 주거/상업 비교 (30회 인용) |

### 2.4 연구 차별성 (Research Gap)

| 기존 연구 한계 | 본 연구의 차별점 |
|---|---|
| 국내 ML 연구는 예측 정확도만 비교 | SHAP으로 결정요인 **해석** 제공 |
| 해외 XAI 연구는 리스본/상하이/런던 등 | **서울** 아파트 시장에 최초 적용 |
| 전통 헤도닉 모형은 선형 가정 | XGBoost로 비선형 관계 포착 |
| 단일 지역 분석 위주 | 지역별(강남/비강남) 비교 분석 |

---

## 3. 연구 방법

### 3.1 연구 흐름도

```
데이터 수집 (공공API)
    ↓
데이터 전처리 (결측치, 이상치, 변수 생성)
    ↓
탐색적 데이터 분석 (EDA, 기술통계)
    ↓
모형 학습 (Linear Reg → Random Forest → XGBoost)
    ↓
모형 성능 비교 (RMSE, MAE, R²)
    ↓
SHAP 분석 (Global + Local Explanation)
    ↓
지역별 비교 분석
    ↓
결론 및 정책 시사점
```

### 3.2 데이터 수집

#### 종속변수
- **서울시 아파트 실거래가** (국토교통부 실거래가 공개시스템)
- 기간: 2020.01 ~ 2025.12 (약 6년)
- 범위: 서울시 25개 자치구

#### 독립변수 (4개 카테고리)

**① 물리적 특성 (Property Features)**
| 변수 | 출처 |
|---|---|
| 전용면적 (㎡) | 국토부 실거래가 |
| 층수 | 국토부 실거래가 |
| 건축연도 (경과연수) | 국토부 실거래가 |
| 총 세대수 | 공동주택 관리정보시스템 |

**② 입지 특성 (Location Features)**
| 변수 | 출처 |
|---|---|
| 지하철역 거리 (m) | 서울 열린데이터광장 |
| 버스정류장 수 (반경 500m) | 서울 열린데이터광장 |
| 초등학교 거리 (m) | 학구도 안내 서비스 |
| 학원 수 (반경 1km) | 교육부 학원정보 |
| 대형마트/백화점 거리 | 소상공인진흥공단 |

**③ 환경 특성 (Environment Features)**
| 변수 | 출처 |
|---|---|
| 공원 면적 (반경 1km) | 서울 열린데이터광장 |
| 범죄율 (자치구별) | 경찰청 범죄통계 |
| 미세먼지 농도 | 에어코리아 |

**④ 거시경제 특성 (Macro Features)**
| 변수 | 출처 |
|---|---|
| 기준금리 | 한국은행 |
| 전세가율 | 한국부동산원 |
| 거래량 (월별) | 한국부동산원 |

### 3.3 분석 모형

#### 비교 모형 (3개)
1. **다중선형회귀 (OLS)** — 베이스라인
2. **Random Forest** — 앙상블 비교군
3. **XGBoost** — 주 모형

#### 하이퍼파라미터 튜닝
- GridSearchCV 또는 Optuna 활용
- 5-Fold Cross Validation

#### 성능 평가 지표
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (결정계수)
- MAPE (Mean Absolute Percentage Error)

### 3.4 SHAP 분석

#### Global Explanation (전체 해석)
- **SHAP Summary Plot** — 전체 변수 중요도 + 영향 방향
- **SHAP Bar Plot** — 평균 절대 SHAP값 기준 중요도 순위
- **SHAP Dependence Plot** — 개별 변수와 가격의 비선형 관계

#### Local Explanation (개별 해석)
- **SHAP Waterfall Plot** — 특정 아파트의 가격이 왜 이렇게 형성되었는지
- **SHAP Force Plot** — 개별 예측의 변수별 기여도 시각화

#### 지역별 비교
- 강남3구 vs 비강남 → SHAP값 차이 비교
- 어떤 요인이 지역에 따라 다르게 작용하는지 분석

### 3.5 분석 도구
- **Python 3.10+**
- pandas, numpy, scikit-learn
- xgboost
- shap
- matplotlib, seaborn
- geopandas (선택: 공간 시각화)

---

## 4. 기대 결과

### 4.1 예상 연구 결과
1. XGBoost가 OLS, RF 대비 높은 예측 성능 (기존 연구와 일관)
2. SHAP 분석을 통해 전용면적, 경과연수, 지하철역 거리 등이 상위 결정요인으로 도출
3. 강남3구에서는 학원 수/학군이, 비강남에서는 교통 접근성이 더 중요하게 나타날 것으로 예상
4. 비선형 관계 발견 (예: 일정 면적 이상에서 가격 상승폭 체감)

### 4.2 학술적 기여
- 국내 최초 XAI 기반 아파트 가격 결정요인 분석
- 해외 XAI 부동산 연구의 한국 적용 가능성 검증
- 전통 헤도닉 모형의 한계 보완

### 4.3 실무적·정책적 시사점
- 지역별 가격 결정요인 차이 → 차별화된 주택 정책 수립 근거
- 부동산 자동감정평가(AVM) 모형의 해석력 향상 방안 제시
- 소비자의 합리적 의사결정 지원

---

## 5. 연구 일정 (안)

| 단계 | 기간 | 내용 |
|---|---|---|
| 1단계 | 1~2개월 | 선행연구 정리, 데이터 수집 및 전처리 |
| 2단계 | 2~3개월 | 모형 구축, 하이퍼파라미터 튜닝 |
| 3단계 | 3~4개월 | SHAP 분석, 시각화, 지역별 비교 |
| 4단계 | 4~5개월 | 논문 집필 |
| 5단계 | 5~6개월 | 수정, 심사 대응, 최종 제출 |

---

## 6. 참고문헌 (주요)

### 국내
- 조민지 (2023). 서울시 아파트 매매가격지수 예측력 비교 연구. 한양대학교 부동산융합대학원 석사학위논문.
- 이학만 (2025). 머신러닝을 활용한 부산시 부동산 지수 분석 및 예측. 부경대학교 석사학위논문.
- 진수정 (2024). SRGCNN 모형을 이용한 서울시 아파트 매매 가격 예측. 서울대학교 석사학위논문.
- 김선현 (2022). 빅데이터 기반의 아파트 가격 형성 요인 분석. 경북대학교 석사학위논문.
- 박재수 (2020). 주택시장 예측을 위한 부동산 감성지수 개발 연구. 강원대학교 석사학위논문.

### 해외
- Neves, F.T. et al. (2024). The Impacts of Open Data and eXplainable AI on Real Estate Price Predictions in Smart Cities. Applied Sciences, 14(5), 2209.
- Dou, M. et al. (2023). Incorporating Neighborhoods with Explainable Artificial Intelligence for Modeling Fine-Scale Housing Prices. Applied Geography, 159, 103073.
- Acharya, D.B. et al. (2024). Explainable and Fair AI: Balancing Performance in Financial and Real Estate ML Models. IEEE Access.
- Tchuente, D. (2024). Real Estate AVM with XAI Based on Shapley Values. J. Real Estate Finance and Economics.
- Kee, T. & Ho, W. (2025). Explainable ML for Real Estate: XGBoost and Shapley Values in Price Prediction. Civil Engineering Journal.
- Krämer, B. et al. (2023). Explainable AI in a Real Estate Context. J. Housing Economics, 50.
- Choy, L.H.T. & Ho, W. (2023). The Use of Machine Learning in Real Estate Research. Land, 12(4), 740.
- Tekouabou, S.C.K. et al. (2024). AI-Based ML for Urban Real Estate Prediction: A Systematic Survey. Springer.
- Lundberg, S.M. & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS. [SHAP 원논문]
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD. [XGBoost 원논문]

---

## 부록: 핵심 Python 코드 스케치

```python
# 데이터 로드
import pandas as pd
import xgboost as xgb
import shap

# 모형 학습
model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    random_state=42
)
model.fit(X_train, y_train)

# SHAP 분석
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 시각화
shap.summary_plot(shap_values, X_test)          # 전체 변수 중요도
shap.dependence_plot("전용면적", shap_values, X_test)  # 개별 변수 영향
shap.waterfall_plot(explainer(X_test)[0])        # 개별 예측 해석
```
