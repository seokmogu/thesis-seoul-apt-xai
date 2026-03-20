# 🏠 XGBoost와 SHAP을 활용한 서울시 아파트 매매가격 결정요인 분석

> 한양대학교 부동산융합대학원 빅데이터전공 석사논문

## 📊 연구 개요

서울시 25개 자치구의 아파트 매매 실거래가 데이터(2019~2024, 308,555건)를 활용하여 XGBoost 모델로 가격을 예측하고, SHAP(SHapley Additive exPlanations)을 통해 가격 결정요인의 영향력을 해석한 연구입니다.

### 핵심 결과

| 모델 | R² | RMSE (만원) | MAPE |
|------|-----|------------|------|
| OLS 다중회귀 | 0.604 | 45,901 | 39.87% |
| Random Forest | **0.919** | **20,754** | **14.54%** |
| XGBoost | 0.918 | 20,851 | 14.99% |

### SHAP 변수 중요도 Top 5
1. **전용면적** — 가격에 가장 큰 영향
2. **강남구분** — 지역 프리미엄 효과
3. **건물연령** — 신축일수록 가격↑
4. **M2(광의통화)** — 유동성 효과
5. **소비자물가지수** — 인플레이션 반영

## 📁 프로젝트 구조

```
thesis-seoul-apt-xai/
├── README.md
├── requirements.txt
├── API_신청_목록.md          # 데이터 API 신청 가이드
│
├── scripts/                  # 데이터 수집 & 분석 코드
│   ├── collect_apartment_trades.py   # 국토교통부 실거래가 수집
│   ├── collect_seoul_data.py         # 서울열린데이터 (지하철/공원)
│   ├── collect_ecos.py               # 한국은행 거시경제 지표
│   ├── collect_schools.py            # NEIS 학교 정보
│   ├── preprocess.py                 # 데이터 전처리 & 변수 생성
│   ├── modeling.py                   # OLS → RF → XGBoost → SHAP
│   ├── utils.py                      # API 키 로더 & 유틸리티
│   └── test_apis.py                  # API 연결 테스트
│
├── data/                     # 수집된 원시 & 가공 데이터
│   ├── apartment_trades.csv          # 실거래가 원본 (308,555건)
│   ├── apartment_final.csv           # 전처리 완료 데이터
│   ├── seoul_subway_stations.csv     # 지하철역 (799개)
│   ├── seoul_parks.csv               # 공원 (131개)
│   ├── seoul_schools.csv             # 학교 (1,415개)
│   ├── seoul_large_stores.csv        # 대규모점포 (36,741건)
│   ├── seoul_department_stores.csv   # 백화점 (505개)
│   └── ecos_macro.csv                # 거시경제 지표 (72개월)
│
├── results/                  # 분석 결과
│   ├── model_comparison.csv          # 모델 성능 비교
│   ├── feature_importance.csv        # 변수 중요도 (OLS/RF/XGB/SHAP)
│   └── shap_values.csv               # SHAP 값 (5,000건 샘플)
│
├── paper/                    # 논문 관련 문서
│   ├── 논문_초안_XAI_아파트가격.md    # 논문 초안 (한글)
│   ├── 논문_초안.html                 # 논문 초안 (HTML)
│   ├── 연구설계서_XAI_아파트가격.md   # 연구설계서
│   └── 데이터_출처_정리.md            # 데이터 출처 표
│
└── references/               # 선행연구 서베이
    ├── 부동산_빅데이터_선행연구_서베이.md       # 기본 (36편)
    └── 부동산_빅데이터_선행연구_서베이_확장.md  # 확장 (55편)
```

## 🚀 실행 방법

### 1. 환경 설정
```bash
pip install -r requirements.txt
```

### 2. API 키 설정
`scripts/` 상위 디렉토리에 `.api_keys` 파일 생성:
```
SEOUL_API_KEY=your_key_here
DATA_GO_KR_KEY_DECODED=your_key_here
ECOS_API_KEY=your_key_here
NEIS_API_KEY=your_key_here
```

### 3. 데이터 수집
```bash
cd scripts
python collect_apartment_trades.py   # ~10분 소요
python collect_seoul_data.py
python collect_ecos.py
python collect_schools.py
```

### 4. 전처리 & 모델링
```bash
python preprocess.py    # 데이터 병합 & 변수 생성
python modeling.py      # OLS → RF → XGBoost → SHAP
```

## 📐 데이터 출처

| 데이터 | 출처 | URL |
|--------|------|-----|
| 아파트 실거래가 | 국토교통부, 공공데이터포털 | data.go.kr |
| 지하철역/공원 | 서울특별시, 서울열린데이터광장 | data.seoul.go.kr |
| 학교 정보 | 교육부, NEIS | open.neis.go.kr |
| 대규모점포 | 서울특별시, 서울열린데이터광장 | data.seoul.go.kr |
| 거시경제지표 | 한국은행, ECOS | ecos.bok.or.kr |

## 📝 참고문헌 (주요)

- 조민지 (2023). 서울시 아파트 매매가격지수 예측력 비교 연구. 한양대학교 부동산융합대학원.
- Neves et al. (2024). Explainable AI for housing price prediction. *Expert Systems with Applications*.
- Chen & Guestrin (2016). XGBoost: A scalable tree boosting system. *KDD*.
- Lundberg & Lee (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Rosen (1974). Hedonic prices and implicit markets. *JPE*.

## ⚖️ License

This project is for academic purposes (Master's thesis at Hanyang University).

<!-- build-date: 2026-03-20 -->
