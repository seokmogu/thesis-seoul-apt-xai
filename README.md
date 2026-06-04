# 서울시 아파트 매매가격 구조 분석

> 한양대학교 부동산융합대학원 도시·부동산빅데이터전공 석사학위논문

## 연구 개요

본 프로젝트는 서울특별시 아파트 매매가격 구조를 거리 기반 접근성과 시공간 이질성의 관점에서 분석한다. 2019년 1월부터 2025년 12월까지의 아파트 실거래 391,826건과 8,601개 단지 좌표를 사용하며, 종속변수는 `log(㎡당 거래가격)`이다.

전용면적은 단위면적당 가격 산식에만 사용하고 설명변수에서는 제외한다. 시설 변수는 행정동 집계값이 아니라 거래 단지 좌표 기준 거리·반경 접근성으로 재구성하고, 시설 개업·폐업 이력을 반영한 연도별 시점 정합 스냅샷을 사용한다. 해석은 XGBoost 예측값에 대한 SHAP 기여도를 전체·권역·연도·연도×권역 단위로 분해한다.

## 최신 핵심 결과

| 분할 기준 | OLS R² | Random Forest R² | XGBoost R² | XGBoost MAPE |
| --- | ---: | ---: | ---: | ---: |
| 무작위 분할 | 0.494 | 0.860 | 0.927 | 10.0% |
| 단지 분할 | 0.434 | 0.538 | 0.638 | 20.4% |
| 시간순 분할 | 0.406 | 0.675 | 0.812 | 14.6% |

전체 SHAP 상위 5개 변수는 다음과 같다.

1. 강남구분 (13.6%)
2. M2 통화량 (11.4%)
3. 건물연령 (10.5%)
4. 어린이집 1km 내 개수 (5.5%)
5. 소비자물가지수 (4.7%)

무작위 분할의 높은 성능은 동일·유사 단지 거래가 학습·평가 표본에 함께 포함되는 구조의 영향을 받을 수 있다. 따라서 본 논문은 시간순 분할과 단지 분할 결과를 함께 제시하고, 신규 단지 외삽 성능은 제한적으로 해석한다.

## 단일 진실과 제출 산출물

| 구분 | 경로 | 설명 |
| --- | --- | --- |
| 본문 소스 | `paper/석사학위논문_박현근.md` | 논문 본문 단일 진실 |
| 제출 DOCX | `paper/석사학위논문_박현근.docx` | 한양대 서식 빌드 산출물 |
| 제출 PDF | `paper/석사학위논문_박현근.pdf` | DOCX 변환 산출물 |
| HWP 안내 | `paper/HWP_변환_가이드.md` | 한컴 변환 단계 안내 |
| 연구윤리 | `paper/연구윤리서약서_안내.md` | HY-in 출력·날인 안내 |
| 작업 인수인계 | `HANDOFF_TO_CODEX.md` | 빌드 결정, 잔존 이슈, 검증 기록 |

## 프로젝트 구조

```text
thesis-seoul-apt-xai/
├── README.md
├── CLAUDE.md
├── HANDOFF_TO_CODEX.md
├── requirements.txt
├── package.json
├── API_신청_목록.md
├── data/
├── figures/
├── paper/
│   ├── 석사학위논문_박현근.md
│   ├── 석사학위논문_박현근.docx
│   ├── 석사학위논문_박현근.pdf
│   ├── HWP_변환_가이드.md
│   └── 연구윤리서약서_안내.md
├── references/
├── results/
├── scripts/
│   ├── build_v8_dataset.py
│   ├── modeling_v8.py
│   ├── modeling_v8_year_region.py
│   ├── generate_figures.py
│   ├── generate_concept_figures.py
│   ├── export_docx_v8.py
│   └── export_docx_v8_compact.py
└── templates/
```

## 실행 방법

### Python 환경

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

일부 분석·문서 빌드 스크립트는 `pandas`, `scikit-learn`, `xgboost`, `shap`, `python-docx`, `matplotlib`, `latex2mathml`, `mathml2omml` 등을 사용한다. 기존 로컬 환경은 `.venv/`에 구성되어 있다.

### API 키

루트의 `.api_keys` 파일을 사용한다. 이 파일은 로컬 비밀 파일이며 커밋하지 않는다.

```text
SEOUL_API_KEY=...
DATA_GO_KR_KEY_DECODED=...
ECOS_API_KEY=...
NEIS_API_KEY=...
KAKAO_REST_API_KEY=...
```

### 데이터와 모델 재현

```bash
.venv/bin/python scripts/build_v8_dataset.py
.venv/bin/python scripts/modeling_v8.py
.venv/bin/python scripts/modeling_v8_year_region.py
.venv/bin/python scripts/generate_figures.py
.venv/bin/python scripts/generate_concept_figures.py
```

대용량 파생 데이터인 `data/apartment_final_v8.csv`와 `data/seoul_large_stores_v2.csv`는 GitHub 100MB 제한 때문에 추적하지 않는다. 필요하면 위 스크립트로 재생성한다.

### 논문 DOCX/PDF 빌드

```bash
.venv/bin/python scripts/export_docx_v8.py
```

빌드는 `paper/석사학위논문_박현근.md`를 읽어 한양대 부동산융합대학원 서식의 DOCX와 PDF를 생성한다. A4, 상하 4.0cm·좌우 3.5cm 여백, HY신명조 11pt, 줄간격 160%, 장 제목 16pt, 절 제목 13pt 기준이다. 기본 빌드는 학위청구논문 심사용 본문이며, 연구윤리서약서는 HY-in에서 별도 출력·날인해 최종 인쇄본 제출 때 첨부한다.

## 주요 결과 파일

| 파일 | 내용 |
| --- | --- |
| `results/modeling_v8_results.json` | 최종 모형 성능 |
| `results/shap_importance_v8.csv` | 전체 SHAP 중요도 |
| `results/ablation_v7_v8.csv` | 행정동 집계·거리 기반·시점 정합 소거분석 |
| `results/v8_year_region_summary.csv` | 연도×권역 성능 |
| `results/v8_year_region_shap_top.csv` | 연도×권역 SHAP 상위 변수 |
| `figures/fig1_research_flow.png` ~ `figures/fig13_top1_timeline.png` | 본문 그림 |

## 데이터 출처

| 데이터 | 출처 |
| --- | --- |
| 아파트 실거래가 | 국토교통부 실거래가 공개시스템, 공공데이터포털 |
| 지하철역·공원·대규모점포·공공시설 | 서울열린데이터광장 |
| 학교 정보 | 교육부 NEIS |
| 병원·백화점·마트·학원 좌표 보정 | Kakao Local API |
| 거시경제지표 | 한국은행 ECOS |

## 참고

- `CLAUDE.md`: 한양대 부동산융합대학원 서식 및 작성 규칙.
- `HANDOFF_TO_CODEX.md`: 빌드 방식, 폰트·정렬·페이지번호 결정, 잔존 이슈.
- `paper/작성규칙_지도교수피드백_20260527.md`: 지도교수 피드백 기준 수정 규칙.
- `paper/지도교수_피드백_수정기록_20260527.md`: 피드백 항목별 반영 기록.

## License

This project is for academic purposes as a master's thesis project at Hanyang University.
