# 한양대 부동산융합대학원 석사학위논문 핸드오프

## 현재 상태

- 프로젝트 경로: `/Users/seokmogu/project/thesis-seoul-apt-xai`
- 논문 주제: 서울시 아파트 단위면적당 매매가격 구조 분석: 거리 기반 접근성과 시공간 이질성을 중심으로
- 저자: 박현근
- 전공: 한양대학교 부동산융합대학원 도시·부동산빅데이터전공
- 본문 단일 진실: `paper/석사학위논문_박현근.md`
- 제출 DOCX: `paper/석사학위논문_박현근.docx`
- 제출 PDF: `paper/석사학위논문_박현근.pdf`

최신 PDF 확인값:

- 생성 시각: 2026-05-27 15:51 KST
- 생성 도구: LibreOffice 26.2.2.2
- 페이지 수: 87쪽
- 용지: A4
- 파일 크기: 약 2.8MB

## 핵심 연구 설계

- 분석 기간: 2019년 1월부터 2025년 12월까지.
- 분석 표본: 서울 아파트 실거래 391,826건, 단지 좌표 8,601개.
- 종속변수: `log(㎡당 거래가격)`.
- 전용면적은 종속변수 산식에만 사용하고 설명변수에서는 제외한다.
- 시설 변수는 행정동 집계값 대신 단지 좌표 기준 거리·반경 접근성 지표로 구성한다.
- 시설 개업·폐업 이력을 반영해 연도별 시점 정합 스냅샷을 사용한다.
- SHAP은 전체, 권역, 연도, 연도×권역 단위로 분해한다.

## 최신 핵심 결과

| 분할 기준 | OLS R² | Random Forest R² | XGBoost R² | XGBoost MAPE |
| --- | ---: | ---: | ---: | ---: |
| 무작위 분할 | 0.494 | 0.860 | 0.927 | 10.0% |
| 단지 분할 | 0.434 | 0.538 | 0.638 | 20.4% |
| 시간순 분할 | 0.406 | 0.675 | 0.812 | 14.6% |

전체 SHAP 상위 5개 변수:

1. 강남구분
2. M2 통화량
3. 건물연령
4. 어린이집 1km 내 개수
5. 소비자물가지수

무작위 분할 성능만으로 일반화 성능을 주장하지 않는다. 단지 분할과 시간순 분할의 성능 하락을 함께 제시해 신규 단지 외삽과 시점 외삽의 한계를 명시한다.

## 주요 파일

| 경로 | 역할 |
| --- | --- |
| `README.md` | 현재 프로젝트 개요와 실행 방법 |
| `CLAUDE.md` | 한양대 부동산융합대학원 서식·작성 규칙 |
| `paper/석사학위논문_박현근.md` | 본문 소스 |
| `paper/석사학위논문_박현근.docx` | 제출용 Word 산출물 |
| `paper/석사학위논문_박현근.pdf` | 제출용 PDF 산출물 |
| `paper/HWP_변환_가이드.md` | HWP 변환 안내 |
| `paper/연구윤리서약서_안내.md` | HY-in 연구윤리서약서 출력 안내 |
| `scripts/export_docx_v8.py` | 본문 Markdown에서 DOCX/PDF 생성 |
| `scripts/export_docx_v8_compact.py` | compact 변형 산출 스크립트 |
| `scripts/modeling_v8.py` | 최종 모델링 |
| `scripts/modeling_v8_year_region.py` | 연도×권역 분석 |
| `scripts/generate_figures.py` | 분석 그림 생성 |
| `scripts/generate_concept_figures.py` | 개념도 생성 |
| `templates/` | 학교 제공 HWP 양식과 텍스트 변환본 |

## 재현 명령

```bash
cd /Users/seokmogu/project/thesis-seoul-apt-xai
.venv/bin/python scripts/build_v8_dataset.py
.venv/bin/python scripts/modeling_v8.py
.venv/bin/python scripts/modeling_v8_year_region.py
.venv/bin/python scripts/generate_figures.py
.venv/bin/python scripts/generate_concept_figures.py
.venv/bin/python scripts/export_docx_v8.py
```

`data/apartment_final_v8.csv`와 `data/seoul_large_stores_v2.csv`는 GitHub 100MB 제한 때문에 커밋하지 않는다.

## 서식 결정

- 용지: A4
- Word 여백: 상하 4.0cm, 좌우 3.5cm, 머리말·꼬리말 1.5cm
- 본문 글꼴: HY신명조 11pt
- 줄간격: 160%
- 큰 제목: 16pt 진하게
- 절 제목: 13pt 진하게
- 페이지 번호: front matter 로마자, 본문 아라비아 숫자
- 표·그림 캡션: 한양대 부동산융합대학원 관행에 맞춰 위쪽 배치

## 검증 기록

- `results/export_docx_v8.log`: DOCX/PDF 빌드 완료 로그.
- `results/render_docx_20260527_final.log`: PDF 페이지 이미지 렌더 완료 로그.
- `paper/render_check_20260527_final/`: 최종 렌더 확인 PNG. 용량이 커서 Git 추적 대상에서 제외한다.

## 남은 제출 단계

- HWP가 필요하면 `paper/HWP_변환_가이드.md`에 따라 한컴독스 또는 한컴오피스로 DOCX를 변환한다.
- 연구윤리서약서는 `paper/연구윤리서약서_안내.md`에 따라 HY-in에서 국문·영문 서약서를 출력하고 날인한다.
- 학과 또는 지도교수가 PDF 제출을 허용하면 현재 PDF 산출물을 우선 제출 후보로 사용한다.
