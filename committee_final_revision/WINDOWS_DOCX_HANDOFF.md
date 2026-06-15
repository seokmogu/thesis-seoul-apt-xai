# Windows DOCX 작업 핸드오프

## 목적

이 문서는 심사위원 피드백 이후 Markdown 수정본을 Windows 환경에서 DOCX로 반영하기 위한 전달 문서이다. Mac/Codex에서는 DOCX/PDF를 생성하지 않는다.

## 기준 파일

- Markdown 원고: `committee_final_revision/paper/석사학위논문_박현근_심사위원피드백후_최종수정본.md`
- 이미지 폴더: `committee_final_revision/figures/`
- 수정 로그: `committee_final_revision/REVISION_LOG.md`
- 체크리스트: `committee_final_revision/CHECKLIST.md`
- 리뷰 전문 재검토: `committee_final_revision/FULL_TRANSCRIPT_RECHECK.md`
- Claude Code 재검토 아티팩트: `.omx/artifacts/ask-claude-thesis-full-transcript-review-cc-20260609.md`
- 조미정 교수 텍스트 피드백 반영표: `committee_final_revision/JOMIJEONG_TEXT_FEEDBACK_RECHECK.md`
- 문서 포맷 전용 체크리스트: `committee_final_revision/WINDOWS_DOCX_FORMAT_FEEDBACK.md`

## Windows에서 반영할 핵심 변경

1. 제목
   - 국문: `서울시 아파트 매매가격 예측 구조 분석: 거리 기반 접근성과 시점 정합을 중심으로`
   - 영문: `An Analysis of Apartment Sale Price Prediction Structure in Seoul: Focusing on Distance-Based Accessibility and Temporal Consistency`

2. 국문초록/영문초록
   - `시공간 이질성` 관점 삭제.
   - AVM 실무 맥락, 시점 정합, 보수적 검증, 권역·연도별 SHAP 비교 중심으로 반영.

3. 목차/표 목차/그림 목차
   - `제7절 연도×권역 교차 SHAP 분석 — 시공간 이질성의 실증`
   - 위 표현을 `제7절 연도별·권역별 SHAP 비교분석 — 시장 국면별 예측 구조`로 반영.
   - `<표 4-14>`와 `<그림 4-9>` 제목도 `연도별·권역별`로 맞춘다.
   - 새 `<표 4-12>` 추가로 이후 표 번호가 `<표 4-16>`까지 밀렸으므로 표 목차를 전체 업데이트한다.

4. 제1장
   - 연구의 기여를 `머신러닝/SHAP/좌표 기반 자체의 신규성`이 아니라 `시점 정합`, `시장 국면별 예측 안정성`, `AVM 지속 검증 필요성`으로 반영.

5. 제2장
   - XAI/SHAP은 해석 도구로 낮춰 서술.
   - 유사 선행연구 대비 차이는 보수적으로 반영.
   - DOCX 생성 전, XAI/SHAP 이론·수식이 제2장과 제3장에 중복되거나 과도하게 부각되지 않는지 확인한다.
   - 심사위원 피드백 기준으로는 AVM/부동산 가격 예측 선행연구 중심축 보강이 아직 가장 큰 내용 리스크다.

6. 제3장
   - 강남3구/비강남 구분의 근거와 한계를 반영.
   - 비강남은 동질 시장이 아니라 `강남3구 외 지역` 비교군임을 명시.
   - 직선거리의 정의·한계와 별도로, 왜 직선거리를 선택했는지 적극적 사유가 들어갔는지 확인한다.

7. 제4장 제7절
   - `교차 분석`이 아니라 연도별·권역별 비교분석임을 명시.
   - 상호작용 효과나 공간통계 모형을 암시하지 않게 반영.

8. 제5장
   - AVM 갱신 주기는 직접 추정하지 않았음을 명시.
   - 시간순 분할, 단지 분할, 과거 학습창 기반 보조 검증, 권역별·연도별 SHAP 점검을 병행해야 한다는 수준으로 정리.
   - 단지 대표좌표 사용으로 동별 위치 차이를 반영하지 못한 한계를 추가했는지 확인한다.

## DOCX 생성 전 추가 보강 권고

다음 항목은 `cc` Claude Code 재검토에서 Windows DOCX 전 필수 보강으로 분리된 사항이다. 실제 DOCX/PDF 생성은 Windows에서만 진행한다.

1. 제2장 XAI/SHAP 비중 축소 및 AVM 이론·선행연구 보강.
2. 직선거리 선택 사유를 한계 서술과 분리하여 1-2문장 명시.
3. 단지 대표좌표 사용의 한계, 특히 동별 위치 미반영을 제5장에 명시.
4. LaTeX 수식이 Word OMML로 제대로 변환되는지 전 수식 육안 검수.
5. 표제지, 제출서, 인준서, 연구윤리 서약서 등 front matter 삽입 확인.
6. `CHECKLIST.md`, `REVISION_LOG.md`, 본문 반영 상태의 정합성 재확인.

## Windows Word 확인 항목

- 문서 포맷 전용 체크리스트(`WINDOWS_DOCX_FORMAT_FEEDBACK.md`)를 먼저 열고 표/그림 배치 대상 목록을 확인한다.
- 표지 제목과 영문 제목 반영 여부.
- 표제지, 제출서, 인준서, 연구윤리 서약서 등 front matter 포함 여부.
- 목차, 표 목차, 그림 목차 업데이트.
- 표/그림 캡션 위치와 번호.
- 조미정 교수 피드백 기준으로 제목·절 제목 바로 아래에 표가 먼저 오지 않고, 설명 문단 뒤에 표가 오도록 각 표의 페이지 배치를 확인한다.
- `<표 3-2>` XGBoost 하이퍼파라미터 표 추가로 표 목차와 이후 표 번호가 자동 갱신됐는지 확인한다.
- `<표 4-1>`, `<표 4-2>`, `<표 4-4>`, `<표 4-5>`, `<표 4-7>`, `<표 4-10>`, `<표 4-11>` 주변 문단과 표 위치가 Word에서 자연스럽게 이어지는지 확인한다.
- `<표 4-4>`는 Markdown 변환 안정성을 위해 첫 열의 분할명을 반복 표기했다. Word에서 행 병합을 적용할 경우 의미가 바뀌지 않는지 확인한다.
- 그림 링크 또는 삽입 이미지 누락 여부.
- 그림 파일명과 본문 그림 번호가 다르므로 수동 삽입 시 대응 관계를 다시 확인한다.
- LaTeX 수식이 Word 수식(OMML)로 정상 변환됐는지 확인한다.
- 한글/영문 초록 페이지의 줄바꿈.
- `시공간 이질성`, `Spatiotemporal`, `연도×권역`, `교차 SHAP` 표현 잔존 여부.
- PDF 변환 후 표/그림 잘림 여부.

## 금지

- Windows 작업 전 Mac에서 DOCX/PDF를 생성하지 않는다.
- 기존 `paper/`의 제출본 DOCX/PDF를 조용히 덮어쓰지 않는다.
- 새 DOCX 산출물은 심사위원 피드백 이후 최종수정본임을 파일명에 명시한다.
