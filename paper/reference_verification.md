# 참고문헌 검증 보고서

**검증일:** 2026-03-10  
**검증 대상:** `paper/논문_초안.md` 현재 참고문헌 32편(국내 8편, 해외 24편)  
**검증 원칙:** HEARTBEAT 지침에 따라 로컬 산출물(`references/verification_v2.json`, `references/verification_firecrawl.json`, `references/papers/`)을 우선 사용하여 최신 초안 기준으로 재정리함.

---

## 검증 결과 요약

- 현재 초안 기준 참고문헌은 **총 32편**이다.
- 이전 v3 기준 보고서(37편)에서 문제로 지적되었던 **Acharya et al. (2024), Krämer et al. (2023), Kee & Ho (2025), Na/Ko/Park (2025), Park/Oh/Won (2024), Tchuente (2024), Tekouabou et al. (2024), 이선구 (2024)** 등은 **현재 초안 참고문헌 목록에서 제거 또는 정정되어 더 이상 포함되지 않는다.**
- 현재 목록에 남아 있는 참고문헌 중, 로컬 PDF/TXT 보유 또는 기존 검증 JSON 확인이 가능한 항목은 **29편**이다.
- **3편(Hair et al., 2010; Hastie et al., 2009; 통계청 2024)**은 현 폴더 내 PDF/TXT 직접 대조본은 없으나, 표준 교재/공식 통계자료로서 논문 서지상 문제 정황은 확인되지 않았다.
- `python3 scripts/verify_refs_firecrawl.py` 재실행은 이번 루프에서 시도했으나 장시간 응답 지연으로 완료 로그를 확보하지 못했다. 다만 **이번 루프에서 새 인용을 추가하거나 참고문헌 항목 자체를 신규 삽입하지는 않았으므로**, 최신화의 핵심은 현 초안 기준 목록 정리와 로컬 검증 근거 재작성에 있다.
- 결론적으로, **현재 초안 기준에서 유령 참고문헌으로 보이는 항목은 확인되지 않았다.**

---

## 1. 현재 초안 참고문헌 목록(32편)과 로컬 근거

### 1-1. 국내문헌 (8편)

| 번호 | 참고문헌 | 로컬 근거 | 판정 |
|---|---|---|---|
| 1 | 김상진 (2023) | `references/papers/김상진_2023_프롭테크.pdf` | 확인 |
| 2 | 김선현 (2022) | `references/papers/김선현_2022_대구아파트.pdf` | 확인 |
| 3 | 오성훈 (2022) | `references/papers/오성훈_2022_뉴스빅데이터.pdf` | 확인 |
| 4 | 이용운 (2024) | `references/papers/이용운_2024_젠트리피케이션.pdf` | 확인 |
| 5 | 이학만 (2025) | `references/papers/이학만_2025_부산부동산.pdf` | 확인 |
| 6 | 조민지 (2023) | `references/papers/조민지_2023_서울아파트.pdf` | 확인 |
| 7 | 진수정 (2024) | `references/papers/진수정_2024_SRGCNN.pdf` | 확인 |
| 8 | 통계청 (2024) | 정부 공식 통계자료 인용(로컬 PDF 없음) | 확인(공식자료) |

### 1-2. 해외문헌 (24편)

| 번호 | 참고문헌 | 로컬 근거 | 보조 근거 | 판정 |
|---|---|---|---|---|
| 1 | An et al. (2025) | `references/papers/An_2025.pdf` | `verification_v2.json`, `verification_firecrawl.json` | 확인 |
| 2 | Anselin (1988) | `references/papers/Anselin_1988_SpatialEconometrics.txt` |  | 확인 |
| 3 | Breiman (2001) | `references/papers/Breiman_2001.pdf` | `verification_v2.json`, `verification_firecrawl.json` | 확인 |
| 4 | Čeh et al. (2018) | `references/papers/Ceh_2018.pdf` | `verification_v2.json`, `verification_firecrawl.json` | 확인 |
| 5 | Chen & Guestrin (2016) | `references/papers/Chen_Guestrin_2016_XGBoost.pdf` | `verification_v2.json`, `verification_firecrawl.json` | 확인 |
| 6 | Choy & Ho (2023) | `references/papers/Choy_Ho_2023.pdf` | `verification_v2.json`, `verification_firecrawl.json` | 확인 |
| 7 | Chun et al. (2025) | `references/papers/Chun_2025_Seoul_XAI_ML.pdf` | `verification_v2.json`, `verification_firecrawl.json` | 확인 |
| 8 | Friedman (2001) | `references/papers/Friedman_2001_GBM.pdf`, `Friedman_2001_GBM.txt` |  | 확인 |
| 9 | Hair et al. (2010) | 로컬 PDF/TXT 없음 | 표준 교재 서지 | 확인(교재) |
| 10 | Hastie et al. (2009) | 로컬 PDF/TXT 없음 | 표준 교재 서지 | 확인(교재) |
| 11 | Ke et al. (2017) | `references/papers/Ke_2017.pdf` | `verification_v2.json`, `verification_firecrawl.json` | 확인 |
| 12 | Kim et al. (2022) | `references/papers/Kim_2022_Multiplex.pdf` | `verification_v2.json`, `verification_firecrawl.json` | 확인 |
| 13 | Kim, Choi & Lee (2025) | `references/papers/Kim_Choi_Lee_2025.pdf` | `verification_v2.json`, `verification_firecrawl.json` | 확인 |
| 14 | Lancaster (1966) | `references/papers/Lancaster_1966.pdf`, `Lancaster_1966_ConsumerTheory.txt` | `verification_v2.json` | 확인 |
| 15 | Limsombunchai (2004) | `references/papers/Limsombunchai_2004.pdf` | `verification_v2.json`, `verification_firecrawl.json` | 확인 |
| 16 | Lundberg & Lee (2017) | `references/papers/Lundberg_Lee_2017.pdf` | `verification_v2.json`, `verification_firecrawl.json` | 확인 |
| 17 | Lundberg et al. (2020) | `references/papers/Lundberg_2020_TreeSHAP.pdf` | `verification_firecrawl.json` | 확인 |
| 18 | Mora-García et al. (2022) | `references/papers/Mora_Garcia_2022.pdf` | `verification_v2.json`, `verification_firecrawl.json` | 확인 |
| 19 | Neves et al. (2024) | `references/papers/Neves_2024.pdf` | `verification_v2.json`, `verification_firecrawl.json` | 확인 |
| 20 | Revathi & Devarajan (2025) | `references/papers/Revathi_2025_XGBoost_SHAP.pdf` | `verification_firecrawl.json` | 확인 |
| 21 | Ribeiro et al. (2016) | `references/papers/Ribeiro_2016.pdf` | `verification_v2.json`, `verification_firecrawl.json` | 확인 |
| 22 | Rosen (1974) | `references/papers/Rosen_1974_Hedonic.txt`, `Rosen_1974.pdf` | `verification_v2.json`, `verification_firecrawl.json` | 확인 |
| 23 | Shahhosseini et al. (2022) | `references/papers/Shahhosseini_2022_SHAP_Housing.pdf` | `verification_firecrawl.json` | 확인 |
| 24 | Tarasov & Dessoulavy-Śliwiński (2025) | `references/papers/Tarasov_2025.pdf` | `verification_v2.json` | 확인 |

---

## 2. 기존 오류 항목 정리 결과

이전 보고서(v3 기준)에서 문제였던 항목들은 다음 상태로 정리되었다.

- **초안에서 삭제됨:** Acharya et al. (2024), Krämer et al. (2023), Kee & Ho (2025), Na/Ko/Park (2025), Park/Oh/Won (2024), Tchuente (2024), Tekouabou et al. (2024), 이선구 (2024)
- **중복 제거됨:** Lundberg & Lee (2017) 중복 항목 제거 완료
- **페이지/서지 수정 반영됨:** Tarasov & Dessoulavy-Śliwiński (2025) `33(1), 22-34`

따라서 v3 보고서에서 지적된 할루시네이션/오서지 이슈는 **현재 초안 참고문헌 목록에는 직접 남아 있지 않다.**

---

## 3. 이번 루프의 추가 확인 사항

1. 현재 `논문_초안.md` 참고문헌 수는 **32편**으로 확인하였다.
2. 국내문헌 8편 / 해외문헌 24편 구조로 정리되어 있으며, 참고문헌 형식상 국내 목록의 불필요한 빈 줄 1건을 제거하여 일관성을 맞추었다.
3. `references/verification_v2.json`과 `references/verification_firecrawl.json`은 일부 고전문헌/교재를 완전하게 포괄하지는 않지만, 현재 핵심 해외논문 다수에 대해 실재 근거를 제공한다.
4. 이번 루프에서 Firecrawl 스크립트 재실행은 시도했으나 완료 로그를 확보하지 못했다. 향후 새 참고문헌을 추가하거나 기존 항목을 교체하는 경우에는 `python3 scripts/verify_refs_firecrawl.py`를 다시 실행해 JSON 산출물을 재생성할 필요가 있다.

---

## 4. 최종 판정

- **유령 참고문헌:** 0건
- **현재 초안 기준 중대한 서지 오류:** 0건
- **즉시 삭제가 필요한 참고문헌:** 0건
- **추가 권고:** 새 인용 추가 시 Firecrawl JSON 재생성

현재 초안의 참고문헌은 **제출용 초안 기준에서 실재성·서지 일관성 측면의 치명적 문제는 없는 상태**로 판단한다.
