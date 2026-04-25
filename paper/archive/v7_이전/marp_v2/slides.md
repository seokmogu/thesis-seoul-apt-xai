---
marp: true
theme: modern-thesis
paginate: true
size: 16:9
style: |
  @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable.css');
  @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700;800&family=Inter:wght@400;600;700&display=swap');
---

<!-- _class: hero -->
<!-- _paginate: false -->

# XGBoost와 SHAP을 활용한<br/><em>서울시 아파트 단위면적당<br/>매매가격의 설명 패턴 분석</em>

<div class="subtitle">단위면적 정규화 · 행정동 분석 · 지역 별도 모형</div>

<div class="meta">
한양대학교 부동산융합대학원 · 도시부동산정책전공<br/>
<strong>석사학위 중간발표 · 박 현 근</strong><br/>
<span style="font-size:18px">2026년 4월</span>
</div>

---

<!-- _class: content -->

<div data-section="CONTENTS"></div>

## 발표 목차

<div class="features" style="grid-template-columns: repeat(5, 1fr)">

<div class="card">
<div class="badge">01</div>
<h3>서 론</h3>
<p>연구 배경, 필요성, 범위, 방법</p>
</div>

<div class="card">
<div class="badge">02</div>
<h3>이론 · 선행연구</h3>
<p>헤도닉 · ML · XAI · 차별성</p>
</div>

<div class="card">
<div class="badge">03</div>
<h3>연구 설계</h3>
<p>변수 · 분할 · 모형 · 강건성</p>
</div>

<div class="card">
<div class="badge">04</div>
<h3>실증 분석</h3>
<p>기술통계 · SHAP · 지역 비교</p>
</div>

<div class="card">
<div class="badge">05</div>
<h3>결 론</h3>
<p>시사점 · 한계 · 향후 과제</p>
</div>

</div>

<div class="callout">
본 발표의 핵심: <strong>총가격이 아닌 단위면적당 가격으로 정규화</strong>하여, 규모효과에 가려져 있던 <strong>주거권·사교육·재건축</strong> 설명 신호를 드러낸다.
</div>

---

<!-- _class: divider -->

<div class="section-num">01</div>

# 서 론

<div class="items">
1. 연구의 배경 및 목적<br/>
2. 연구의 범위 및 방법
</div>

---

<!-- _class: content -->

<div data-section="01 서론 · 연구 배경"></div>

## 왜 서울 아파트 단위가격인가

<div class="split one-two">

<div>

### 한국 가계의 현실

<div class="stat-card" style="margin-bottom:16px">
<div class="number">75.2<span style="font-size:48px">%</span></div>
<div class="label">한국 가계 자산 중 실물자산 비중</div>
<div class="sub">통계청 (2024) 가계금융복지조사</div>
</div>

<p style="font-size:18px;margin-top:24px">
서울 아파트는 <strong>자산 축적·주거 안정·거시경제</strong>의 핵심 매개.<br/>
매매가격의 예측요인 파악은 학술·정책 모두 핵심.
</p>

</div>

<div class="right">

### 기존 연구의 3대 공백

<div class="features" style="grid-template-columns: 1fr; gap:16px; margin-top:16px">

<div class="card" style="padding:20px">
<div class="badge">공백 1</div>
<h3 style="font-size:18px;margin-bottom:4px">공간 단위 거친 집계</h3>
<p style="font-size:14px">대부분 연구가 자치구(25개) 단위 → 동네 수준 이질성 유실</p>
</div>

<div class="card" style="padding:20px">
<div class="badge">공백 2</div>
<h3 style="font-size:18px;margin-bottom:4px">단일 모형 예측 비교</h3>
<p style="font-size:14px">OLS vs ML 성능 비교에 머무름 → SHAP 해석 체계 부재</p>
</div>

<div class="card" style="padding:20px">
<div class="badge">공백 3</div>
<h3 style="font-size:18px;margin-bottom:4px">총가격 모형의 면적 독점</h3>
<p style="font-size:14px">면적이 SHAP 22% 독점 → 질적·입지적 신호 가려짐</p>
</div>

</div>

</div>

</div>

---

<!-- _class: content -->

<div data-section="01 서론 · 연구 목적"></div>

## 본 연구의 3가지 차별성

<div class="features">

<div class="card">
<div class="badge">축 1</div>
<h3>단위면적 정규화</h3>
<p><strong>Y = log(거래금액 / 전용면적)</strong></p>
<p>규모효과를 분모로 제거하여 질적·입지적 설명 신호 부각. 헤도닉 문헌의 log-price 전통에 연결.</p>
</div>

<div class="card">
<div class="badge">축 2</div>
<h3>행정동 세분화</h3>
<p><strong>215개 행정동 단위</strong></p>
<p>자치구 25개 대비 8.6배 세분화. Nominatim 지오코딩 + GeoJSON 공간조인으로 매핑 방법론 구축.</p>
</div>

<div class="card">
<div class="badge">축 3</div>
<h3>지역 별도 모형</h3>
<p><strong>전체 · 강남3구 · 비강남 22개구</strong></p>
<p>강남더미를 제거한 17변수 독립 적합. 지역 내부 설명 구조 직접 비교.</p>
</div>

</div>

<div class="callout warm">
<strong>연구 목적</strong> — 서울 아파트 단위가격 결정의 숨은 질적 구조를 드러내고, AVM 실무의 <strong>설명 책무</strong>를 뒷받침하는 XAI 프레임워크 구축
</div>

---

<!-- _class: content -->

<div data-section="01 서론 · 연구 범위"></div>

## 연구 범위 및 데이터 스케일

<div class="stat-grid">

<div class="stat-card">
<div class="number">391,826</div>
<div class="label">아파트 매매 실거래</div>
<div class="sub">2019.01 ~ 2025.12 (84개월)</div>
</div>

<div class="stat-card accent">
<div class="number">215</div>
<div class="label">서울 행정동</div>
<div class="sub">25개 자치구 세분화</div>
</div>

<div class="stat-card warm">
<div class="number">18</div>
<div class="label">독립변수</div>
<div class="sub">물리 3 + 입지 2 + 환경 9 + 거시 4</div>
</div>

</div>

<div class="split two-one" style="margin-top:20px">

<div>

### 4대 공공데이터 원천

<div class="features" style="grid-template-columns: 1fr 1fr; gap:10px; margin-top:0">

<div class="card" style="padding:14px 18px">
<h3 style="font-size:17px;margin-bottom:2px">🏠 국토교통부</h3>
<p style="font-size:14px;margin:0">실거래가 (data.go.kr)</p>
</div>
<div class="card" style="padding:14px 18px">
<h3 style="font-size:17px;margin-bottom:2px">🚇 서울열린데이터광장</h3>
<p style="font-size:14px;margin:0">지하철·백화점·공원·도서관·어린이집</p>
</div>
<div class="card" style="padding:14px 18px">
<h3 style="font-size:17px;margin-bottom:2px">🎓 NEIS</h3>
<p style="font-size:14px;margin:0">학교·학원 (open.neis.go.kr)</p>
</div>
<div class="card" style="padding:14px 18px">
<h3 style="font-size:17px;margin-bottom:2px">💰 한국은행 ECOS</h3>
<p style="font-size:14px;margin:0">기준금리·CD금리·CPI·M2</p>
</div>

</div>

</div>

<div class="right" style="padding:20px 24px">
<h3 style="font-size:20px;margin-bottom:8px">시간 범위의 의의</h3>
<p style="font-size:15px;margin:0">2019–2025년은 <strong>COVID-19 팬데믹 · 금리 인상 · 인하 전환</strong>의 <em>3대 거시 국면</em>을 모두 포함하여, 시장 국면별 설명 패턴 안정성 검증에 적합.</p>
</div>

</div>

---

<!-- _class: divider -->

<div class="section-num">02</div>

# 이론적 배경 및<br/>선행연구

<div class="items">
1. 헤도닉 가격모형과 정규화<br/>
2. 머신러닝 기반 예측<br/>
3. XAI와 SHAP<br/>
4. 선행연구 종합
</div>

---

<!-- _class: content -->

<div data-section="02 이론 · 헤도닉"></div>

## 헤도닉 가격모형과 본 연구의 설계

<div class="compare">

<div class="before">
<h3>기존 헤도닉 OLS</h3>

**P = β₀ + β₁z₁ + ... + βₙzₙ + ε**

- Lancaster(1966) · Rosen(1974)
- 총가격을 종속변수로 사용

<p style="margin-top:20px;color:#E53E3E"><strong>3대 한계</strong></p>

- 선형 가정 → 비선형 미포착
- 상호작용 수동 지정 필요
- 다중공선성 취약 (VIF 110+)

</div>

<div class="arrow">→</div>

<div class="after">
<h3>본 연구: log(㎡당 가격)</h3>

**ln(P/A) = β₀ + Σβᵢxᵢ + ε**

- Malpezzi(2003), Sirmans et al.(2005)
- 단위면적 정규화 log-price 전통

<p style="margin-top:20px;color:#006B7F"><strong>3대 이점</strong></p>

- 규모효과를 분모로 제거
- 이분산성 자연 완화
- 준탄력성 (e^β − 1)·100% 해석

</div>

</div>

---

<!-- _class: content -->

<div data-section="02 이론 · 머신러닝"></div>

## 머신러닝 기반 예측

<div class="split">

<div>

### Random Forest
<div class="tag">Breiman (2001)</div>
<div class="tag">배깅 기반</div>

<p>독립 트리 앙상블 → 예측값 평균. 분산 감소에 강점.</p>

<table style="margin-top:16px;font-size:16px">
<tr><td>n_estimators</td><td>200</td></tr>
<tr><td>max_depth</td><td>15</td></tr>
<tr><td>min_samples_leaf</td><td>5</td></tr>
</table>

</div>

<div>

### XGBoost <span class="tag accent">대표 해석 모형</span>
<div class="tag">Chen & Guestrin (2016)</div>
<div class="tag">부스팅 + 정규화</div>

<p>이전 트리 잔차를 순차 학습 → 편향 감소에 강점. L1/L2 정규화 내장.</p>

<table style="margin-top:16px;font-size:16px">
<tr><td>max_depth</td><td>8</td></tr>
<tr><td>learning_rate</td><td>0.1</td></tr>
<tr><td>reg_alpha / lambda</td><td>0.1 / 1.0</td></tr>
<tr><td>최대 boosting 라운드</td><td>2,000</td></tr>
</table>

</div>

</div>

<p style="margin-top:32px;text-align:center;color:#718096;font-size:18px">
모든 분할 · 모든 지역 모형에 <strong>동일 하이퍼파라미터</strong>를 적용하여 비교 가능성 확보
</p>

---

<!-- _class: content -->

<div data-section="02 이론 · XAI"></div>

## SHAP과 본 연구의 활용

<div class="mega-stat">
<div class="big">SHAP</div>
<div class="desc">
<strong>SHapley Additive exPlanations</strong><br/>
Lundberg & Lee (2017) — 게임이론의 Shapley value를 ML 예측값 분해에 적용.<br/>
각 특성이 개별 예측에 미치는 기여를 <em>이론적으로 일관되게 분해</em>.
</div>
</div>

<div class="features">

<div class="card">
<div class="badge">글로벌</div>
<h3>변수 중요도 서열</h3>
<p>mean(|SHAP_log|) 기준 18개 변수 서열과 비중(%)</p>
</div>

<div class="card">
<div class="badge">Dependence</div>
<h3>비선형 패턴 검증</h3>
<p>변수값 ↔ SHAP 산점도 — U자형, 체감, 임계구간 확인</p>
</div>

<div class="card">
<div class="badge">Local</div>
<h3>개별 거래 분해</h3>
<p>Force/Waterfall — base value에서 최종 예측까지 변수별 기여</p>
</div>

</div>

---

<!-- _class: content -->

<div data-section="02 이론 · 선행연구 종합"></div>

## 선행연구 비교 및 본 연구의 위치

<table>
<thead>
<tr><th>연구</th><th>종속변수</th><th>공간 단위</th><th>모형</th><th>해석</th></tr>
</thead>
<tbody>
<tr><td>Čeh et al. (2018)</td><td>총가격</td><td>Ljubljana</td><td>OLS vs RF</td><td>성능만</td></tr>
<tr><td>Chun et al. (2025)</td><td>총가격</td><td>서울 구</td><td>XGB + SHAP</td><td>통합 SHAP</td></tr>
<tr><td>Kim et al. (2025)</td><td>총가격</td><td>한국</td><td>XAI 앙상블</td><td>통합 SHAP</td></tr>
<tr><td>Neves et al. (2024)</td><td>총가격</td><td>리스본</td><td>XGB + SHAP</td><td>공원 거리</td></tr>
<tr style="background:#E6F7FA">
<td><strong>본 연구 (2026)</strong></td>
<td><strong>log(㎡당 가격)</strong></td>
<td><strong>서울 215 행정동</strong></td>
<td><strong>OLS + RF + XGB + SHAP</strong></td>
<td><strong>지역 별도 3모형</strong></td>
</tr>
</tbody>
</table>

<div class="callout">
<strong>차별화된 기여</strong> — 단위면적 정규화 × 행정동 세분화 × 지역 별도 모형의 <em>3축 결합</em>은<br/>
국내 부동산 XAI 문헌에서 <strong>최초 적용</strong>
</div>

---

<!-- _class: divider -->

<div class="section-num">03</div>

# 연구 설계 및 방법

<div class="items">
1. 연구 흐름 및 변수 설정<br/>
2. 3가지 데이터 분할<br/>
3. 모형 · 평가 지표 · SHAP<br/>
4. 강건성 점검 및 지역 별도 모형
</div>

---

<!-- _class: content -->

<div data-section="03 연구 설계 · 흐름"></div>

## 연구의 5단계 흐름

<div class="features" style="grid-template-columns: repeat(5, 1fr); gap:16px">

<div class="card">
<div class="badge">01</div>
<h3 style="font-size:20px">데이터 수집</h3>
<p style="font-size:14px">국토부 · 서울데이터<br/>NEIS · ECOS</p>
</div>

<div class="card">
<div class="badge">02</div>
<h3 style="font-size:20px">전처리 · 파생</h3>
<p style="font-size:14px">법정동→행정동 매핑<br/>Y=log(P/A) 파생</p>
</div>

<div class="card">
<div class="badge">03</div>
<h3 style="font-size:20px">모형 적합</h3>
<p style="font-size:14px">OLS · RF · XGBoost<br/>3가지 분할</p>
</div>

<div class="card">
<div class="badge">04</div>
<h3 style="font-size:20px">SHAP 해석</h3>
<p style="font-size:14px">TreeSHAP 5,000 샘플<br/>글로벌 + Dependence</p>
</div>

<div class="card">
<div class="badge">05</div>
<h3 style="font-size:20px">강건성 점검</h3>
<p style="font-size:14px">지역 별도 · Ablation<br/>Moran's I · 시기별</p>
</div>

</div>

<div class="split" style="margin-top:48px">

<div class="right">
<h3>종속변수 설계</h3>

**Y = ln(거래금액[만원] / 전용면적[㎡])**

<p>규모효과를 분모로 제거 → 잔여 질적·입지적 설명 신호 추출. 헤도닉 문헌의 표준 log-price 전통 연결.</p>
</div>

<div class="right">
<h3>독립변수 18개</h3>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:14px">
<div><span class="tag">물리 3</span> 전용면적·층·건물연령</div>
<div><span class="tag">입지 2</span> 강남구분·지하철역수</div>
<div><span class="tag">환경 9</span> 학교·CCTV·백화점·공원·도서관·학원·어린이집</div>
<div><span class="tag">거시 4</span> 기준금리·CD금리·CPI·M2</div>
</div>

</div>

</div>

---

<!-- _class: content -->

<div data-section="03 연구 설계 · 분할"></div>

## 3가지 데이터 분할 방식

<div class="features">

<div class="card">
<div class="badge">① 무작위</div>
<h3>Random 70/10/20</h3>
<p>학습 <strong>274,277</strong> / 검증 <strong>39,183</strong> / 테스트 <strong>78,366</strong></p>
<p style="margin-top:12px;color:#E53E3E;font-size:14px"><strong>측정 대상:</strong> 동일 단지 내 예측 성능</p>
<p style="font-size:14px">동일 단지 반복거래 공존 가능 → 낙관적 추정</p>
</div>

<div class="card">
<div class="badge">② Group</div>
<h3>단지 기준 분할</h3>
<p>법정동 + 아파트명 기준 <strong>8,601</strong>개 단지, overlap = 0</p>
<p style="margin-top:12px;color:#D69E2E;font-size:14px"><strong>측정 대상:</strong> 미경험 단지 일반화</p>
<p style="font-size:14px">단지 반복거래 누수 차단 → 가장 보수적</p>
</div>

<div class="card">
<div class="badge">③ 시간순</div>
<h3>Chronological</h3>
<p>학습 <strong>≤ 2023.12</strong> / 검증 <strong>2024.01–06</strong> / 테스트 <strong>2024.07 이후</strong></p>
<p style="margin-top:12px;color:#805AD5;font-size:14px"><strong>측정 대상:</strong> 미래 시점 예측</p>
<p style="font-size:14px">시장 국면 이동(금리 인하 전환) 반영</p>
</div>

</div>

<div class="callout warm" style="margin-top:16px">
<strong>이중 기준 해석</strong> — 무작위 R²는 "동일 단지 내", Group/시간순 R²는 "미경험 단지·미래 시점". 두 기준 병행으로 <em>낙관 편향</em> 완화
</div>

---

<!-- _class: divider -->

<div class="section-num">04</div>

# 실증 분석 결과

<div class="items">
1. 기술통계 및 지역 분포<br/>
2. 상관관계 · VIF · Ablation<br/>
3. Moran's I 공간 자기상관<br/>
4. 모형별 · 분할별 성능<br/>
5. SHAP 글로벌 + 지역 별도
</div>

---

<!-- _class: content -->

<div data-section="04 실증 · 기술통계"></div>

## 단위면적당 가격의 지역 분포

<div class="stat-grid">

<div class="stat-card dark">
<div class="number">1,338<span style="font-size:36px"> 만원/㎡</span></div>
<div class="label">전체 평균</div>
<div class="sub">n = 391,826 · 중앙값 1,143</div>
</div>

<div class="stat-card">
<div class="number">2,208</div>
<div class="label">강남3구 평균</div>
<div class="sub">n = 65,077 · 중앙값 2,040</div>
</div>

<div class="stat-card accent">
<div class="number">1,165</div>
<div class="label">비강남 22구 평균</div>
<div class="sub">n = 326,749 · 중앙값 1,054</div>
</div>

</div>

<div class="mega-stat" style="margin-top:32px">
<div class="big">1.90<span style="font-size:48px">×</span></div>
<div class="desc">
<strong>강남 : 비강남 단위가격 배율</strong><br/>
총가격 배율 <em>2.20×</em>보다 작음 → 강남에 대형 평형이 상대적으로 많이 분포하여<br/>
총가격 격차의 일부가 규모효과에서 비롯됨을 단위면적 정규화가 실증.
</div>
</div>

---

<!-- _class: content -->

<div data-section="04 실증 · 모형 성능"></div>

## 모형별 · 분할별 예측 성능

<table>
<thead>
<tr><th>분할</th><th>OLS R²</th><th>RF R²</th><th>XGB R²</th><th>XGB Median APE</th></tr>
</thead>
<tbody>
<tr><td>무작위 (동일 단지 내)</td><td>0.5061</td><td>0.8937</td><td><strong>0.9554</strong></td><td><strong>4.16%</strong></td></tr>
<tr><td>Group (미경험 단지)</td><td>0.4618</td><td>0.7414</td><td><strong>0.8005</strong></td><td>11.72%</td></tr>
<tr><td>시간순 (미래 시점)</td><td>0.3939</td><td>0.6955</td><td><strong>0.7972</strong></td><td>12.61%</td></tr>
</tbody>
</table>

<div class="features" style="margin-top:20px">

<div class="card">
<div class="badge">관찰 1</div>
<h3>일관된 서열</h3>
<p>모든 분할에서 <strong>XGB &gt; RF &gt;&gt; OLS</strong> — 비선형 모형의 구조적 우위</p>
</div>

<div class="card">
<div class="badge">관찰 2</div>
<h3>Group ≈ 시간순</h3>
<p>두 분할의 R² 차이 <strong>0.003</strong> — 단지 누수 &gt; 국면 이동 효과</p>
</div>

<div class="card">
<div class="badge">관찰 3</div>
<h3>실무 일반화 기준선</h3>
<p>미경험 단지 예측 <strong>R² ≈ 0.80, Median APE ≈ 12%</strong></p>
</div>

</div>

---

<!-- _class: content -->

<div data-section="04 실증 · SHAP 재편"></div>

## SHAP Top 6 — 규모 정규화가 가져온 재편

<div class="compare">

<div class="before">
<h3>v6 총가격 모형</h3>

<table style="font-size:18px">
<tr><td>1. 전용면적</td><td><strong>22.2%</strong></td></tr>
<tr><td>2. 강남구분</td><td>16.0%</td></tr>
<tr><td>3. 건물연령</td><td>11.4%</td></tr>
<tr><td>4. M2 통화량</td><td>8.3%</td></tr>
<tr><td>5. 백화점수</td><td>6.8%</td></tr>
</table>

<p style="margin-top:16px;color:#E53E3E;font-size:16px">
<strong>면적이 SHAP 상위 독점</strong> → 질적·입지적 신호 가려짐
</p>

</div>

<div class="arrow">→</div>

<div class="after">
<h3>v7 단위가격 모형</h3>

<table style="font-size:18px">
<tr><td>1. 건물연령</td><td><strong>12.4%</strong></td></tr>
<tr><td>2. 강남구분</td><td>11.7%</td></tr>
<tr style="background:#FFF4E0"><td>3. 어린이집수 ⭐</td><td><strong>11.5%</strong></td></tr>
<tr><td>4. M2 통화량</td><td>10.8%</td></tr>
<tr style="background:#FFF4E0"><td>5. 학원수 ⭐</td><td><strong>9.6%</strong></td></tr>
<tr><td>6. 전용면적</td><td>9.6% <span style="color:#718096">(1위→6위)</span></td></tr>
</table>

<p style="margin-top:16px;color:#006B7F;font-size:16px">
<strong>주거권 · 사교육 · 재건축</strong> 신호 전면화
</p>

</div>

</div>

---

<!-- _class: content -->

<div data-section="04 실증 · 지역 별도 모형"></div>

## 지역 별도 모형 — 질적으로 다른 두 시장

<div class="split">

<div class="right" style="border-left-color:#FF6B6B">
<h3 style="color:#C53030">🏙️ 강남 3구 <span class="tag accent">n = 65,077</span></h3>

<p><strong>XGB R² = 0.9356 · Median APE 4.10%</strong></p>

<table style="font-size:17px;margin-top:12px">
<tr><th>순위</th><th>변수</th><th>비중</th></tr>
<tr><td>1</td><td>건물연령</td><td><strong>16.5%</strong></td></tr>
<tr><td>2</td><td>소비자물가</td><td>13.3%</td></tr>
<tr><td>3</td><td>백화점수</td><td>13.0%</td></tr>
<tr><td>4</td><td>전용면적</td><td>12.7%</td></tr>
<tr><td>5</td><td>M2</td><td>8.5%</td></tr>
<tr style="color:#A0AEC0"><td>15/17</td><td>어린이집수</td><td>1.7%</td></tr>
</table>

<p style="margin-top:16px;color:#C53030"><strong>주도 축:</strong> 재건축 기대 · 상업 집적</p>

</div>

<div class="right" style="border-left-color:#08A5C1">
<h3 style="color:#006B7F">🏘️ 비강남 22구 <span class="tag">n = 326,749</span></h3>

<p><strong>XGB R² = 0.9421 · Median APE 4.07%</strong></p>

<table style="font-size:17px;margin-top:12px">
<tr><th>순위</th><th>변수</th><th>비중</th></tr>
<tr><td>1</td><td>건물연령</td><td>14.8%</td></tr>
<tr><td>2</td><td>소비자물가</td><td>11.8%</td></tr>
<tr><td>3</td><td>전용면적</td><td>11.2%</td></tr>
<tr style="background:#E6F7FA"><td>4</td><td>어린이집수</td><td><strong>10.6%</strong></td></tr>
<tr><td>5</td><td>M2</td><td>10.1%</td></tr>
<tr style="background:#E6F7FA"><td>6</td><td>학원수</td><td><strong>8.6%</strong></td></tr>
</table>

<p style="margin-top:16px;color:#006B7F"><strong>주도 축:</strong> 주거권 · 사교육 · 거시 유동성</p>

</div>

</div>

---

<!-- _class: content -->

<div data-section="04 실증 · 강건성"></div>

## 강건성 점검 — Ablation + Moran's I

<div class="split">

<div class="right">
<h3>Ablation — 프록시 변수 제거</h3>

<p>학원수 · 어린이집수 (구 단위 프록시) 제거 시 R² 변화</p>

<table style="margin-top:16px;font-size:18px">
<tr><th>모형</th><th>ΔR²_log</th></tr>
<tr><td>OLS</td><td style="color:#E53E3E">−0.0542</td></tr>
<tr><td>Random Forest</td><td>−0.0062</td></tr>
<tr style="background:#E6F7FA"><td>XGBoost</td><td style="color:#006B7F"><strong>−0.0004</strong></td></tr>
</table>

<p style="margin-top:16px;font-size:16px"><strong>→ XGB는 두 프록시 없이도 동등 구조 유지</strong><br/>
구 단위 배분의 측정 한계가 전체 결과를 왜곡하지 않음</p>
</div>

<div class="right">
<h3>Moran's I — 잔차 공간 자기상관</h3>

<p>214개 행정동 평균 잔차 · 같은 구 내 인접 가중 · 499회 순열</p>

<table style="margin-top:16px;font-size:18px">
<tr><th>모형</th><th>I</th><th>Z</th><th>p</th></tr>
<tr><td>OLS</td><td style="color:#E53E3E"><strong>0.3247</strong></td><td>8.882</td><td>&lt; 0.001</td></tr>
<tr style="background:#E6F7FA"><td>XGBoost</td><td style="color:#006B7F">0.0042</td><td>0.252</td><td>0.880</td></tr>
</table>

<p style="margin-top:16px;font-size:16px"><strong>→ XGB의 비선형 조합이 공간 구조 흡수</strong><br/>
공간계량 모형 없이도 공간 이질성 대응</p>
</div>

</div>

---

<!-- _class: divider -->

<div class="section-num">05</div>

# 결론 및 시사점

<div class="items">
1. 학술 · 정책 · 실무 시사점<br/>
2. 연구의 한계 및 향후 과제
</div>

---

<!-- _class: content -->

<div data-section="05 결론 · 시사점"></div>

## 3층위 시사점

<div class="features">

<div class="card">
<div class="badge">학술적</div>
<h3>📚 XAI 국내 적용 확장</h3>
<p>단위면적 정규화 + 행정동 세분화 + 지역 별도 모형의 3축 결합은 국내 부동산 XAI 문헌 최초 적용.</p>
<p>공간 이질성의 질적 구조 직접 실증.</p>
</div>

<div class="card">
<div class="badge">정책적</div>
<h3>🏛️ 이원 시장 설계</h3>
<p>강남 = 재건축·상업, 비강남 = 주거권·사교육.<br/>가격 수준만 다른 것이 아닌 <strong>결정 구조 자체가 상이</strong>.</p>
<p>정책 수단의 지역별 차별적 파급효과 시사.</p>
</div>

<div class="card">
<div class="badge">실무적</div>
<h3>💼 AVM 설명 책무</h3>
<p>단위가격 SHAP 분해 → 면적에 가려지지 않은 질적 프리미엄 투명화.<br/>프롭테크·감정평가 <em>설명 책무</em> 지원.</p>
</div>

</div>

<div class="callout">
<strong>핵심 메시지</strong> — 단위면적 정규화로 <strong>주거권·사교육·재건축 신호</strong>가 전면화되었고, SHAP Dependence에서 <strong>U자형·체감형 비선형 패턴</strong>이 드러났으며, 지역별 설명 구조는 <em>질적으로 상이</em>함이 실증되었다.
</div>

---

<!-- _class: content -->

<div data-section="05 결론 · 한계와 과제"></div>

## 연구의 한계 및 향후 과제

<div class="split">

<div class="right" style="border-left-color:#FBB040">
<h3>⚠️ 현재 한계</h3>

<ul style="margin-top:16px">
<li><strong>시설 변수 stock 병합</strong> — 연도별 변동 미반영 <span class="tag warm">교수 지적</span></li>
<li><strong>학원·어린이집 프록시</strong> — 구 단위 균등 배분 한계</li>
<li><strong>질적 변수 미반영</strong> — 브랜드·조망·재건축 추진 단계</li>
<li><strong>Moran's I 간이 가중</strong> — 정밀 경계 기반 미적용</li>
<li><strong>공간계량(SAR/SEM)</strong> 직접 비교 미수행</li>
<li><strong>강남 표본 규모</strong> — 비강남의 1/5</li>
</ul>

</div>

<div class="right">
<h3>🚀 향후 과제</h3>

<ul style="margin-top:16px">
<li><strong>연도별 시설 패널</strong> — 행안부·교육부 연간 스냅샷</li>
<li><strong>GIS 최근접 거리</strong> — nearest distance 변수</li>
<li><strong>Queen/Rook 가중</strong> — 경계 기반 정밀 공간 행렬</li>
<li><strong>ML 확장</strong> — LightGBM · CatBoost · TabNet</li>
<li><strong>XAI 교차</strong> — LIME · ALE · Permutation Importance</li>
<li><strong>범위 확장</strong> — 수도권·지방 대도시로 비교</li>
</ul>

</div>

</div>

---

<!-- _class: thanks -->
<!-- _paginate: false -->

# 감사합니다

경청해 주셔서 감사드립니다.

<p style="margin-top:80px;font-size:20px;color:#A0AEC0">
질문 · 피드백 환영합니다
</p>

<p style="font-size:18px;color:#08A5C1;margin-top:40px">
박 현 근 · 한양대학교 부동산융합대학원
</p>
