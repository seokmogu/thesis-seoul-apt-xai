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

# XGBoost와 SHAP으로 본<br/><em>서울 아파트 단위면적당 가격</em>

<div class="subtitle">비선형 설명 구조 · 행정동 · 지역 별도 모형</div>

<div class="meta">
한양대학교 부동산융합대학원 · 도시부동산정책전공<br/>
<strong>석사학위 중간발표 · 박 현 근</strong><br/>
<span style="font-size:18px">2026년 4월 · 5분 발표</span>
</div>

---

<!-- _class: content -->

<div data-section="01 연구 차별성"></div>

## 왜 단위면적당 가격인가 — 3가지 차별성

<div class="features">

<div class="card">
<div class="badge">축 1</div>
<h3>단위면적 정규화</h3>
<p><strong>Y = log(거래금액 / 전용면적)</strong></p>
<p>총가격 모형에서 면적 1변수가 SHAP 22% 독점 → 규모효과 제거로 <strong>질적·입지적 신호</strong> 전면화</p>
</div>

<div class="card">
<div class="badge">축 2</div>
<h3>행정동 세분화</h3>
<p><strong>215개 행정동 단위</strong> (자치구 대비 8.6배)</p>
<p>Nominatim 지오코딩 + GeoJSON 공간조인</p>
</div>

<div class="card">
<div class="badge">축 3</div>
<h3>지역 별도 3모형</h3>
<p><strong>전체 · 강남3구 · 비강남 22개구</strong></p>
<p>강남더미 제거 17변수 독립 적합 → 지역 내부 설명 구조 <strong>직접 비교</strong></p>
</div>

</div>

<div class="callout">
기존 ML 논문과의 차별: ① 종속변수 정규화  ② 행정동 세분화  ③ 지역 3모형 <strong>동시 결합</strong> — 국내 최초
</div>

---

<!-- _class: content -->

<div data-section="02 데이터 · 설계"></div>

## 데이터 · 설계 요약

<div class="stat-grid">

<div class="stat-card">
<div class="number">391,826</div>
<div class="label">실거래 (2019–2025)</div>
<div class="sub">84개월, COVID·금리 인상·인하 전환</div>
</div>

<div class="stat-card accent">
<div class="number">215</div>
<div class="label">행정동</div>
<div class="sub">25 자치구 세분화</div>
</div>

<div class="stat-card warm">
<div class="number">18</div>
<div class="label">독립변수</div>
<div class="sub">물리3·입지2·환경9·거시4</div>
</div>

</div>

<div class="features" style="margin-top:16px">

<div class="card">
<div class="badge">분할 ①</div>
<h3>무작위 70/10/20</h3>
<p>동일 단지 내 예측 성능</p>
</div>

<div class="card">
<div class="badge">분할 ②</div>
<h3>Group (단지 기준)</h3>
<p>법정동+아파트명 8,601단지 · 미경험 단지 일반화</p>
</div>

<div class="card">
<div class="badge">분할 ③</div>
<h3>시간순</h3>
<p>2024.07 이후 테스트 · 미래 시점 예측</p>
</div>

</div>

---

<!-- _class: content -->

<div data-section="03 모형 성능"></div>

## 모형별·분할별 예측 성능

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
<p>모든 분할에서 <strong>XGB &gt; RF &gt;&gt; OLS</strong> — 비선형의 구조적 우위</p>
</div>

<div class="card">
<div class="badge">관찰 2</div>
<h3>비선형 설명력</h3>
<p>XGB − OLS 격차 <strong>+0.45p</strong> — OLS가 포착 못 한 비선형 조합이 주된 설명원</p>
</div>

<div class="card">
<div class="badge">관찰 3</div>
<h3>실무 일반화</h3>
<p>미경험 단지 예측 R² ≈ 0.80, Median APE ≈ 12%</p>
</div>

</div>

---

<!-- _class: content -->

<div data-section="04 SHAP 재편"></div>

## SHAP Top 6 — 규모 정규화가 가져온 재편

<div class="compare">

<div class="before">
<h3>v6 총가격 모형</h3>

<table style="font-size:16px">
<tr><td>1. 전용면적</td><td><strong>22.2%</strong></td></tr>
<tr><td>2. 강남구분</td><td>16.0%</td></tr>
<tr><td>3. 건물연령</td><td>11.4%</td></tr>
<tr><td>4. M2 통화량</td><td>8.3%</td></tr>
<tr><td>5. 백화점수</td><td>6.8%</td></tr>
</table>

<p style="margin-top:12px;color:#E53E3E;font-size:15px">
<strong>면적이 SHAP 독점</strong> → 질적·입지적 신호 가려짐
</p>

</div>

<div class="arrow">→</div>

<div class="after">
<h3>v7 단위가격 모형</h3>

<table style="font-size:16px">
<tr><td>1. 건물연령</td><td><strong>12.4%</strong></td></tr>
<tr><td>2. 강남구분</td><td>11.7%</td></tr>
<tr style="background:#FFF4E0"><td>3. 어린이집수 ⭐</td><td><strong>11.5%</strong></td></tr>
<tr><td>4. M2 통화량</td><td>10.8%</td></tr>
<tr style="background:#FFF4E0"><td>5. 학원수 ⭐</td><td><strong>9.6%</strong></td></tr>
<tr><td>6. 전용면적 <span style="color:#718096">(1→6)</span></td><td>9.6%</td></tr>
</table>

<p style="margin-top:12px;color:#006B7F;font-size:15px">
<strong>주거권 · 사교육 · 재건축</strong> 신호 전면화
</p>

</div>

</div>

---

<!-- _class: content -->

<div data-section="05 지역 3모형"></div>

## 지역 별도 3모형 — 질적으로 다른 두 시장

<div class="split">

<div class="right" style="border-left-color:#FF6B6B">
<h3 style="color:#C53030">🏙️ 강남 3구 <span class="tag accent">n=65,077</span></h3>
<p><strong>XGB R²=0.9356 · MedAPE 4.10%</strong></p>
<table style="font-size:15px;margin-top:8px">
<tr><th>순위</th><th>변수</th><th>비중</th></tr>
<tr><td>1</td><td>건물연령</td><td><strong>16.5%</strong></td></tr>
<tr><td>2</td><td>소비자물가</td><td>13.3%</td></tr>
<tr><td>3</td><td>백화점수</td><td>13.0%</td></tr>
<tr><td>4</td><td>전용면적</td><td>12.7%</td></tr>
<tr style="color:#A0AEC0"><td>15/17</td><td>어린이집수</td><td>1.7%</td></tr>
</table>
<p style="margin-top:10px;color:#C53030;font-size:15px"><strong>주도 축:</strong> 재건축 · 상업 집적</p>
</div>

<div class="right" style="border-left-color:#08A5C1">
<h3 style="color:#006B7F">🏘️ 비강남 22구 <span class="tag">n=326,749</span></h3>
<p><strong>XGB R²=0.9421 · MedAPE 4.07%</strong></p>
<table style="font-size:15px;margin-top:8px">
<tr><th>순위</th><th>변수</th><th>비중</th></tr>
<tr><td>1</td><td>건물연령</td><td>14.8%</td></tr>
<tr><td>2</td><td>소비자물가</td><td>11.8%</td></tr>
<tr><td>3</td><td>전용면적</td><td>11.2%</td></tr>
<tr style="background:#E6F7FA"><td>4</td><td>어린이집수</td><td><strong>10.6%</strong></td></tr>
<tr style="background:#E6F7FA"><td>6</td><td>학원수</td><td><strong>8.6%</strong></td></tr>
</table>
<p style="margin-top:10px;color:#006B7F;font-size:15px"><strong>주도 축:</strong> 주거권 · 사교육 · 거시</p>
</div>

</div>

---

<!-- _class: content -->

<div data-section="06 비선형 실증"></div>

## 비선형 설명 구조의 증거

<div class="split">

<div class="right">
<h3>🎯 SHAP Dependence — U자형 건물연령</h3>
<p style="font-size:15px">연령 증가 → SHAP 감소, 그러나 <strong>25~30년 이상 구간에서 반등</strong>. OLS 선형 가정이 포착 못 하는 재건축 기대 신호.</p>
<p style="font-size:14px;color:#718096;margin-top:8px">전용면적: 소형(+)↔대형(−) 전환, 규모 정규화 후 잔여 체감형 비선형</p>
</div>

<div class="right">
<h3>🧪 Moran's I · Ablation</h3>
<table style="font-size:16px;margin-top:8px">
<tr><th></th><th>OLS</th><th>XGB</th></tr>
<tr><td>Moran I (잔차)</td><td style="color:#E53E3E">0.325</td><td style="color:#006B7F">0.004</td></tr>
<tr><td>Moran p-value</td><td>&lt;0.001</td><td>0.880</td></tr>
<tr><td>Ablation ΔR²</td><td style="color:#E53E3E">−0.054</td><td style="color:#006B7F">−0.0004</td></tr>
</table>
<p style="font-size:14px;margin-top:10px"><strong>→ XGB의 비선형 조합</strong>이 공간 구조·프록시 한계를 모두 흡수</p>
</div>

</div>

<div class="callout">
본 연구의 핵심은 <strong>비선형 설명 구조</strong> — 선형 OLS가 포착 못 한 U자·체감·임계 패턴을 XGBoost+SHAP으로 실증
</div>

---

<!-- _class: content -->

<div data-section="07 결론 · 한계"></div>

## 결론 · 한계 · 향후 과제

<div class="features">

<div class="card">
<div class="badge">학술</div>
<h3>📚 국내 최초 3축 결합</h3>
<p>단위면적 정규화 × 행정동 × 지역 3모형을 하나의 XAI 프레임워크에 결합</p>
</div>

<div class="card">
<div class="badge">정책</div>
<h3>🏛️ 이원 시장</h3>
<p>강남=재건축·상업, 비강남=주거권·사교육 — 가격 수준이 아닌 <strong>결정 구조 자체가 상이</strong></p>
</div>

<div class="card">
<div class="badge">실무</div>
<h3>💼 AVM 설명 책무</h3>
<p>단위가격 SHAP 분해 → 면적에 가려지지 않은 질적 프리미엄 투명화</p>
</div>

</div>

<div class="split" style="margin-top:16px">

<div class="right" style="border-left-color:#FBB040;padding:16px 24px">
<h3 style="font-size:20px">⚠️ 한계</h3>
<ul style="font-size:15px;margin-top:4px">
<li>시설 변수 stock 병합 (연도별 변동 미반영) <span class="tag warm">교수 지적</span></li>
<li>학원·어린이집 구 단위 프록시 배분</li>
<li>공간계량 모형 직접 비교 미수행</li>
</ul>
</div>

<div class="right" style="padding:16px 24px">
<h3 style="font-size:20px">🚀 향후</h3>
<ul style="font-size:15px;margin-top:4px">
<li>연도별 시설 패널 (행안부·교육부)</li>
<li>GIS 최근접 거리 변수 · 경계 기반 Queen 가중</li>
<li>평형대(60/85/132㎡)별 보조 분석</li>
</ul>
</div>

</div>

---

<!-- _class: thanks -->
<!-- _paginate: false -->

# 감사합니다

<p style="margin-top:60px;font-size:24px;color:#A0AEC0">
경청해 주셔서 감사드립니다 · 질의 환영
</p>

<p style="font-size:20px;color:#08A5C1;margin-top:40px">
박 현 근 · 한양대학교 부동산융합대학원
</p>
