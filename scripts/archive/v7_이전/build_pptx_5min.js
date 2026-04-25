// 논문 v7 기반 중간발표 5분용 10장 PPTX
// 논문 "XGBoost와 SHAP을 활용한 서울시 아파트 단위면적당 매매가격의 설명 패턴 분석"
// 2026-04 박현근 · 한양대 부동산융합대학원
// pptxgenjs (Anthropic PPTX 스킬 기반) · 인치 단위 명시 좌표

const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";   // 13.333 × 7.5 inches
pres.title = "서울시 아파트 단위면적당 매매가격의 설명 패턴 분석";
pres.author = "박현근";

const W = 13.333;
const H = 7.5;

// Palette
const C = {
  bgDark:   "0A1628",
  bgMid:    "1A365D",
  primary:  "08A5C1",
  primaryD: "006B7F",
  primaryL: "E6F7FA",
  accent:   "FF6B6B",
  accentD:  "C53030",
  warm:     "FBB040",
  warmL:    "FFF4E0",
  textD:    "1A202C",
  textM:    "4A5568",
  muted:    "A0AEC0",
  border:   "E2E8F0",
  bgSoft:   "F7F9FC",
  white:    "FFFFFF",
};
const shadow = () => ({ type: "outer", color: "000000", blur: 6, offset: 2, angle: 135, opacity: 0.08 });
const FONT = "Pretendard";
const FONT_NUM = "Montserrat";

// 공통 헤더
function header(slide, tag, title) {
  slide.addText(tag, {
    x: 0.5, y: 0.32, w: 12.3, h: 0.28,
    fontSize: 10, fontFace: FONT, color: C.primary, bold: true, charSpacing: 2, margin: 0,
  });
  slide.addText(title, {
    x: 0.5, y: 0.6, w: 12.3, h: 0.78,
    fontSize: 30, fontFace: FONT, color: C.textD, bold: true, margin: 0,
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.42, w: 0.55, h: 0.04,
    fill: { color: C.primary }, line: { color: C.primary },
  });
}

function card(slide, x, y, w, h, opts = {}) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: opts.fill || C.white }, line: { color: opts.border || C.border, width: 1 },
    shadow: shadow(),
  });
  if (opts.accent) {
    slide.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 0.08, h,
      fill: { color: opts.accent }, line: { color: opts.accent },
    });
  }
}

function badge(slide, x, y, text, w = 0.9, bg = C.primaryL, fg = C.primaryD) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h: 0.3,
    fill: { color: bg }, line: { color: bg },
  });
  slide.addText(text, {
    x, y, w, h: 0.3,
    fontSize: 10, fontFace: FONT, color: fg, bold: true, align: "center", valign: "middle", margin: 0,
  });
}

function pageNum(slide, n) {
  slide.addText(`${n} / 10`, {
    x: W - 1.2, y: H - 0.4, w: 0.9, h: 0.3,
    fontSize: 10, fontFace: FONT, color: C.muted, align: "right", margin: 0,
  });
}

// ============================================================
// 1. 표지
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bgDark };
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: W, h: 0.1,
    fill: { color: C.primary }, line: { color: C.primary },
  });

  s.addText("XGBoost와 SHAP을 활용한", {
    x: 0.8, y: 1.9, w: 11.8, h: 0.7,
    fontSize: 28, fontFace: FONT, color: C.white, bold: true, align: "center", valign: "middle", margin: 0,
  });
  s.addText([
    { text: "서울시 아파트 ", options: { color: C.white } },
    { text: "단위면적당 매매가격", options: { color: C.primary } },
    { text: "의", options: { color: C.white } },
  ], {
    x: 0.8, y: 2.7, w: 11.8, h: 1.0,
    fontSize: 44, fontFace: FONT, bold: true, align: "center", valign: "middle", margin: 0,
  });
  s.addText("설명 패턴 분석", {
    x: 0.8, y: 3.75, w: 11.8, h: 0.9,
    fontSize: 44, fontFace: FONT, color: C.white, bold: true, align: "center", valign: "middle", margin: 0,
  });
  s.addText("Explanatory Patterns of Apartment Unit-Area Sale Prices in Seoul", {
    x: 0.8, y: 4.8, w: 11.8, h: 0.4,
    fontSize: 14, fontFace: FONT, color: C.muted, italic: true, align: "center", valign: "middle", margin: 0,
  });

  s.addText([
    { text: "한양대학교 부동산융합대학원 · 도시부동산정책전공", options: { color: C.muted, fontSize: 15, breakLine: true } },
    { text: "석사학위 중간발표", options: { color: C.muted, fontSize: 15, breakLine: true } },
    { text: " " , options: { fontSize: 6, breakLine: true } },
    { text: "박 현 근", options: { color: C.primary, fontSize: 22, bold: true, breakLine: true } },
    { text: "2026. 04", options: { color: C.muted, fontSize: 13 } },
  ], { x: 0.8, y: 5.6, w: 11.8, h: 1.7, fontFace: FONT, align: "center", valign: "top", margin: 0, paraSpaceAfter: 2 });
}

// ============================================================
// 2. 연구 배경 · 3대 공백 · 차별성 3축
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bgSoft };
  header(s, "01  서 론", "연구 배경 · 기존 문헌 3대 공백 · 본 연구의 차별성");

  // 상단 좌: 배경 수치 (한국 가계 실물자산 75.2%)
  card(s, 0.7, 1.7, 4.0, 2.4);
  s.addText("75.2%", {
    x: 0.9, y: 1.85, w: 3.6, h: 1.0,
    fontSize: 56, fontFace: FONT_NUM, color: C.primary, bold: true, margin: 0,
  });
  s.addText("한국 가계 자산 중 실물자산 비중", {
    x: 0.9, y: 2.9, w: 3.6, h: 0.4,
    fontSize: 13, fontFace: FONT, color: C.textD, bold: true, margin: 0,
  });
  s.addText("통계청(2024) 가계금융복지조사. 서울 아파트는 자산 축적·주거 안정의 핵심 매개로, 가격 결정 구조 이해가 학술·정책적 과제.", {
    x: 0.9, y: 3.3, w: 3.6, h: 0.75,
    fontSize: 11, fontFace: FONT, color: C.textM, margin: 0,
  });

  // 상단 우: 기존 문헌 3대 공백
  card(s, 4.9, 1.7, 7.7, 2.4, { accent: C.accent });
  s.addText("기존 국내 서울 아파트 ML 연구의 3대 공백", {
    x: 5.1, y: 1.85, w: 7.4, h: 0.4,
    fontSize: 16, fontFace: FONT, color: C.accentD, bold: true, margin: 0,
  });
  const gaps = [
    ["공백 1", "공간 단위 거친 집계", "대부분 자치구(25개) 단위 → 동네 수준 이질성 유실"],
    ["공백 2", "단일 모형 예측 비교", "OLS vs ML 성능 비교에 머무름 → SHAP 해석 체계 부재"],
    ["공백 3", "총가격 모형의 면적 독점", "면적이 SHAP 상위 22% 독점 → 질적·입지적 신호 가려짐"],
  ];
  gaps.forEach(([tag, title, body], i) => {
    const gx = 5.1 + i * 2.5, gy = 2.35;
    badge(s, gx, gy, tag, 0.6, "FFE5E5", C.accentD);
    s.addText(title, {
      x: gx, y: gy + 0.4, w: 2.4, h: 0.4,
      fontSize: 13, fontFace: FONT, color: C.textD, bold: true, margin: 0,
    });
    s.addText(body, {
      x: gx, y: gy + 0.8, w: 2.4, h: 0.95,
      fontSize: 10.5, fontFace: FONT, color: C.textM, margin: 0,
    });
  });

  // 하단: 본 연구 차별성 3축
  const axes = [
    { tag: "축 1", title: "단위면적 정규화",
      lead: "Y = ln(거래금액[만원] / 전용면적[㎡])",
      body: "규모효과를 분모로 제거. 소형·대형 평형 간 단위가격 프리미엄을 독립변수로 보존(Malpezzi 2003, Sirmans et al. 2005 log-price 전통)." },
    { tag: "축 2", title: "행정동 215개 세분화",
      lead: "자치구 25개 대비 약 8.6배",
      body: "Nominatim 지오코딩 + GeoJSON 공간조인으로 법정동→행정동 매핑 파이프라인 구축. 세분화된 공간 단위에서 설명 패턴 탐색." },
    { tag: "축 3", title: "지역 별도 3모형",
      lead: "전체 · 강남3구 · 비강남 22개구",
      body: "강남더미 제거한 17변수로 각 지역 독립 적합. 통합 모형 SHAP 재집계가 아닌 별도 모형으로 공간 이질성 실증." },
  ];
  const aW = 3.95, aH = 2.85, aGap = 0.14;
  const aStart = (W - aW * 3 - aGap * 2) / 2;
  axes.forEach((a, i) => {
    const x = aStart + i * (aW + aGap), y = 4.3;
    card(s, x, y, aW, aH, { accent: C.primary });
    badge(s, x + 0.25, y + 0.25, a.tag, 0.65);
    s.addText(a.title, {
      x: x + 0.3, y: y + 0.65, w: aW - 0.6, h: 0.45,
      fontSize: 16, fontFace: FONT, color: C.textD, bold: true, margin: 0,
    });
    s.addText(a.lead, {
      x: x + 0.3, y: y + 1.1, w: aW - 0.6, h: 0.5,
      fontSize: 12, fontFace: FONT, color: C.primaryD, bold: true, margin: 0,
    });
    s.addText(a.body, {
      x: x + 0.3, y: y + 1.65, w: aW - 0.6, h: 1.15,
      fontSize: 11, fontFace: FONT, color: C.textM, margin: 0,
    });
  });

  pageNum(s, 2);
}

// ============================================================
// 3. 데이터 · 변수 체계 (18개 4범주)
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bgSoft };
  header(s, "02  연구 설계", "데이터 · 18개 독립변수 체계");

  // 상단 3개 stat
  const stats = [
    { num: "391,826", color: C.primary, label: "서울 아파트 매매 실거래", sub: "2019.01 ~ 2025.12 (84개월)" },
    { num: "215",     color: C.accent,  label: "서울시 행정동",             sub: "자치구 25개 8.6배 세분화" },
    { num: "18",      color: C.warm,    label: "독립변수",                  sub: "물리 3 · 입지 2 · 환경 9 · 거시 4" },
  ];
  const sW = 3.95, sH = 1.8, sGap = 0.14;
  const sStart = (W - sW * 3 - sGap * 2) / 2;
  stats.forEach((st, i) => {
    const x = sStart + i * (sW + sGap), y = 1.7;
    card(s, x, y, sW, sH);
    s.addText(st.num, { x: x + 0.3, y: y + 0.18, w: sW - 0.6, h: 0.85,
      fontSize: 42, fontFace: FONT_NUM, color: st.color, bold: true, margin: 0 });
    s.addText(st.label, { x: x + 0.3, y: y + 1.05, w: sW - 0.6, h: 0.35,
      fontSize: 13, fontFace: FONT, color: C.textD, bold: true, margin: 0 });
    s.addText(st.sub, { x: x + 0.3, y: y + 1.4, w: sW - 0.6, h: 0.35,
      fontSize: 10.5, fontFace: FONT, color: C.muted, margin: 0 });
  });

  // 하단 변수 4범주
  const cats = [
    { tag: "물리", n: 3, color: C.primary,
      items: ["전용면적(㎡)", "층(층)", "건물연령(년)"] },
    { tag: "입지", n: 2, color: C.primaryD,
      items: ["강남구분(더미)", "지하철역수(개)"] },
    { tag: "환경", n: 9, color: C.warm,
      items: ["초/중/고교수", "CCTV수", "백화점수(매장)", "공원/도서관수", "학원수/어린이집수 (구 단위 프록시)"] },
    { tag: "거시", n: 4, color: C.accent,
      items: ["기준금리(%)", "CD금리(%)", "소비자물가지수", "M2 광의통화"] },
  ];
  const cW = 2.96, cH = 3.35, cGap = 0.13;
  const cStart = (W - cW * 4 - cGap * 3) / 2;
  cats.forEach((cat, i) => {
    const x = cStart + i * (cW + cGap), y = 3.7;
    card(s, x, y, cW, cH, { accent: cat.color });
    // 뱃지 + N
    s.addText(`${cat.tag}   ${cat.n}`, {
      x: x + 0.25, y: y + 0.25, w: cW - 0.5, h: 0.4,
      fontSize: 15, fontFace: FONT, color: cat.color, bold: true, margin: 0,
    });
    const items = cat.items.map((it, k) => ({
      text: it, options: { bullet: true, breakLine: k < cat.items.length - 1 },
    }));
    s.addText(items, {
      x: x + 0.3, y: y + 0.7, w: cW - 0.55, h: cH - 0.9,
      fontSize: 11.5, fontFace: FONT, color: C.textM, margin: 0, paraSpaceAfter: 2,
    });
  });

  // 하단 안내
  s.addText("종속변수: Y = ln(거래금액 ÷ 전용면적). 계수 β → 준탄력성 (e^β − 1)·100% 로 해석",
    { x: 0.5, y: 7.1, w: 12.3, h: 0.3,
      fontSize: 12, fontFace: FONT, color: C.textM, align: "center", italic: true, margin: 0 });

  pageNum(s, 3);
}

// ============================================================
// 4. 분석 방법 · 3가지 데이터 분할
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bgSoft };
  header(s, "02  연구 설계", "3가지 데이터 분할 · OLS · RF · XGBoost · SHAP");

  // 상단 3개 분할 방식
  const splits = [
    { tag: "① 무작위", title: "Random 70/10/20",
      stat: "학습 274,277 · 검증 39,183 · 테스트 78,366",
      meaning: "동일 단지 내 예측 성능", note: "동일 단지 반복거래 공존 가능 → 낙관적 추정",
      color: C.primary },
    { tag: "② Group", title: "단지 기준 분할",
      stat: "법정동 + 아파트명 기준 8,601 단지 · overlap = 0",
      meaning: "미경험 신규 단지 일반화 성능", note: "단지 누수 차단 → 가장 보수적",
      color: C.warm },
    { tag: "③ 시간순", title: "Chronological",
      stat: "학습 ≤ 2023.12 · 검증 2024.01–06 · 테스트 ≥ 2024.07",
      meaning: "미래 시점 예측 성능", note: "시장 국면 이동(금리 인하 전환) 반영",
      color: C.accent },
  ];
  const pW = 3.95, pH = 3.0, pGap = 0.14;
  const pStart = (W - pW * 3 - pGap * 2) / 2;
  splits.forEach((sp, i) => {
    const x = pStart + i * (pW + pGap), y = 1.7;
    card(s, x, y, pW, pH, { accent: sp.color });
    badge(s, x + 0.25, y + 0.25, sp.tag, 1.1);
    s.addText(sp.title, {
      x: x + 0.3, y: y + 0.65, w: pW - 0.6, h: 0.45,
      fontSize: 17, fontFace: FONT, color: C.textD, bold: true, margin: 0 });
    s.addText(sp.stat, {
      x: x + 0.3, y: y + 1.1, w: pW - 0.6, h: 0.75,
      fontSize: 11, fontFace: FONT, color: C.textM, margin: 0 });
    s.addText("📌 " + sp.meaning, {
      x: x + 0.3, y: y + 1.9, w: pW - 0.6, h: 0.4,
      fontSize: 12, fontFace: FONT, color: sp.color, bold: true, margin: 0 });
    s.addText(sp.note, {
      x: x + 0.3, y: y + 2.3, w: pW - 0.6, h: 0.6,
      fontSize: 10.5, fontFace: FONT, color: C.muted, italic: true, margin: 0 });
  });

  // 하단 모형 정의
  card(s, 0.7, 4.95, 12.1, 2.0);
  s.addText("세 모형 공통 평가 · 동일 하이퍼파라미터 · SHAP (TreeSHAP 5,000 샘플) 해석", {
    x: 0.95, y: 5.1, w: 11.7, h: 0.4,
    fontSize: 14, fontFace: FONT, color: C.textD, bold: true, margin: 0,
  });
  const models = [
    { name: "OLS",  desc: "선형 · 베이스라인", params: "준탄력성 해석, 잔차 진단 Moran's I", color: C.textM },
    { name: "Random Forest", desc: "배깅 · 분산 감소", params: "n=200, max_depth=15, min_leaf=5", color: C.primaryD },
    { name: "XGBoost", desc: "부스팅 · 편향 감소 · 대표 해석 모형", params: "max_depth=8, lr=0.1, rounds≤2000, early stop 50", color: C.primary },
  ];
  models.forEach((m, i) => {
    const x = 0.95 + i * 3.95, y = 5.6;
    s.addText(m.name, { x, y, w: 3.8, h: 0.35,
      fontSize: 15, fontFace: FONT, color: m.color, bold: true, margin: 0 });
    s.addText(m.desc, { x, y: y + 0.35, w: 3.8, h: 0.3,
      fontSize: 11, fontFace: FONT, color: C.textD, margin: 0 });
    s.addText(m.params, { x, y: y + 0.65, w: 3.8, h: 0.55,
      fontSize: 10, fontFace: FONT, color: C.muted, italic: true, margin: 0 });
  });

  pageNum(s, 4);
}

// ============================================================
// 5. 기술통계 · OLS 결과 핵심
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bgSoft };
  header(s, "03  실증 분석", "기술통계 · OLS 회귀 결과 (베이스라인)");

  // 상단 좌: 단위가격 지역 분포 (mega stat)
  const mx = 0.7, my = 1.7, mw = 6.3, mh = 2.25;
  s.addShape(pres.shapes.RECTANGLE, {
    x: mx, y: my, w: mw, h: mh,
    fill: { color: C.bgDark }, line: { color: C.bgDark }, shadow: shadow(),
  });
  s.addText("1.90×", {
    x: mx + 0.3, y: my + 0.4, w: 2.3, h: 1.4,
    fontSize: 72, fontFace: FONT_NUM, color: C.primary, bold: true, margin: 0, align: "left",
  });
  s.addText([
    { text: "강남 : 비강남 단위가격 배율", options: { color: C.white, fontSize: 14, bold: true, breakLine: true } },
    { text: " ", options: { fontSize: 4, breakLine: true } },
    { text: "전체 평균 1,337.7 만원/㎡ · 중앙값 1,142.7", options: { color: C.muted, fontSize: 12, breakLine: true } },
    { text: "강남3구 2,207.5 / 비강남 22구 1,164.5", options: { color: C.muted, fontSize: 12, breakLine: true } },
    { text: "총가격 배율 2.20×보다 작음 → 강남 대형 평형 집중이", options: { color: C.muted, fontSize: 11, breakLine: true } },
    { text: "총가격 격차의 일부를 설명함을 단위면적 정규화가 실증", options: { color: C.muted, fontSize: 11 } },
  ], { x: mx + 2.7, y: my + 0.3, w: mw - 3.0, h: mh - 0.6, fontFace: FONT, margin: 0, paraSpaceAfter: 2 });

  // 상단 우: OLS 요약 (카드)
  const ox = 7.2, oy = 1.7, ow = 5.6, oh = 2.25;
  card(s, ox, oy, ow, oh, { accent: C.warm });
  s.addText("OLS 베이스라인 결과", {
    x: ox + 0.25, y: oy + 0.2, w: ow - 0.5, h: 0.4,
    fontSize: 16, fontFace: FONT, color: C.textD, bold: true, margin: 0,
  });
  s.addText([
    { text: "R² = 0.506", options: { color: C.warm, bold: true, fontSize: 14 } },
    { text: "   |   Adj R² = 0.506   |   N = 391,826", options: { color: C.textM, fontSize: 12 } },
  ], { x: ox + 0.25, y: oy + 0.65, w: ow - 0.5, h: 0.4, fontFace: FONT, margin: 0 });
  s.addText("단위가격 총 변동의 약 50.6%만 18개 변수의 선형 결합으로 설명 → OLS 선형 가정이 비선형 가격구조(재건축·입지 상호작용·국면 전환)를 충분히 포착하지 못함을 시사. 설명적 기준선으로만 활용.", {
    x: ox + 0.25, y: oy + 1.05, w: ow - 0.5, h: 1.1,
    fontSize: 11, fontFace: FONT, color: C.textM, margin: 0,
  });

  // 하단: OLS 주요 계수 표
  const tx = 0.7, ty = 4.15, tw = 12.1, th = 2.85;
  card(s, tx, ty, tw, th);
  s.addText("OLS 주요 준탄력성 (log 계수 B, (e^B − 1)·100% 해석)", {
    x: tx + 0.25, y: ty + 0.2, w: tw - 0.5, h: 0.35,
    fontSize: 14, fontFace: FONT, color: C.textD, bold: true, margin: 0,
  });
  const olsRows = [
    [
      { text: "변수",     options: { bold: true, color: C.white, fill: { color: C.bgDark }, align: "left" } },
      { text: "B",        options: { bold: true, color: C.white, fill: { color: C.bgDark }, align: "center" } },
      { text: "t",        options: { bold: true, color: C.white, fill: { color: C.bgDark }, align: "center" } },
      { text: "준탄력성", options: { bold: true, color: C.white, fill: { color: C.bgDark }, align: "right" } },
      { text: "해석",     options: { bold: true, color: C.white, fill: { color: C.bgDark }, align: "left" } },
    ],
    [
      { text: "강남구분", options: { bold: true, color: C.textD } },
      { text: "+0.368",   options: { align: "center" } },
      { text: "177.7",    options: { align: "center" } },
      { text: "+44.46%",  options: { align: "right", color: C.primaryD, bold: true } },
      { text: "비강남 대비 강남 단위가격 차이", options: { color: C.textM } },
    ],
    [
      { text: "지하철역수", options: { bold: true, color: C.textD } },
      { text: "+0.061",    options: { align: "center" } },
      { text: "104.8",     options: { align: "center" } },
      { text: "+6.27%/역", options: { align: "right", color: C.primaryD, bold: true } },
      { text: "역세권 프리미엄", options: { color: C.textM } },
    ],
    [
      { text: "도서관수", options: { bold: true, color: C.textD } },
      { text: "+0.028",  options: { align: "center" } },
      { text: "27.3",    options: { align: "center" } },
      { text: "+2.79%/개", options: { align: "right", color: C.primaryD, bold: true } },
      { text: "문화 인프라 프리미엄", options: { color: C.textM } },
    ],
    [
      { text: "건물연령", options: { bold: true, color: C.textD } },
      { text: "−0.0035", options: { align: "center" } },
      { text: "−63.8",   options: { align: "center" } },
      { text: "−0.35%/년", options: { align: "right", color: C.accentD, bold: true } },
      { text: "연식 1년당 단위가격 감소 (U자 반등 구간 존재)", options: { color: C.textM } },
    ],
    [
      { text: "어린이집수", options: { bold: true, color: C.textD } },
      { text: "−0.0182",  options: { align: "center" } },
      { text: "−199.8",   options: { align: "center" } },
      { text: "−1.80%/개", options: { align: "right", color: C.accentD, bold: true } },
      { text: "주거밀집지 혼재로 음의 계수(지역 별도 모형서 재검토)", options: { color: C.textM } },
    ],
  ];
  s.addTable(olsRows, {
    x: tx + 0.25, y: ty + 0.6, w: tw - 0.5,
    colW: [2.0, 1.0, 0.9, 1.6, 6.1],
    rowH: [0.34, 0.34, 0.34, 0.34, 0.34, 0.34],
    fontFace: FONT, fontSize: 11, color: C.textM,
    border: { pt: 0.5, color: C.border },
  });

  s.addText("※ 기준금리·CD금리 VIF 110.1 / 96.4 → 거시변수는 개별 효과가 아닌 시장 국면 대리 묶음으로 해석", {
    x: 0.7, y: 7.05, w: 12.1, h: 0.3,
    fontSize: 10.5, fontFace: FONT, color: C.muted, italic: true, margin: 0,
  });

  pageNum(s, 5);
}

// ============================================================
// 6. 모형별 · 분할별 예측 성능
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bgSoft };
  header(s, "03  실증 분석", "모형별 · 분할별 예측 성능 (종속변수: log ㎡당 가격)");

  // 표
  const tRows = [
    [
      { text: "분할",            options: { bold: true, color: C.white, fill: { color: C.bgDark }, align: "left" } },
      { text: "OLS R²",          options: { bold: true, color: C.white, fill: { color: C.bgDark }, align: "center" } },
      { text: "RF R²",           options: { bold: true, color: C.white, fill: { color: C.bgDark }, align: "center" } },
      { text: "XGBoost R²",      options: { bold: true, color: C.white, fill: { color: C.bgDark }, align: "center" } },
      { text: "XGB Median APE",  options: { bold: true, color: C.white, fill: { color: C.bgDark }, align: "center" } },
      { text: "XGB MAPE",        options: { bold: true, color: C.white, fill: { color: C.bgDark }, align: "center" } },
    ],
    [
      { text: "무작위 (동일 단지 내)", options: { bold: true, color: C.textD } },
      { text: "0.5061", options: { align: "center", color: C.textM } },
      { text: "0.8937", options: { align: "center", color: C.textM } },
      { text: "0.9554", options: { align: "center", color: C.primaryD, bold: true, fill: { color: C.primaryL } } },
      { text: "4.16%",  options: { align: "center", color: C.primaryD, bold: true, fill: { color: C.primaryL } } },
      { text: "6.73%",  options: { align: "center", color: C.textM } },
    ],
    [
      { text: "Group (미경험 단지)", options: { bold: true, color: C.textD } },
      { text: "0.4618", options: { align: "center", color: C.textM } },
      { text: "0.7414", options: { align: "center", color: C.textM } },
      { text: "0.8005", options: { align: "center", color: C.primaryD, bold: true, fill: { color: C.primaryL } } },
      { text: "11.72%", options: { align: "center", color: C.textM } },
      { text: "15.52%", options: { align: "center", color: C.textM } },
    ],
    [
      { text: "시간순 (미래 시점)", options: { bold: true, color: C.textD } },
      { text: "0.3939", options: { align: "center", color: C.textM } },
      { text: "0.6955", options: { align: "center", color: C.textM } },
      { text: "0.7972", options: { align: "center", color: C.primaryD, bold: true, fill: { color: C.primaryL } } },
      { text: "12.61%", options: { align: "center", color: C.textM } },
      { text: "15.15%", options: { align: "center", color: C.textM } },
    ],
  ];
  s.addTable(tRows, {
    x: 0.7, y: 1.75, w: 11.95,
    colW: [4.5, 1.4, 1.4, 1.65, 1.65, 1.35],
    rowH: [0.5, 0.5, 0.5, 0.5],
    fontFace: FONT, fontSize: 14,
    border: { pt: 1, color: C.border },
  });

  // 하단 3개 해석 카드
  const obs = [
    { tag: "관찰 1", title: "일관된 서열: XGB > RF >> OLS",
      body: "세 분할 모두에서 동일 서열. OLS R² 0.39~0.51 vs XGB 0.80~0.96 — 약 +0.30~0.45p 비선형 설명력 차이." },
    { tag: "관찰 2", title: "Group ≈ 시간순 (R² 차이 0.003)",
      body: "Group 0.8005 vs 시간순 0.7972. 단지 누수 효과(≈0.15p)가 국면 이동 효과보다 예측력에 더 큰 영향." },
    { tag: "관찰 3", title: "이원 기준 해석",
      body: "무작위 R²=0.9554는 '동일 단지 내', Group/시간순 R²≈0.80은 '미경험 단지·미래 시점' 보수적 일반화 기준." },
  ];
  const oW = 3.95, oH = 2.4, oGap = 0.14;
  const oStart = (W - oW * 3 - oGap * 2) / 2;
  obs.forEach((o, i) => {
    const x = oStart + i * (oW + oGap), y = 4.5;
    card(s, x, y, oW, oH, { accent: C.primary });
    badge(s, x + 0.25, y + 0.25, o.tag);
    s.addText(o.title, {
      x: x + 0.3, y: y + 0.65, w: oW - 0.6, h: 0.55,
      fontSize: 14, fontFace: FONT, color: C.textD, bold: true, margin: 0 });
    s.addText(o.body, {
      x: x + 0.3, y: y + 1.2, w: oW - 0.6, h: 1.15,
      fontSize: 11, fontFace: FONT, color: C.textM, margin: 0 });
  });

  pageNum(s, 6);
}

// ============================================================
// 7. SHAP 전체 Top 6 — v6 vs v7 재편
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bgSoft };
  header(s, "04  SHAP 분석", "단위면적 정규화가 가져온 SHAP Top 6 재편");

  // before (v6)
  const bx = 0.7, by = 1.75, bw = 5.5, bh = 4.5;
  card(s, bx, by, bw, bh);
  s.addText("v6 총가격 모형 (log 거래금액)", {
    x: bx + 0.3, y: by + 0.25, w: bw - 0.6, h: 0.45,
    fontSize: 17, fontFace: FONT, color: C.textM, bold: true, margin: 0 });

  const v6 = [
    ["1", "전용면적",    "22.2%"],
    ["2", "강남구분",    "16.0%"],
    ["3", "건물연령",    "11.4%"],
    ["4", "M2 통화량",   "8.3%"],
    ["5", "백화점수",    "6.8%"],
    ["6", "어린이집수",  "5.2%"],
  ];
  s.addTable(v6.map(([r, v, p]) => [
    { text: r, options: { align: "center", color: C.muted } },
    { text: v, options: { align: "left",   color: C.textD, bold: true } },
    { text: p, options: { align: "right",  color: C.textD, bold: true } },
  ]), {
    x: bx + 0.3, y: by + 0.85, w: bw - 0.6,
    colW: [0.55, 3.1, 1.25], rowH: 0.42,
    fontFace: FONT, fontSize: 13,
    border: { pt: 0.5, color: C.border },
  });
  s.addText("면적이 SHAP 상위 22% 독점 → 질적·입지적 신호 가려짐", {
    x: bx + 0.3, y: by + 3.6, w: bw - 0.6, h: 0.75,
    fontSize: 12, fontFace: FONT, color: C.accentD, bold: true, margin: 0 });

  // arrow
  s.addText("→", {
    x: 6.35, y: 3.6, w: 0.6, h: 0.8,
    fontSize: 36, fontFace: FONT_NUM, color: C.primary, bold: true, align: "center", valign: "middle", margin: 0,
  });

  // after (v7)
  const ax = 7.1, ay = 1.75, aw = 5.7, ah = 4.5;
  s.addShape(pres.shapes.RECTANGLE, {
    x: ax, y: ay, w: aw, h: ah,
    fill: { color: C.primaryL }, line: { color: C.primary, width: 2 }, shadow: shadow(),
  });
  s.addText("v7 단위가격 모형 (log ㎡당 가격)", {
    x: ax + 0.3, y: ay + 0.25, w: aw - 0.6, h: 0.45,
    fontSize: 17, fontFace: FONT, color: C.primaryD, bold: true, margin: 0 });
  const v7 = [
    ["1", "건물연령",       "12.43%", false],
    ["2", "강남구분",       "11.74%", false],
    ["3", "어린이집수 ⭐",  "11.53%", true],
    ["4", "M2 통화량",      "10.77%", false],
    ["5", "학원수 ⭐",      "9.61%",  true],
    ["6", "전용면적 (1→6)", "9.55%",  false],
  ];
  s.addTable(v7.map(([r, v, p, hl]) => [
    { text: r, options: { align: "center", color: C.textD, fill: hl ? { color: C.warmL } : undefined } },
    { text: v, options: { align: "left",   color: C.textD, bold: true, fill: hl ? { color: C.warmL } : undefined } },
    { text: p, options: { align: "right",  color: C.primaryD, bold: true, fill: hl ? { color: C.warmL } : undefined } },
  ]), {
    x: ax + 0.3, y: ay + 0.85, w: aw - 0.6,
    colW: [0.55, 3.3, 1.25], rowH: 0.42,
    fontFace: FONT, fontSize: 13,
    border: { pt: 0.5, color: C.border },
  });
  s.addText("주거권(어린이집) · 사교육(학원) · 재건축(건물연령) 신호 전면화", {
    x: ax + 0.3, y: ay + 3.6, w: aw - 0.6, h: 0.75,
    fontSize: 12, fontFace: FONT, color: C.primaryD, bold: true, margin: 0 });

  // 하단 해설 bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.7, y: 6.4, w: 12.1, h: 0.75,
    fill: { color: C.primary }, line: { color: C.primary }, shadow: shadow(),
  });
  s.addText([
    { text: "핵심 변화  ", options: { bold: true } },
    { text: "전용면적 1위 22.2% → 6위 9.55%로 규모효과 분리 · 거시(M2+CPI+금리) 통합 기여도 17.8% → 21.35%로 국면 민감도 증가" },
  ], {
    x: 0.7, y: 6.4, w: 12.1, h: 0.75,
    fontSize: 13, fontFace: FONT, color: C.white, align: "center", valign: "middle", margin: 0,
  });

  pageNum(s, 7);
}

// ============================================================
// 8. 지역 별도 3모형 비교 (강남 vs 비강남)
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bgSoft };
  header(s, "04  지역 비교", "지역 별도 3모형 — 질적으로 다른 두 시장");

  // 강남 panel
  const gx = 0.7, gy = 1.75, gw = 5.95, gh = 5.15;
  card(s, gx, gy, gw, gh, { accent: C.accent });
  s.addText("🏙️ 강남 3구", {
    x: gx + 0.3, y: gy + 0.2, w: 3.5, h: 0.5,
    fontSize: 20, fontFace: FONT, color: C.accentD, bold: true, margin: 0 });
  s.addText("n = 65,077   (16.6%)", {
    x: gx + 3.7, y: gy + 0.3, w: 2.0, h: 0.4,
    fontSize: 12, fontFace: FONT, color: C.muted, align: "right", margin: 0 });
  s.addText("XGB R² = 0.9356 · MedAPE 4.10%   |   OLS R² = 0.379", {
    x: gx + 0.3, y: gy + 0.7, w: gw - 0.6, h: 0.35,
    fontSize: 12, fontFace: FONT, color: C.textD, bold: true, margin: 0 });

  const gnRows = [
    [
      { text: "순위", options: { bold: true, color: C.white, fill: { color: C.bgDark }, align: "center" } },
      { text: "변수",    options: { bold: true, color: C.white, fill: { color: C.bgDark }, align: "left" } },
      { text: "SHAP %", options: { bold: true, color: C.white, fill: { color: C.bgDark }, align: "right" } },
    ],
    [{ text: "1", options: { align: "center" } }, { text: "건물연령",     options: { bold: true, color: C.textD } }, { text: "16.46%", options: { align: "right", bold: true } }],
    [{ text: "2", options: { align: "center" } }, { text: "소비자물가지수", options: { color: C.textD } },         { text: "13.25%", options: { align: "right" } }],
    [{ text: "3", options: { align: "center" } }, { text: "백화점수",      options: { bold: true, color: C.textD } }, { text: "13.04%", options: { align: "right", bold: true } }],
    [{ text: "4", options: { align: "center" } }, { text: "전용면적",      options: { color: C.textD } },          { text: "12.68%", options: { align: "right" } }],
    [{ text: "5", options: { align: "center" } }, { text: "M2",            options: { color: C.textD } },          { text: "8.47%",  options: { align: "right" } }],
    [{ text: "15/17", options: { align: "center", color: C.muted } }, { text: "어린이집수", options: { color: C.muted } }, { text: "1.69%", options: { align: "right", color: C.muted } }],
  ];
  s.addTable(gnRows, {
    x: gx + 0.3, y: gy + 1.15, w: gw - 0.6,
    colW: [1.1, 3.1, 1.15], rowH: [0.36, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38],
    fontFace: FONT, fontSize: 13, color: C.textM,
    border: { pt: 0.5, color: C.border },
  });
  s.addText("주도 축: 재건축 기대 · 상업 집적", {
    x: gx + 0.3, y: gy + 4.1, w: gw - 0.6, h: 0.5,
    fontSize: 14, fontFace: FONT, color: C.accentD, bold: true, margin: 0 });
  s.addText("압구정·대치·잠실 등 노후 단지 재건축과 명품·상업 집적이 단위가격을 주도. 어린이집·학원 프록시는 하위권.", {
    x: gx + 0.3, y: gy + 4.55, w: gw - 0.6, h: 0.5,
    fontSize: 11, fontFace: FONT, color: C.textM, margin: 0 });

  // 비강남 panel
  const bx = 6.85, by = 1.75, bw = 5.95, bh = 5.15;
  card(s, bx, by, bw, bh, { accent: C.primary });
  s.addText("🏘️ 비강남 22구", {
    x: bx + 0.3, y: by + 0.2, w: 3.5, h: 0.5,
    fontSize: 20, fontFace: FONT, color: C.primaryD, bold: true, margin: 0 });
  s.addText("n = 326,749   (83.4%)", {
    x: bx + 3.7, y: by + 0.3, w: 2.0, h: 0.4,
    fontSize: 12, fontFace: FONT, color: C.muted, align: "right", margin: 0 });
  s.addText("XGB R² = 0.9421 · MedAPE 4.07%   |   OLS R² = 0.381", {
    x: bx + 0.3, y: by + 0.7, w: bw - 0.6, h: 0.35,
    fontSize: 12, fontFace: FONT, color: C.textD, bold: true, margin: 0 });

  const nsRows = [
    [
      { text: "순위", options: { bold: true, color: C.white, fill: { color: C.bgDark }, align: "center" } },
      { text: "변수", options: { bold: true, color: C.white, fill: { color: C.bgDark }, align: "left" } },
      { text: "SHAP %", options: { bold: true, color: C.white, fill: { color: C.bgDark }, align: "right" } },
    ],
    [{ text: "1", options: { align: "center" } }, { text: "건물연령",       options: { color: C.textD } }, { text: "14.81%", options: { align: "right" } }],
    [{ text: "2", options: { align: "center" } }, { text: "소비자물가지수", options: { color: C.textD } }, { text: "11.79%", options: { align: "right" } }],
    [{ text: "3", options: { align: "center" } }, { text: "전용면적",       options: { color: C.textD } }, { text: "11.19%", options: { align: "right" } }],
    [
      { text: "4", options: { align: "center", fill: { color: C.primaryL } } },
      { text: "어린이집수", options: { bold: true, color: C.textD, fill: { color: C.primaryL } } },
      { text: "10.57%", options: { align: "right", bold: true, color: C.primaryD, fill: { color: C.primaryL } } },
    ],
    [{ text: "5", options: { align: "center" } }, { text: "M2",    options: { color: C.textD } },        { text: "10.11%", options: { align: "right" } }],
    [
      { text: "6", options: { align: "center", fill: { color: C.primaryL } } },
      { text: "학원수", options: { bold: true, color: C.textD, fill: { color: C.primaryL } } },
      { text: "8.63%", options: { align: "right", bold: true, color: C.primaryD, fill: { color: C.primaryL } } },
    ],
  ];
  s.addTable(nsRows, {
    x: bx + 0.3, y: by + 1.15, w: bw - 0.6,
    colW: [1.1, 3.1, 1.15], rowH: [0.36, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38],
    fontFace: FONT, fontSize: 13, color: C.textM,
    border: { pt: 0.5, color: C.border },
  });
  s.addText("주도 축: 주거권 · 사교육 인프라 · 거시 유동성", {
    x: bx + 0.3, y: by + 4.1, w: bw - 0.6, h: 0.5,
    fontSize: 14, fontFace: FONT, color: C.primaryD, bold: true, margin: 0 });
  s.addText("어린이집·학원 프록시가 상위권에 진입 — 비강남 주거권 특성·사교육 수요가 단위가격 차별화 신호.", {
    x: bx + 0.3, y: by + 4.55, w: bw - 0.6, h: 0.5,
    fontSize: 11, fontFace: FONT, color: C.textM, margin: 0 });

  s.addText("※ 어린이집수 대비: 강남 15/17위 1.69% ↔ 비강남 4위 10.57% — 동일 변수가 지역에 따라 질적으로 다른 설명 역할",
    { x: 0.5, y: 7.05, w: 12.3, h: 0.3,
      fontSize: 11, fontFace: FONT, color: C.textM, align: "center", italic: true, margin: 0 });

  pageNum(s, 8);
}

// ============================================================
// 9. 비선형 · 공간 자기상관 · Ablation 강건성
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bgSoft };
  header(s, "04  강건성 점검", "비선형 패턴 · Moran's I · Ablation");

  // 좌: 건물연령 U자 + 전용면적 체감 설명
  const dx = 0.7, dy = 1.75, dw = 6.1, dh = 4.5;
  card(s, dx, dy, dw, dh, { accent: C.primary });
  s.addText("🎯 SHAP Dependence Plot — 비선형 패턴", {
    x: dx + 0.3, y: dy + 0.25, w: dw - 0.6, h: 0.4,
    fontSize: 16, fontFace: FONT, color: C.primaryD, bold: true, margin: 0 });
  s.addText([
    { text: "건물연령 U자형", options: { bold: true, color: C.textD, fontSize: 14, breakLine: true } },
    { text: " ", options: { fontSize: 4, breakLine: true } },
    { text: "연령 증가 → SHAP 감소 기본 추세 위에, 25~30년 이상 구간에서 ", options: {} },
    { text: "반등 U자 패턴", options: { bold: true, color: C.primaryD } },
    { text: ". 재건축 기대·입지·표본 조합의 비선형 결합. OLS 선형 가정으로는 포착 불가.", options: { breakLine: true } },
    { text: " ", options: { fontSize: 6, breakLine: true } },
    { text: "전용면적 체감형", options: { bold: true, color: C.textD, fontSize: 14, breakLine: true } },
    { text: " ", options: { fontSize: 4, breakLine: true } },
    { text: "소형(40–60㎡) SHAP +, 중·대형(85㎡+) SHAP − 전환. 규모 정규화 후에도 ", options: {} },
    { text: "소형 프리미엄", options: { bold: true, color: C.primaryD } },
    { text: "이 비선형 체감 형태로 잔존.", options: {} },
  ], {
    x: dx + 0.3, y: dy + 0.75, w: dw - 0.6, h: 3.6,
    fontSize: 12, fontFace: FONT, color: C.textM, margin: 0, paraSpaceAfter: 2,
  });

  // 우: Moran's I + Ablation 표
  const rx = 6.9, ry = 1.75, rw = 5.9, rh = 4.5;
  card(s, rx, ry, rw, rh, { accent: C.warm });
  s.addText("🧪 공간 자기상관 · Ablation", {
    x: rx + 0.3, y: ry + 0.25, w: rw - 0.6, h: 0.4,
    fontSize: 16, fontFace: FONT, color: C.textD, bold: true, margin: 0 });
  s.addText("214개 행정동 잔차 · 같은 구 인접 가중 · 499 순열", {
    x: rx + 0.3, y: ry + 0.65, w: rw - 0.6, h: 0.3,
    fontSize: 10, fontFace: FONT, color: C.muted, italic: true, margin: 0 });

  const robustRows = [
    [
      { text: "진단",      options: { bold: true, color: C.white, fill: { color: C.bgDark }, align: "left" } },
      { text: "OLS",       options: { bold: true, color: C.white, fill: { color: C.bgDark }, align: "center" } },
      { text: "XGBoost",   options: { bold: true, color: C.white, fill: { color: C.bgDark }, align: "center" } },
    ],
    [
      { text: "Moran I (잔차)", options: { bold: true, color: C.textD } },
      { text: "0.3247",  options: { align: "center", color: C.accentD, bold: true } },
      { text: "0.0042",  options: { align: "center", color: C.primaryD, bold: true } },
    ],
    [
      { text: "Z-score", options: { color: C.textD } },
      { text: "8.882",   options: { align: "center", color: C.accentD } },
      { text: "0.252",   options: { align: "center", color: C.primaryD } },
    ],
    [
      { text: "p-value (순열)", options: { color: C.textD } },
      { text: "< 0.001", options: { align: "center", color: C.accentD, bold: true } },
      { text: "0.880",   options: { align: "center", color: C.primaryD, bold: true } },
    ],
    [
      { text: "Ablation ΔR² (학원·어린이집 제거)", options: { bold: true, color: C.textD } },
      { text: "−0.0542", options: { align: "center", color: C.accentD, bold: true } },
      { text: "−0.0004", options: { align: "center", color: C.primaryD, bold: true } },
    ],
  ];
  s.addTable(robustRows, {
    x: rx + 0.3, y: ry + 1.0, w: rw - 0.6,
    colW: [3.2, 1.0, 1.0], rowH: [0.38, 0.38, 0.38, 0.38, 0.6],
    fontFace: FONT, fontSize: 11.5,
    border: { pt: 0.5, color: C.border },
  });
  s.addText([
    { text: "→ XGB의 비선형 조합이 공간 구조 흡수", options: { bold: true, color: C.textD, breakLine: true } },
    { text: "→ 프록시 두 변수 없이도 예측 구조 유지 (측정 한계 비왜곡)", options: { color: C.textM } },
  ], {
    x: rx + 0.3, y: ry + 3.35, w: rw - 0.6, h: 1.0,
    fontSize: 11.5, fontFace: FONT, margin: 0,
  });

  // 하단 결론 bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.7, y: 6.5, w: 12.1, h: 0.8,
    fill: { color: C.primary }, line: { color: C.primary }, shadow: shadow(),
  });
  s.addText([
    { text: "핵심  ", options: { bold: true } },
    { text: "OLS가 포착 못 한 ", options: {} },
    { text: "U자·체감·임계 비선형 패턴", options: { bold: true } },
    { text: "을 XGBoost+SHAP이 정량 실증 — 공간계량 모형 없이 공간 구조 흡수 · 프록시 한계 비의존" },
  ], {
    x: 0.7, y: 6.5, w: 12.1, h: 0.8,
    fontSize: 13, fontFace: FONT, color: C.white, align: "center", valign: "middle", margin: 0,
  });

  pageNum(s, 9);
}

// ============================================================
// 10. 결론 · 시사점 · 한계 · 감사
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.bgSoft };
  header(s, "05  결 론", "학술·정책·실무 시사점 · 한계 및 향후 과제");

  // 상단 3층위 시사점 카드
  const impl = [
    { tag: "학술적", title: "📚 XAI 국내 확장",
      body: "단위면적 정규화 × 행정동(215) × 지역 별도 3모형 — 국내 부동산 XAI 문헌 최초 3축 결합" },
    { tag: "정책적", title: "🏛️ 이원 시장 설계",
      body: "강남=재건축·상업, 비강남=주거권·사교육. 가격 수준이 아닌 결정 구조 자체가 상이 → 지역별 차별 정책 시사" },
    { tag: "실무적", title: "💼 AVM 설명 책무",
      body: "단위가격 SHAP 분해로 면적에 가려지지 않은 질적 프리미엄 투명화 — 감정평가·프롭테크 설명 책무 지원" },
  ];
  const iW = 3.95, iH = 1.95, iGap = 0.14;
  const iStart = (W - iW * 3 - iGap * 2) / 2;
  impl.forEach((it, i) => {
    const x = iStart + i * (iW + iGap), y = 1.7;
    card(s, x, y, iW, iH);
    badge(s, x + 0.25, y + 0.22, it.tag, 0.9);
    s.addText(it.title, {
      x: x + 0.3, y: y + 0.6, w: iW - 0.6, h: 0.45,
      fontSize: 15, fontFace: FONT, color: C.textD, bold: true, margin: 0 });
    s.addText(it.body, {
      x: x + 0.3, y: y + 1.05, w: iW - 0.6, h: 0.85,
      fontSize: 11, fontFace: FONT, color: C.textM, margin: 0 });
  });

  // 하단 한계 / 향후
  const limX = 0.7, limY = 3.85, limW = 5.95, limH = 2.9;
  card(s, limX, limY, limW, limH, { accent: C.warm });
  s.addText("⚠️ 연구의 한계", {
    x: limX + 0.3, y: limY + 0.2, w: limW - 0.6, h: 0.4,
    fontSize: 16, fontFace: FONT, color: C.textD, bold: true, margin: 0 });
  s.addText([
    { text: "시설 변수 stock 병합 — 연도별 변동 미반영 ", options: { bullet: true } },
    { text: "[교수 지적]", options: { color: C.warm, bold: true, breakLine: true } },
    { text: "학원·어린이집 구 단위 프록시 균등 배분 한계", options: { bullet: true, breakLine: true } },
    { text: "도보 반경·최단거리 변수 미사용 (행정동 경계 count 기반)", options: { bullet: true, breakLine: true } },
    { text: "질적 변수(브랜드·조망·재건축 단계) 미반영", options: { bullet: true, breakLine: true } },
    { text: "공간계량(SAR/SEM) 직접 비교 미수행", options: { bullet: true } },
  ], {
    x: limX + 0.3, y: limY + 0.65, w: limW - 0.6, h: 2.1,
    fontSize: 11, fontFace: FONT, color: C.textM, margin: 0, paraSpaceAfter: 2,
  });

  const futX = 6.85, futY = 3.85, futW = 5.95, futH = 2.9;
  card(s, futX, futY, futW, futH, { accent: C.primary });
  s.addText("🚀 향후 과제", {
    x: futX + 0.3, y: futY + 0.2, w: futW - 0.6, h: 0.4,
    fontSize: 16, fontFace: FONT, color: C.textD, bold: true, margin: 0 });
  s.addText([
    { text: "연도별 시설 패널 (행안부·교육부 연간 스냅샷)", options: { bullet: true, breakLine: true } },
    { text: "GIS 기반 최근접 거리 변수 · Queen/Rook 경계 가중", options: { bullet: true, breakLine: true } },
    { text: "평형대(60/85/132㎡)별 ㎡단가 보조 분석", options: { bullet: true, breakLine: true } },
    { text: "LightGBM·CatBoost·TabNet 확장, LIME·ALE 교차 XAI", options: { bullet: true, breakLine: true } },
    { text: "수도권·지방 대도시로 범위 확장 비교", options: { bullet: true } },
  ], {
    x: futX + 0.3, y: futY + 0.65, w: futW - 0.6, h: 2.1,
    fontSize: 11, fontFace: FONT, color: C.textM, margin: 0, paraSpaceAfter: 2,
  });

  // 하단 감사 bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 6.95, w: W - 1, h: 0.4,
    fill: { color: C.bgDark }, line: { color: C.bgDark },
  });
  s.addText("경청해 주셔서 감사드립니다 · 질의 환영합니다    박 현 근 · 한양대 부동산융합대학원", {
    x: 0.5, y: 6.95, w: W - 1, h: 0.4,
    fontSize: 12, fontFace: FONT, color: C.white, align: "center", valign: "middle", margin: 0,
  });

  pageNum(s, 10);
}

// ============================================================
pres.writeFile({ fileName: "/Users/seokmogu/project/thesis-seoul-apt-xai/paper/중간발표_v7_5min.pptx" })
  .then((n) => console.log("생성 완료:", n));
