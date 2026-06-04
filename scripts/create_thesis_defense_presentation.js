const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");
const sizeOf = require("image-size");

const ROOT = path.join(__dirname, "..");
const OUT = path.join(ROOT, "paper", "논문심사_발표_박현근.pptx");
const SCRIPT_OUT = path.join(ROOT, "paper", "논문심사_발표_스크립트_박현근.md");
const FIG = path.join(ROOT, "figures");

const pptx = new pptxgen();
const FONT = "Pretendard";
const W = 13.333;
const H = 7.5;
const KOR_TITLE = "서울시 아파트 매매가격 구조 분석";
const KOR_SUBTITLE = "거리 기반 접근성과 시공간 이질성을 중심으로";
const ENG_TITLE = "An Analysis of Apartment Sale Price Structure in Seoul:\nFocusing on Distance-Based Accessibility and Spatiotemporal Heterogeneity";
const SCHOOL = "한양대학교 부동산융합대학원";
const MAJOR = "도시·부동산빅데이터전공";
const DEFENSE_LABEL = "석사학위논문 최종심사 발표";
const AUTHOR = "박현근";

pptx.defineLayout({ name: "WIDE", width: W, height: H });
pptx.layout = "WIDE";
pptx.author = AUTHOR;
pptx.company = SCHOOL;
pptx.subject = DEFENSE_LABEL;
pptx.title = KOR_TITLE;
pptx.lang = "ko-KR";
pptx.theme = { headFontFace: FONT, bodyFontFace: FONT, lang: "ko-KR" };

const C = {
  navy: "26354F",
  navy2: "344865",
  red: "B8443E",
  yellow: "F2D16B",
  text: "2F3542",
  muted: "646B78",
  line: "D4DAE3",
  soft: "F4F6F9",
  soft2: "EAF0F5",
  white: "FFFFFF",
  green: "4B7F68",
  blue: "426A9B",
};

const deckNotes = [];

function addText(slide, text, opts) {
  slide.addText(text, {
    fontFace: FONT,
    color: C.text,
    margin: 0.04,
    fit: "shrink",
    breakLine: false,
    ...opts,
  });
}

function addBox(slide, x, y, w, h, color, line = color) {
  slide.addShape(pptx.ShapeType.rect, {
    x, y, w, h,
    fill: { color },
    line: { color: line || color, width: 0.6 },
  });
}

function addLine(slide, x, y, w, color = C.line, width = 1) {
  slide.addShape(pptx.ShapeType.line, {
    x, y, w, h: 0,
    line: { color, width },
  });
}

function addHeader(slide, section, title, no) {
  addBox(slide, 0, 0, W, 0.5, C.navy);
  addText(slide, section, {
    x: 0.45, y: 0.14, w: 4.2, h: 0.2,
    fontSize: 10.5, bold: true, color: C.yellow,
  });
  addText(slide, String(no).padStart(2, "0"), {
    x: 12.35, y: 0.12, w: 0.55, h: 0.22,
    fontSize: 10.5, bold: true, color: C.white, align: "right",
  });
  addText(slide, title, {
    x: 0.55, y: 0.82, w: 11.4, h: 0.48,
    fontSize: title.length > 26 ? 22.2 : 24, bold: true, color: C.navy,
    breakLine: true,
  });
  addLine(slide, 0.55, 1.4, 12.2, C.line, 1.2);
}

function addFooter(slide) {
  addText(slide, `${DEFENSE_LABEL} | ${AUTHOR}`, {
    x: 0.55, y: 7.14, w: 4.8, h: 0.18,
    fontSize: 8.2, color: C.muted,
  });
}

function addBullets(slide, items, x, y, w, h, opts = {}) {
  const lines = items.map((item) => `• ${item}`).join("\n");
  addText(slide, lines, {
    x, y, w, h,
    fontSize: opts.fontSize ?? 16.6,
    color: opts.color ?? C.text,
    bold: opts.bold ?? false,
    breakLine: true,
    fit: "shrink",
    valign: opts.valign ?? "mid",
    paraSpaceAfterPt: opts.spaceAfter ?? 7,
    breakLineOnHyphen: false,
  });
}

function addStat(slide, value, label, x, y, w, h, color = C.navy) {
  addBox(slide, x, y, w, h, C.soft, C.line);
  addText(slide, value, {
    x: x + 0.12, y: y + 0.2, w: w - 0.24, h: 0.36,
    fontSize: 24, bold: true, color,
    align: "center",
  });
  addText(slide, label, {
    x: x + 0.12, y: y + 0.73, w: w - 0.24, h: 0.38,
    fontSize: 12.2, color: C.muted,
    align: "center",
  });
}

function addImageContain(slide, file, x, y, w, h) {
  const filePath = path.join(FIG, file);
  if (!fs.existsSync(filePath)) {
    addBox(slide, x, y, w, h, C.soft, C.line);
    addText(slide, `그림 파일 없음: ${file}`, {
      x: x + 0.2, y: y + h / 2 - 0.15, w: w - 0.4, h: 0.3,
      fontSize: 12, color: C.red, align: "center",
    });
    return;
  }
  const dim = sizeOf.imageSize(filePath);
  const imgRatio = dim.width / dim.height;
  const boxRatio = w / h;
  let iw = w;
  let ih = h;
  if (imgRatio > boxRatio) {
    ih = w / imgRatio;
  } else {
    iw = h * imgRatio;
  }
  slide.addImage({
    path: filePath,
    x: x + (w - iw) / 2,
    y: y + (h - ih) / 2,
    w: iw,
    h: ih,
  });
}

function addMiniTable(slide, headers, rows, x, y, w, rowH, colWs, opts = {}) {
  const headerH = opts.headerH ?? 0.48;
  addBox(slide, x, y, w, headerH + rows.length * rowH, C.white, C.line);
  let cx = x;
  headers.forEach((h, i) => {
    addBox(slide, cx, y, colWs[i], headerH, C.soft2, C.line);
    addText(slide, h, {
      x: cx + 0.04, y: y + 0.11, w: colWs[i] - 0.08, h: 0.18,
      fontSize: opts.headerSize ?? 12.5, bold: true, color: C.navy,
      align: "center",
    });
    cx += colWs[i];
  });
  rows.forEach((r, ridx) => {
    cx = x;
    const ry = y + headerH + ridx * rowH;
    const fill = ridx % 2 === 0 ? C.white : C.soft;
    r.forEach((cell, i) => {
      addBox(slide, cx, ry, colWs[i], rowH, fill, C.line);
      addText(slide, cell, {
        x: cx + 0.05, y: ry + 0.06, w: colWs[i] - 0.1, h: rowH - 0.12,
        fontSize: opts.fontSize ?? 12.5,
        bold: opts.boldFirst && i === 0,
        color: i === 0 ? C.navy : C.text,
        align: i === 0 ? "center" : "center",
        valign: "mid",
      });
      cx += colWs[i];
    });
  });
}

function addR2Bar(slide, label, value, x, y, maxW, color) {
  addText(slide, label, { x, y: y - 0.04, w: 1.35, h: 0.3, fontSize: 15, bold: true, color: C.text });
  addBox(slide, x + 1.45, y, maxW, 0.28, "E3E8EF", "E3E8EF");
  addBox(slide, x + 1.45, y, maxW * value, 0.28, color, color);
  addText(slide, value.toFixed(3), { x: x + 1.45 + maxW + 0.12, y: y - 0.03, w: 0.78, h: 0.3, fontSize: 13.4, bold: true, color: C.navy });
}

function addBigMetric(slide, value, label, x, y, w, h, color = C.navy) {
  addBox(slide, x, y, w, h, C.soft, C.line);
  addText(slide, value, {
    x: x + 0.12, y: y + 0.18, w: w - 0.24, h: 0.48,
    fontSize: 25, bold: true, color, align: "center",
  });
  addText(slide, label, {
    x: x + 0.15, y: y + 0.78, w: w - 0.3, h: 0.42,
    fontSize: 13.2, color: C.text, align: "center", breakLine: true,
  });
}

function addWideBar(slide, label, value, maxValue, x, y, w, color, opts = {}) {
  const labelW = opts.labelW ?? 1.95;
  const barH = opts.barH ?? 0.34;
  addText(slide, label, {
    x, y: y - 0.04, w: labelW, h: 0.32,
    fontSize: opts.labelSize ?? 14.2, bold: true, color: C.text,
  });
  addBox(slide, x + labelW + 0.1, y, w, barH, "E3E8EF", "E3E8EF");
  addBox(slide, x + labelW + 0.1, y, w * (value / maxValue), barH, color, color);
  addText(slide, value.toFixed(3), {
    x: x + labelW + 0.25 + w, y: y - 0.03, w: 0.72, h: 0.3,
    fontSize: opts.valueSize ?? 12.8, bold: true, color: C.navy,
  });
}

function addRankRow(slide, rank, label, value, x, y, w, color, maxValue) {
  addText(slide, String(rank), {
    x, y: y + 0.01, w: 0.32, h: 0.24,
    fontSize: 12.5, bold: true, color: C.white, align: "center",
    fill: { color },
  });
  addText(slide, label, {
    x: x + 0.45, y: y - 0.02, w: 2.5, h: 0.34,
    fontSize: 14.2, bold: true, color: C.text,
  });
  addBox(slide, x + 3.08, y + 0.05, w, 0.24, "E3E8EF", "E3E8EF");
  addBox(slide, x + 3.08, y + 0.05, w * (value / maxValue), 0.24, color, color);
  addText(slide, `${(value * 100).toFixed(1)}%`, {
    x: x + 3.18 + w, y: y - 0.02, w: 0.72, h: 0.32,
    fontSize: 12.8, bold: true, color: C.navy,
  });
}

function addFlowStep(slide, no, title, desc, x, y, w, h) {
  addBox(slide, x, y, w, h, C.white, C.line);
  addBox(slide, x + 0.18, y + 0.17, 0.48, 0.48, C.red, C.red);
  addText(slide, String(no), {
    x: x + 0.18, y: y + 0.29, w: 0.48, h: 0.18,
    fontSize: 13.2, bold: true, color: C.white, align: "center",
  });
  addText(slide, title, {
    x: x + 0.85, y: y + 0.14, w: w - 1.1, h: 0.22,
    fontSize: 15.2, bold: true, color: C.navy,
  });
  addText(slide, desc, {
    x: x + 0.85, y: y + 0.43, w: w - 1.1, h: 0.22,
    fontSize: 13.2, color: C.muted,
  });
}

function addNotes(slide, title, seconds, notes) {
  slide.addNotes(`[${seconds}초] ${title}\n\n${notes}`);
  deckNotes.push({ title, seconds, notes });
}

function addSlideWithHeader(section, title, no) {
  const slide = pptx.addSlide();
  slide.background = { color: C.white };
  addHeader(slide, section, title, no);
  addFooter(slide);
  return slide;
}

function buildDeck() {
  let slide = pptx.addSlide();
  slide.background = { color: C.white };
  addBox(slide, 0, 0, W, 0.55, C.navy);
  addText(slide, DEFENSE_LABEL, {
    x: 0.55, y: 0.16, w: 4.6, h: 0.2,
    fontSize: 11, bold: true, color: C.yellow,
  });
  addBox(slide, 0.75, 1.35, 11.85, 2.55, C.navy);
  addText(slide, KOR_TITLE, {
    x: 1.05, y: 1.8, w: 11.25, h: 0.5,
    fontSize: 30, bold: true, color: C.white,
    align: "center",
  });
  addText(slide, KOR_SUBTITLE, {
    x: 1.05, y: 2.53, w: 11.25, h: 0.42,
    fontSize: 20.5, bold: true, color: C.yellow,
    align: "center",
  });
  addText(slide, ENG_TITLE, {
    x: 1.2, y: 3.13, w: 10.9, h: 0.58,
    fontSize: 12.4, color: "DDE4EE",
    align: "center", breakLine: true,
  });
  addText(slide, SCHOOL, {
    x: 0.8, y: 5.36, w: 11.8, h: 0.25,
    fontSize: 12.6, color: C.text, align: "center",
  });
  addText(slide, MAJOR, {
    x: 0.8, y: 5.68, w: 11.8, h: 0.25,
    fontSize: 12.6, color: C.text, align: "center",
  });
  addText(slide, AUTHOR, {
    x: 0.8, y: 6.08, w: 11.8, h: 0.36,
    fontSize: 18, bold: true, color: C.navy, align: "center",
  });
  addText(slide, "2026. 06.", {
    x: 0.8, y: 6.57, w: 11.8, h: 0.24,
    fontSize: 12, color: C.muted, align: "center",
  });
  addNotes(slide, "표지", 30,
    `안녕하십니까. ${SCHOOL} ${MAJOR} ${AUTHOR}입니다. 발표 주제는 '${KOR_TITLE}: ${KOR_SUBTITLE}'입니다. 본 발표는 피드백을 반영한 최종 논문심사 자료입니다. 본 연구는 단순히 가격을 잘 예측하는 모형을 만드는 데서 끝나지 않고, 어떤 접근성 조건과 시장 국면이 예측값 구조에 반영되는지를 설명하는 데 초점을 두었습니다.`);

  slide = addSlideWithHeader("01. INTRODUCTION", "연구 배경과 연구 질문", 2);
  addBox(slide, 0.7, 1.72, 3.75, 3.75, C.soft, C.line);
  addBox(slide, 4.8, 1.72, 3.75, 3.75, C.soft, C.line);
  addBox(slide, 8.9, 1.72, 3.75, 3.75, C.soft, C.line);
  addText(slide, "공간 단위", { x: 0.95, y: 2.02, w: 3.25, h: 0.3, fontSize: 18, bold: true, color: C.navy, align: "center" });
  addText(slide, "시간 정합", { x: 5.05, y: 2.02, w: 3.25, h: 0.3, fontSize: 18, bold: true, color: C.navy, align: "center" });
  addText(slide, "해석 단면", { x: 9.15, y: 2.02, w: 3.25, h: 0.3, fontSize: 18, bold: true, color: C.navy, align: "center" });
  addBullets(slide, [
    "행정동 집계는 접근성을 희석",
    "공간 단위가 결과를 바꿈",
  ], 0.95, 2.64, 3.25, 1.95, { fontSize: 18, spaceAfter: 10 });
  addBullets(slide, [
    "현재 시설은 미래 정보를 소급",
    "연도별 활동 시설 기준 필요",
  ], 5.05, 2.64, 3.25, 1.95, { fontSize: 18, spaceAfter: 10 });
  addBullets(slide, [
    "단일 평균은 차이를 희석",
    "권역·연도 단위 분해 필요",
  ], 9.15, 2.64, 3.25, 1.95, { fontSize: 18, spaceAfter: 10 });
  addBox(slide, 0.7, 5.78, 11.95, 0.82, C.navy);
  addText(slide, "연구 질문: 거리 기반 접근성과 시점 정합은 가격 예측 구조를 어떻게 바꾸는가?", {
    x: 0.95, y: 6.02, w: 11.45, h: 0.28,
    fontSize: 16.2, bold: true, color: C.white,
    align: "center",
  });
  addNotes(slide, "연구 배경과 연구 질문", 60,
    "이 연구의 출발점은 세 가지 한계입니다. 첫째, 기존 연구가 행정동이나 자치구 단위 집계를 많이 사용하면서 단지 주변의 실제 접근성을 충분히 반영하지 못했다는 점입니다. 둘째, 시설 정보를 분석 시점의 스냅샷으로 붙이면 2019년 거래에 2026년 시설 정보가 들어가는 시간역전 정보누수가 생길 수 있습니다. 셋째, SHAP을 쓰더라도 전체 평균만 제시하면 강남과 비강남, 그리고 시장 국면별 차이를 설명하기 어렵습니다. 따라서 본 연구는 거리 기반 접근성, 시점 정합, 권역·연도별 SHAP 분해를 결합해 가격 예측 구조를 분석했습니다.");

  slide = addSlideWithHeader("02. DATA", "데이터와 분석 범위", 3);
  addStat(slide, "391,826건", "서울 아파트 매매 실거래\n2019.01~2025.12", 0.7, 1.7, 2.65, 1.25, C.red);
  addStat(slide, "8,601개", "최종 단지 좌표", 3.65, 1.7, 2.25, 1.25, C.navy);
  addStat(slide, "424 / 426", "행정동 분포", 6.2, 1.7, 2.25, 1.25, C.green);
  addStat(slide, "34개", "최종 주모형 변수", 8.75, 1.7, 2.0, 1.25, C.blue);
  addStat(slide, "전용면적 제외", "종속변수 산식에만 사용", 11.05, 1.7, 1.55, 1.25, C.red);
  addBox(slide, 0.7, 3.25, 5.85, 2.95, C.soft, C.line);
  addText(slide, "표본 구성", { x: 0.95, y: 3.48, w: 2.2, h: 0.3, fontSize: 16.2, bold: true, color: C.navy });
  addBullets(slide, [
    "강남3구 65,077건 (16.6%)",
    "비강남 326,749건 (83.4%)",
    "2022년 거래량 전년 대비 약 70% 감소",
    "종속변수: log(㎡당 거래가격)",
  ], 0.95, 3.91, 5.35, 1.95, { fontSize: 16.2, spaceAfter: 6 });
  addBox(slide, 6.85, 3.25, 5.75, 2.95, C.soft, C.line);
  addText(slide, "자료 출처와 변수군", { x: 7.1, y: 3.48, w: 2.8, h: 0.3, fontSize: 16.2, bold: true, color: C.navy });
  addBullets(slide, [
    "주요 출처: 실거래가·서울열린데이터·NEIS·ECOS",
    "단지·주택: 층, 건물연령",
    "접근성: 교통·교육·보육·의료·상업",
    "거시경제: M2, CPI, 금리",
  ], 7.1, 3.91, 5.25, 1.95, { fontSize: 16.2, spaceAfter: 6 });
  addNotes(slide, "데이터와 분석 범위", 60,
    "분석 대상은 2019년 1월부터 2025년 12월까지 서울 아파트 매매 실거래 391,826건입니다. 단지 좌표는 8,601개이고, 서울시 426개 행정동 중 424개 행정동에 표본이 분포했습니다. 종속변수는 단위면적당 거래가격의 자연로그값입니다. 중요한 점은 전용면적을 설명변수에서 제외했다는 것입니다. 단위면적당 가격을 종속변수로 만들 때 이미 면적이 사용되므로, 다시 설명변수로 넣으면 기계적 결합이 생길 수 있기 때문입니다.");

  slide = addSlideWithHeader("03. METHODS", "분석 설계: 거리 기반 접근성과 시점 정합", 4);
  addBox(slide, 0.65, 1.65, 5.75, 4.75, C.soft, C.line);
  addText(slide, "분석 흐름", { x: 0.95, y: 1.9, w: 2.2, h: 0.3, fontSize: 16.5, bold: true, color: C.navy });
  addFlowStep(slide, 1, "원자료 수집·정제", "실거래·시설·거시자료 결합", 0.95, 2.35, 5.15, 0.78);
  addText(slide, "↓", { x: 3.3, y: 3.14, w: 0.25, h: 0.16, fontSize: 11, bold: true, color: C.red, align: "center" });
  addFlowStep(slide, 2, "단지 좌표 기반 변수", "거리·반경 접근성 생성", 0.95, 3.30, 5.15, 0.78);
  addText(slide, "↓", { x: 3.3, y: 4.09, w: 0.25, h: 0.16, fontSize: 11, bold: true, color: C.red, align: "center" });
  addFlowStep(slide, 3, "모형 학습·검증", "OLS·RF·XGBoost 비교", 0.95, 4.25, 5.15, 0.78);
  addText(slide, "↓", { x: 3.3, y: 5.04, w: 0.25, h: 0.16, fontSize: 11, bold: true, color: C.red, align: "center" });
  addFlowStep(slide, 4, "SHAP 해석", "권역·연도별 기여도 분해", 0.95, 5.20, 5.15, 0.78);
  addBox(slide, 6.75, 1.65, 5.85, 4.75, C.soft, C.line);
  addText(slide, "분석의 핵심 설계", { x: 7.05, y: 1.92, w: 3.1, h: 0.3, fontSize: 16.5, bold: true, color: C.navy });
  addBullets(slide, [
    "단지 좌표 기준 거리·반경 접근성",
    "거래연도별 활동 시설만 사용",
    "무작위·시간순·단지 분할 비교",
    "SHAP을 권역·연도 단위로 분해",
  ], 7.05, 2.52, 5.25, 2.55, { fontSize: 16.6, spaceAfter: 8 });
  addBox(slide, 7.05, 5.35, 5.25, 0.62, C.navy);
  addText(slide, "거리 접근성은 성능, 시점 정합은 타당성을 점검하는 장치", {
    x: 7.25, y: 5.53, w: 4.85, h: 0.22,
    fontSize: 14.4, bold: true, color: C.white,
    align: "center",
  });
  addNotes(slide, "분석 설계", 65,
    "연구 설계는 네 단계입니다. 먼저 원자료를 수집하고 단지 좌표와 결합했습니다. 다음으로 시설 변수를 행정동 경계 안의 개수가 아니라 거래 단지에서 시설까지의 거리와 반경 내 개수로 바꾸었습니다. 또한 시설의 개업과 폐업 이력을 반영해서 해당 거래연도에 실제로 이용 가능했던 시설만 남겼습니다. 모형은 OLS, 랜덤 포레스트, XGBoost를 비교했고, 검증은 무작위 분할뿐 아니라 시간순 분할과 단지 분할까지 함께 사용했습니다. 마지막으로 XGBoost에 SHAP을 적용해 전체, 권역, 연도, 권역×연도별 예측 기여 구조를 분석했습니다.");

  slide = addSlideWithHeader("04. RESULTS", "모형 성능: XGBoost는 우수하지만 분할 조건별로 다르게 읽어야 함", 5);
  addMiniTable(slide, ["분할", "OLS R²", "RF R²", "XGB R²", "MAPE"], [
    ["무작위", "0.494", "0.860", "0.927", "10.0%"],
    ["시간순", "0.406", "0.675", "0.812", "14.6%"],
    ["단지", "0.434", "0.538", "0.638", "20.4%"],
  ], 0.75, 1.72, 5.65, 0.76, [1.0, 1.1, 1.1, 1.45, 1.0], { fontSize: 14.4, headerSize: 13.8, headerH: 0.62 });
  addBox(slide, 6.85, 1.75, 5.75, 3.0, C.soft, C.line);
  addText(slide, "XGBoost R²", { x: 7.1, y: 2.03, w: 2.5, h: 0.32, fontSize: 17, bold: true, color: C.navy });
  addR2Bar(slide, "무작위", 0.927, 7.1, 2.65, 3.45, C.red);
  addR2Bar(slide, "시간순", 0.812, 7.1, 3.25, 3.45, C.blue);
  addR2Bar(slide, "단지", 0.638, 7.1, 3.85, 3.45, C.green);
  addBox(slide, 0.75, 5.2, 11.85, 1.15, C.navy);
  addText(slide, "해석 포인트", { x: 1.05, y: 5.39, w: 1.8, h: 0.3, fontSize: 15.2, bold: true, color: C.yellow });
  addBullets(slide, [
    "무작위는 가장 쉬운 조건, 시간순·단지는 외삽 조건",
    "핵심은 높은 R²보다 분할 조건별 성능 격차",
  ], 2.45, 5.27, 9.6, 0.84, { fontSize: 16, color: C.white, spaceAfter: 5 });
  addNotes(slide, "모형 성능", 65,
    "성능 결과를 보면 XGBoost가 세 분할 모두에서 OLS와 랜덤 포레스트를 상회했습니다. 무작위 분할 R²는 0.927로 높고, 시간순 분할은 0.812, 단지 분할은 0.638입니다. 여기서 중요한 점은 무작위 성능만 강조하지 않는 것입니다. 무작위 분할은 같은 단지 또는 유사 단지가 학습과 평가에 같이 들어갈 수 있어 가장 쉬운 조건입니다. 반면 시간순 분할은 이후 시점 예측이고, 단지 분할은 학습에서 보지 못한 단지에 대한 외삽입니다. 그래서 본 연구는 높은 성능보다 일반화 조건별 성능 격차를 함께 해석했습니다.");

  slide = addSlideWithHeader("04. RESULTS", "소거분석: 거리 기반 전환과 시점 정합의 역할 분리", 6);
  addBigMetric(slide, "A", "행정동 집계\n기준선", 0.85, 1.7, 2.25, 1.32, C.navy);
  addBigMetric(slide, "B", "거리 기반\n전환", 3.35, 1.7, 2.25, 1.32, C.blue);
  addBigMetric(slide, "C", "시점 정합\n누수 제거", 5.85, 1.7, 2.25, 1.32, C.green);
  addBox(slide, 8.55, 1.7, 3.75, 1.32, C.navy);
  addText(slide, "+15.0%p", {
    x: 8.75, y: 1.93, w: 3.35, h: 0.38,
    fontSize: 25, bold: true, color: C.yellow, align: "center",
  });
  addText(slide, "A → B 시간순 R² 개선", {
    x: 8.75, y: 2.42, w: 3.35, h: 0.26,
    fontSize: 14.2, bold: true, color: C.white, align: "center",
  });
  addBox(slide, 0.85, 3.34, 11.45, 2.45, C.soft, C.line);
  addText(slide, "시간순 분할 XGBoost R²", {
    x: 1.15, y: 3.62, w: 3.1, h: 0.3,
    fontSize: 17, bold: true, color: C.navy,
  });
  addWideBar(slide, "A 기준선", 0.665, 1.0, 1.15, 4.18, 7.85, C.navy, { labelSize: 14.5, valueSize: 13.2 });
  addWideBar(slide, "B 거리", 0.816, 1.0, 1.15, 4.78, 7.85, C.blue, { labelSize: 14.5, valueSize: 13.2 });
  addWideBar(slide, "C 정합", 0.799, 1.0, 1.15, 5.38, 7.85, C.green, { labelSize: 14.5, valueSize: 13.2 });
  addBullets(slide, [
    "거리 기반 전환은 시간순 예측 성능을 크게 개선",
    "시점 정합은 미래 정보 소급을 제거하는 타당성 장치",
  ], 1.15, 6.08, 10.7, 0.62, { fontSize: 16.2, spaceAfter: 2 });
  addNotes(slide, "소거분석", 60,
    "소거분석은 공간 정합과 시간 정합의 역할을 분리하기 위해 설계했습니다. A는 행정동 경계 내 시설 개수를 쓰는 기준선이고, B는 단지 좌표 기준 거리 변수로 바꾼 모형이며, C는 여기에 연도별 활동 시설 스냅샷을 적용한 모형입니다. 시간순 분할에서 A의 R²는 0.665였고, 거리 기반 전환 후 B는 0.816으로 약 15.0%p 개선되었습니다. C는 B보다 소폭 낮지만, 이는 미래 시설 정보를 과거 거래에 소급하지 않도록 한 보수적 설계입니다. 따라서 시점 정합은 성능을 높이는 장치라기보다 정보누수를 제거하는 장치로 해석했습니다.");

  slide = addSlideWithHeader("04. RESULTS", "전체 SHAP: 가격 예측 기여 구조", 7);
  addBox(slide, 0.75, 1.58, 7.65, 4.92, C.soft, C.line);
  addText(slide, "상위 예측 기여 변수", { x: 1.05, y: 1.88, w: 3.2, h: 0.32, fontSize: 17, bold: true, color: C.navy });
  addRankRow(slide, 1, "강남구분", 0.136, 1.05, 2.45, 3.55, C.red, 0.136);
  addRankRow(slide, 2, "M2 통화량", 0.114, 1.05, 2.95, 3.55, C.navy2, 0.136);
  addRankRow(slide, 3, "건물연령", 0.105, 1.05, 3.45, 3.55, C.blue, 0.136);
  addRankRow(slide, 4, "어린이집 1km 개수", 0.055, 1.05, 3.95, 3.55, C.green, 0.136);
  addRankRow(slide, 5, "소비자물가지수", 0.047, 1.05, 4.45, 3.55, C.navy2, 0.136);
  addRankRow(slide, 6, "지하철 1km 역수", 0.044, 1.05, 4.95, 3.55, C.green, 0.136);
  addRankRow(slide, 7, "학원 1km 개수", 0.042, 1.05, 5.45, 3.55, C.green, 0.136);
  addBox(slide, 8.75, 1.58, 3.75, 4.92, C.soft, C.line);
  addText(slide, "해석 포인트", { x: 9.05, y: 1.95, w: 2.0, h: 0.32, fontSize: 17, bold: true, color: C.navy });
  addBigMetric(slide, "9개", "상위 15개 중\n거리 기반 변수", 9.05, 2.55, 3.15, 1.18, C.green);
  addBigMetric(slide, "32.6%", "거리 기반 변수\nSHAP 비중 합계", 9.05, 4.0, 3.15, 1.18, C.red);
  addText(slide, "SHAP은 인과효과가 아니라\n예측 기여도로 해석", {
    x: 9.05, y: 5.55, w: 3.15, h: 0.42,
    fontSize: 14.8, bold: true, color: C.navy, align: "center",
    breakLine: true,
  });
  addNotes(slide, "전체 SHAP", 70,
    "전체 SHAP 분석에서는 XGBoost 예측값에 어떤 변수가 평균적으로 크게 기여했는지를 확인했습니다. 상위 변수는 강남구분, M2 통화량, 건물연령, 어린이집 1km 내 개수, 소비자물가지수 순입니다. 상위 15개 변수 중 9개가 거리 기반 변수였고, 이들의 SHAP 비중 합계는 약 32.6%였습니다. 즉 단일 최상위 변수는 권역이나 거시 변수일 수 있지만, 여러 생활 인프라 접근성 변수가 중위권에 넓게 분포한다는 점이 중요합니다. 다만 SHAP은 인과효과가 아니라 모형 내부의 예측 기여도이므로, 이 변수들이 가격을 직접 올린다고 해석하지 않고 예측 신호로 해석했습니다.");

  slide = addSlideWithHeader("04. RESULTS", "권역별 SHAP: 강남3구와 비강남의 구성 차이", 8);
  addBox(slide, 0.75, 1.65, 5.75, 4.6, C.soft, C.line);
  addBox(slide, 6.85, 1.65, 5.75, 4.6, C.soft, C.line);
  addText(slide, "강남3구", { x: 1.05, y: 1.98, w: 2.0, h: 0.35, fontSize: 19, bold: true, color: C.red });
  addText(slide, "비강남", { x: 7.15, y: 1.98, w: 2.0, h: 0.35, fontSize: 19, bold: true, color: C.blue });
  addMiniTable(slide, ["순위", "변수"], [
    ["1", "건물연령"],
    ["2", "백화점 최근접"],
    ["3", "어린이집 1km 내 개수"],
    ["4", "초등학교 최근접"],
    ["5", "종합병원 최근접"],
  ], 1.05, 2.55, 5.15, 0.54, [0.65, 4.5], { fontSize: 14.6, headerSize: 14.2, headerH: 0.5 });
  addMiniTable(slide, ["순위", "변수"], [
    ["1", "건물연령"],
    ["2", "지하철 1km 내 역수"],
    ["3", "어린이집 1km 내 개수"],
    ["4", "학원 1km 내 개수"],
    ["5", "지하철 최근접"],
  ], 7.15, 2.55, 5.15, 0.54, [0.65, 4.5], { fontSize: 14.6, headerSize: 14.2, headerH: 0.5 });
  addBox(slide, 0.75, 6.42, 11.85, 0.55, C.navy);
  addText(slide, "강남3구는 상업·교육·의료 접근성, 비강남은 교통·생활 인프라가 반복적으로 중요", {
    x: 1.05, y: 6.58, w: 11.25, h: 0.22,
    fontSize: 15.4, bold: true, color: C.white, align: "center",
  });
  addNotes(slide, "권역별 SHAP", 60,
    "권역별로 보면 강남3구와 비강남은 단순히 가격 수준만 다른 것이 아니라, 예측값을 구성하는 변수의 종류도 다릅니다. 강남3구에서는 건물연령과 함께 백화점 최근접거리, 초등학교 최근접거리, 종합병원 최근접거리처럼 상업 중심지, 교육, 고차 의료 접근성이 상위권에 들어왔습니다. 반면 비강남에서는 지하철 1km 내 역수, 어린이집 1km 내 개수, 학원 1km 내 개수, 지하철 최근접거리처럼 교통과 생활권 접근성이 반복적으로 중요하게 나타났습니다. 이는 전체 평균만으로는 보이지 않는 권역별 예측 기여 구조의 차이를 보여줍니다.");

  slide = addSlideWithHeader("04. RESULTS", "연도×권역: 2022년 조정기와 시공간 이질성", 9);
  addBox(slide, 0.75, 1.65, 3.4, 4.95, C.navy);
  addText(slide, "2022", {
    x: 0.95, y: 2.05, w: 3.0, h: 0.65,
    fontSize: 36, bold: true, color: C.yellow, align: "center",
  });
  addText(slide, "금리 급등과\n거래량 급감이 겹친\n조정기", {
    x: 1.0, y: 3.05, w: 2.9, h: 1.05,
    fontSize: 18, bold: true, color: C.white, align: "center", breakLine: true,
  });
  addText(slide, "세 권역 모두\n예측 안정성 저하", {
    x: 1.0, y: 4.75, w: 2.9, h: 0.7,
    fontSize: 15.5, color: "DDE4EE", align: "center", breakLine: true,
  });
  addBox(slide, 4.45, 1.65, 7.95, 3.25, C.soft, C.line);
  addText(slide, "연도별 예측 안정성", { x: 4.75, y: 1.95, w: 3.0, h: 0.32, fontSize: 17.5, bold: true, color: C.navy });
  addMiniTable(slide, ["구분", "2019", "2022", "2025"], [
    ["강남3구 R²", "0.894", "0.831", "0.906"],
    ["비강남 R²", "0.887", "0.788", "0.891"],
    ["전체 R²", "0.921", "0.841", "0.916"],
  ], 4.75, 2.55, 7.25, 0.66, [2.1, 1.7, 1.7, 1.75], { fontSize: 15.2, headerSize: 14.6, headerH: 0.56 });
  addBox(slide, 4.45, 5.22, 7.95, 1.38, C.soft, C.line);
  addBullets(slide, [
    "비강남은 건물연령·교통 변수의 반복성이 강함",
    "강남3구는 상위 변수가 시장 국면별로 이동",
  ], 4.75, 5.53, 7.25, 0.7, { fontSize: 16.2, spaceAfter: 3 });
  addNotes(slide, "연도×권역 분석", 65,
    "연도별로 보면 2022년이 가장 중요한 관찰 지점입니다. 2022년은 기준금리가 급등하고 거래량이 전년 대비 약 70% 감소한 조정기였습니다. 이 해에는 강남3구 R²가 0.831, 비강남 R²가 0.788, 전체 R²가 0.841로 모두 7년 중 낮은 수준을 보였습니다. 이는 시장 국면이 바뀔 때 기존 변수 조합의 예측 안정성이 약해질 수 있음을 보여줍니다. 또한 비강남은 건물연령, 교통, 생활권 변수가 반복적으로 상위권을 유지한 반면, 강남3구는 백화점, CCTV, 종합병원, 어린이집 등 상위 변수가 국면별로 이동했습니다.");

  slide = addSlideWithHeader("05. CONCLUSION", "결론: 예측 성능보다 설명 가능한 가격 구조", 10);
  addBox(slide, 0.75, 1.65, 3.8, 4.65, C.soft, C.line);
  addBox(slide, 4.8, 1.65, 3.8, 4.65, C.soft, C.line);
  addBox(slide, 8.85, 1.65, 3.8, 4.65, C.soft, C.line);
  addText(slide, "연구 결과", { x: 1.05, y: 2.0, w: 3.2, h: 0.34, fontSize: 18.2, bold: true, color: C.navy, align: "center" });
  addText(slide, "학술·실무 기여", { x: 5.1, y: 2.0, w: 3.2, h: 0.34, fontSize: 18.2, bold: true, color: C.navy, align: "center" });
  addText(slide, "한계와 후속 과제", { x: 9.15, y: 2.0, w: 3.2, h: 0.34, fontSize: 18.2, bold: true, color: C.navy, align: "center" });
  addBullets(slide, [
    "XGBoost가 비교 모형 상회",
    "거리 변수는 시간순 조건에서 유용",
    "권역·연도별 기여 구조 차이",
  ], 1.05, 2.58, 3.2, 2.35, { fontSize: 16.2, spaceAfter: 8 });
  addBullets(slide, [
    "단지 좌표 기반 접근성 적용",
    "연도별 활동 시설로 정보누수 방지",
    "권역·연도 SHAP으로 AVM 해석",
  ], 5.1, 2.58, 3.2, 2.35, { fontSize: 16.2, spaceAfter: 8 });
  addBullets(slide, [
    "SHAP은 예측 기여도",
    "도보 시간은 대체 근사",
    "신규 단지는 공간계량 보완 필요",
  ], 9.15, 2.58, 3.2, 2.35, { fontSize: 16.2, spaceAfter: 8 });
  addBox(slide, 0.75, 6.5, 11.9, 0.55, C.navy);
  addText(slide, "핵심 결론: 가격 구조는 거리 접근성·시장 국면·권역 조건이 결합된 예측 기여 구조로 읽어야 한다.", {
    x: 1.0, y: 6.64, w: 11.4, h: 0.26,
    fontSize: 14.6, bold: true, color: C.white,
    align: "center",
  });
  addNotes(slide, "결론", 65,
    "마지막으로 결론입니다. 본 연구는 서울 아파트 매매가격 구조를 거리 기반 접근성과 시공간 이질성의 관점에서 분석했습니다. XGBoost는 OLS와 랜덤 포레스트보다 높은 성능을 보였지만, 본 연구의 핵심은 성능 수치 자체보다 그 성능을 어떻게 보수적으로 해석할 것인가에 있습니다. 거리 기반 변수는 시간순 분할에서 유용했고, 시점 정합은 정보누수를 방지하는 방법론적 장치였습니다. 또한 강남3구와 비강남, 그리고 연도별 시장 국면에 따라 예측 기여 구조가 달랐습니다. 다만 SHAP은 인과효과가 아니며, 도보 시간 근사와 신규 단지 외삽의 한계가 있습니다. 후속 연구에서는 네트워크 거리, 단지 고정효과, 공간계량모형을 결합해 보완할 필요가 있습니다. 이상으로 발표를 마치겠습니다.");
}

function writeScriptMarkdown() {
  const lines = [
    "# 석사학위논문 최종심사 발표 스크립트",
    "",
    `- 발표자: ${AUTHOR}`,
    `- 소속: ${SCHOOL} ${MAJOR}`,
    "- 발표 분량: 약 10분",
    "- 기준 원문: `paper/석사학위논문_박현근.md`",
    "",
  ];
  deckNotes.forEach((item, index) => {
    lines.push(`## ${index + 1}. ${item.title} (${item.seconds}초)`);
    lines.push("");
    lines.push(item.notes);
    lines.push("");
  });
  const total = deckNotes.reduce((sum, item) => sum + item.seconds, 0);
  lines.push(`총 예상 시간: ${Math.floor(total / 60)}분 ${total % 60}초`);
  lines.push("");
  fs.writeFileSync(SCRIPT_OUT, lines.join("\n"), "utf8");
}

async function main() {
  buildDeck();
  writeScriptMarkdown();
  await pptx.writeFile({ fileName: OUT });
  console.log(OUT);
  console.log(SCRIPT_OUT);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
