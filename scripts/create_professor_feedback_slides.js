const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const OUT = path.join(__dirname, "..", "paper", "교수별_심사의견_반영내용_박현근.pptx");

const pptx = new pptxgen();
const FONT = "Pretendard";
pptx.layout = "LAYOUT_WIDE";
pptx.author = "박현근";
pptx.company = "한양대학교 부동산융합대학원";
pptx.subject = "석사학위논문 심사 의견 반영 내용";
pptx.title = "교수별 심사 의견 반영 내용";
pptx.lang = "ko-KR";
pptx.theme = {
  headFontFace: FONT,
  bodyFontFace: FONT,
  lang: "ko-KR",
};
pptx.defineLayout({ name: "WIDE", width: 13.333, height: 7.5 });
pptx.layout = "WIDE";

const C = {
  navy: "262B43",
  navy2: "1E2438",
  yellow: "F4D35E",
  gray: "AEB4C0",
  line: "B7BDC6",
  header: "DCEBEC",
  body: "F7F8FA",
  white: "FFFFFF",
  text: "2B2F3A",
  muted: "555D6A",
};

function addTop(slide, title = "심사 의견 반영 내용") {
  slide.background = { color: "FFFFFF" };
  slide.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: 13.333, h: 0.72, fill: { color: C.navy }, line: { color: C.navy } });
  slide.addShape(pptx.ShapeType.rect, { x: 0, y: 0.72, w: 13.333, h: 0.06, fill: { color: C.gray }, line: { color: C.gray } });
  slide.addShape(pptx.ShapeType.rect, { x: 11.15, y: 0, w: 0.75, h: 0.72, fill: { color: "E6E8ED" }, line: { color: "E6E8ED" } });
  slide.addShape(pptx.ShapeType.rect, { x: 11.9, y: 0, w: 0.75, h: 0.72, fill: { color: "C8CCD5" }, line: { color: "C8CCD5" } });
  slide.addShape(pptx.ShapeType.rect, { x: 12.65, y: 0, w: 0.68, h: 0.72, fill: { color: C.navy }, line: { color: C.navy } });
  slide.addText(title, {
    x: 0.75, y: 0.1, w: 5.2, h: 0.46, margin: 0,
    fontFace: FONT, fontSize: 21, bold: true, color: C.yellow,
  });
}

function addProfessor(slide, name, subtitle) {
  slide.addText(name, {
    x: 0.83, y: 0.98, w: 3.9, h: 0.34, margin: 0,
    fontFace: FONT, fontSize: 21, bold: true, color: "515879",
  });
  slide.addShape(pptx.ShapeType.line, { x: 0.83, y: 1.42, w: 11.75, h: 0, line: { color: C.gray, width: 1.2 } });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 8.1, y: 1.02, w: 4.5, h: 0.24, margin: 0,
      fontFace: FONT, fontSize: 8.5, color: "6D7280", align: "right",
    });
  }
}

function fitFont(txt, base, min = 8.2) {
  const len = String(txt).replace(/\s+/g, "").length;
  if (len > 160) return Math.max(min, base - 3.8);
  if (len > 125) return Math.max(min, base - 3.0);
  if (len > 95) return Math.max(min, base - 2.0);
  if (len > 70) return Math.max(min, base - 1.2);
  return base;
}

function addWrappedText(slide, text, opts) {
  slide.addText(text, {
    fontFace: FONT,
    breakLine: false,
    fit: "shrink",
    valign: "mid",
    margin: opts.margin ?? 0.06,
    ...opts,
  });
}

function drawTable(slide, rows, options = {}) {
  const x = 0.83;
  const y = 1.72;
  const w = 11.75;
  const headerH = 0.52;
  const rowH = options.rowH ?? 0.82;
  const widths = options.widths ?? [4.75, 5.95, 1.05];
  const totalH = headerH + rowH * rows.length;
  const headers = ["심사 의견", "반영 사항", "반영 페이지"];

  slide.addShape(pptx.ShapeType.rect, { x, y, w, h: totalH, fill: { color: C.white }, line: { color: C.line, width: 0.7 } });
  let cx = x;
  headers.forEach((h, i) => {
    slide.addShape(pptx.ShapeType.rect, { x: cx, y, w: widths[i], h: headerH, fill: { color: C.header }, line: { color: C.line, width: 0.7 } });
    slide.addText(h, { x: cx, y: y + 0.14, w: widths[i], h: 0.22, margin: 0, fontFace: FONT, fontSize: 13, bold: true, color: C.text, align: "center", valign: "mid" });
    cx += widths[i];
  });

  rows.forEach((r, idx) => {
    const ry = y + headerH + rowH * idx;
    const fill = idx % 2 === 0 ? C.body : C.white;
    cx = x;
    [r.opinion, r.response, r.page].forEach((txt, i) => {
      slide.addShape(pptx.ShapeType.rect, { x: cx, y: ry, w: widths[i], h: rowH, fill: { color: fill }, line: { color: C.line, width: 0.55 } });
      const fontSize = i === 0 ? fitFont(txt, 10.8, 8.0) : i === 1 ? fitFont(txt, 9.8, 7.9) : fitFont(txt, 9.5, 7.7);
      addWrappedText(slide, txt, {
        x: cx + 0.07,
        y: ry + 0.06,
        w: widths[i] - 0.14,
        h: rowH - 0.12,
        fontSize,
        color: i === 2 ? C.muted : C.text,
        bold: i === 0,
        align: i === 2 ? "center" : "left",
        valign: "mid",
      });
      cx += widths[i];
    });
  });
}

function addFeedbackSlide(name, subtitle, rows, opts = {}) {
  const slide = pptx.addSlide();
  addTop(slide);
  addProfessor(slide, name, subtitle);
  drawTable(slide, rows, opts);
}

function bullets(items) {
  return items.map((item) => `• ${item}`).join("\n");
}

const jomijeongRows = [
  {
    opinion: "1. 제1장에 연구 방법 및 과정과 연구 흐름도를 제시",
    response: bullets([
      "제1장 제3절을 '연구 방법 및 과정'으로 정리",
      "본문 설명 뒤에 <그림 1-1> 연구의 흐름도 배치",
    ]),
    page: "p.7~9",
  },
  {
    opinion: "2. 이론적 배경은 논문 주제와 결론에 연결되도록 재구성",
    response: bullets([
      "아파트 가격 결정요인, 머신러닝 가격 예측, XAI 해석 연구로 범주 축소",
      "논문 주제와 결론 흐름에 맞게 선행연구 재배열",
    ]),
    page: "p.10~28",
  },
  {
    opinion: "3. 제3장 제목과 절 구성을 연구 설계 중심으로 정리",
    response: bullets([
      "제3장 제목을 '분석의 틀'로 변경",
      "연구 방법, 변수의 정의 및 구축, 연구 모형 순서로 재편",
    ]),
    page: "p.29~43",
  },
  {
    opinion: "4. 제목 직후 그림·표를 두지 말고 본문 설명 뒤 배치",
    response: bullets([
      "연구 흐름도와 주요 표·그림을 본문 설명 뒤로 이동",
      "그림 제목은 그림 아래 가운데 정렬로 통일",
    ]),
    page: "p.9, p.44~72",
  },
  {
    opinion: "5. 변수 설명에서 내부 영문 코드와 불명확한 주석 제거",
    response: bullets([
      "원자료 필드명과 코드식 표현 제거",
      "개교일·설립일·폐업일 등 본문 설명형 표현으로 대체",
    ]),
    page: "p.30~37",
  },
  {
    opinion: "6. 표 제목은 좌측, 그림 제목은 아래 가운데로 정리",
    response: bullets([
      "표 캡션은 좌측 정렬로 통일",
      "그림 캡션은 하단 중앙 정렬 적용 후 렌더 이미지로 확인",
    ]),
    page: "전체",
  },
  {
    opinion: "7. 결론은 요약 중심으로 두고 시사점은 분석 결과에서 처리",
    response: bullets([
      "'시사점' 독립 절 삭제",
      "제5장을 연구 결과 요약, 한계 및 향후 과제 중심으로 재구성",
    ]),
    page: "p.74~79",
  },
  {
    opinion: "8. 신청서 기준 심사위원명과 인준서 순서 확인",
    response: bullets([
      "학위청구논문제출신청서 기준 심사위원명 확인",
      "인준서에 조미정 위원장, 엄선용·고준호 위원 순서 반영",
    ]),
    page: "인준서",
  },
];

const goRowsA = [
  {
    opinion: "1. 제목에서 '단위면적당'과 'Unit-Area' 표현 삭제",
    response: bullets([
      "국문 제목에서 '단위면적당' 삭제",
      "영문 제목에서 'Unit-Area' 삭제 후 동일 의미 축으로 정리",
    ]),
    page: "표지·제출서",
  },
  {
    opinion: "2. 주요어의 '권역×연도 이질성' 표현 부적절",
    response: bullets([
      "국문 주요어의 해당 표현 삭제",
      "Keywords는 'spatiotemporal heterogeneity'로 수정",
    ]),
    page: "초록, p.89",
  },
  {
    opinion: "3. 외국 저자 인용 표기와 et al. 띄어쓰기 정리",
    response: bullets([
      "Lundberg & Lee, Choy & Ho, Kim, Choi & Lee 표기 통일",
      "et al. (연도) 띄어쓰기와 형식을 본문 전체에 적용",
    ]),
    page: "전체",
  },
  {
    opinion: "4. 헤도닉 가격모형 한계 주장에 인용문헌 필요",
    response: bullets([
      "헤도닉 모형의 선형성 한계 설명 보강",
      "Čeh et al. (2018), Limsombunchai (2004) 인용 맥락 유지",
    ]),
    page: "p.10~13",
  },
  {
    opinion: "5. 보정계수 1.35의 근거 또는 문헌 필요",
    response: bullets([
      "Boeing (2019)을 circuity 개념 근거로 추가",
      "1.35는 문헌 상수가 아닌 휴리스틱 환산값으로 한계 명시",
    ]),
    page: "p.34~36",
  },
  {
    opinion: "6. GroupKFold 같은 전문용어는 근거와 개념 설명 필요",
    response: bullets([
      "GroupKFold 구현명 노출 제거",
      "동일 단지 반복 거래가 학습·평가에 동시에 들어가지 않는 그룹 분할 검증으로 설명",
    ]),
    page: "p.40~43",
  },
  {
    opinion: "7. n_estimators, max_depth 등 코드 기호는 본문에 불필요",
    response: bullets([
      "n_estimators, max_depth 등 코드식 변수명 삭제",
      "트리의 수, 최대 깊이, 최소 리프 노드 표본 수 등 원 용어로 재작성",
    ]),
    page: "p.14~18, p.38~43",
  },
];

const goRowsB = [
  {
    opinion: "8. '전체 수준 해석:', '결정계수(R²):' 등 나열식 표현 지양",
    response: bullets([
      "콜론형·사전형 설명을 일반 논문 문장으로 재구성",
      "R²·RMSE·MAE 설명도 문장형으로 정리",
    ]),
    page: "p.38~43",
  },
  {
    opinion: "9. 표만 제시하지 말고 설명 추가",
    response: bullets([
      "<표 4-7> 앞에 분석 목적 문단 추가",
      "표 뒤에 결과 해석 문단 추가",
    ]),
    page: "p.53~55",
  },
  {
    opinion: "10. 제목에 표 번호를 넣지 말고 다른 부분도 확인",
    response: bullets([
      "소제목의 '(<표 4-x>)' 패턴 모두 제거",
      "표 번호는 표 캡션과 본문 설명에서만 사용하도록 정리",
    ]),
    page: "p.50~72",
  },
  {
    opinion: "11. SHAP 상위 변수 표의 비교 의미 설명 필요",
    response: bullets([
      "권역별·연도별 SHAP 상위 변수 표 앞에 비교 목적 추가",
      "해석 방향을 문장으로 보강",
    ]),
    page: "p.62~72",
  },
  {
    opinion: "12. '— 핵심 해석' 등 불필요한 강조형 소제목 삭제",
    response: bullets([
      "강조형 소제목 삭제",
      "'권역별 예측 기여 구조의 종합'으로 바꾸고 학술 문장으로 재구성",
    ]),
    page: "p.64~66",
  },
  {
    opinion: "13. 표지·영문 제목 줄바꿈과 글꼴 불일치 확인",
    response: bullets([
      "Word 화면 기준 제목 줄바꿈을 의미 단위 3줄로 정리",
      "영문 제목 3줄의 글꼴·크기·굵기 통일",
    ]),
    page: "표지·제출서",
  },
  {
    opinion: "14. 목차 번호와 실제 페이지 번호 불일치 확인",
    response: bullets([
      "Word 렌더링 기준 실제 페이지 번호 추출",
      "목차·표목차·그림목차 번호 고정 및 제목 불일치 0건 확인",
    ]),
    page: "목차",
  },
];

addFeedbackSlide("조미정 교수님 (1)", "근거: 음성 전사, 수정 PDF·스캔, 참고논문·한양대 양식", jomijeongRows.slice(0, 4), {
  rowH: 1.02,
  footer: "근거 파일: 논문_조미정교수.m4a, 조미정_수정.pdf, Scan_20260531_152358.pdf, review_inputs/transcripts/thesis_jomijeong_20260601.txt",
});
addFeedbackSlide("조미정 교수님 (2)", "근거: 음성 전사, 수정 PDF·스캔, 참고논문·한양대 양식", jomijeongRows.slice(4), {
  rowH: 1.02,
  footer: "근거 파일: 논문_조미정교수.m4a, 조미정_수정.pdf, Scan_20260531_152358.pdf, review_inputs/transcripts/thesis_jomijeong_20260601.txt",
});
addFeedbackSlide("고준호 교수님 (1)", "근거: Word 메모 28개, 원문-메모 교차분석", goRowsA.slice(0, 4), {
  rowH: 1.02,
  footer: "근거 파일: paper/석사학위논문_박현근_의견0603.docx, paper/지도교수_피드백_원문교차분석_20260604.md",
});
addFeedbackSlide("고준호 교수님 (2)", "근거: Word 메모 28개, 원문-메모 교차분석", goRowsA.slice(4), {
  rowH: 1.22,
  footer: "근거 파일: paper/석사학위논문_박현근_의견0603.docx, paper/지도교수_피드백_원문교차분석_20260604.md",
});
addFeedbackSlide("고준호 교수님 (3)", "근거: Word 메모 28개, 최종 검증 결과", goRowsB.slice(0, 4), {
  rowH: 1.02,
  footer: "검증: 잔여 패턴 검색 0건, DOCX 내부 PAGEREF 0건, 목차 제목 불일치 0건",
});
addFeedbackSlide("고준호 교수님 (4)", "근거: Word 메모 28개, 최종 검증 결과", goRowsB.slice(4), {
  rowH: 1.22,
  footer: "검증: 잔여 패턴 검색 0건, DOCX 내부 PAGEREF 0건, 목차 제목 불일치 0건",
});

pptx.writeFile({ fileName: OUT });
console.log(OUT);
