#!/usr/bin/env python3
"""
논문 HWPX 변환 — 한양대학교 부동산융합대학원 석사논문
scripts/export_docx_v2.py 기반 포팅.

주의: python-hwpx 2.9.0 의 HwpxDocument 는 문단/표/이미지를 추가할 수는 있으나
폰트 크기·굵기·정렬 등 세밀한 스타일 제어 API 가 제한적이다.
따라서 본 스크립트는 '최소 동작(제목 + 본문 텍스트 + 표 + 이미지)' 까지를
목표로 하며, 세부 서식(바탕체 11pt, 160% 줄간격, 중앙정렬 등)은
한컴오피스에서 서식 템플릿을 적용하거나 DOCX 결과물을 참고해
수동으로 맞출 것을 권장한다.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from hwpx import HwpxDocument

BASE_DIR = Path(__file__).resolve().parent.parent
PAPER_DIR = BASE_DIR / "paper"
MD_PATH = PAPER_DIR / "논문_초안.md"
OUT_PATH = PAPER_DIR / "논문_초안.hwpx"


# ---------- 유틸 ----------

def add_blank(doc: HwpxDocument, n: int = 1) -> None:
    for _ in range(n):
        doc.add_paragraph("")


def add_centered(doc: HwpxDocument, text: str) -> None:
    # API 가 정렬 제어를 쉽게 노출하지 않으므로 텍스트만 삽입.
    # 중앙정렬은 한컴에서 템플릿 스타일로 처리.
    doc.add_paragraph(text)


def add_heading(doc: HwpxDocument, text: str, level: int = 1) -> None:
    # 레벨 표시만 남겨서 후처리에서 인식 가능하게 한다.
    if level == 1:
        doc.add_paragraph("")  # 장 사이 여백
        doc.add_paragraph(text)
        doc.add_paragraph("")
    elif level == 2:
        doc.add_paragraph(text)
    else:
        doc.add_paragraph(text)


def add_body(doc: HwpxDocument, text: str) -> None:
    if not text.strip():
        return
    # 볼드 마커(**) 제거 — API 가 부분 run 스타일을 쉽게 지원하지 않음
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    doc.add_paragraph(cleaned)


def is_sep(line: str) -> bool:
    return bool(re.match(r"^\s*\|[\s:\-]+\|\s*$", line))


def parse_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def add_table(doc: HwpxDocument, rows: list[list[str]], caption: str = "") -> None:
    if caption:
        doc.add_paragraph(caption)
    if not rows:
        return
    ncols = max(len(r) for r in rows)
    try:
        tbl = doc.add_table(len(rows), ncols)
    except Exception as e:
        # 표 생성 실패 시 텍스트로 대체
        doc.add_paragraph(f"[표 생성 실패: {e}]")
        for r in rows:
            doc.add_paragraph(" | ".join(r))
        return
    for i, row in enumerate(rows):
        for j in range(ncols):
            txt = row[j] if j < len(row) else ""
            try:
                tbl.set_cell_text(i, j, txt)
            except Exception:
                pass
    doc.add_paragraph("")


def add_image(doc: HwpxDocument, path: Path, caption: str = "") -> None:
    if not path.exists():
        doc.add_paragraph(f"[그림 파일 없음: {path.name}]")
        if caption:
            doc.add_paragraph(caption)
        return
    try:
        data = path.read_bytes()
        fmt = path.suffix.lstrip(".").lower() or "png"
        item_id = doc.add_image(data, fmt)
        # 이미지 참조만 매니페스트에 등록. 실제 본문 배치 <hp:pic> 구성은
        # API 수준에서 간단히 제공되지 않으므로 캡션+참조id 문단으로 표기.
        doc.add_paragraph(f"[그림: {path.name} (id={item_id})]")
    except Exception as e:
        doc.add_paragraph(f"[이미지 삽입 실패 {path.name}: {e}]")
    if caption:
        doc.add_paragraph(caption)


# ---------- 표지/제출서/인준서 ----------

def add_cover(doc: HwpxDocument) -> None:
    add_blank(doc, 4)
    add_centered(doc, "석 사 학 위 논 문")
    add_blank(doc, 2)
    add_centered(doc, "XGBoost와 SHAP을 활용한")
    add_centered(doc, "서울시 아파트 매매가격 결정요인 분석")
    add_blank(doc)
    add_centered(doc, "Analysis of Determinants of Apartment Sale Prices")
    add_centered(doc, "in Seoul Using XGBoost and SHAP")
    add_blank(doc, 5)
    add_centered(doc, "[성 명]")
    add_blank(doc, 2)
    add_centered(doc, "한 양 대 학 교  부 동 산 융 합 대 학 원")
    add_blank(doc)
    add_centered(doc, "2026 년  2 월")
    add_blank(doc, 2)


def add_submission(doc: HwpxDocument) -> None:
    add_blank(doc, 4)
    add_centered(doc, "석 사 학 위 논 문")
    add_blank(doc, 2)
    add_centered(doc, "XGBoost와 SHAP을 활용한")
    add_centered(doc, "서울시 아파트 매매가격 결정요인 분석")
    add_blank(doc)
    add_centered(doc, "Analysis of Determinants of Apartment Sale Prices")
    add_centered(doc, "in Seoul Using XGBoost and SHAP")
    add_blank(doc, 2)
    add_centered(doc, "지도교수  [지도교수명]")
    add_blank(doc, 2)
    add_centered(doc, "이 논문을 공학 석사학위논문으로 제출합니다.")
    add_blank(doc, 2)
    add_centered(doc, "2026 년  2 월")
    add_blank(doc)
    add_centered(doc, "한 양 대 학 교  부 동 산 융 합 대 학 원")
    add_blank(doc)
    add_centered(doc, "도시·부동산빅데이터 전공")
    add_blank(doc)
    add_centered(doc, "[성 명]")
    add_blank(doc, 2)


def add_approval(doc: HwpxDocument) -> None:
    add_blank(doc, 3)
    add_centered(doc, "이 논문을 [성명]의")
    add_centered(doc, "석사학위 논문으로 인준함.")
    add_blank(doc, 2)
    add_centered(doc, "2026 년  2 월")
    add_blank(doc, 4)
    for role in ["심 사 위 원 장", "심  사  위  원", "심  사  위  원"]:
        doc.add_paragraph(f"{role} :  ________________  (인)")
        add_blank(doc)
    add_blank(doc, 3)
    add_centered(doc, "한 양 대 학 교  부 동 산 융 합 대 학 원")
    add_blank(doc, 2)


# ---------- Markdown 파서 ----------

def convert_md(doc: HwpxDocument, md_path: Path) -> None:
    lines = md_path.read_text(encoding="utf-8").splitlines()

    start = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("# 국문초록") or s.startswith("# 국문 초록"):
            start = i
            break

    i = start
    caption = ""

    while i < len(lines):
        line = lines[i].rstrip()

        # 수식 블록
        if line.strip() == "$$":
            parts = []
            i += 1
            while i < len(lines) and lines[i].strip() != "$$":
                parts.append(lines[i].strip())
                i += 1
            doc.add_paragraph(" ".join(parts))
            i += 1
            continue

        # 표 캡션
        m = re.match(r"^\*\*(<표[^>]*>[^*]*)\*\*$", line.strip())
        if m:
            caption = m.group(1)
            i += 1
            continue

        # 표
        if "|" in line and line.strip().startswith("|") and not is_sep(line):
            rows = []
            while i < len(lines):
                l = lines[i].rstrip()
                if "|" in l and l.strip().startswith("|"):
                    if not is_sep(l):
                        rows.append(parse_row(l))
                    i += 1
                else:
                    break
            add_table(doc, rows, caption)
            caption = ""
            continue

        if is_sep(line):
            i += 1
            continue

        # 그림
        m = re.match(r"!\[(.+?)\]\((.+?)\)", line)
        if m:
            cap = m.group(1)
            rel = m.group(2)
            absp = (PAPER_DIR / rel).resolve()
            add_image(doc, absp, cap)
            i += 1
            continue

        # 제목
        if line.startswith("# "):
            add_heading(doc, line[2:].strip(), 1)
            i += 1
            continue
        if line.startswith("## "):
            add_heading(doc, line[3:].strip(), 2)
            i += 1
            continue
        if line.startswith("### "):
            add_heading(doc, line[4:].strip(), 3)
            i += 1
            continue

        if line.strip() == "---":
            add_blank(doc)
            i += 1
            continue

        if line.strip().startswith("- <표") or line.strip().startswith("- <그림"):
            i += 1
            continue

        if not line.strip():
            i += 1
            continue

        add_body(doc, line)
        i += 1


def main() -> int:
    print("논문 HWPX 생성 (한양대 공식 서식 기반, 최소 포맷)")

    doc = HwpxDocument.new()

    add_cover(doc)
    add_submission(doc)
    add_approval(doc)
    convert_md(doc, MD_PATH)

    doc.save(str(OUT_PATH))

    size_kb = OUT_PATH.stat().st_size // 1024
    print(f"완료: {OUT_PATH}")
    print(f"크기: {size_kb}KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
