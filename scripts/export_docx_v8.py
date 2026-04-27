#!/usr/bin/env python3
"""
논문 DOCX 변환 v2 — 한양대학교 부동산융합대학원 석사논문 공식 서식
출처: https://gupd.hanyang.ac.kr/front/ko/bachelor/thesis
참고: 조민지(2023) 한양대 석사논문

편집용지: A4
워드 여백: 위 4cm, 아래 4cm, 좌 3.5cm, 우 3.5cm, 머리말 1.5cm, 꼬리말 1.5cm
본문: 10~11pt 바탕체, 줄간격 160~200%
큰제목(장): 16pt 진하게 / 중간제목(절): 13pt 진하게
"""
import os, re, io
import latex2mathml.converter as _l2m
import mathml2omml as _m2o
from docx import Document

OMML_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/math'

from docx.shared import Pt, Cm, Mm, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PAPER_DIR = os.path.join(BASE_DIR, 'paper')

FONT_KR = '바탕'


def set_font(run, size=11, bold=False, font=FONT_KR, italic=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = font
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = parse_xml(f'<w:rFonts {nsdecls("w")} w:eastAsia="{font}"/>')
        rPr.append(rFonts)
    else:
        rFonts.set(qn('w:eastAsia'), font)


def set_shading(cell, color):
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color}"/>')
    cell._tc.get_or_add_tcPr().append(shading)


def add_page_number(section, start_num=None, fmt='decimal', placeholder_fmt=None):
    """페이지 번호 추가 (하단 중앙, "- N -" 형식). fmt='decimal'|'lowerRoman'.

    placeholder_fmt: LibreOffice 호환을 위해 sectPr에 pgNumType을 2개 두는데,
    placeholder는 reset되지 않은 fmt이고 그 다음 element가 reset.
    fmt와 placeholder_fmt가 다를 때만 LibreOffice가 reset 인식. None이면 fmt 사용.
    """
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)

    r1 = p.add_run('- ')
    set_font(r1, 10)

    # 페이지 번호 필드
    fld_xml = (
        f'<w:fldSimple {nsdecls("w")} w:instr=" PAGE \\* {fmt} ">'
        f'<w:r><w:rPr><w:sz w:val="20"/></w:rPr><w:t>1</w:t></w:r>'
        f'</w:fldSimple>'
    )
    p._element.append(parse_xml(fld_xml))

    r2 = p.add_run(' -')
    set_font(r2, 10)

    # 시작 번호 지정 (front matter는 i부터, body는 1부터 재시작)
    # NOTE: LibreOffice/Word는 두 번째 섹션의 pgNumType.start reset이 sectPr 안에
    #       pgNumType element가 2개 이상 있을 때만 인식하는 동작이 있다. 따라서
    #       기존 element 모두 제거 + placeholder + reset 두 개를 추가한다.
    if start_num is not None:
        sectPr = section._sectPr
        for ex in list(sectPr.findall(qn('w:pgNumType'))):
            sectPr.remove(ex)
        ph_fmt = placeholder_fmt or ('lowerRoman' if fmt == 'decimal' else 'decimal')
        placeholder = parse_xml(
            f'<w:pgNumType {nsdecls("w")} w:fmt="{ph_fmt}"/>'
        )
        new_el = parse_xml(
            f'<w:pgNumType {nsdecls("w")} w:start="{start_num}" w:fmt="{fmt}"/>'
        )
        cols = sectPr.find(qn('w:cols'))
        if cols is not None:
            cols.addprevious(placeholder)
            cols.addprevious(new_el)
        else:
            sectPr.append(placeholder)
            sectPr.append(new_el)



_BM_ID = [1000]


def _next_bm_id():
    _BM_ID[0] += 1
    return _BM_ID[0]


def wrap_paragraph_with_bookmark(p, name):
    """기존 paragraph p의 콘텐츠를 bookmarkStart/End로 감싼다."""
    bid = _next_bm_id()
    bs = parse_xml(f'<w:bookmarkStart {nsdecls("w")} w:id="{bid}" w:name="{name}"/>')
    be = parse_xml(f'<w:bookmarkEnd {nsdecls("w")} w:id="{bid}"/>')
    pPr = p._p.find(qn('w:pPr'))
    if pPr is not None:
        pPr.addnext(bs)
    else:
        p._p.insert(0, bs)
    p._p.append(be)


def append_pageref_field(p, name, font='바탕', size=11, bold=False):
    """Paragraph p 끝에 PAGEREF <name> \\h 필드를 삽입.
    PAGE 자리 표시는 0으로 두고 settings.xml의 updateFields가 채운다."""
    rPr_xml = f'<w:rPr><w:rFonts {nsdecls("w")} w:eastAsia="{font}"/><w:sz w:val="{size*2}"/>{("<w:b/>") if bold else ""}</w:rPr>'
    begin = parse_xml(f'<w:r {nsdecls("w")}>{rPr_xml}<w:fldChar w:fldCharType="begin"/></w:r>')
    instr = parse_xml(
        f'<w:r {nsdecls("w")}>{rPr_xml}<w:instrText xml:space="preserve"> PAGEREF {name} \\h </w:instrText></w:r>'
    )
    sep = parse_xml(f'<w:r {nsdecls("w")}>{rPr_xml}<w:fldChar w:fldCharType="separate"/></w:r>')
    placeholder = parse_xml(f'<w:r {nsdecls("w")}>{rPr_xml}<w:t>0</w:t></w:r>')
    end = parse_xml(f'<w:r {nsdecls("w")}>{rPr_xml}<w:fldChar w:fldCharType="end"/></w:r>')
    for el in (begin, instr, sep, placeholder, end):
        p._p.append(el)


def add_toc_entry(doc, text, bm_name, level=0, bold=False, tab_pos_cm=14.0):
    """점 리더 + 우측 정렬 페이지번호로 구성된 목차 항목."""
    from docx.enum.text import WD_TAB_ALIGNMENT, WD_TAB_LEADER
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.left_indent = Cm(0.7 * level)
    p.paragraph_format.line_spacing = 1.6
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.tab_stops.add_tab_stop(
        Cm(tab_pos_cm), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS
    )
    r = p.add_run(text)
    set_font(r, 11, bold)
    rt = p.add_run('\t')
    set_font(rt, 11, bold)
    if bm_name:
        append_pageref_field(p, bm_name, size=11, bold=bold)
    return p


def _empty(doc, n=1):
    """빈 줄 추가 (줄간격 1.0 강제 — 표지/제출서 페이지 초과 방지)"""
    from docx.shared import Pt as _Pt
    for _ in range(n):
        p = doc.add_paragraph('')
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_before = _Pt(0)
        p.paragraph_format.space_after = _Pt(0)

def create_doc():
    doc = Document()
    
    # 기본 섹션 설정
    section = doc.sections[0]
    section.page_width = Mm(210)
    section.page_height = Mm(297)
    section.top_margin = Cm(4.0)
    section.bottom_margin = Cm(4.0)
    section.left_margin = Cm(3.5)
    section.right_margin = Cm(3.5)
    section.header_distance = Cm(1.5)
    section.footer_distance = Cm(1.5)
    
    # 기본 스타일
    style = doc.styles['Normal']
    style.font.name = FONT_KR
    style.font.size = Pt(11)
    style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    style.paragraph_format.line_spacing = 1.6
    style.paragraph_format.first_line_indent = Cm(0.8)
    rPr = style.element.get_or_add_rPr()
    rFonts = parse_xml(f'<w:rFonts {nsdecls("w")} w:eastAsia="{FONT_KR}"/>')
    rPr.append(rFonts)
    
    return doc


def latex_to_omml(latex, block=False):
    """LaTeX -> MathML -> OMML XML element (with m: namespace). block=True wraps in oMathPara."""
    mml = _l2m.convert(latex)
    omml_str = _m2o.convert(mml)  # starts with <m:oMath>...</m:oMath>
    # Bug workaround: mathml2omml 0.0.2 emits malformed closing tag for groupChrPr.
    omml_str = re.sub(
        r'(<m:groupChrPr>[^<]*<m:chr[^/]*/>[^<]*<m:pos[^/]*/>)\s*</m:groupChr>',
        r'\1</m:groupChrPr>',
        omml_str,
    )
    # Inject namespace on root element so parse_xml can resolve m: prefix.
    omml_ns = omml_str.replace('<m:oMath>', f'<m:oMath xmlns:m="{OMML_NS}">', 1)
    if block:
        omml_ns = (
            f'<m:oMathPara xmlns:m="{OMML_NS}">'
            f'{omml_ns}'
            f'</m:oMathPara>'
        )
    return parse_xml(omml_ns)


def add_block_math(doc, latex):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    try:
        elem = latex_to_omml(latex, block=True)
        p._p.append(elem)
    except Exception as e:
        r = p.add_run(latex)
        r.font.size = Pt(10)
        r.font.name = 'Cambria Math'
        r.italic = True
        print(f'  [math fallback-block] {latex[:60]} -> {e}')
    return p


def add_inline_math_run(paragraph, latex):
    try:
        elem = latex_to_omml(latex, block=False)
        paragraph._p.append(elem)
        return None
    except Exception as e:
        r = paragraph.add_run(latex)
        r.font.name = 'Cambria Math'
        r.italic = True
        print(f'  [math fallback-inline] {latex[:60]} -> {e}')
        return r


def centered_text(doc, text, size=14, bold=False, space_before=0, space_after=0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.line_spacing = 1.15
    if space_before:
        p.paragraph_format.space_before = Pt(space_before)
    if space_after:
        p.paragraph_format.space_after = Pt(space_after)
    r = p.add_run(text)
    set_font(r, size, bold)
    return p


def add_cover(doc):
    """양식1: 표지"""
    _empty(doc, 4)
    centered_text(doc, '석 사 학 위 논 문', 16, True)
    _empty(doc, 2)
    centered_text(doc, 'XGBoost와 SHAP을 활용한', 18, True)
    centered_text(doc, '서울시 아파트 단위면적당 매매가격의', 18, True)
    centered_text(doc, '설명 패턴 분석', 18, True)
    _empty(doc)
    centered_text(doc, 'Explanatory Patterns of Apartment Unit-Area', 14, True)
    centered_text(doc, 'Sale Prices in Seoul Using XGBoost and SHAP', 14, True)
    _empty(doc, 5)
    centered_text(doc, '2026 년  2 월', 14, True)
    _empty(doc)
    centered_text(doc, '한 양 대 학 교  부 동 산 융 합 대 학 원', 14, True)
    _empty(doc)
    centered_text(doc, '도 시 부 동 산 정 책 전 공', 13)
    _empty(doc)
    centered_text(doc, '박  현  근', 16, True)
    doc.add_page_break()


def add_submission(doc):
    """양식2: 제출서"""
    _empty(doc, 4)
    centered_text(doc, '석 사 학 위 논 문', 16, True)
    _empty(doc, 2)
    centered_text(doc, 'XGBoost와 SHAP을 활용한', 18, True)
    centered_text(doc, '서울시 아파트 단위면적당 매매가격의', 18, True)
    centered_text(doc, '설명 패턴 분석', 18, True)
    _empty(doc)
    centered_text(doc, 'Explanatory Patterns of Apartment Unit-Area', 14, True)
    centered_text(doc, 'Sale Prices in Seoul Using XGBoost and SHAP', 14, True)
    _empty(doc, 2)
    centered_text(doc, '지도교수  고  준  호', 14, True)
    _empty(doc, 2)
    centered_text(doc, '이 논문을 부동산학 석사학위논문으로 제출합니다.', 12)
    _empty(doc, 2)
    centered_text(doc, '2026 년  2 월', 14, True)
    _empty(doc)
    centered_text(doc, '한 양 대 학 교  부 동 산 융 합 대 학 원', 14, True)
    _empty(doc)
    centered_text(doc, '도 시 부 동 산 정 책 전 공', 13)
    _empty(doc)
    centered_text(doc, '박  현  근', 16, True)
    doc.add_page_break()


def add_approval(doc):
    """양식3: 인준서"""
    _empty(doc, 3)
    centered_text(doc, '이 논문을 박현근의', 14)
    centered_text(doc, '석사학위 논문으로 인준함.', 14)
    _empty(doc, 2)
    centered_text(doc, '2026 년  2 월', 14, True)
    _empty(doc, 4)
    
    for role in ['심 사 위 원 장', '심  사  위  원', '심  사  위  원']:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.space_after = Pt(24)
        r = p.add_run(f'{role} :  ________________  (인)')
        set_font(r, 14)
    
    _empty(doc, 3)
    centered_text(doc, '한 양 대 학 교  부 동 산 융 합 대 학 원', 14, True)
    doc.add_page_break()


def add_heading(doc, text, level=1, bm_name=None):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0)
    # 한양대 표준: 헤딩 줄간격 160%, 들여쓰기 0
    p.paragraph_format.line_spacing = 1.6
    # 표준 Heading 스타일 적용 (TOC·목차 인식 + grep false positive 제거)
    style_name = f'Heading {level}'
    try:
        p.style = doc.styles[style_name]
    except Exception:
        pass

    if level == 1:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(24)
        p.paragraph_format.space_after = Pt(18)
        r = p.add_run(text)
        set_font(r, 16, True)
    elif level == 2:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(18)
        p.paragraph_format.space_after = Pt(12)
        r = p.add_run(text)
        set_font(r, 13, True)
    elif level == 3:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(8)
        r = p.add_run(text)
        set_font(r, 11, True)
    elif level == 4:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run(text)
        set_font(r, 12, True)
    if bm_name:
        wrap_paragraph_with_bookmark(p, bm_name)
    return p


def add_body(doc, text):
    if not text.strip():
        return
    p = doc.add_paragraph()
    # 한국 학위논문 표준: 양쪽 정렬, 줄간격 160%, 첫 줄 들여쓰기 2자(약 0.7cm)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.6
    p.paragraph_format.first_line_indent = Cm(0.7)
    p.paragraph_format.space_after = Pt(0)
    # split by bold first
    bold_parts = re.split(r'(\*\*[^*]+\*\*)', text)
    for bp in bold_parts:
        is_bold = bp.startswith('**') and bp.endswith('**')
        segment = bp[2:-2] if is_bold else bp
        # split by inline math $...$
        math_parts = re.split(r'(\$[^$\n]+\$)', segment)
        for mp in math_parts:
            if mp.startswith('$') and mp.endswith('$') and len(mp) > 2:
                add_inline_math_run(p, mp[1:-1])
            elif mp:
                # 인라인 코드 `...` 처리 — monospace 표시
                code_parts = re.split(r'(`[^`\n]+`)', mp)
                for cp in code_parts:
                    if cp.startswith('`') and cp.endswith('`') and len(cp) > 2:
                        r = p.add_run(cp[1:-1])
                        r.font.name = 'D2Coding'
                        rPr = r._element.get_or_add_rPr()
                        rFonts = rPr.find(qn('w:rFonts'))
                        if rFonts is None:
                            rPr.append(parse_xml(f'<w:rFonts {nsdecls("w")} w:ascii="D2Coding" w:hAnsi="D2Coding" w:eastAsia="D2Coding"/>'))
                        r.font.size = Pt(10)
                        r.font.bold = is_bold
                    elif cp:
                        # 단일 별표 *italic* 처리 (참고문헌 학술지명·책명 등)
                        italic_parts = re.split(r'(\*[^*\n]+\*)', cp)
                        for ip in italic_parts:
                            if ip.startswith('*') and ip.endswith('*') and len(ip) > 2 and not ip.startswith('**'):
                                r = p.add_run(ip[1:-1])
                                set_font(r, 11, bold=is_bold, italic=True)
                            elif ip:
                                r = p.add_run(ip)
                                set_font(r, 11, is_bold)
    return p


def is_sep(line):
    s = line.strip()
    if not (s.startswith('|') and s.endswith('|')):
        return False
    # 컬럼별로 split 후 각 칸이 정렬 마커(- : 공백)만 포함하는지
    cells = [c.strip() for c in s.strip('|').split('|')]
    if not cells:
        return False
    return all(re.fullmatch(r':?-+:?', c) for c in cells if c != '')


def parse_row(line):
    return [c.strip() for c in line.strip().strip('|').split('|')]


def add_table(doc, rows, caption='', bm_name=None):
    if caption:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.space_before = Pt(8)
        r = p.add_run(caption)
        set_font(r, 10, True, '휴먼명조')  # 공식: 표/그림 내용 10pt 휴먼명조
        if bm_name:
            wrap_paragraph_with_bookmark(p, bm_name)
    
    if not rows:
        return
    
    ncols = max(len(r) for r in rows)
    tbl = doc.add_table(rows=len(rows), cols=ncols)
    tbl.style = 'Table Grid'
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    for i, row_data in enumerate(rows):
        for j, txt in enumerate(row_data):
            if j >= ncols:
                continue
            cell = tbl.rows[i].cells[j]
            cell.text = ''
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            # 셀 내 마크다운 **bold**, *italic*, `code` 처리
            parts = re.split(r'(\*\*[^*]+\*\*|\*[^*\n]+\*|`[^`]+`)', txt)
            for part in parts:
                if not part:
                    continue
                if part.startswith('**') and part.endswith('**'):
                    r = p.add_run(part[2:-2])
                    set_font(r, 9, bold=True, font='휴먼명조')
                elif part.startswith('*') and part.endswith('*') and len(part) > 2 and not part.startswith('**'):
                    r = p.add_run(part[1:-1])
                    set_font(r, 9, bold=False, font='휴먼명조', italic=True)
                elif part.startswith('`') and part.endswith('`'):
                    r = p.add_run(part[1:-1])
                    r.font.name = 'D2Coding'
                    rPr = r._element.get_or_add_rPr()
                    rFonts = rPr.find(qn('w:rFonts'))
                    if rFonts is None:
                        rPr.append(parse_xml(f'<w:rFonts {nsdecls("w")} w:ascii="D2Coding" w:hAnsi="D2Coding" w:eastAsia="D2Coding"/>'))
                    r.font.size = Pt(9)
                    r.font.bold = (i == 0)
                else:
                    r = p.add_run(part)
                    set_font(r, 9, bold=(i == 0), font='휴먼명조')
            if i == 0:
                set_shading(cell, 'D9E2F3')
    
    _empty(doc)
def add_image(doc, path, caption='', bm_name=None):
    if not os.path.exists(path):
        add_body(doc, f'[그림 파일 없음: {os.path.basename(path)}]')
        return
    if caption:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Cm(0)
        r = p.add_run(caption)
        set_font(r, 10, font='휴먼명조')
        if bm_name:
            wrap_paragraph_with_bookmark(p, bm_name)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    r = p.add_run()
    r.add_picture(path, width=Cm(13))


_TOC_HEADINGS = ('목차', '표 목차', '그림 목차')


def prebuild_bookmark_map(md_path):
    """본문 헤딩(H1·H2)·표 캡션·그림 캡션에 부여할 bookmark 사전.

    목차/표목차/그림목차 영역(H1) 안의 H2 라인은 등록하지 않는다.
    """
    headings = {}  # full text → bm name
    tables = {}    # 'N-M' → bm name
    figures = {}   # 'N-M' → bm name

    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    in_toc_zone = False
    seq = 0
    for ln in lines:
        s = ln.rstrip()
        if s.startswith('# ') and not s.startswith('## '):
            txt = s[2:].strip()
            seq += 1
            bm = f'_h_{seq}'
            headings[txt] = bm
            in_toc_zone = txt in _TOC_HEADINGS
            continue
        if s.startswith('## '):
            if in_toc_zone:
                continue
            txt = s[3:].strip()
            seq += 1
            bm = f'_h_{seq}'
            headings[txt] = bm
            continue
        m = re.match(r'^\*\*<표\s*([0-9-]+)>[^*]*\*\*$', s.strip())
        if m:
            num = m.group(1)
            tables[num] = f'_t_{num.replace("-", "_")}'
            continue
        m = re.match(r'^!\[<그림\s*([0-9-]+)>[^]]*\]\(.+?\)\s*$', s.strip())
        if m:
            num = m.group(1)
            figures[num] = f'_f_{num.replace("-", "_")}'
            continue
    return {'headings': headings, 'tables': tables, 'figures': figures}


def convert_md(doc, md_path, bm_map=None):
    if bm_map is None:
        bm_map = {'headings': {}, 'tables': {}, 'figures': {}}

    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 본문 시작은 표제지/제출서/인준서 다음의 첫 marker(목차 또는 국문초록)부터
    start = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith('# 목차') or s.startswith('# 국문초록') or s.startswith('# 국문 초록'):
            start = i
            break

    i = start
    caption = ''
    body_section_started = False  # 첫 '제N장' 헤딩을 만나면 새 섹션 시작
    in_references = False  # 참고문헌 섹션 hanging indent 모드
    toc_mode = None  # 'chapter' | 'table' | 'figure' | None

    while i < len(lines):
        line = lines[i].rstrip()
        
        # 블록 수식 (한 줄 $$...$$)
        m_one = re.match(r'^\s*\$\$(.+)\$\$\s*$', line)
        if m_one:
            add_block_math(doc, m_one.group(1).strip())
            i += 1
            continue
        # 블록 수식 (다중 라인 $$ ... $$)
        if line.strip() == '$$':
            parts = []
            i += 1
            while i < len(lines) and lines[i].strip() != '$$':
                parts.append(lines[i].strip())
                i += 1
            add_block_math(doc, ' '.join(parts))
            i += 1
            continue
        
        # 표 캡션
        m = re.match(r'^\*\*(<표[^>]*>[^*]*)\*\*$', line.strip())
        if m:
            caption = m.group(1)
            i += 1
            continue

        # 표
        if '|' in line and line.strip().startswith('|') and not is_sep(line):
            rows = []
            while i < len(lines):
                l = lines[i].rstrip()
                if '|' in l and l.strip().startswith('|'):
                    if not is_sep(l):
                        rows.append(parse_row(l))
                    i += 1
                else:
                    break
            tbm = None
            mn = re.match(r'^<표\s*([0-9-]+)>', caption)
            if mn:
                tbm = bm_map['tables'].get(mn.group(1))
            add_table(doc, rows, caption, bm_name=tbm)
            caption = ''
            continue

        if is_sep(line):
            i += 1
            continue

        # 그림
        m = re.match(r'!\[(.+?)\]\((.+?)\)', line)
        if m:
            cap = m.group(1)
            rel = m.group(2)
            absp = os.path.normpath(os.path.join(PAPER_DIR, rel))
            fbm = None
            mn = re.match(r'^<그림\s*([0-9-]+)>', cap)
            if mn:
                fbm = bm_map['figures'].get(mn.group(1))
            add_image(doc, absp, cap, bm_name=fbm)
            i += 1
            continue

        # 제목
        if line.startswith('# '):
            txt = line[2:].strip()
            # 참고문헌/Abstract 헤딩에서 모드 전환
            if txt.startswith('참고문헌'):
                in_references = True
            elif txt.startswith('Abstract') or txt.startswith('ABSTRACT'):
                in_references = False
            # 목차/표목차/그림목차 영역 진입 플래그
            if txt == '목차':
                toc_mode = 'chapter'
            elif txt == '표 목차':
                toc_mode = 'table'
            elif txt == '그림 목차':
                toc_mode = 'figure'
            else:
                toc_mode = None
            # 장 제목은 새 페이지. 첫 제N장 만남 → body 섹션 시작 (페이지 번호 1부터 아라비아)
            if txt.startswith('제') and '장' in txt:
                if not body_section_started:
                    from docx.enum.section import WD_SECTION
                    new_sec = doc.add_section(WD_SECTION.NEW_PAGE)
                    new_sec.page_width = Mm(210)
                    new_sec.page_height = Mm(297)
                    new_sec.top_margin = Cm(4.0)
                    new_sec.bottom_margin = Cm(4.0)
                    new_sec.left_margin = Cm(3.5)
                    new_sec.right_margin = Cm(3.5)
                    new_sec.header_distance = Cm(1.5)
                    new_sec.footer_distance = Cm(1.5)
                    # Explicit nextPage type so LibreOffice honors pgNumType.start reset
                    sectPr = new_sec._sectPr
                    if sectPr.find(qn('w:type')) is None:
                        type_el = parse_xml(f'<w:type {nsdecls("w")} w:val="nextPage"/>')
                        # Insert after footerReference / before pgSz
                        pgSz = sectPr.find(qn('w:pgSz'))
                        if pgSz is not None:
                            pgSz.addprevious(type_el)
                        else:
                            sectPr.append(type_el)
                    add_page_number(new_sec, start_num=1, fmt='decimal')
                    body_section_started = True
                else:
                    doc.add_page_break()
            hbm = bm_map['headings'].get(txt)
            add_heading(doc, txt, 1, bm_name=hbm)
            i += 1
            continue
        if line.startswith('## '):
            txt = line[3:].strip()
            # 챕터 목차 영역에서는 ## 제N장 ... 라인을 TOC 항목으로 변환
            if toc_mode == 'chapter':
                bm = bm_map['headings'].get(txt)
                add_toc_entry(doc, txt, bm, level=0, bold=True)
                i += 1
                continue
            hbm = bm_map['headings'].get(txt)
            add_heading(doc, txt, 2, bm_name=hbm)
            i += 1
            continue
        if line.startswith('#### '):
            add_heading(doc, line[5:].strip(), 4)
            i += 1
            continue
        if line.startswith('### '):
            add_heading(doc, line[4:].strip(), 3)
            i += 1
            continue
        
        if line.strip() == '---':
            doc.add_page_break()
            i += 1
            continue
        
        # 표 목차·그림 목차의 항목 → TOC entry (점 리더 + PAGEREF)
        if toc_mode in ('table', 'figure') and (
            line.strip().startswith('- <표') or line.strip().startswith('- <그림')
        ):
            txt = line.strip()[2:]  # "- " 제거
            mn = re.match(r'^<(표|그림)\s*([0-9-]+)>', txt)
            bm = None
            if mn:
                key = mn.group(2)
                if mn.group(1) == '표':
                    bm = bm_map['tables'].get(key)
                else:
                    bm = bm_map['figures'].get(key)
            add_toc_entry(doc, txt, bm, level=0)
            i += 1
            continue

        if not line.strip():
            i += 1
            continue

        # 마크다운 리스트 → docx bullet/number 스타일
        m_dash = re.match(r'^[ \t]*-\s+(.*)$', line)
        m_num = re.match(r'^[ \t]*(\d+)\.\s+(.*)$', line)
        if m_dash:
            txt = m_dash.group(1)
            # 챕터 TOC 영역의 bullet 항목 → TOC entry
            if toc_mode == 'chapter':
                # 절 라인은 들여쓰기 1단, 그 외(국문초록·표목차 등)는 들여쓰기 0단·굵게
                if txt.startswith('제') and '절' in txt[:6]:
                    bm = bm_map['headings'].get(txt)
                    add_toc_entry(doc, txt, bm, level=1)
                else:
                    bm = bm_map['headings'].get(txt)
                    add_toc_entry(doc, txt, bm, level=0, bold=True)
                i += 1
                continue
            p = add_body(doc, txt)
            if p is not None:
                if in_references:
                    # 참고문헌: bullet 제거, hanging indent (1.0cm)
                    p.paragraph_format.first_line_indent = Cm(-1.0)
                    p.paragraph_format.left_indent = Cm(1.0)
                else:
                    p.style = doc.styles['List Bullet']
                    p.paragraph_format.first_line_indent = Cm(0)
            i += 1
            continue
        if m_num:
            p = add_body(doc, m_num.group(2))
            if p is not None:
                p.style = doc.styles['List Number']
                p.paragraph_format.first_line_indent = Cm(0)
            i += 1
            continue

        add_body(doc, line)
        i += 1


def enable_update_fields(doc):
    """settings.xml에 <w:updateFields w:val='true'/> 추가 — open 시 PAGE 등 필드 자동 재계산."""
    settings = doc.settings.element
    if settings.find(qn('w:updateFields')) is not None:
        return
    el = parse_xml(f'<w:updateFields {nsdecls("w")} w:val="true"/>')
    settings.insert(0, el)


def freeze_fields_via_libreoffice(docx_path):
    """LibreOffice headless로 docx 재저장 → PAGE/PAGEREF 필드 평가값을 OOXML cache에 굽기.

    Word for Mac AppleScript는 macOS sandbox로 fields update를 OOXML까지 굽지 못한다
    (sdef 비어있음, do Visual Basic 컴파일 차단). LibreOffice는 변환 단계에서 fields를
    평가하고 cache를 굽기 때문에 후처리로 사용한다. 한글 폰트·표 정렬에 미세한 변동이
    있을 수 있어 결과 docx는 시각 검증 필요.
    """
    import subprocess, shutil, tempfile
    soffice = '/Applications/LibreOffice.app/Contents/MacOS/soffice'
    if not os.path.exists(soffice):
        print('  [skip] LibreOffice 미설치 — 필드는 placeholder로 남음')
        return False
    with tempfile.TemporaryDirectory() as tmp:
        try:
            subprocess.run(
                [soffice, '--headless', '--convert-to', 'docx',
                 '--outdir', tmp, docx_path],
                check=True, timeout=240,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f'  [warn] field freeze 실패: {e}')
            return False
        fname = os.path.basename(docx_path)
        src = os.path.join(tmp, fname)
        if not os.path.exists(src):
            print('  [warn] LibreOffice 출력 파일 없음')
            return False
        shutil.move(src, docx_path)
        print(f'  ✓ PAGE/PAGEREF 필드 cache 채움 (LibreOffice 재저장)')
        return True


def export_pdf_via_libreoffice(docx_path):
    """docx → PDF (LibreOffice headless)."""
    import subprocess
    soffice = '/Applications/LibreOffice.app/Contents/MacOS/soffice'
    if not os.path.exists(soffice):
        return False
    try:
        subprocess.run(
            [soffice, '--headless', '--convert-to', 'pdf',
             '--outdir', os.path.dirname(docx_path), docx_path],
            check=True, timeout=240,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f'  ✓ PDF 변환 완료')
        return True
    except Exception as e:
        print(f'  [warn] PDF 변환 실패: {e}')
        return False


def main():
    print("📄 논문 DOCX v2 생성 (한양대 공식 서식)")

    doc = create_doc()
    enable_update_fields(doc)
    
    # 1. 표지
    add_cover(doc)
    
    # 2. 제출서
    add_submission(doc)
    
    # 3. 인준서
    add_approval(doc)
    
    # 첫 섹션 (표지/제출서/인준서/목차/표목차/그림목차/국문초록): 로마자 i부터
    add_page_number(doc.sections[0], start_num=1, fmt='lowerRoman')

    # 4. 본문 (목차~참고문헌~Abstract 포함). 제1장 만나면 새 섹션 시작 → 아라비아 1부터
    md_path = os.path.join(PAPER_DIR, '석사학위논문_박현근.md')
    bm_map = prebuild_bookmark_map(md_path)
    convert_md(doc, md_path, bm_map=bm_map)
    
    out = os.path.join(PAPER_DIR, '석사학위논문_박현근.docx')
    doc.save(out)

    # 후처리: LibreOffice로 PAGE/PAGEREF 필드 cache 굽기 + PDF 변환
    print('🔧 fields freeze + PDF 생성')
    freeze_fields_via_libreoffice(out)
    export_pdf_via_libreoffice(out)

    print(f"✅ 완료: {out}")
    print(f"   크기: {os.path.getsize(out)//1024}KB")
    print(f"   문단: {len(doc.paragraphs)}, 표: {len(doc.tables)}")
    print(f"   여백: 상하 4cm, 좌우 3.5cm (공식 워드 기준)")
    print(f"   폰트: 바탕 11pt, 표/그림 휴먼명조 10pt")


if __name__ == '__main__':
    main()
