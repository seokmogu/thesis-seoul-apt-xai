#!/usr/bin/env python3
"""논문 초안 Markdown → PDF 변환"""
import os, sys
import markdown
from weasyprint import HTML

PAPER_DIR = os.path.join(os.path.dirname(__file__), '..', 'paper')
INPUT = os.path.join(PAPER_DIR, '논문_초안.md')
OUTPUT_HTML = os.path.join(PAPER_DIR, '논문_초안.html')
OUTPUT_PDF = os.path.join(PAPER_DIR, '논문_초안.pdf')

CSS = """
@page {
    size: A4;
    margin: 2.5cm 2cm 2.5cm 2cm;
    @bottom-center { content: counter(page); font-size: 10pt; }
}
body {
    font-family: 'NanumSquare_ac', 'Noto Sans CJK KR', 'Malgun Gothic', sans-serif;
    font-size: 11pt;
    line-height: 1.8;
    color: #222;
    max-width: 100%;
}
h1 {
    font-size: 18pt;
    font-weight: bold;
    text-align: center;
    margin-top: 2em;
    margin-bottom: 1em;
    page-break-before: always;
}
h1:first-of-type { page-break-before: avoid; }
h2 {
    font-size: 14pt;
    font-weight: bold;
    margin-top: 1.5em;
    border-bottom: 1px solid #ccc;
    padding-bottom: 0.3em;
}
h3 { font-size: 12pt; font-weight: bold; margin-top: 1.2em; }
h4 { font-size: 11pt; font-weight: bold; margin-top: 1em; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    font-size: 9.5pt;
}
th, td {
    border: 1px solid #999;
    padding: 4px 8px;
    text-align: left;
}
th { background-color: #f0f0f0; font-weight: bold; }
p { text-indent: 1em; margin: 0.5em 0; }
blockquote { border-left: 3px solid #ccc; padding-left: 1em; color: #555; }
code { font-size: 9pt; background: #f5f5f5; padding: 2px 4px; }
hr { border: none; border-top: 1px solid #999; margin: 2em 0; }
img { max-width: 90%; display: block; margin: 1em auto; }
strong { font-weight: bold; }
"""

def main():
    print("=== 논문 PDF 생성 ===")
    
    with open(INPUT, 'r', encoding='utf-8') as f:
        md_text = f.read()
    
    # Markdown → HTML
    extensions = ['tables', 'fenced_code', 'toc']
    html_body = markdown.markdown(md_text, extensions=extensions)
    
    full_html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<style>{CSS}</style>
</head>
<body>
{html_body}
</body>
</html>"""
    
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(full_html)
    print(f"  HTML: {OUTPUT_HTML}")
    
    # HTML → PDF
    HTML(string=full_html, base_url=PAPER_DIR).write_pdf(OUTPUT_PDF)
    
    size_mb = os.path.getsize(OUTPUT_PDF) / 1024 / 1024
    print(f"  PDF: {OUTPUT_PDF} ({size_mb:.1f} MB)")
    print("✅ 완료!")

if __name__ == '__main__':
    main()
