#!/usr/bin/env python3
"""DOI 기반 참고문헌 PDF 다운로드"""
import requests, os, time, re

PAPERS_DIR = os.path.join(os.path.dirname(__file__), '..', 'references', 'papers')

# (filename, DOI or direct URL)
PAPERS = [
    ("Breiman_2001.pdf", "https://link.springer.com/content/pdf/10.1023/A:1010933404324.pdf"),
    ("Chen_Guestrin_2016.pdf", "https://dl.acm.org/doi/pdf/10.1145/2939672.2939785"),
    ("Choy_Ho_2023.pdf", "https://www.mdpi.com/2073-445X/12/4/740/pdf"),
    ("Friedman_2001.pdf", "https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.pdf"),
    ("Lancaster_1966.pdf", "10.1086/259131"),  # DOI
    ("Mora_Garcia_2022.pdf", "https://www.mdpi.com/2073-445X/11/11/2100/pdf"),
    ("Neves_2024.pdf", "https://www.mdpi.com/2076-3417/14/5/2209/pdf"),
    ("Ke_2017.pdf", "https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf"),
    ("Lundberg_Lee_2017.pdf", "https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf"),
    ("An_2025.pdf", "10.1186/s40854-025-00874-w"),
    ("Chun_2025.pdf", "10.3837/tiis.2025.01.004"),
    ("Shahhosseini_2022.pdf", "10.1016/j.mlwa.2021.100251"),
    ("Revathi_2025.pdf", "10.1051/itmconf/20257601023"),
    ("Tarasov_2025.pdf", "10.2478/remav-2025-0003"),
    ("Kim_Choi_Lee_2025.pdf", "10.3846/ijspm.2025.25137"),
]

def download(url, filepath):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; academic-research)"}
    try:
        r = requests.get(url, timeout=30, stream=True, headers=headers, allow_redirects=True)
        if r.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            size = os.path.getsize(filepath)
            if size > 5000:
                return size
            os.remove(filepath)
    except Exception as e:
        print(f"  Error: {e}")
    return 0

def try_unpaywall(doi, filepath):
    """Unpaywall API (free, email required)"""
    try:
        r = requests.get(f"https://api.unpaywall.org/v2/{doi}?email=research@example.com", timeout=10)
        if r.status_code == 200:
            data = r.json()
            oa = data.get('best_oa_location', {})
            if oa and oa.get('url_for_pdf'):
                return download(oa['url_for_pdf'], filepath)
    except:
        pass
    return 0

for fname, source in PAPERS:
    filepath = os.path.join(PAPERS_DIR, fname)
    
    if os.path.exists(filepath) and os.path.getsize(filepath) > 10000:
        print(f"📄 {fname} — already have ({os.path.getsize(filepath)//1024}KB)")
        continue
    
    print(f"⬇️  {fname}...", end=" ")
    
    if source.startswith("http"):
        size = download(source, filepath)
    else:
        # DOI — try Unpaywall first, then direct
        size = try_unpaywall(source, filepath)
        if not size:
            size = download(f"https://doi.org/{source}", filepath)
    
    if size:
        print(f"✅ ({size//1024}KB)")
    else:
        print(f"❌ failed")
    
    time.sleep(0.5)

# 최종 확인
print("\n" + "=" * 50)
print("PDF 보유 현황:")
for f in sorted(os.listdir(PAPERS_DIR)):
    if f.endswith('.pdf'):
        size = os.path.getsize(os.path.join(PAPERS_DIR, f))
        print(f"  {'✅' if size > 10000 else '⚠️'} {f} ({size//1024}KB)")
