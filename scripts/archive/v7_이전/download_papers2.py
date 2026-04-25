#!/usr/bin/env python3
"""2차 다운로드 — 직접 URL로 open access 논문 수집"""
import os, urllib.request, time

SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'references', 'papers')

# 직접 URL 매핑 (MDPI, arXiv, etc.)
DIRECT_URLS = {
    "Chen_Guestrin_2016_XGBoost": "https://arxiv.org/pdf/1603.02754.pdf",
    "Breiman_2001_RandomForests": "https://link.springer.com/content/pdf/10.1023/A:1010933404324.pdf",
    "Mora_Garcia_2022_COVID_ML": "https://www.mdpi.com/2073-445X/11/11/2100/pdf",
    "Choy_Ho_2023_ML_RealEstate": "https://www.mdpi.com/2073-445X/12/4/740/pdf",
    "Friedman_2001_GBM": "https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.pdf",
    "Fotheringham_2002_GWR": None,  # book
    "Anselin_1988_SpatialEconometrics": None,  # book
    # RISS 한국 논문 (접근 제한적이지만 시도)
}

# MDPI는 /pdf URL이 바로 PDF
MDPI_DOIS = {
    "Neves_2024_OpenData_XAI": "https://www.mdpi.com/2076-3417/14/5/2209/pdf",
    "Ceh_2018_RF_Apartments": "https://www.mdpi.com/2220-9964/7/5/168/pdf",
    "Kim_2022_Multiplex_Spatial": "https://www.mdpi.com/2071-1050/14/5/2891/pdf",
}

def download(url, filepath):
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
            'Accept': 'application/pdf,*/*'
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            ct = resp.headers.get('Content-Type', '')
            content = resp.read()
            if len(content) > 5000 and (content[:5] == b'%PDF-' or 'pdf' in ct.lower()):
                with open(filepath, 'wb') as f:
                    f.write(content)
                return True, len(content)
    except Exception as e:
        return False, str(e)
    return False, "not PDF"

def main():
    count = 0
    for name, url in DIRECT_URLS.items():
        if not url:
            continue
        fp = os.path.join(SAVE_DIR, f"{name}.pdf")
        if os.path.exists(fp) and os.path.getsize(fp) > 5000:
            print(f"✅ {name} (exists)")
            count += 1
            continue
        print(f"📥 {name}...")
        ok, info = download(url, fp)
        if ok:
            print(f"   ✅ {info:,} bytes")
            count += 1
        else:
            print(f"   ❌ {info}")
        time.sleep(1)
    
    print(f"\n총 다운로드: {count}")
    print("\n=== 전체 파일 현황 ===")
    for f in sorted(os.listdir(SAVE_DIR)):
        fp = os.path.join(SAVE_DIR, f)
        sz = os.path.getsize(fp)
        icon = "📄" if f.endswith('.pdf') and sz > 5000 else "📝"
        print(f"  {icon} {f} ({sz:,} bytes)")

if __name__ == '__main__':
    main()
