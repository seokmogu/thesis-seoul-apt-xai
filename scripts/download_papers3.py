#!/usr/bin/env python3
"""3차 다운로드 — 남은 16편 추가 시도"""
import os, time, json, urllib.request, urllib.parse

SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'references', 'papers')

def download(url, filepath, timeout=30):
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
            'Accept': 'application/pdf,*/*'
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read()
            if len(content) > 5000 and (content[:5] == b'%PDF-' or b'%PDF' in content[:20]):
                with open(filepath, 'wb') as f:
                    f.write(content)
                return True
    except:
        pass
    return False

def try_unpaywall(doi, filepath):
    try:
        url = f"https://api.unpaywall.org/v2/{doi}?email=research@example.com"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            for loc in data.get('oa_locations', []):
                pdf_url = loc.get('url_for_pdf') or loc.get('url')
                if pdf_url and download(pdf_url, filepath):
                    return True
    except:
        pass
    return False

def try_semantic_scholar(query, filepath):
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={urllib.parse.quote(query)}&limit=3&fields=openAccessPdf,title"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            for paper in data.get('data', []):
                oa = paper.get('openAccessPdf', {})
                if oa and oa.get('url'):
                    if download(oa['url'], filepath):
                        return True, paper.get('title','')
        return False, None
    except Exception as e:
        return False, str(e)

# 미확보 논문 - DOI + 검색어
MISSING = [
    ("Rosen_1974_Hedonic", "10.1086/260169", "hedonic prices implicit markets Rosen 1974"),
    ("Lancaster_1966_ConsumerTheory", "10.1086/259131", "new approach consumer theory Lancaster 1966"),
    ("Friedman_2001_GBM", "10.1214/aos/1013203451", "greedy function approximation gradient boosting Friedman 2001"),
    ("Anselin_1988_SpatialEconometrics", None, "spatial econometrics methods models Anselin 1988"),
    ("Dou_2023_Neighborhoods_XAI", "10.1016/j.apgeog.2023.103073", "neighborhoods explainable AI housing prices Dou 2023"),
    ("Kee_Ho_2025_XGBoost_SHAP", None, "explainable machine learning real estate XGBoost Shapley Kee Ho 2025 civil engineering"),
    ("Kramer_2023_XAI_RealEstate", None, "explainable AI real estate residential values Kramer Nagl Just 2023"),
    ("Tchuente_2024_AVM_SHAP", None, "automated valuation model explainable AI Shapley Tchuente 2024"),
    ("Rico_Juan_2021_ML_Hedonic", "10.1016/j.eswa.2021.114590", "machine learning explainability spatial hedonics Alicante Rico-Juan 2021"),
    ("Acharya_2024_FairAI", None, "explainable fair AI financial real estate Acharya 2024 IEEE"),
    ("Tekouabou_2024_AIUrban", None, "AI machine learning urban real estate systematic survey Tekouabou 2024"),
    ("Matic_2025_XGBoost_RF", None, "housing price prediction XGBoost random forest Matic Kalinic 2025"),
    ("Limsombunchai_2004_Hedonic_ANN", None, "house price hedonic artificial neural network Limsombunchai 2004"),
    ("Na_Ko_Park_2025_Gangnam", None, "machine learning apartment prices Gangnam Seoul Na Ko Park 2025"),
    ("Park_Oh_Won_2024_XAI_Housing", None, "explainable machine learning housing price regional temporal Park Oh Won 2024"),
    ("Oh_Lee_2025_Seoul_Apartment", None, "Seoul apartment market long-term equilibrium causal Oh Lee 2025"),
]

def main():
    ok = 0
    fail = []
    for name, doi, query in MISSING:
        fp = os.path.join(SAVE_DIR, f"{name}.pdf")
        if os.path.exists(fp) and os.path.getsize(fp) > 5000:
            print(f"✅ {name} (exists)")
            ok += 1
            continue
        
        print(f"\n🔍 {name}")
        success = False
        
        # Unpaywall
        if doi:
            print(f"   Unpaywall DOI: {doi}")
            if try_unpaywall(doi, fp):
                print(f"   ✅ Unpaywall")
                success = True
                ok += 1
        
        # Semantic Scholar (with delay)
        if not success:
            time.sleep(3)  # avoid 429
            print(f"   S2: {query[:50]}...")
            found, info = try_semantic_scholar(query, fp)
            if found:
                print(f"   ✅ S2: {info[:60] if info else ''}")
                success = True
                ok += 1
            elif info:
                print(f"   ℹ️ {info[:80]}")
        
        if not success:
            fail.append(name)
            print(f"   ❌ Not available")
        
        time.sleep(1)
    
    # 국내 논문 RISS 접근 시도
    print("\n=== 국내 논문 (RISS) ===")
    korean_papers = [
        ("조민지_2023_서울아파트", "https://www.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=81076c5eb4b5f3c5ffe0bdc3ef48d419"),
        ("김선현_2022_대구아파트", None),
        ("진수정_2024_SRGCNN", None),
    ]
    for name, url in korean_papers:
        print(f"  📝 {name} — RISS 로그인 필요 (자동 다운로드 불가)")
    
    print(f"\n{'='*60}")
    print(f"✅ 이번 다운로드: {ok}")
    print(f"❌ 미확보: {len(fail)}")
    for f in fail:
        print(f"   - {f}")
    
    # 최종 현황
    print(f"\n=== 최종 파일 현황 ===")
    pdfs = [f for f in os.listdir(SAVE_DIR) if f.endswith('.pdf') and os.path.getsize(os.path.join(SAVE_DIR, f)) > 5000]
    txts = [f for f in os.listdir(SAVE_DIR) if f.endswith('.txt')]
    print(f"PDF: {len(pdfs)}편 / 미확보: {len(txts)}편")

if __name__ == '__main__':
    main()
