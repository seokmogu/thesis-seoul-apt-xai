#!/usr/bin/env python3
"""참고문헌 PDF 다운로드 — Semantic Scholar API + direct DOI"""
import os, time, re, json, urllib.request, urllib.parse, urllib.error

SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'references', 'papers')
os.makedirs(SAVE_DIR, exist_ok=True)

# 논문 목록: (파일명, 검색 쿼리, DOI if known)
PAPERS = [
    # 해외 논문
    ("Chen_Guestrin_2016_XGBoost", "XGBoost scalable tree boosting system Chen Guestrin 2016", "10.1145/2939672.2939785"),
    ("Lundberg_Lee_2017_SHAP", "unified approach interpreting model predictions Lundberg Lee 2017", None),
    ("Rosen_1974_Hedonic", "hedonic prices implicit markets Rosen 1974", "10.1086/260169"),
    ("Breiman_2001_RandomForests", "random forests Breiman 2001 machine learning", "10.1023/A:1010933404324"),
    ("Lancaster_1966_ConsumerTheory", "new approach consumer theory Lancaster 1966", "10.1086/259131"),
    ("Neves_2024_OpenData_XAI", "impacts open data explainable AI real estate price predictions smart cities Neves 2024", "10.3390/app14052209"),
    ("Kee_Ho_2025_XGBoost_SHAP", "explainable machine learning real estate XGBoost Shapley Kee Ho 2025", None),
    ("Dou_2023_Neighborhoods_XAI", "incorporating neighborhoods explainable artificial intelligence housing prices Dou 2023", "10.1016/j.apgeog.2023.103073"),
    ("Kramer_2023_XAI_RealEstate", "explainable AI real estate context residential values Kramer 2023", None),
    ("Tchuente_2024_AVM_SHAP", "real estate automated valuation model explainable AI Shapley Tchuente 2024", None),
    ("Mora_Garcia_2022_COVID_ML", "housing price prediction machine learning COVID-19 Mora-Garcia 2022", "10.3390/land11112100"),
    ("Friedman_2001_GBM", "greedy function approximation gradient boosting machine Friedman 2001", None),
    ("Ke_2017_LightGBM", "LightGBM highly efficient gradient boosting decision tree Ke 2017", None),
    ("Ribeiro_2016_LIME", "why should I trust you explaining predictions classifier Ribeiro 2016", "10.1145/2939672.2939778"),
    ("Ceh_2018_RF_Apartments", "estimating performance random forest multiple regression apartment prices Ceh 2018", "10.3390/ijgi7050168"),
    ("Rico_Juan_2021_ML_Hedonic", "machine learning explainability spatial hedonics housing Alicante Rico-Juan 2021", "10.1016/j.eswa.2021.114590"),
    ("Choy_Ho_2023_ML_RealEstate", "use machine learning real estate research Choy Ho 2023", "10.3390/land12040740"),
    ("Na_Ko_Park_2025_Gangnam", "machine learning determinants apartment prices Gangnam Seoul Na Ko Park 2025", None),
    ("Park_Oh_Won_2024_XAI_Housing", "explainable machine learning housing price determinants regional temporal Park Oh Won 2024", None),
    ("Anselin_1988_SpatialEconometrics", "spatial econometrics methods models Anselin 1988", None),
    ("Kim_2022_Multiplex_Spatial", "price determinants multiplex houses spatial regression Kim Cho Lee 2022", "10.3390/su14052891"),
    ("Acharya_2024_FairAI", "explainable fair AI financial real estate machine learning Acharya 2024", None),
    ("Tekouabou_2024_AIUrban", "AI based machine learning urban real estate prediction systematic survey Tekouabou 2024", None),
    ("Matic_2025_XGBoost_RF", "housing price prediction XGBoost random forest Matic Kalinic 2025", None),
    ("Limsombunchai_2004_Hedonic_ANN", "house price prediction hedonic price model artificial neural network Limsombunchai 2004", None),
    ("Oh_Lee_2025_Seoul_Apartment", "determinants prices Seoul apartment market long-term equilibrium causal Oh Lee 2025", None),
]

def search_semantic_scholar(query, limit=3):
    """Semantic Scholar API로 논문 검색"""
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={urllib.parse.quote(query)}&limit={limit}&fields=title,externalIds,openAccessPdf,url"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            return data.get('data', [])
    except Exception as e:
        print(f"    S2 search error: {e}")
        return []

def try_download_pdf(url, filepath):
    """PDF 다운로드 시도"""
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
            'Accept': 'application/pdf,*/*'
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
            if len(content) > 1000 and (content[:5] == b'%PDF-' or b'%PDF' in content[:20]):
                with open(filepath, 'wb') as f:
                    f.write(content)
                return True
    except Exception as e:
        pass
    return False

def try_doi_pdf(doi, filepath):
    """DOI로 직접 PDF 접근 시도 (Unpaywall + Sci-Hub fallback)"""
    # Unpaywall
    try:
        url = f"https://api.unpaywall.org/v2/{doi}?email=test@example.com"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            oa = data.get('best_oa_location', {})
            if oa and oa.get('url_for_pdf'):
                if try_download_pdf(oa['url_for_pdf'], filepath):
                    return True
            # try url_for_landing_page PDF
            if oa and oa.get('url'):
                if try_download_pdf(oa['url'], filepath):
                    return True
    except:
        pass
    return False

def main():
    downloaded = 0
    failed = []
    
    for filename, query, doi in PAPERS:
        filepath = os.path.join(SAVE_DIR, f"{filename}.pdf")
        if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
            print(f"✅ {filename} (already exists)")
            downloaded += 1
            continue
        
        print(f"\n🔍 {filename}")
        success = False
        
        # 1. DOI → Unpaywall
        if doi and not success:
            print(f"   DOI: {doi}")
            if try_doi_pdf(doi, filepath):
                print(f"   ✅ Downloaded via Unpaywall")
                success = True
                downloaded += 1
        
        # 2. Semantic Scholar
        if not success:
            results = search_semantic_scholar(query)
            time.sleep(1)  # rate limit
            for paper in results:
                oa_pdf = paper.get('openAccessPdf', {})
                if oa_pdf and oa_pdf.get('url'):
                    if try_download_pdf(oa_pdf['url'], filepath):
                        print(f"   ✅ Downloaded via S2: {paper.get('title','')[:60]}")
                        success = True
                        downloaded += 1
                        break
        
        # 3. ArXiv fallback (for ML papers)
        if not success:
            arxiv_ids = {
                "Lundberg_Lee_2017_SHAP": "1705.07874",
                "Ke_2017_LightGBM": "1711.08229",
                "Ribeiro_2016_LIME": "1602.04938",
            }
            if filename in arxiv_ids:
                arxiv_url = f"https://arxiv.org/pdf/{arxiv_ids[filename]}.pdf"
                if try_download_pdf(arxiv_url, filepath):
                    print(f"   ✅ Downloaded via arXiv")
                    success = True
                    downloaded += 1
        
        if not success:
            # Save a stub with search info
            stub_path = os.path.join(SAVE_DIR, f"{filename}.txt")
            with open(stub_path, 'w') as f:
                f.write(f"Paper: {filename}\nQuery: {query}\nDOI: {doi or 'N/A'}\nStatus: Not found as open access\n")
                if results:
                    f.write(f"\nSemantic Scholar results:\n")
                    for p in results:
                        f.write(f"  - {p.get('title','')}\n    URL: {p.get('url','')}\n")
            failed.append(filename)
            print(f"   ❌ Not found as open access → stub saved")
        
        time.sleep(0.5)
    
    print(f"\n{'='*60}")
    print(f"✅ Downloaded: {downloaded}/{len(PAPERS)}")
    print(f"❌ Failed: {len(failed)}")
    for f in failed:
        print(f"   - {f}")

if __name__ == '__main__':
    main()
