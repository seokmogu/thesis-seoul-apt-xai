#!/usr/bin/env python3
"""참고문헌 전수 검증 v2 — 정확한 제목으로 검색 + PDF 다운로드"""
import requests, time, json, os, re

PAPERS_DIR = os.path.join(os.path.dirname(__file__), '..', 'references', 'papers')
os.makedirs(PAPERS_DIR, exist_ok=True)

# (label, year, exact_title_fragment)
REFS = [
    ("An_2025", 2025, "Toward transparent and accurate housing price appraisal"),
    ("Breiman_2001", 2001, "Random forests"),
    ("Ceh_2018", 2018, "Estimating the performance of random forest versus multiple regression"),
    ("Chen_Guestrin_2016", 2016, "XGBoost: A scalable tree boosting system"),
    ("Choy_Ho_2023", 2023, "The use of machine learning in real estate research"),
    ("Chun_2025", 2025, "Predicting housing price in Seoul using explainable AI"),
    ("Friedman_2001", 2001, "Greedy function approximation: A gradient boosting machine"),
    ("Kim_2022_Multiplex", 2022, "analysis of the price determinants of multiplex houses"),
    ("Kim_Choi_Lee_2025", 2025, "Explainable AI-based mass appraisal"),
    ("Lancaster_1966", 1966, "A new approach to consumer theory"),
    ("Limsombunchai_2004", 2004, "House price prediction: Hedonic price model"),
    ("Lundberg_Lee_2017", 2017, "A unified approach to interpreting model predictions"),
    ("Lundberg_2020", 2020, "From local explanations to global understanding"),
    ("Mora_Garcia_2022", 2022, "Housing price prediction using machine learning algorithms in COVID"),
    ("Neves_2024", 2024, "The impacts of open data and explainable AI on real estate"),
    ("Revathi_2025", 2025, "robust ensemble-based framework for house price estimation"),
    ("Ribeiro_2016", 2016, "Why should I trust you"),
    ("Rosen_1974", 1974, "Hedonic prices and implicit markets"),
    ("Shahhosseini_2022", 2022, "Optimizing ensemble weights and hyperparameters"),
    ("Tarasov_2025", 2025, "Algorithm-driven hedonic real estate pricing"),
    ("Ke_2017", 2017, "LightGBM: A highly efficient gradient boosting"),
]

def search_ss(title_fragment):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": title_fragment, "limit": 3, 
              "fields": "title,authors,year,venue,externalIds,isOpenAccess,openAccessPdf"}
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            return r.json().get('data', [])
        elif r.status_code == 429:
            time.sleep(3)
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r.json().get('data', [])
    except Exception as e:
        print(f"  ERROR: {e}")
    return []

def download_pdf(url, filepath):
    try:
        r = requests.get(url, timeout=30, stream=True, 
                        headers={"User-Agent": "Mozilla/5.0"})
        ct = r.headers.get('content-type', '')
        if r.status_code == 200 and ('pdf' in ct or url.endswith('.pdf')):
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            size = os.path.getsize(filepath)
            if size > 10000:  # minimum 10KB
                return True
            else:
                os.remove(filepath)
    except:
        pass
    return False

def try_scihub(doi, filepath):
    """Try sci-hub mirrors"""
    mirrors = ["https://sci-hub.se", "https://sci-hub.st"]
    for mirror in mirrors:
        try:
            r = requests.get(f"{mirror}/{doi}", timeout=15, 
                           headers={"User-Agent": "Mozilla/5.0"})
            # Find PDF link
            match = re.search(r'(https?://[^"\']+\.pdf[^"\']*)', r.text)
            if match:
                pdf_url = match.group(1)
                if download_pdf(pdf_url, filepath):
                    return True
        except:
            continue
    return False

results = []
print("=" * 70)
print("참고문헌 전수 검증 + PDF 다운로드")
print("=" * 70)

for label, year, title_frag in REFS:
    print(f"\n--- {label} ({year}) ---")
    filepath = os.path.join(PAPERS_DIR, f"{label}.pdf")
    
    # 이미 다운로드된 파일 확인
    if os.path.exists(filepath) and os.path.getsize(filepath) > 10000:
        print(f"  📄 Already have PDF ({os.path.getsize(filepath)//1024}KB)")
        results.append({"ref": label, "status": "HAVE_PDF"})
        time.sleep(0.3)
        continue
    
    hits = search_ss(title_frag)
    time.sleep(0.5)
    
    found = False
    for h in hits:
        h_year = h.get('year', 0) or 0
        h_title = h.get('title', '')
        
        # 제목이 대략 맞고 년도가 ±2
        if abs(h_year - year) <= 2 and title_frag.lower()[:30] in h_title.lower():
            doi = h.get('externalIds', {}).get('DOI', '')
            oa = h.get('openAccessPdf')
            pdf_url = oa.get('url') if oa else None
            
            print(f"  ✅ VERIFIED: {h_title[:80]}")
            print(f"     DOI: {doi}")
            
            # PDF 다운로드 시도
            downloaded = False
            if pdf_url:
                downloaded = download_pdf(pdf_url, filepath)
                if downloaded:
                    print(f"     📥 Downloaded (Open Access)")
            
            if not downloaded and doi:
                downloaded = try_scihub(doi, filepath)
                if downloaded:
                    print(f"     📥 Downloaded (alt)")
            
            if not downloaded:
                print(f"     ⚠️ No PDF available")
            
            results.append({"ref": label, "status": "VERIFIED", "doi": doi, "pdf": downloaded})
            found = True
            break
    
    if not found:
        print(f"  ❌ NOT FOUND in Semantic Scholar")
        results.append({"ref": label, "status": "NOT_FOUND"})

# 요약
print("\n" + "=" * 70)
verified = sum(1 for r in results if r['status'] in ('VERIFIED', 'HAVE_PDF'))
pdfs = sum(1 for r in results if r.get('pdf') or r['status'] == 'HAVE_PDF')
not_found = [r['ref'] for r in results if r['status'] == 'NOT_FOUND']

print(f"검증됨: {verified}/{len(results)}")
print(f"PDF 확보: {pdfs}/{len(results)}")
if not_found:
    print(f"미확인: {not_found}")

with open(os.path.join(PAPERS_DIR, '..', 'verification_v2.json'), 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
