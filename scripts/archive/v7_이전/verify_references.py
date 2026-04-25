#!/usr/bin/env python3
"""참고문헌 전수 검증 — Semantic Scholar API"""
import requests, time, json, os

PAPERS_DIR = os.path.join(os.path.dirname(__file__), '..', 'references', 'papers')

# 해외 논문 목록 (저자, 년도, 제목 키워드, DOI/venue)
FOREIGN_REFS = [
    ("An et al.", 2025, "transparent accurate housing price appraisal hedonic machine learning", "Financial Innovation"),
    ("Anselin", 1988, "Spatial Econometrics Methods Models", None),  # 교과서
    ("Breiman", 2001, "Random forests", "Machine Learning"),
    ("Čeh et al.", 2018, "random forest multiple regression predicting prices apartments", "ISPRS"),
    ("Chen & Guestrin", 2016, "XGBoost scalable tree boosting system", "KDD"),
    ("Choy & Ho", 2023, "machine learning real estate research", "Land"),
    ("Chun et al.", 2025, "Predicting housing price Seoul explainable AI XAI machine learning", "KSII"),
    ("Friedman", 2001, "Greedy function approximation gradient boosting machine", "Annals of Statistics"),
    ("Hair et al.", 2010, "Multivariate Data Analysis", None),  # 교과서
    ("Hastie et al.", 2009, "Elements of Statistical Learning", None),  # 교과서
    ("Ke et al.", 2017, "LightGBM efficient gradient boosting decision tree", "NeurIPS"),
    ("Kim et al.", 2022, "price determinants multiplex houses spatial regression", "Sustainability"),
    ("Kim Choi Lee", 2025, "Explainable AI mass appraisal Korea residential property", "Strategic Property Management"),
    ("Lancaster", 1966, "new approach consumer theory", "Journal of Political Economy"),
    ("Limsombunchai", 2004, "House price prediction hedonic neural network", "New Zealand Economic Papers"),
    ("Lundberg & Lee", 2017, "unified approach interpreting model predictions", "NeurIPS"),
    ("Lundberg et al.", 2020, "local explanations global understanding explainable AI trees", "Nature Machine Intelligence"),
    ("Mora-García et al.", 2022, "Housing price prediction machine learning COVID-19", "Land"),
    ("Neves et al.", 2024, "open data explainable AI real estate price predictions smart cities", "Applied Sciences"),
    ("Revathi & Devarajan", 2025, "ensemble framework house price XGBoost SHAP web deployment", "ITM Web of Conferences"),
    ("Ribeiro et al.", 2016, "Why should I trust you explaining predictions classifier", "KDD"),
    ("Rosen", 1974, "Hedonic prices implicit markets product differentiation", "Journal of Political Economy"),
    ("Shahhosseini et al.", 2022, "Optimizing ensemble weights hyperparameters machine learning regression", "Machine Learning with Applications"),
    ("Tarasov & Dessoulavy", 2025, "Algorithm-driven hedonic real estate pricing explainable AI", "Real Estate Management and Valuation"),
]

def search_semantic_scholar(query, limit=3):
    """Semantic Scholar에서 논문 검색"""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": query, "limit": limit, "fields": "title,authors,year,venue,externalIds,isOpenAccess,openAccessPdf"}
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            return r.json().get('data', [])
    except:
        pass
    return []

def download_pdf(url, filepath):
    """PDF 다운로드"""
    try:
        r = requests.get(url, timeout=30, stream=True)
        if r.status_code == 200 and 'pdf' in r.headers.get('content-type', '').lower():
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            return True
    except:
        pass
    return False

def main():
    os.makedirs(PAPERS_DIR, exist_ok=True)
    results = []
    
    print("=" * 70)
    print("해외 참고문헌 전수 검증 (Semantic Scholar)")
    print("=" * 70)
    
    for author, year, query, venue in FOREIGN_REFS:
        print(f"\n--- {author} ({year}) ---")
        hits = search_semantic_scholar(query)
        time.sleep(0.5)
        
        found = False
        for h in hits:
            h_year = h.get('year', 0)
            h_title = h.get('title', '')
            h_authors = ', '.join([a['name'] for a in h.get('authors', [])[:3]])
            
            if abs((h_year or 0) - year) <= 1:  # 년도 ±1
                print(f"  ✅ FOUND: {h_title}")
                print(f"     Authors: {h_authors}")
                print(f"     Year: {h_year}, Venue: {h.get('venue','')}")
                
                # PDF 다운로드 시도
                pdf_url = None
                oa_pdf = h.get('openAccessPdf')
                if oa_pdf and oa_pdf.get('url'):
                    pdf_url = oa_pdf['url']
                
                ext_ids = h.get('externalIds', {})
                doi = ext_ids.get('DOI', '')
                
                if pdf_url:
                    safe_name = f"{author.split()[0]}_{year}.pdf".replace(' ', '_')
                    filepath = os.path.join(PAPERS_DIR, safe_name)
                    if not os.path.exists(filepath):
                        if download_pdf(pdf_url, filepath):
                            print(f"     📥 Downloaded: {filepath}")
                        else:
                            print(f"     ⚠️ Download failed: {pdf_url}")
                    else:
                        print(f"     📄 Already exists: {filepath}")
                elif doi:
                    print(f"     DOI: {doi} (no open access PDF)")
                
                results.append({"ref": f"{author} ({year})", "status": "FOUND", "title": h_title, "doi": doi})
                found = True
                break
        
        if not found:
            # 교과서는 검색 안 될 수 있음
            if venue is None:
                print(f"  📚 TEXTBOOK (not in Semantic Scholar — OK)")
                results.append({"ref": f"{author} ({year})", "status": "TEXTBOOK"})
            else:
                print(f"  ❌ NOT FOUND")
                results.append({"ref": f"{author} ({year})", "status": "NOT_FOUND"})
    
    # 결과 저장
    print("\n" + "=" * 70)
    print("검증 결과 요약")
    print("=" * 70)
    
    found_count = sum(1 for r in results if r['status'] == 'FOUND')
    textbook_count = sum(1 for r in results if r['status'] == 'TEXTBOOK')
    not_found = [r for r in results if r['status'] == 'NOT_FOUND']
    
    print(f"  FOUND: {found_count}/{len(results)}")
    print(f"  TEXTBOOK: {textbook_count}/{len(results)}")
    print(f"  NOT FOUND: {len(not_found)}/{len(results)}")
    
    if not_found:
        print("\n미확인 논문:")
        for r in not_found:
            print(f"  ❌ {r['ref']}")
    
    with open(os.path.join(PAPERS_DIR, '..', 'verification_results.json'), 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
