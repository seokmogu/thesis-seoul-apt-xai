#!/usr/bin/env python3
"""Reference verification via Firecrawl search API."""
import json
import os
import re
import time
from pathlib import Path

import requests
from requests import RequestException

BASE_DIR = Path(__file__).resolve().parent.parent
PAPERS_DIR = BASE_DIR / "references" / "papers"
PAPERS_DIR.mkdir(parents=True, exist_ok=True)
ENV_FILE = BASE_DIR / ".env.firecrawl"
VERIFY_JSON = BASE_DIR / "references" / "verification_firecrawl.json"

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


def load_api_key() -> str:
    if os.getenv("FIRECRAWL_API_KEY"):
        return os.environ["FIRECRAWL_API_KEY"]
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if line.startswith("FIRECRAWL_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("FIRECRAWL_API_KEY not found")


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def extract_doi(text: str):
    match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", text or "", re.I)
    return match.group(0) if match else None


def search_firecrawl(session: requests.Session, query: str):
    url = "https://api.firecrawl.dev/v1/search"
    payload = {
        "query": query,
        "limit": 5,
        "scrapeOptions": {
            "formats": ["markdown"],
            "onlyMainContent": True,
            "maxAge": 86400000,
        },
    }
    for attempt in range(4):
        try:
            response = session.post(url, json=payload, timeout=20)
        except RequestException:
            time.sleep((attempt + 1) * 3)
            continue
        if response.status_code == 429:
            time.sleep((attempt + 1) * 10)
            continue
        if response.status_code >= 400:
            return []
        try:
            data = response.json().get("data", [])
        except ValueError:
            return []
        return data if isinstance(data, list) else []
    return []


def verify_one(session: requests.Session, label: str, year: int, title_fragment: str):
    query = f'"{title_fragment}" {year}'
    results = search_firecrawl(session, query)
    target = normalize(title_fragment)
    for item in results:
        title = item.get("title", "") or ""
        desc = item.get("description", "") or ""
        markdown = item.get("markdown", "") or ""
        haystack = " ".join([title, desc, markdown])
        if target[:28] and target[:28] in normalize(haystack):
            doi = extract_doi(haystack)
            return {
                "ref": label,
                "status": "VERIFIED",
                "query": query,
                "matched_title": title,
                "url": item.get("url"),
                "doi": doi,
            }
    return {
        "ref": label,
        "status": "NOT_FOUND",
        "query": query,
        "matched_title": None,
        "url": None,
        "doi": None,
    }


def main() -> None:
    api_key = load_api_key()
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    })

    results = []
    for label, year, title_fragment in REFS:
        result = verify_one(session, label, year, title_fragment)
        results.append(result)
        print(f"{label}: {result['status']}")
        time.sleep(1)

    VERIFY_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    verified = sum(1 for r in results if r["status"] == "VERIFIED")
    print(f"verified={verified}/{len(results)}")
    print(f"output={VERIFY_JSON}")


if __name__ == "__main__":
    main()
