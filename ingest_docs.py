import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

SEED_URLS = [
    "https://olake.io/docs/",
    "https://olake.io/docs/getting-started/creating-first-pipeline/",
    "https://olake.io/docs/getting-started/playground/",
    "https://olake.io/docs/community/setting-up-a-dev-env/",
]
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT = PROJECT_ROOT / "data" / "olake_docs.json"
MAX_PAGES = 35
HEADERS = {"User-Agent": "AskOLakeHackathonBot/1.0 (documentation indexing for a student hackathon)"}

def clean_text(soup):
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text("\n", strip=True)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text[:25000]

def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)
    urls = list(dict.fromkeys(SEED_URLS))
    found = []
    visited = set()

    # Use sitemap if available, but keep only official docs URLs.
    for sitemap in ["https://olake.io/sitemap.xml"]:
        try:
            r = session.get(sitemap, timeout=12)
            if r.ok:
                soup = BeautifulSoup(r.text, "xml")
                for loc in soup.find_all("loc"):
                    u = loc.get_text(strip=True)
                    if "olake.io/docs/" in u and u not in urls:
                        urls.append(u)
        except requests.RequestException:
            pass

    while urls and len(found) < MAX_PAGES:
        url = urls.pop(0)
        if url in visited:
            continue
        visited.add(url)
        try:
            response = session.get(url, timeout=15)
            if response.status_code != 200 or "text/html" not in response.headers.get("content-type", ""):
                continue
            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.title.get_text(" ", strip=True) if soup.title else url
            body = clean_text(soup)
            if len(body) > 250:
                found.append({"title": title, "url": url, "text": body})
                print(f"[{len(found)}] Indexed: {title}")
            # Follow internal docs links only.
            for a in soup.select("a[href]"):
                href = urljoin(url, a.get("href", "").split("#")[0])
                parsed = urlparse(href)
                if parsed.netloc in ("olake.io", "www.olake.io") and "/docs/" in parsed.path:
                    href = href.rstrip("/") + "/"
                    if href not in visited and href not in urls:
                        urls.append(href)
            time.sleep(0.15)
        except requests.RequestException as exc:
            print(f"Skipped {url}: {exc}")

    if not found:
        raise RuntimeError(
            "No OLake documentation pages were downloaded; "
            "the existing index was left unchanged."
        )

    OUTPUT.write_text(json.dumps(found, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(found)} documentation pages to {OUTPUT}")

if __name__ == "__main__":
    main()
