#!/usr/bin/env python3
"""
探索性抓取（低优先级，可做可不做，不阻塞主线）：
从 OSF 预印本平台（SocArXiv/PsyArXiv/EdArXiv/LawArXiv/MediArXiv）各抓一小批元数据+PDF，
用于后续探索社科 E1/E2（综述结构解析）在预印本全文上是否可行。

每个 provider 抓 MAX_PER_PROVIDER 篇（分页拉取元数据），标题/摘要含 survey/review 优先，
下载 PDF（如有）。限速：每次请求间隔 2 秒。
"""
import os
import time
import urllib.request
import urllib.parse
import json
import re

PROVIDERS = ["socarxiv", "psyarxiv", "edarxiv", "lawarxiv", "mediarxiv"]
MAX_PER_PROVIDER = 150
OUT_DIR = "work/corpus_rebuild/osf_preprints"
SLEEP_SECONDS = 2
SURVEY_KEYWORDS = re.compile(r"\b(survey|review|meta-analysis|systematic review)\b", re.IGNORECASE)


def api_get(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SLRGP-NMI-data-pipeline/0.1"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"  retry due to: {e}")
            time.sleep(3)
    return None


def fetch_provider(provider):
    out_dir = f"{OUT_DIR}/{provider}"
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    url = (
        f"https://api.osf.io/v2/preprints/"
        f"?filter%5Bprovider%5D={provider}&page%5Bsize%5D=100"
        f"&sort=-date_created"
    )
    collected = 0
    while url and collected < MAX_PER_PROVIDER:
        data = api_get(url)
        if not data:
            break
        for item in data.get("data", []):
            attrs = item.get("attributes", {})
            title = attrs.get("title", "")
            abstract = attrs.get("description", "") or ""
            pid = item["id"]
            is_survey = bool(SURVEY_KEYWORDS.search(title) or SURVEY_KEYWORDS.search(abstract))
            manifest.append({
                "id": pid, "title": title, "abstract": abstract,
                "date_created": attrs.get("date_created"), "is_survey_like": is_survey,
            })
            # 优先下载疑似综述的PDF
            if is_survey:
                links = item.get("links", {})
                pdf_url = links.get("preprint_doi") or f"https://osf.io/{pid}/download"
                pdf_path = f"{out_dir}/{pid}.pdf"
                if not os.path.exists(pdf_path):
                    try:
                        req = urllib.request.Request(pdf_url, headers={"User-Agent": "SLRGP-NMI/0.1"})
                        with urllib.request.urlopen(req, timeout=30) as resp:
                            content = resp.read()
                        if content[:4] == b"%PDF":
                            with open(pdf_path, "wb") as f:
                                f.write(content)
                    except Exception:
                        pass
                    time.sleep(SLEEP_SECONDS)
            collected += 1
            if collected >= MAX_PER_PROVIDER:
                break
        url = data.get("links", {}).get("next")
        time.sleep(SLEEP_SECONDS)

    with open(f"{out_dir}/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    n_survey = sum(1 for m in manifest if m["is_survey_like"])
    print(f"{provider}: 抓取 {len(manifest)} 条元数据，疑似综述 {n_survey} 篇")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for p in PROVIDERS:
        print(f"=== {p} ===")
        fetch_provider(p)
    print("\nOSF 探索性抓取完成")


if __name__ == "__main__":
    main()
