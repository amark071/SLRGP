#!/usr/bin/env python3
"""Resolve OA PDFs for S1a social-science review candidates via Semantic Scholar."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


API = "https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=title,openAccessPdf"
BATCH_API = "https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,openAccessPdf,externalIds"
OPENALEX_API = "https://api.openalex.org/works/{doi}?api_key={key}"


def request_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "SLRGP-S1a-research/1.0"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read())


def request_batch(dois: list[str]) -> dict[str, str]:
    payload = json.dumps({"ids": [f"DOI:{doi}" for doi in dois]}).encode()
    request = urllib.request.Request(
        BATCH_API,
        data=payload,
        headers={"User-Agent": "SLRGP-S1a-research/1.0", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        records = json.loads(response.read())
    urls = {}
    for record in records:
        if not record:
            continue
        doi = str((record.get("externalIds") or {}).get("DOI") or "").lower()
        url = (record.get("openAccessPdf") or {}).get("url")
        if doi and url:
            urls[doi] = url
    return urls


def request_openalex_pdf(doi: str, api_key: str) -> str | None:
    work_url = "https://doi.org/" + doi
    url = OPENALEX_API.format(doi=urllib.parse.quote(work_url, safe=""), key=urllib.parse.quote(api_key, safe=""))
    work = request_json(url)
    location = work.get("best_oa_location") or {}
    return location.get("pdf_url") or location.get("landing_page_url")


def fetch_pdf(url: str, target: Path) -> bool:
    request = urllib.request.Request(url, headers={"User-Agent": "SLRGP-S1a-research/1.0", "Accept": "application/pdf"})
    with urllib.request.urlopen(request, timeout=90) as response:
        content_type = response.headers.get("Content-Type", "")
        data = response.read()
    if len(data) < 10_000 or not data.startswith(b"%PDF"):
        return False
    if "pdf" not in content_type.lower() and not data.startswith(b"%PDF"):
        return False
    target.write_bytes(data)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--per-discipline", type=int, default=50)
    parser.add_argument("--sleep", type=float, default=1.2)
    parser.add_argument("--openalex-key", default=os.environ.get("OPENALEX_API_KEY", ""))
    args = parser.parse_args()
    if not args.openalex_key:
        raise SystemExit("OPENALEX_API_KEY or --openalex-key is required")
    rows = [json.loads(line) for line in args.candidates.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = defaultdict(int)
    selected_rows = []
    for row in rows:
        if selected[row["discipline"]] >= args.per_discipline:
            continue
        selected[row["discipline"]] += 1
        selected_rows.append(row)
    ledger = []
    counts = Counter()
    for row in selected_rows:
        discipline = row["discipline"]
        doi = row.get("doi", "").replace("https://doi.org/", "").strip()
        outcome = row | {"doi_normalized": doi, "pdf_url": "", "downloaded": False, "failure_reason": ""}
        if not doi:
            outcome["failure_reason"] = "missing_doi"
            ledger.append(outcome)
            counts[(discipline, outcome["failure_reason"])] += 1
            continue
        target = args.out_dir / discipline / f"{row['review_id']}.pdf"
        if target.exists() and target.stat().st_size > 10_000:
            outcome["downloaded"] = True
            outcome["failure_reason"] = "already_downloaded"
            ledger.append(outcome)
            counts[(discipline, "already_downloaded")] += 1
            continue
        try:
            pdf = request_openalex_pdf(doi, args.openalex_key)
            if not pdf:
                outcome["failure_reason"] = "no_open_access_pdf"
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                outcome["downloaded"] = fetch_pdf(pdf, target)
                outcome["pdf_url"] = pdf
                if not outcome["downloaded"]:
                    outcome["failure_reason"] = "pdf_download_or_content_validation_failed"
        except Exception as exc:  # noqa: BLE001 - persist every acquisition failure
            outcome["failure_reason"] = f"lookup_error:{type(exc).__name__}"
        ledger.append(outcome)
        counts[(discipline, "downloaded" if outcome["downloaded"] else outcome["failure_reason"])] += 1
        time.sleep(args.sleep)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "oa_acquisition_ledger.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in ledger) + "\n", encoding="utf-8"
    )
    summary = {
        "n_attempted": len(ledger),
        "n_downloaded": sum(row["downloaded"] for row in ledger),
        "by_discipline_outcome": [
            {"discipline": discipline, "outcome": outcome, "n": n}
            for (discipline, outcome), n in sorted(counts.items())
        ],
    }
    (args.out_dir / "oa_acquisition_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
