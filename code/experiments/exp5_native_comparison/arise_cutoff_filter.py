#!/usr/bin/env python3
"""Resolve ARISE web citations through OpenAlex and enforce the S3 cutoff."""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


DOI_PATTERN = re.compile(r"(?:doi\.org/|doi:\s*)(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.I)
SPACE_PATTERN = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    return SPACE_PATTERN.sub(" ", text.lower()).strip()


def match_score(title: str, citation: str) -> float:
    left = normalize(title)
    right = normalize(citation)
    if left and left in right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


def fetch_json(url: str, attempts: int = 4) -> dict[str, Any]:
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "SLRGP-S3/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except Exception:
            if attempt + 1 >= attempts:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def resolve(citation: str) -> dict[str, Any] | None:
    doi_match = DOI_PATTERN.search(citation)
    if doi_match:
        doi = doi_match.group(1).rstrip(".,;)")
        url = "https://api.openalex.org/works/https://doi.org/" + urllib.parse.quote(
            doi, safe="/"
        )
        try:
            item = fetch_json(url)
            return {
                "id": item.get("id"),
                "doi": item.get("doi"),
                "title": item.get("title"),
                "publication_date": item.get("publication_date"),
                "match_score": 1.0,
                "resolution": "doi",
            }
        except Exception:
            pass
    params = urllib.parse.urlencode(
        {
            "search": citation,
            "per-page": 5,
            "select": "id,title,publication_date,doi",
        }
    )
    payload = fetch_json("https://api.openalex.org/works?" + params)
    candidates = payload.get("results") or []
    if not candidates:
        return None
    item = max(candidates, key=lambda row: match_score(row.get("title") or "", citation))
    score = match_score(item.get("title") or "", citation)
    if score < 0.45:
        return None
    return {
        "id": item.get("id"),
        "doi": item.get("doi"),
        "title": item.get("title"),
        "publication_date": item.get("publication_date"),
        "match_score": score,
        "resolution": "title_search",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--minimum-kept", type=int, default=10)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    kept = []
    audit_rows = []
    for row in rows:
        citation = row.get("Result") or ""
        try:
            resolved = resolve(citation)
            status = "unresolved" if resolved is None else "resolved"
        except Exception as exc:
            resolved = None
            status = f"resolution_error:{type(exc).__name__}"
        eligible = bool(
            resolved
            and resolved.get("publication_date")
            and resolved["publication_date"] <= args.cutoff
        )
        audit_rows.append(
            {
                "citation": citation,
                "status": status,
                "eligible": eligible,
                "resolved": resolved,
            }
        )
        if eligible:
            kept.append(row)
        time.sleep(0.1)
    if len(kept) < args.minimum_kept:
        raise SystemExit(
            f"ARISE cutoff gate failed: only {len(kept)} of {len(rows)} citations eligible"
        )
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Index", "Result"])
        writer.writeheader()
        for index, row in enumerate(kept, 1):
            writer.writerow({"Index": index, "Result": row["Result"]})
    args.audit.write_text(
        json.dumps(
            {
                "cutoff": args.cutoff,
                "input_count": len(rows),
                "eligible_count": len(kept),
                "rows": audit_rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
