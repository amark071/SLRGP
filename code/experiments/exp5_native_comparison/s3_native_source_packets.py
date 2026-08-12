#!/usr/bin/env python3
"""Resolve extracted claim citations into auditable title/abstract packets."""
from __future__ import annotations

import argparse
import http.client
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


ARXIV_API = "https://export.arxiv.org/api/query"
SYSTEMS = ("slrgp", "autosurvey", "surveyforge", "surveygen")
ATOM = {"a": "http://www.w3.org/2005/Atom"}


def clean_arxiv_id(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^(?:arxiv_|https?://arxiv\.org/(?:abs|pdf)/)", "", text, flags=re.I)
    text = re.sub(r"v\d+$", "", text)
    return text.rstrip(".pdf")


def row_arxiv_id(row: dict) -> str:
    for key in ("arxiv_id", "doc_id", "value", "source_key"):
        value = clean_arxiv_id(row.get(key))
        if re.match(r"^\d{4}\.\d{4,5}$", value) or re.match(
            r"^[a-z-]+/\d{7}$", value, re.I
        ):
            return value
    doi = str(row.get("doi") or "")
    match = re.search(r"arxiv[./](\d{4}\.\d{4,5})", doi, re.I)
    return match.group(1) if match else ""


def fetch_arxiv(ids: list[str], cache_path: Path) -> dict:
    cache = (
        json.loads(cache_path.read_text(encoding="utf-8"))
        if cache_path.is_file()
        else {}
    )
    missing = [
        value
        for value in sorted(set(ids))
        if value and (value not in cache or cache[value].get("resolution_error"))
    ]
    for start in range(0, len(missing), 25):
        chunk = missing[start : start + 25]
        url = ARXIV_API + "?" + urllib.parse.urlencode(
            {"id_list": ",".join(chunk), "max_results": len(chunk)}
        )
        request = urllib.request.Request(
            url, headers={"User-Agent": "SLRGP-S3-source-packets/1.0"}
        )
        error: Exception | None = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    root = ET.fromstring(response.read())
                break
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                http.client.IncompleteRead,
                ET.ParseError,
            ) as exc:
                error = exc
                time.sleep(min(60, 5 * (2**attempt)))
        else:
            raise RuntimeError(f"arXiv resolution failed for {chunk!r}: {error!r}")
        returned = set()
        for entry in root.findall("a:entry", ATOM):
            identifier = clean_arxiv_id(entry.findtext("a:id", "", ATOM))
            returned.add(identifier)
            cache[identifier] = {
                "arxiv_id": identifier,
                "title": " ".join(
                    (entry.findtext("a:title", "", ATOM) or "").split()
                ),
                "abstract": " ".join(
                    (entry.findtext("a:summary", "", ATOM) or "").split()
                ),
                "published": entry.findtext("a:published", "", ATOM),
                "source_url": f"https://arxiv.org/abs/{identifier}",
            }
        for identifier in set(chunk) - returned:
            cache[identifier] = {"arxiv_id": identifier, "resolution_error": "not_found"}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"arxiv resolved {start + len(chunk)}/{len(missing)}", flush=True)
        time.sleep(3)
    return cache


def reference_map(system: str, references: list[dict]) -> dict[str, dict]:
    mapping = {}
    for index, row in enumerate(references, 1):
        keys = {str(index)}
        for key in ("citation_number", "source_key", "doc_id", "arxiv_id"):
            if row.get(key):
                keys.add(str(row[key]))
        for key in keys:
            mapping[key] = row
    return mapping


def marker_keys(markers: list[str], system: str) -> list[str]:
    joined = " ".join(markers)
    if system == "slrgp":
        return re.findall(r"(?:arxiv_|ss_|oa_)[A-Za-z0-9._/-]+", joined)
    return re.findall(r"\d+", joined)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--topics", type=Path, required=True)
    args = parser.parse_args()
    topics = json.loads(args.topics.read_text(encoding="utf-8"))["topics"]
    claim_root = args.root / "claim_support"
    references_by_output = {}
    extracted_by_output = {}
    all_arxiv_ids = []
    for topic in topics:
        for system in SYSTEMS:
            key = (topic["topic_id"], system)
            refs = json.loads(
                (
                    args.root
                    / "systems"
                    / system
                    / topic["topic_id"]
                    / "references.json"
                ).read_text(encoding="utf-8")
            )
            extracted = json.loads(
                (
                    claim_root
                    / "extracted_claims"
                    / f"{topic['topic_id']}__{system}.json"
                ).read_text(encoding="utf-8")
            )
            references_by_output[key] = refs
            extracted_by_output[key] = extracted
            mapping = reference_map(system, refs)
            for claim in extracted["claims"]:
                for marker in marker_keys(claim["citation_markers"], system):
                    row = mapping.get(marker)
                    if row is not None:
                        all_arxiv_ids.append(row_arxiv_id(row))
    arxiv_cache = fetch_arxiv(
        all_arxiv_ids, claim_root / "source_cache" / "arxiv_metadata.json"
    )
    packet_dir = claim_root / "source_packets"
    packet_dir.mkdir(parents=True, exist_ok=True)
    for topic in topics:
        for system in SYSTEMS:
            output = packet_dir / f"{topic['topic_id']}__{system}.json"
            key = (topic["topic_id"], system)
            extracted = extracted_by_output[key]
            refs = references_by_output[key]
            mapping = reference_map(system, refs)
            claims = []
            for claim in extracted["claims"]:
                sources = []
                unresolved = []
                for key in marker_keys(claim["citation_markers"], system):
                    row = mapping.get(key)
                    if row is None:
                        unresolved.append(key)
                        continue
                    source = dict(row)
                    arxiv_id = row_arxiv_id(row)
                    if arxiv_id and arxiv_cache.get(arxiv_id):
                        resolved = arxiv_cache[arxiv_id]
                        source.setdefault("title", resolved.get("title"))
                        source.setdefault("abstract", resolved.get("abstract"))
                        source["resolved_arxiv"] = resolved
                    source["source_accessible"] = bool(
                        source.get("title") and source.get("abstract")
                    )
                    sources.append(source)
                claims.append(
                    {
                        **claim,
                        "sources": sources,
                        "unresolved_citation_keys": unresolved,
                        "source_accessible": bool(sources)
                        and all(row["source_accessible"] for row in sources)
                        and not unresolved,
                    }
                )
            output.write_text(
                json.dumps(
                    {
                        "topic_id": topic["topic_id"],
                        "system": system,
                        "claims": claims,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(topic["topic_id"], system, "packetized", flush=True)


if __name__ == "__main__":
    main()
