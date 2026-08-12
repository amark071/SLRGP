#!/usr/bin/env python3
"""Convert GROBID TEI reviews into the E1/S1a parsed-review contract.

The adapter deliberately records PDF/TEI extraction failures as parser failures:
they must never be interpreted as evidence against LOTCF-LR.  Successful records
have the same tree and citation-assignment interface used by the LaTeX S1a path.
"""
from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any


TEI = "{http://www.tei-c.org/ns/1.0}"
MIN_SECTIONS = 3
MIN_CITATIONS = 3


def clean_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return re.sub(r"\s+", " ", "".join(element.itertext())).strip()


def citation_ids(element: ET.Element) -> list[str]:
    ids = []
    for ref in element.iter(f"{TEI}ref"):
        if ref.get("type") != "bibr":
            continue
        target = (ref.get("target") or "").lstrip("#").strip()
        ids.append(target or clean_text(ref))
    return sorted(set(identifier for identifier in ids if identifier))


def dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def heading_level(div: ET.Element, inherited: int) -> int:
    raw = div.get("n") or ""
    match = re.match(r"\s*(\d+)", raw)
    if match:
        return max(1, min(3, int(match.group(1))))
    return min(3, max(1, inherited))


def direct_body_citations(div: ET.Element) -> list[str]:
    """Citations owned by a div, excluding nested div subtrees."""
    values: list[str] = []
    for child in div:
        if child.tag == f"{TEI}div":
            continue
        values.extend(citation_ids(child))
    return dedupe(values)


def parse_div(div: ET.Element, inherited_level: int) -> dict[str, Any]:
    level = heading_level(div, inherited_level)
    head = div.find(f"{TEI}head")
    title = clean_text(head) or f"Untitled section level {level}"
    children = [
        parse_div(child, level + 1)
        for child in div
        if child.tag == f"{TEI}div" and clean_text(child)
    ]
    own = direct_body_citations(div)
    total = dedupe(own + [cite for child in children for cite in child["total_cite_keys"]])
    return {
        "level": level,
        "title": title,
        "own_cite_keys": own,
        "total_cite_keys": total,
        "children": children,
    }


def parse_record(row: dict[str, Any]) -> dict[str, Any]:
    review_id = str(row["review_id"])
    discipline = str(row["discipline"])
    tei_path = Path(row["tei_path"])
    base = {
        "arxiv_id": review_id,
        "discipline": discipline,
        "source": "social_science_pdf",
        "parser": "grobid_tei",
        "source_path": str(tei_path),
        "metadata": row.get("metadata", {}),
    }
    if not tei_path.exists():
        return base | {"parse_status": "failed", "fail_reason": "tei_missing"}
    try:
        root = ET.parse(tei_path).getroot()
    except ET.ParseError as exc:
        return base | {"parse_status": "failed", "fail_reason": "pdf_parse_failed", "parser_diagnostics": {"error": str(exc)}}
    body = root.find(f".//{TEI}text/{TEI}body")
    if body is None:
        return base | {"parse_status": "failed", "fail_reason": "no_body_extracted"}
    tree = [parse_div(div, 1) for div in body if div.tag == f"{TEI}div" and clean_text(div)]
    n_sections = sum(1 for _ in body.iter(f"{TEI}div"))
    all_cites = sorted({cite for node in tree for cite in node["total_cite_keys"]})
    if n_sections < MIN_SECTIONS:
        return base | {"parse_status": "failed", "fail_reason": "too_few_sections", "n_sections": n_sections}
    if len(all_cites) < MIN_CITATIONS:
        return base | {"parse_status": "failed", "fail_reason": "too_few_citations", "n_sections": n_sections}
    if len(tree) < 2:
        return base | {"parse_status": "failed", "fail_reason": "fewer_than_two_top_level_sections", "n_sections": n_sections}
    return base | {
        "parse_status": "ok",
        "n_sections": n_sections,
        "n_unique_cite_keys": len(all_cites),
        "tree": tree,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL: review_id, discipline, tei_path, optional metadata")
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts = Counter()
    for row in rows:
        record = parse_record(row)
        discipline = record["discipline"]
        out = args.out_root / discipline / f"{record['arxiv_id']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        counts[(discipline, record["parse_status"], record.get("fail_reason", "ok"))] += 1
    summary = {
        "n_input": len(rows),
        "n_ok": sum(count for (_, status, _), count in counts.items() if status == "ok"),
        "by_discipline_status": [
            {"discipline": discipline, "parse_status": status, "reason": reason, "n": count}
            for (discipline, status, reason), count in sorted(counts.items())
        ],
        "contract": "E1-compatible parsed-review JSON for S2e/S1a; PDF parser failures are recorded separately.",
    }
    (args.out_root / "_parse_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
