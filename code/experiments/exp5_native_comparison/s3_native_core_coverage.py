#!/usr/bin/env python3
"""Compute reference-derived core-paper coverage for native S3 outputs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SYSTEMS = ("slrgp", "autosurvey", "surveyforge", "surveygen")


def normalize_doi(value: object) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", text)


def normalize_title(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def normalize_arxiv(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(
        r"^(?:arxiv_|https?://arxiv\.org/(?:abs|pdf)/|"
        r"(?:https?://doi\.org/)?10\.48550/arxiv\.)",
        "",
        text,
    )
    return re.sub(r"v\d+$", "", text).rstrip(".pdf")


def identifiers(row: dict) -> tuple[set[str], set[str], set[str]]:
    dois = {
        normalize_doi(row.get(key))
        for key in ("doi", "DOI")
        if normalize_doi(row.get(key))
    }
    titles = {
        normalize_title(row.get(key))
        for key in ("title", "display_name", "paper_title")
        if normalize_title(row.get(key))
    }
    arxiv_ids = {
        normalize_arxiv(row.get(key))
        for key in ("arxiv_id", "value", "doc_id", "doi", "DOI")
        if normalize_arxiv(row.get(key))
        and re.fullmatch(
            r"(?:\d{4}\.\d{4,5}|[a-z-]+/\d{7})",
            normalize_arxiv(row.get(key)),
        )
    }
    return dois, titles, arxiv_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--core-sets", type=Path, required=True)
    parser.add_argument("--topics", type=Path)
    args = parser.parse_args()
    core = json.loads(args.core_sets.read_text(encoding="utf-8"))["topics"]
    selected_topics = (
        {
            row["topic_id"]
            for row in json.loads(args.topics.read_text(encoding="utf-8"))["topics"]
        }
        if args.topics
        else set(core)
    )
    rows = []
    for topic_id, topic_core in sorted(core.items()):
        if topic_id not in selected_topics:
            continue
        core_rows = topic_core["core_references"]
        for system in SYSTEMS:
            path = args.root / "systems" / system / topic_id / "references.json"
            generated = json.loads(path.read_text(encoding="utf-8"))
            generated_dois = set()
            generated_titles = set()
            generated_arxiv_ids = set()
            for row in generated:
                dois, titles, arxiv_ids = identifiers(row)
                generated_dois.update(dois)
                generated_titles.update(titles)
                generated_arxiv_ids.update(arxiv_ids)
            matched = []
            for row in core_rows:
                dois, titles, arxiv_ids = identifiers(row)
                if (dois and dois & generated_dois) or (
                    titles and titles & generated_titles
                ) or (
                    arxiv_ids and arxiv_ids & generated_arxiv_ids
                ):
                    matched.append(row["openalex_id"])
            rows.append(
                {
                    "topic_id": topic_id,
                    "system": system,
                    "core_size": len(core_rows),
                    "matched_core_count": len(matched),
                    "recall_at_core": len(matched) / len(core_rows),
                    "generated_reference_count": len(generated),
                    "coverage_per_generated_reference": (
                        len(matched) / len(generated) if generated else 0.0
                    ),
                    "matched_openalex_ids": matched,
                }
            )
    out = args.root / "audit" / "core_reference_coverage.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"schema_version": "1.0", "rows": rows}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
