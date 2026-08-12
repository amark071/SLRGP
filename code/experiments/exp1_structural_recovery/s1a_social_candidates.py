#!/usr/bin/env python3
"""Build a reproducible social-science review-candidate manifest for S1a."""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path


TARGETS = {
    "economics_social": ("经济学",),
    "management": ("管理学", "运筹与管理"),
    "sociology": ("社会学",),
    "political_science": ("政治学",),
    "law": ("法学",),
    "education": ("教育学",),
    "psychology": ("心理学", "心理学（综合）", "应用心理学", "社会心理学", "实验心理学"),
}
REVIEW_RE = re.compile(r"\b(?:systematic\s+review|literature\s+review|scoping\s+review|meta[- ]analysis|review|survey)\b", re.I)


def metadata(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def is_review(row: dict) -> bool:
    return bool(REVIEW_RE.search(f"{row['title']} {row['abstract']}"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-discipline", type=int, default=150)
    args = parser.parse_args()
    conn = sqlite3.connect(args.db)
    rows_by_target = defaultdict(list)
    for target, labels in TARGETS.items():
        placeholders = ",".join("?" for _ in labels)
        query = f"SELECT doc_id, discipline, tier_intl, tier_source, author_names, data FROM papers WHERE discipline IN ({placeholders})"
        for doc_id, source_discipline, tier, tier_source, author_names, raw in conn.execute(query, labels):
            data = metadata(raw)
            row = {
                "review_id": f"ss_{doc_id}",
                "discipline": target,
                "source_discipline": source_discipline,
                "title": str(data.get("title") or ""),
                "abstract": str(data.get("abstract") or ""),
                "doi": str(data.get("doi") or ""),
                "year": str(data.get("year") or ""),
                "venue": str(data.get("journal") or ""),
                "authors": str(data.get("authors") or author_names or ""),
                "tier": tier,
                "tier_source": tier_source,
                "cited_by_count": int(data.get("cited_by") or 0),
                "candidate_rule": "frozen_title_or_abstract_review_regex_v1",
            }
            if is_review(row):
                rows_by_target[target].append(row)
    manifest = []
    for target, rows in rows_by_target.items():
        rows.sort(key=lambda row: (row["tier"] == "T1", row["cited_by_count"], row["year"]), reverse=True)
        manifest.extend(rows[: args.per_discipline])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in manifest) + "\n", encoding="utf-8")
    summary = {
        "per_discipline_limit": args.per_discipline,
        "n_total": len(manifest),
        "by_target": {target: sum(row["discipline"] == target for row in manifest) for target in TARGETS},
        "note": "This is metadata-only candidate selection. Full-text OA acquisition and TEI parse audit determine S1a eligibility.",
    }
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
