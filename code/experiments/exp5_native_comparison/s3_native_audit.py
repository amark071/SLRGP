#!/usr/bin/env python3
"""Deterministic completeness, provenance, citation, and compliance audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


SYSTEMS = ("slrgp", "autosurvey", "surveyforge", "surveygen")
LEAKAGE_PATTERNS = (
    re.compile(r"\bI(?:'ll| will| need to) (?:carefully )?(?:check|verify|evaluate)", re.I),
    re.compile(r"\bcitation(?:s)? (?:check|verification)\b", re.I),
    re.compile(r"\bthe paper content described above\b", re.I),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(topic_dir: Path) -> tuple[bool, list[str]]:
    path = topic_dir / "artifact_manifest.json"
    if not path.is_file():
        return False, ["artifact_manifest_missing"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    issues = []
    for name, record in (manifest.get("files") or {}).items():
        artifact = topic_dir / name
        if not artifact.is_file():
            issues.append(f"missing:{name}")
        elif sha256_file(artifact) != record.get("sha256"):
            issues.append(f"hash_mismatch:{name}")
    return not issues, issues


def split_body(text: str) -> str:
    return re.split(r"(?im)^##+\s+(?:references|bibliography)\s*$", text, 1)[0]


def numeric_citations(text: str) -> set[int]:
    values = set()
    for match in re.finditer(r"\[([0-9][0-9,;\s–-]*)\]", split_body(text)):
        values.update(int(value) for value in re.findall(r"\d+", match.group(1)))
    return values


def identifier_citations(text: str) -> set[str]:
    values = set()
    for match in re.finditer(r"\[([^\]]+)\]", split_body(text)):
        for value in re.split(r"[,;\s]+", match.group(1)):
            value = value.strip().rstrip(".")
            if value.startswith(("arxiv_", "ss_", "oa_")):
                values.add(value)
    return values


def reference_ids(references: list[dict]) -> set[str]:
    values = set()
    for row in references:
        for key in ("source_key", "doc_id", "id", "arxiv_id"):
            if row.get(key):
                values.add(str(row[key]))
    return values


def cutoff_violations(value: object, cutoff: str) -> list[str]:
    violations = []
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = key.lower()
            if item and (
                lowered in {"date", "publicationdate", "publication_date"}
                or lowered.endswith("_date")
            ):
                date = str(item)[:10]
                if re.match(r"^\d{4}(?:-\d{2}(?:-\d{2})?)?$", date) and date > cutoff:
                    violations.append(date)
            elif item and lowered in {"year", "publication_year"}:
                if str(item)[:4].isdigit() and int(str(item)[:4]) > int(cutoff[:4]):
                    violations.append(str(item))
            violations.extend(cutoff_violations(item, cutoff))
    elif isinstance(value, list):
        for item in value:
            violations.extend(cutoff_violations(item, cutoff))
    return violations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--topics", type=Path, required=True)
    args = parser.parse_args()
    topics = json.loads(args.topics.read_text(encoding="utf-8"))["topics"]
    rows = []
    for topic in topics:
        for system in SYSTEMS:
            topic_dir = args.root / "systems" / system / topic["topic_id"]
            status_path = topic_dir / "status.json"
            status = (
                json.loads(status_path.read_text(encoding="utf-8"))
                if status_path.is_file()
                else {"status": "missing"}
            )
            survey_path = topic_dir / "survey.md"
            refs_path = topic_dir / "references.json"
            text = (
                survey_path.read_text(encoding="utf-8")
                if survey_path.is_file()
                else ""
            )
            references = (
                json.loads(refs_path.read_text(encoding="utf-8"))
                if refs_path.is_file()
                else []
            )
            manifest_valid, manifest_issues = verify_manifest(topic_dir)
            if system == "slrgp":
                citations = identifier_citations(text)
                valid_ids = reference_ids(references)
                invalid = sorted(citations - valid_ids)
                cited_count = len(citations & valid_ids)
            else:
                citations = numeric_citations(text)
                explicit = {
                    int(str(row["citation_number"]))
                    for row in references
                    if str(row.get("citation_number") or "").isdigit()
                }
                valid_numbers = explicit or set(range(1, len(references) + 1))
                invalid = sorted(citations - valid_numbers)
                cited_count = len(citations & valid_numbers)
            headings = [
                line.strip()
                for line in text.splitlines()
                if re.match(r"^#{1,6}\s+", line)
            ]
            leakage = [
                pattern.pattern
                for pattern in LEAKAGE_PATTERNS
                if pattern.search(text)
            ]
            rows.append(
                {
                    "topic_id": topic["topic_id"],
                    "system": system,
                    "status": status.get("status"),
                    "files_complete_and_hash_valid": manifest_valid,
                    "manifest_issues": manifest_issues,
                    "word_count": len(text.split()),
                    "length_compliant": (
                        topic["min_words"] <= len(text.split()) <= topic["max_words"]
                    ),
                    "reference_count": len(references),
                    "reference_compliant": (
                        topic["target_references_min"]
                        <= len(references)
                        <= topic["target_references_max"]
                    ),
                    "unique_citations": len(citations),
                    "invalid_citations": invalid,
                    "citation_closure": (
                        cited_count / len(references) if references else 0.0
                    ),
                    "cutoff_violations": cutoff_violations(
                        references, topic["publication_cutoff"]
                    ),
                    "heading_count": len(headings),
                    "duplicate_heading_count": len(headings) - len(set(headings)),
                    "prompt_leakage_patterns": leakage,
                }
            )
    audit_dir = args.root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "n_rows": len(rows),
        "all_generation_success": all(row["status"] == "ok" for row in rows),
        "all_hash_valid": all(
            row["files_complete_and_hash_valid"] for row in rows
        ),
        "rows": rows,
    }
    (audit_dir / "deterministic_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}))


if __name__ == "__main__":
    main()
