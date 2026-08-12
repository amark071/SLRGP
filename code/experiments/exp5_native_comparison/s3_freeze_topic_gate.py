#!/usr/bin/env python3
"""Atomically freeze confirmatory topics after all registered gates pass."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_counts(path: Path, system: str) -> dict[str, int]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if system == "surveygen":
        return {
            key: int(row.get("valid_count") or 0)
            for key, row in value["topics"].items()
        }
    return {key: int(count) for key, count in value.items()}


def atomic_write(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", type=Path, required=True)
    parser.add_argument("--autosurvey", type=Path, required=True)
    parser.add_argument("--surveyforge", type=Path, required=True)
    parser.add_argument("--slrgp", type=Path, required=True)
    parser.add_argument("--surveygen", type=Path, required=True)
    parser.add_argument("--core-sets", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--minimum", type=int, default=30)
    args = parser.parse_args()
    manifest = json.loads(args.topics.read_text(encoding="utf-8"))
    systems = {
        "autosurvey": read_counts(args.autosurvey, "autosurvey"),
        "surveyforge": read_counts(args.surveyforge, "surveyforge"),
        "slrgp": read_counts(args.slrgp, "slrgp"),
        "surveygen": read_counts(args.surveygen, "surveygen"),
    }
    core = json.loads(args.core_sets.read_text(encoding="utf-8"))
    rows = []
    failures = []
    for topic in manifest["topics"]:
        topic_id = topic["topic_id"]
        counts = {system: values.get(topic_id, 0) for system, values in systems.items()}
        core_topic = (core.get("topics") or {}).get(topic_id) or {}
        row = {
            "topic_id": topic_id,
            "category": topic.get("category"),
            "pre_cutoff_candidate_counts": counts,
            "retrieval_pass": all(value >= args.minimum for value in counts.values()),
            "source_review_count": len(core_topic.get("source_reviews") or []),
            "core_reference_count": len(core_topic.get("core_references") or []),
        }
        row["core_set_pass"] = (
            row["source_review_count"] >= 3 and row["core_reference_count"] >= 20
        )
        if not row["retrieval_pass"] or not row["core_set_pass"]:
            failures.append(topic_id)
        rows.append(row)
    audit = {
        "schema_version": "1.0",
        "minimum_pre_cutoff_candidates": args.minimum,
        "all_passed": not failures,
        "failed_topics": failures,
        "rows": rows,
        "core_reference_sets_sha256": hashlib.sha256(
            args.core_sets.read_bytes()
        ).hexdigest(),
    }
    atomic_write(args.audit_output, audit)
    if failures:
        raise SystemExit(f"Topic gate failed: {failures}")
    manifest["status"] = "frozen_after_all_system_retrieval_gate"
    manifest["selection_audit_sha256"] = sha256_json(audit)
    atomic_write(args.topics, manifest)


if __name__ == "__main__":
    main()
