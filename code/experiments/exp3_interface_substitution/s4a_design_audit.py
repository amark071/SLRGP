#!/usr/bin/env python3
"""Audit S4a topic-arm comparability before blinded quality evaluation."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ARMS = ("intact", "l_default", "r_pass", "o_rank_slab", "no_recursion")
CONFIG_KEYS = (
    "seed",
    "prompt_version",
    "model",
    "model_id",
    "model_g",
    "max_depth",
    "d_max",
    "d_max_effective",
    "target_words",
    "word_budget",
    "candidate_limit",
    "n_candidates",
)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def first_value(*mappings: dict[str, Any], key: str) -> Any:
    for mapping in mappings:
        if key in mapping:
            return mapping[key]
    return None


def audit_arm(path: Path, arm: str, topic: str) -> dict[str, Any]:
    survey = path / "survey.md"
    meta = read_json(path / "meta.json")
    trace = read_json(path / "trace.json")
    provenance = read_json(path / "leaf_evidence_provenance.json")
    text = survey.read_text(encoding="utf-8") if survey.exists() else ""
    leaf_writes = trace.get("leaf_writes", [])
    if not isinstance(leaf_writes, list):
        leaf_writes = []
    evidence_counts = [
        len(item.get("candidate_ids", []))
        for item in leaf_writes
        if isinstance(item, dict) and isinstance(item.get("candidate_ids"), list)
    ]
    groups = meta.get("groups", [])
    if not isinstance(groups, list):
        groups = []
    row = {
        "topic": topic,
        "arm": arm,
        "path": str(path),
        "has_survey": survey.exists(),
        "has_meta": bool(meta),
        "has_trace": bool(trace),
        "has_provenance": bool(provenance),
        "words": len(text.split()),
        "reported_words": meta.get("word_count"),
        "n_groups": len(groups),
        "group_sizes": [item.get("n") for item in groups if isinstance(item, dict)],
        "n_leaf_writes": len(leaf_writes),
        "leaf_evidence_counts": evidence_counts,
    }
    for key in CONFIG_KEYS:
        row[key] = first_value(meta, trace, key=key)
    return row


def locate_arm(root: Path, arm: str, topic: str) -> Path:
    name = safe_name(topic)
    candidates = (root / arm / name, root / name / arm, root / arm / topic)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--topics-file", type=Path)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-word-ratio", type=float, default=1.15)
    args = parser.parse_args()

    arms = tuple(item.strip() for item in args.arms.split(",") if item.strip())
    if args.topics_file:
        topics = [
            line.strip()
            for line in args.topics_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        topics = sorted(
            path.name.replace("_", " ")
            for path in args.root.iterdir()
            if path.is_dir() and all((path / arm).is_dir() for arm in arms)
        )
    rows = [audit_arm(locate_arm(args.root, arm, topic), arm, topic) for topic in topics for arm in arms]
    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_topic[row["topic"]].append(row)

    verdicts = []
    for topic, topic_rows in by_topic.items():
        reasons = []
        if len(topic_rows) != len(arms) or not all(row["has_survey"] for row in topic_rows):
            reasons.append("missing_arm_or_survey")
        words = [row["words"] for row in topic_rows if row["words"]]
        if words and max(words) / min(words) > args.max_word_ratio:
            reasons.append(f"word_ratio>{args.max_word_ratio:g}")
        for key in ("seed", "prompt_version", "model", "model_id"):
            values = {str(row[key]) for row in topic_rows if row[key] is not None}
            if len(values) > 1:
                reasons.append(f"mismatched_{key}")
        verdicts.append(
            {
                "topic": topic,
                "primary_eligible": not reasons,
                "reasons": reasons,
                "words_by_arm": {row["arm"]: row["words"] for row in topic_rows},
                "groups_by_arm": {row["arm"]: row["n_groups"] for row in topic_rows},
            }
        )

    report = {
        "root": str(args.root),
        "arms": arms,
        "max_word_ratio": args.max_word_ratio,
        "rows": rows,
        "topic_verdicts": verdicts,
        "n_primary_eligible": sum(item["primary_eligible"] for item in verdicts),
        "n_topics": len(verdicts),
    }
    out = args.out or args.root / "s4a_design_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}; eligible={report['n_primary_eligible']}/{report['n_topics']}")


if __name__ == "__main__":
    main()
