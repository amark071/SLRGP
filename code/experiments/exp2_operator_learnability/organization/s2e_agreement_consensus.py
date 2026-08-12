#!/usr/bin/env python3
"""
S2e/O Phase 3 — two-pass agreement diagnostics and consensus labels.

The two MODEL-A passes are treated as independent blinded annotations from the
same annotation instrument. Consensus is conservative:
  - if pass-1 and pass-2 agree on an axis, keep the shared label;
  - otherwise keep the higher-confidence label for that axis and flag it.

This keeps all records usable for learning while making disagreement explicit
for audit and stratified sensitivity reporting.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

IN_DIR = Path(os.environ.get(
    "S2E_OUT_DIR",
    Path(__file__).resolve().parents[4] / "data/exp2_operator_learnability/organization/confirmatory",
))
OUT_DIR = Path(os.environ.get("S2E_WORK_DIR", Path(__file__).resolve().parents[4] / "work/exp2_organization"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
FACETS = ["topic", "method", "theory", "object", "level", "controversy", "application_task", "other"]
RELATIONS = [
    "chronological", "foundational_to_applied", "simple_to_complex",
    "consensus_to_controversy", "general_to_specific", "enumerative_none", "other",
]


def load_latest(path: Path) -> dict[str, dict]:
    latest = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            latest[row["node_id"]] = row
    return latest


def cohen_kappa(a: list[str], b: list[str], labels: list[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    if abs(1 - pe) < 1e-12:
        return 1.0 if abs(po - 1) < 1e-12 else 0.0
    return (po - pe) / (1 - pe)


def summarize_pair(rows: list[tuple[dict, dict]], axis: str, labels: list[str]) -> dict:
    a = [p1[axis] for p1, _ in rows]
    b = [p2[axis] for _, p2 in rows]
    n = len(rows)
    agree = sum(x == y for x, y in zip(a, b))
    disagreements = Counter((x, y) for x, y in zip(a, b) if x != y)
    return {
        "n": n,
        "agreement": agree / n if n else None,
        "cohen_kappa": cohen_kappa(a, b, labels),
        "pass1_distribution": dict(Counter(a)),
        "pass2_distribution": dict(Counter(b)),
        "top_disagreements": [
            {"pass1": x, "pass2": y, "n": c}
            for (x, y), c in disagreements.most_common(12)
        ],
    }


def choose_axis(p1: dict, p2: dict, axis: str) -> tuple[str, str]:
    if p1[axis] == p2[axis]:
        return p1[axis], "agree"
    c1 = float(p1.get("confidence") or 0.0)
    c2 = float(p2.get("confidence") or 0.0)
    if c2 > c1:
        return p2[axis], "pass2_higher_confidence"
    return p1[axis], "pass1_higher_confidence"


def main() -> None:
    p1 = load_latest(IN_DIR / "annotations_pass1.jsonl")
    p2 = load_latest(IN_DIR / "annotations_pass2.jsonl")
    common = sorted(set(p1) & set(p2))
    if len(common) != 14332:
        raise RuntimeError(f"expected 14332 common nodes, got {len(common)}")
    rows = [(p1[nid], p2[nid]) for nid in common]

    report = {
        "n_nodes": len(rows),
        "facet": summarize_pair(rows, "facet", FACETS),
        "relation": summarize_pair(rows, "relation", RELATIONS),
        "by_depth": {},
        "by_review_length_bucket": {},
        "by_split": {},
    }
    for group_name, key_fn in {
        "by_depth": lambda r: str(r[0]["node_depth"]),
        "by_review_length_bucket": lambda r: r[0]["review_length_bucket"],
        "by_split": lambda r: r[0]["split"],
    }.items():
        buckets = defaultdict(list)
        for row in rows:
            buckets[key_fn(row)].append(row)
        for bucket, sub in sorted(buckets.items()):
            report[group_name][bucket] = {
                "facet": summarize_pair(sub, "facet", FACETS),
                "relation": summarize_pair(sub, "relation", RELATIONS),
            }

    consensus_path = OUT_DIR / "consensus_labels.jsonl"
    with open(consensus_path, "w", encoding="utf-8") as f:
        for p1r, p2r in rows:
            facet, facet_source = choose_axis(p1r, p2r, "facet")
            relation, relation_source = choose_axis(p1r, p2r, "relation")
            out = {
                "node_id": p1r["node_id"],
                "arxiv_id": p1r["arxiv_id"],
                "node_path": p1r["node_path"],
                "split": p1r["split"],
                "node_depth": p1r["node_depth"],
                "review_length_bucket": p1r["review_length_bucket"],
                "discipline": p1r.get("discipline", ""),
                "facet": facet,
                "relation": relation,
                "facet_consensus_source": facet_source,
                "relation_consensus_source": relation_source,
                "facet_pass1": p1r["facet"],
                "facet_pass2": p2r["facet"],
                "relation_pass1": p1r["relation"],
                "relation_pass2": p2r["relation"],
                "confidence_pass1": p1r.get("confidence"),
                "confidence_pass2": p2r.get("confidence"),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    with open(OUT_DIR / "agreement_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({
        "n_nodes": report["n_nodes"],
        "facet_agreement": report["facet"]["agreement"],
        "facet_kappa": report["facet"]["cohen_kappa"],
        "relation_agreement": report["relation"]["agreement"],
        "relation_kappa": report["relation"]["cohen_kappa"],
        "consensus_path": str(consensus_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
