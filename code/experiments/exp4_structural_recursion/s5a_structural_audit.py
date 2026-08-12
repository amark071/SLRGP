#!/usr/bin/env python3
"""Audit six long-form structural-recursion outputs against S5a controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


TOPICS = (
    "3D Gaussian Splatting",
    "Graph Neural Networks",
    "Multimodal Large Language Models",
    "Retrieval-Augmented Generation for Large Language Models",
    "Federated Learning",
    "Efficient Inference for Large Language Models",
)


def safe_name(text: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def load_rows(path: Path) -> dict[tuple[str, str, str], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {(row["topic"], row["tier"], row["arm"]): row for row in data["rows"]}


def structural_leaf_ids(root: Path, topic: str) -> set[str]:
    trace_path = root / safe_name(topic) / "structural_lotcf_recursive" / "trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    return {
        doc_id
        for leaf in trace.get("leaf_writes", [])
        for doc_id in leaf.get("candidate_ids", [])
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-summary", type=Path, required=True)
    parser.add_argument("--structural-qual-summary", type=Path, required=True)
    parser.add_argument("--structural-qual-root", type=Path, required=True)
    parser.add_argument("--structural-quant-summary", type=Path, required=True)
    parser.add_argument("--structural-quant-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    old = load_rows(args.old_summary)
    qual = load_rows(args.structural_qual_summary)
    quant = load_rows(args.structural_quant_summary)
    rows = []
    issues = []
    for topic in TOPICS:
        structural = qual.get((topic, "long", "structural_lotcf_recursive")) or quant.get(
            (topic, "long", "structural_lotcf_recursive")
        )
        structural_root = args.structural_qual_root if topic in {
            "Retrieval-Augmented Generation for Large Language Models"
        } else args.structural_quant_root
        old_recursive = old[(topic, "long", "recursive_slrgp")]
        old_flat = old[(topic, "long", "flat_single_pass")]
        old_fixed = old[(topic, "long", "fixed_outline_no_reentry")]
        leaf_ids = structural_leaf_ids(structural_root, topic)
        expected_leaf_ids = set(old_recursive.get("leaf_evidence_ids", []))
        control_leaf_sets = {
            "naive_recursive_chunking": set(old_recursive.get("leaf_evidence_ids", [])),
            "fixed_outline_no_reentry": set(old_fixed.get("leaf_evidence_ids", [])),
            "flat_single_pass": set(old_flat.get("leaf_evidence_ids", [])),
        }
        row_issues = []
        if not structural or not structural.get("ok"):
            row_issues.append("structural_generation_failed")
        if not (7600 <= structural.get("word_count", 0) <= 11250):
            row_issues.append("structural_word_count_out_of_window")
        if structural.get("n_invalid_citations", 0):
            row_issues.append("structural_invalid_citations")
        if not structural.get("n_o_calls") or not structural.get("n_merge_events"):
            row_issues.append("missing_structural_recursion_trace")
        if any(leaf_ids != control_set for control_set in control_leaf_sets.values()):
            row_issues.append("leaf_evidence_set_mismatch")
        if any(row.get("n_invalid_citations", 0) for row in (old_recursive, old_flat, old_fixed)):
            row_issues.append("control_invalid_citations")
        row = {
            "topic": topic,
            "structural_words": structural.get("word_count"),
            "structural_refs": structural.get("n_refs"),
            "structural_o_calls": structural.get("n_o_calls"),
            "structural_descend_events": structural.get("n_descend_events"),
            "structural_merge_events": structural.get("n_merge_events"),
            "structural_leaf_card_count": len(leaf_ids),
            "control_leaf_card_count": len(expected_leaf_ids),
            "frozen_pool_hash_match": all(
                structural.get("evidence_pool_hash") == control.get("evidence_pool_hash")
                for control in (old_recursive, old_flat, old_fixed)
            ),
            "leaf_evidence_set_match": {
                arm: leaf_ids == control_set for arm, control_set in control_leaf_sets.items()
            },
            "control_words": {
                "naive_recursive_chunking": old_recursive.get("word_count"),
                "fixed_outline_no_reentry": old_fixed.get("word_count"),
                "flat_single_pass": old_flat.get("word_count"),
            },
            "primary_eligible": not row_issues,
            "issues": row_issues,
        }
        rows.append(row)
        issues.extend({"topic": topic, "issue": issue} for issue in row_issues)
    output = {
        "topics": list(TOPICS),
        "n_topics": len(TOPICS),
        "n_primary_eligible": sum(row["primary_eligible"] for row in rows),
        "issues": issues,
        "rows": rows,
        "interpretation": (
            "The structural arm is eligible only when it uses the same frozen leaf evidence set as the "
            "old long-tier controls and exhibits both recursive O calls and upward merge events."
        ),
    }
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}; eligible={output['n_primary_eligible']}/{output['n_topics']}; issues={len(issues)}")


if __name__ == "__main__":
    main()
