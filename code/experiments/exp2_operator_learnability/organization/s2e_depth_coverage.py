#!/usr/bin/env python3
"""
S2e/O Phase 5 — recursion depth, review length and structural coverage profile.

This report separates two notions that should not be conflated:
  1. structural citation-child coverage: whether citation-bearing evidence under
     a parsed section is represented in child branches;
  2. learned semantic organization: whether the model can predict facet/relation
     labels for the organization node.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[4]
IN_DIR = Path(os.environ.get(
    "S2E_OUT_DIR",
    _PKG_ROOT / "data/exp2_operator_learnability/organization/confirmatory",
))
OUT_DIR = Path(os.environ.get("S2E_WORK_DIR", _PKG_ROOT / "work/exp2_organization"))
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[len(s) // 2]


def summarize_nodes(nodes: list[dict]) -> dict:
    by_review = defaultdict(list)
    for n in nodes:
        by_review[n["arxiv_id"]].append(n)
    review_rows = []
    for rid, ns in by_review.items():
        review_rows.append({
            "arxiv_id": rid,
            "split": ns[0]["split"],
            "review_length_bucket": ns[0]["review_length_bucket"],
            "review_n_sections": ns[0]["review_n_sections"],
            "n_nodes": len(ns),
            "max_node_depth": max(int(n["node_depth"]) for n in ns),
        })

    profile = {}
    for bucket in sorted({r["review_length_bucket"] for r in review_rows}):
        sub = [r for r in review_rows if r["review_length_bucket"] == bucket]
        profile[bucket] = {
            "n_reviews": len(sub),
            "mean_review_n_sections": mean([r["review_n_sections"] for r in sub]),
            "mean_organization_nodes": mean([r["n_nodes"] for r in sub]),
            "median_organization_nodes": median([r["n_nodes"] for r in sub]),
            "mean_max_node_depth": mean([r["max_node_depth"] for r in sub]),
            "max_depth_distribution": dict(Counter(str(r["max_node_depth"]) for r in sub)),
        }
    return {
        "review_length_recursion_profile": profile,
        "overall_max_depth_distribution": dict(Counter(str(r["max_node_depth"]) for r in review_rows)),
    }


def structural_coverage(nodes: list[dict]) -> dict:
    out = {}
    for key in ["node_depth", "review_length_bucket"]:
        buckets = defaultdict(list)
        for n in nodes:
            cov = n.get("citation_child_coverage")
            if cov is not None:
                buckets[str(n[key])].append(float(cov))
        out[f"by_{key}"] = {
            b: {"n_nodes": len(v), "mean": mean(v), "median": median(v)}
            for b, v in sorted(buckets.items())
        }
    return out


def prediction_profile(preds: list[dict]) -> dict:
    out = {}
    for axis in ["facet", "relation"]:
        for key in ["node_depth", "review_length_bucket"]:
            buckets = defaultdict(list)
            for r in preds:
                ok = r[f"gold_{axis}"] == r[f"pred_{axis}_tfidf_logreg"]
                buckets[str(r[key])].append(1.0 if ok else 0.0)
            out[f"{axis}_accuracy_by_{key}"] = {
                b: {"n_nodes": len(v), "accuracy": mean(v)}
                for b, v in sorted(buckets.items())
            }
    return out


def main() -> None:
    nodes = load_jsonl(IN_DIR / "nodes_train.jsonl") + load_jsonl(IN_DIR / "nodes_heldout.jsonl")
    # heldout_predictions 优先读 s2e_train_eval.py 本轮再生成的工作副本，否则读随包冻结版本。
    preds_path = OUT_DIR / "heldout_predictions.jsonl"
    if not preds_path.exists():
        preds_path = IN_DIR / "heldout_predictions.jsonl"
    preds = load_jsonl(preds_path)
    structure = json.load(open(IN_DIR / "structure_summary.json", encoding="utf-8")) if (IN_DIR / "structure_summary.json").exists() else {}
    report = {
        "interpretation": {
            "structural_citation_child_coverage": "Deterministic parse-tree coverage of citation-bearing descendants, not semantic label correctness.",
            "semantic_organization_learning": "Held-out prediction of consensus facet/relation labels for organization nodes.",
            "depth_policy": "Depth is treated as observed natural recursion, with review-level split and length-bucket stratification; no fixed-depth normalization is imposed.",
        },
        "n_nodes_total": len(nodes),
        "n_heldout_prediction_nodes": len(preds),
        "root_node_repair": {
            "n_reviews_with_persisted_root_node": structure.get("n_reviews_with_persisted_root_node"),
            "n_parseable_reviews": structure.get("n_parseable_reviews"),
        },
        **summarize_nodes(nodes),
        "structural_citation_child_coverage": structural_coverage(nodes),
        "heldout_tfidf_logreg_semantic_accuracy": prediction_profile(preds),
    }
    with open(OUT_DIR / "s2e_depth_coverage_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({
        "n_nodes_total": report["n_nodes_total"],
        "root_node_repair": report["root_node_repair"],
        "review_length_recursion_profile": report["review_length_recursion_profile"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
