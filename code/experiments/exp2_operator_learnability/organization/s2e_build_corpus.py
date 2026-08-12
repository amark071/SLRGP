#!/usr/bin/env python3
"""
S2e/O Phase 1 — deterministic, depth-aware LOTCF-LR node corpus.

Reads the raw E1 LaTeX parse output only. It does not read Qwen annotations,
clusters or first-round labels. It explicitly materializes each review's
synthetic root -> top-level-sections node, fixing the first-round omission.

Outputs:
  <OUT_DIR>/s2e_review_split.json       SHA-256 review-level frozen manifest
  <OUT_DIR>/nodes_{train,heldout}.jsonl depth-aware organization nodes
  <OUT_DIR>/structure_summary.json      parse/length/depth/coverage diagnostics
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from split_utils import SplitManifest

PARSED_DIR = os.environ.get("S2E_PARSED_DIR", "data/exp1_structural_recovery/lotcf_trees")
OUT_DIR = os.environ.get("S2E_OUT_DIR", "data/exp2_operator_learnability/organization/confirmatory")


def node_record(rec: dict, node: dict, path: str, depth: int, parent_title: str | None) -> dict:
    children = node.get("children", [])
    own = set(node.get("own_cite_keys") or [])
    total = set(node.get("total_cite_keys") or [])
    child_union = set()
    for child in children:
        child_union.update(child.get("total_cite_keys") or [])
    return {
        "arxiv_id": rec["arxiv_id"],
        "discipline": rec.get("discipline", ""),
        "node_path": path,
        "node_depth": depth,
        "latex_level": int(node.get("level", 0)),
        "is_review_root": depth == 0,
        "parent_title": parent_title,
        "node_title": node.get("title", "__root__") if depth else "__review_root__",
        "child_titles": [c.get("title", "") for c in children],
        "n_children": len(children),
        "n_own_cites": len(own),
        "n_total_cites": len(total),
        "n_child_union_cites": len(child_union),
        # Whether all evidence below this node is represented by a child branch.
        # This is deterministic citation-bearing coverage, not semantic coverage.
        "citation_child_coverage": (len(child_union) / len(total)) if total else None,
    }


def walk(rec: dict) -> list[dict]:
    root = {
        "level": 0,
        "title": "__review_root__",
        "own_cite_keys": [],
        "total_cite_keys": sorted({k for c in rec["tree"] for k in (c.get("total_cite_keys") or [])}),
        "children": rec["tree"],
    }
    out = []

    def visit(node: dict, path: str, depth: int, parent_title: str | None):
        children = node.get("children", [])
        if len(children) >= 2:
            out.append(node_record(rec, node, path, depth, parent_title))
        for i, child in enumerate(children):
            visit(child, f"{path}.{i}", depth + 1, node.get("title"))

    visit(root, "root", 0, None)
    return out


def quantile_bucket(value: int, boundaries: tuple[float, float]) -> str:
    if value <= boundaries[0]:
        return "short"
    if value <= boundaries[1]:
        return "medium"
    return "long"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-manifest", action="store_true")
    args = ap.parse_args()
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    records, failures = [], Counter()
    for p in sorted(glob.glob(os.path.join(PARSED_DIR, "*", "*.json"))):
        rec = json.load(open(p, encoding="utf-8"))
        if rec.get("parse_status") != "ok":
            failures[rec.get("fail_reason", "unknown")] += 1
            continue
        if len(rec.get("tree") or []) < 2:
            failures["fewer_than_two_top_level_sections"] += 1
            continue
        records.append(rec)
    if not records:
        raise RuntimeError(f"no parseable records in {PARSED_DIR}")

    manifest_path = out / "s2e_review_split.json"
    if manifest_path.exists() and not args.force_manifest:
        manifest = SplitManifest.from_json(str(manifest_path))
        missing = {r["arxiv_id"] for r in records} - set(manifest.assignments)
        if missing:
            raise RuntimeError(f"existing manifest missing {len(missing)} reviews; use --force-manifest only before annotation")
    else:
        manifest = SplitManifest.build(
            experiment="s2e_review_level",
            keys=sorted(r["arxiv_id"] for r in records),
            ratios={"train": 0.8, "heldout": 0.2},
            salt="s2e-review-level-2026-07-10",
        )
        manifest.to_json(str(manifest_path))

    review_n_sections = sorted(int(r.get("n_sections", 0)) for r in records)
    q1 = review_n_sections[int(0.33 * (len(review_n_sections) - 1))]
    q2 = review_n_sections[int(0.67 * (len(review_n_sections) - 1))]
    nodes_by_split = defaultdict(list)
    by_depth, by_length, root_reviews = Counter(), Counter(), 0
    coverage_by_depth = defaultdict(list)

    for rec in records:
        split = manifest.assignments[rec["arxiv_id"]]
        length_bucket = quantile_bucket(int(rec.get("n_sections", 0)), (q1, q2))
        nodes = walk(rec)
        if any(n["is_review_root"] for n in nodes):
            root_reviews += 1
        for node in nodes:
            node["split"] = split
            node["review_length_bucket"] = length_bucket
            node["review_n_sections"] = int(rec.get("n_sections", 0))
            nodes_by_split[split].append(node)
            by_depth[str(node["node_depth"])] += 1
            by_length[length_bucket] += 1
            if node["citation_child_coverage"] is not None:
                coverage_by_depth[str(node["node_depth"])].append(node["citation_child_coverage"])

    for split, nodes in nodes_by_split.items():
        with open(out / f"nodes_{split}.jsonl", "w", encoding="utf-8") as f:
            for node in nodes:
                f.write(json.dumps(node, ensure_ascii=False) + "\n")

    summary = {
        "source": "deterministic E1 parsed LaTeX trees only; no first-round semantic labels",
        "n_parseable_reviews": len(records),
        "n_reviews_with_persisted_root_node": root_reviews,
        "split_counts": manifest.to_json if False else {
            k: sum(1 for v in manifest.assignments.values() if v == k) for k in manifest.ratios
        },
        "node_counts": {k: len(v) for k, v in nodes_by_split.items()},
        "depth_counts": dict(by_depth),
        "length_bucket_node_counts": dict(by_length),
        "length_bucket_section_boundaries": {"short_le": q1, "medium_le": q2, "long_gt": q2},
        "citation_child_coverage_by_depth": {
            d: {
                "n": len(v),
                "mean": sum(v) / len(v),
                "median": sorted(v)[len(v) // 2],
            } for d, v in coverage_by_depth.items()
        },
        "excluded_parse_failures": dict(failures),
    }
    with open(out / "structure_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
