#!/usr/bin/env python3
"""Design-gate and provenance audit for the matched S4a rerun."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_ARMS = (
    "intact",
    "o_rank_slab_matched",
    "flat_no_reentry",
    "v_guarded_stress",
    "v_unguarded_stress",
)


def citation_ids(text: str) -> list[str]:
    out = []
    for match in re.finditer(r"\[([^\]]+)\]", text):
        for value in re.split(r"[,;\s]+", match.group(1)):
            value = value.strip().rstrip(".")
            if value.startswith(("arxiv_", "ss_", "oa_")) and value not in out:
                out.append(value)
    return out


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    args = parser.parse_args()
    arms = tuple(arm.strip() for arm in args.arms.split(",") if arm.strip())

    summary = load_json(args.root / "_run_summary.json")
    min_words = int(summary["controls"]["min_words"])
    max_words = int(summary["controls"]["max_words"])
    topics = summary["topics"]
    rows = []
    issues = []

    for topic in topics:
        tdir = args.root / re.sub(r"[^A-Za-z0-9._-]+", "_", topic).strip("_")
        bundle = load_json(tdir / "frozen_bundle.json")
        intact_vector = None
        for arm in arms:
            adir = tdir / arm
            meta = load_json(adir / "meta.json")
            trace = load_json(adir / "trace.json")
            text = (adir / "survey.md").read_text(encoding="utf-8")
            leaf = load_json(adir / "leaf_evidence_provenance.json")
            cited = citation_ids(text)
            invalid = [cid for cid in cited if cid not in leaf]
            vector = meta.get("group_size_vector", [])
            if arm == "intact":
                intact_vector = vector
            comparable_vector = arm == "v_unguarded_stress" or vector == intact_vector
            n_descend = len(trace.get("descend_events", []))
            if arm in {"flat_no_reentry", "v_guarded_stress", "v_unguarded_stress"} and n_descend != 0:
                issues.append({"topic": topic, "arm": arm, "issue": "unexpected_descend_events", "n": n_descend})
            if not meta.get("ok"):
                issues.append({"topic": topic, "arm": arm, "issue": "not_ok"})
            if not (min_words <= int(meta.get("word_count", 0)) <= max_words):
                issues.append({"topic": topic, "arm": arm, "issue": "word_count_out_of_window", "words": meta.get("word_count")})
            if invalid:
                issues.append({"topic": topic, "arm": arm, "issue": "citation_outside_leaf_provenance", "ids": invalid})
            if not comparable_vector:
                issues.append({"topic": topic, "arm": arm, "issue": "group_vector_mismatch", "vector": vector, "intact": intact_vector})
            rows.append({
                "topic": topic,
                "arm": arm,
                "ok": bool(meta.get("ok")),
                "words": int(meta.get("word_count", 0)),
                "n_refs": int(meta.get("n_refs", 0)),
                "group_size_vector": vector,
                "bundle_hash": meta.get("bundle_hash"),
                "bundle_hash_matches": meta.get("bundle_hash") == bundle.get("bundle_hash"),
                "n_descend_events": n_descend,
                "n_leaf_writes": len(trace.get("leaf_writes", [])),
                "n_citations": len(cited),
                "n_invalid_citations": len(invalid),
                "primary_eligible": bool(meta.get("ok")) and min_words <= int(meta.get("word_count", 0)) <= max_words and not invalid and comparable_vector,
                "llm_calls": (meta.get("llm_usage_delta") or {}).get("n_calls"),
                "prompt_tokens": (meta.get("llm_usage_delta") or {}).get("prompt_tokens"),
                "completion_tokens": (meta.get("llm_usage_delta") or {}).get("completion_tokens"),
            })

    audit = {
        "root": str(args.root),
        "n_rows": len(rows),
        "n_eligible": sum(1 for row in rows if row["primary_eligible"]),
        "issues": issues,
        "rows": rows,
        "summary": summary,
    }
    out = args.out or args.root / "s4a_matched_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}; eligible={audit['n_eligible']}/{audit['n_rows']}; issues={len(issues)}")


if __name__ == "__main__":
    main()
