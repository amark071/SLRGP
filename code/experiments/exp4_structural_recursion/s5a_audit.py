#!/usr/bin/env python3
"""Design-gate audit for S5a length/depth outputs."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


ARMS = ("recursive_slrgp", "fixed_outline_no_reentry", "flat_single_pass")


def safe_name(text: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    summary_path = args.summary or args.root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = []
    issues = []
    row_by_key = {}
    for meta in summary["rows"]:
        tier_cfg = summary["tiers"][meta["tier"]]
        topic_dir = args.root / meta["tier"] / safe_name(meta["topic"]) / meta["arm"]
        trace_path = topic_dir / "trace.json"
        trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else {}
        words = meta.get("word_count") or 0
        issues_here = []
        if not meta.get("ok"):
            issues_here.append("generation_failed")
        if not (tier_cfg["min_words"] <= words <= tier_cfg["max_words"]):
            issues_here.append("word_count_out_of_window")
        if meta.get("n_invalid_citations", 0) != 0:
            issues_here.append("invalid_citations")
        if meta["arm"] == "recursive_slrgp" and tier_cfg["d_max"] > 1 and not trace.get("descend_events"):
            issues_here.append("missing_recursive_trace")
        row = {
            "topic": meta["topic"],
            "tier": meta["tier"],
            "arm": meta["arm"],
            "ok": bool(meta.get("ok")),
            "words": words,
            "target_words": tier_cfg["target_words"],
            "d_max": tier_cfg["d_max"],
            "n_refs": meta.get("n_refs", 0),
            "n_invalid_citations": meta.get("n_invalid_citations", 0),
            "n_leaf_writes": len(trace.get("leaf_writes", [])),
            "n_descend_events": len(trace.get("descend_events", [])),
            "evidence_pool_hash": meta.get("evidence_pool_hash"),
            "primary_eligible": not issues_here,
            "issues": issues_here,
            "llm_calls": (meta.get("llm_usage_delta") or {}).get("n_calls", 0),
            "prompt_tokens": (meta.get("llm_usage_delta") or {}).get("prompt_tokens", 0),
            "completion_tokens": (meta.get("llm_usage_delta") or {}).get("completion_tokens", 0),
        }
        rows.append(row)
        row_by_key[(row["topic"], row["tier"], row["arm"])] = row
        for issue in issues_here:
            issues.append({"topic": row["topic"], "tier": row["tier"], "arm": row["arm"], "issue": issue, "words": words})

    pairability = []
    topics = summary["topics"]
    tiers = list(summary["tiers"].keys())
    for tier in tiers:
        for contrast, arm_b in (("recursive vs flat", "flat_single_pass"), ("recursive vs fixed outline", "fixed_outline_no_reentry")):
            primary_topics = []
            all_ok_topics = []
            for topic in topics:
                a = row_by_key.get((topic, tier, "recursive_slrgp"))
                b = row_by_key.get((topic, tier, arm_b))
                if not a or not b:
                    continue
                if a["ok"] and b["ok"]:
                    all_ok_topics.append(topic)
                if a["primary_eligible"] and b["primary_eligible"]:
                    primary_topics.append(topic)
            pairability.append({
                "tier": tier,
                "contrast": contrast,
                "arm_a": "recursive_slrgp",
                "arm_b": arm_b,
                "primary_n": len(primary_topics),
                "primary_topics": primary_topics,
                "all_ok_n": len(all_ok_topics),
            })

    by_tier = defaultdict(list)
    for row in rows:
        by_tier[row["tier"]].append(row)
    output = {
        "root": str(args.root),
        "n_rows": len(rows),
        "n_ok": sum(1 for row in rows if row["ok"]),
        "n_primary_eligible": sum(1 for row in rows if row["primary_eligible"]),
        "issues": issues,
        "rows": rows,
        "summary": {
            "topics": topics,
            "tiers": summary["tiers"],
            "arms": summary["arms"],
            "pairability": pairability,
            "by_tier": {
                tier: {
                    "n": len(tier_rows),
                    "n_primary_eligible": sum(1 for row in tier_rows if row["primary_eligible"]),
                    "mean_words": sum(row["words"] for row in tier_rows) / len(tier_rows) if tier_rows else 0,
                }
                for tier, tier_rows in by_tier.items()
            },
        },
    }
    out = args.out or args.root / "s5a_design_audit.json"
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}; eligible={output['n_primary_eligible']}/{output['n_rows']}; issues={len(issues)}")


if __name__ == "__main__":
    main()
