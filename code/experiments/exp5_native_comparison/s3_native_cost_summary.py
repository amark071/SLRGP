#!/usr/bin/env python3
"""Aggregate comparable generation usage and cost-efficiency for S3."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


SYSTEMS = ("slrgp", "autosurvey", "surveyforge", "surveygen")
INPUT_USD_PER_MILLION = 3.0
OUTPUT_USD_PER_MILLION = 15.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--topics", type=Path, required=True)
    args = parser.parse_args()
    topics = json.loads(args.topics.read_text(encoding="utf-8"))["topics"]
    audit_rows = json.loads(
        (args.root / "audit" / "deterministic_audit.json").read_text(
            encoding="utf-8"
        )
    )["rows"]
    words = {
        (row["topic_id"], row["system"]): int(row["word_count"])
        for row in audit_rows
    }
    totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "n_outputs": 0,
            "n_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "generated_words": 0,
        }
    )
    per_output = []
    for topic in topics:
        topic_id = topic["topic_id"]
        for system in SYSTEMS:
            usage = json.loads(
                (
                    args.root / "systems" / system / topic_id / "usage.json"
                ).read_text(encoding="utf-8")
            )
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            estimated_cost = (
                prompt_tokens / 1_000_000 * INPUT_USD_PER_MILLION
                + completion_tokens / 1_000_000 * OUTPUT_USD_PER_MILLION
            )
            word_count = words[(topic_id, system)]
            row = {
                "topic_id": topic_id,
                "system": system,
                "n_calls": int(usage.get("n_calls") or 0),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "generated_words": word_count,
                "official_price_equivalent_usd": estimated_cost,
            }
            per_output.append(row)
            total = totals[system]
            total["n_outputs"] += 1
            total["n_calls"] += row["n_calls"]
            total["prompt_tokens"] += prompt_tokens
            total["completion_tokens"] += completion_tokens
            total["generated_words"] += word_count
    summary = {}
    for system, total in totals.items():
        cost = (
            total["prompt_tokens"] / 1_000_000 * INPUT_USD_PER_MILLION
            + total["completion_tokens"] / 1_000_000 * OUTPUT_USD_PER_MILLION
        )
        summary[system] = {
            **total,
            "official_price_equivalent_usd": cost,
            "usd_per_output": cost / total["n_outputs"],
            "usd_per_1000_generated_words": (
                cost / total["generated_words"] * 1000
            ),
            "tokens_per_generated_word": (
                (total["prompt_tokens"] + total["completion_tokens"])
                / total["generated_words"]
            ),
        }
    output = {
        "schema_version": "1.0",
        "pricing_assumption": {
            "model": "Anthropic Claude Sonnet 4.6 common backbone",
            "input_usd_per_million_tokens": INPUT_USD_PER_MILLION,
            "output_usd_per_million_tokens": OUTPUT_USD_PER_MILLION,
            "note": (
                "Official-price equivalent from recorded provider tokens; "
                "not an API-relay invoice total."
            ),
        },
        "systems": summary,
        "per_output": per_output,
    }
    out = args.root / "analysis" / "cost_efficiency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
