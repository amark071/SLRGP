#!/usr/bin/env python3
"""
S2c/F gold eligibility audit with MODEL-A.

This validates the semantic/qualification meaning of the weak-supervised F result.
It does NOT ask whether the paper was cited. It asks whether the candidate is
eligible to be considered for the target review evidence space.

Usage:
  export OFOX_API_KEY=...
  python3 s2c_audit_verify.py

Input:
  data/exp2_operator_learnability/filtering/s2c_gold_audit_sample.jsonl (package data)
Output:
  work/s2c_gold_audit_verdicts.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "slrgp"))
from llm_client import LLMClient  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ITEMS = os.path.join(PKG_ROOT, "data", "exp2_operator_learnability", "filtering", "s2c_gold_audit_sample.jsonl")
OUT = os.path.join(HERE, "work", "s2c_gold_audit_verdicts.json")

MODEL = "openai/gpt-5.4-mini"
OFOX = os.environ.get("OFOX_API_KEY", "")

PROMPT = """You are auditing the eligibility filter F for an automated literature-review system.

Task:
Given a target review (title + abstract) and one retrieved candidate paper, decide whether the candidate is eligible to be considered in the review's evidence space. Do NOT judge whether the human review actually cited it. Uncited papers can still be eligible.

Eligibility means:
- The candidate is within the target review's conceptual scope or a clearly relevant neighbouring method/application/theory.
- It is a scholarly research/survey paper or a legitimate technical source for the review.
- It is temporally admissible (not after the review year, based on the metadata).
- It has enough metadata/text to judge.

Labels:
- eligible: clearly admissible evidence for this review.
- borderline: plausibly admissible but peripheral, too broad/narrow, or requires expert judgement.
- out_of_scope: topically unrelated or only superficially similar.
- wrong_document_type: editorial, book review, erratum, call for papers, dataset notice, etc.
- post_cutoff_or_temporal_issue: appears after the review year or otherwise temporally inadmissible.
- insufficient_metadata: not enough title/abstract/metadata to judge.

Target review:
title: {review_title}
abstract: {review_abstract}
year: {review_year}
discipline: {review_discipline}

Candidate paper:
title: {cand_title}
abstract: {cand_abstract}
source: {source}
tier: {tier}
venue: {venue}
doi: {doi}
year_diff(review_year - candidate_year): {year_diff}
same_discipline: {same_discipline}

Respond ONLY as JSON:
{{"label": "eligible|borderline|out_of_scope|wrong_document_type|post_cutoff_or_temporal_issue|insufficient_metadata",
  "confidence": 0.0-1.0,
  "reason": "one concise sentence"}}"""

VALID = {
    "eligible",
    "borderline",
    "out_of_scope",
    "wrong_document_type",
    "post_cutoff_or_temporal_issue",
    "insufficient_metadata",
}


def load_items() -> list[dict]:
    out = []
    with open(ITEMS, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def verify_one(item: dict) -> dict:
    llm = LLMClient(
        base_url="https://api.ofox.ai/v1",
        model=MODEL,
        api_key=OFOX,
        send_thinking_kwarg=False,
        default_timeout=90,
    )
    rev = item.get("review", {})
    cand = item.get("candidate", {})
    meta = item.get("metadata", {})
    msg = [{
        "role": "user",
        "content": PROMPT.format(
            review_title=rev.get("title", ""),
            review_abstract=rev.get("abstract", ""),
            review_year=rev.get("year", ""),
            review_discipline=rev.get("discipline", ""),
            cand_title=cand.get("title", ""),
            cand_abstract=cand.get("abstract_preview", ""),
            source=meta.get("source", ""),
            tier=meta.get("tier", ""),
            venue=meta.get("venue", ""),
            doi=meta.get("doi", ""),
            year_diff=meta.get("year_diff", ""),
            same_discipline=meta.get("same_discipline", ""),
        ),
    }]
    try:
        r = llm.chat_json(msg, max_tokens=800, temperature=0.0)
        label = str(r.get("label", "insufficient_metadata")).strip().lower()
        if label not in VALID:
            label = "insufficient_metadata"
        conf = float(r.get("confidence", 0.0))
        conf = max(0.0, min(1.0, conf))
        return {**item, "gold_label": label, "gold_confidence": conf, "gold_reason": r.get("reason", "")}
    except Exception as e:
        return {**item, "gold_label": "error", "gold_confidence": 0.0, "gold_reason": f"{type(e).__name__}: {e}"}


def summarize(results: list[dict]) -> dict:
    counts = Counter(r["gold_label"] for r in results)
    by_stratum = defaultdict(Counter)
    weak_vs_gold = defaultdict(Counter)
    for r in results:
        by_stratum[r["stratum"]][r["gold_label"]] += 1
        weak = "cited" if r.get("weak_label_cited") else "uncited"
        weak_vs_gold[weak][r["gold_label"]] += 1

    def positive(label: str, include_borderline: bool) -> bool:
        return label == "eligible" or (include_borderline and label == "borderline")

    # These are sample-level diagnostics over a deliberately stratified audit set,
    # not population estimates.
    diagnostics = {}
    for include_borderline in [False, True]:
        key = "eligible_only" if not include_borderline else "eligible_or_borderline"
        cited = [r for r in results if r.get("weak_label_cited")]
        uncited = [r for r in results if not r.get("weak_label_cited")]
        diagnostics[key] = {
            "cited_gold_positive_rate": (
                sum(positive(r["gold_label"], include_borderline) for r in cited) / len(cited)
                if cited else 0.0
            ),
            "uncited_gold_positive_rate": (
                sum(positive(r["gold_label"], include_borderline) for r in uncited) / len(uncited)
                if uncited else 0.0
            ),
            "model_high_uncited_positive_rate": (
                sum(positive(r["gold_label"], include_borderline) for r in results
                    if r["stratum"] == "model_high_uncited")
                / max(1, sum(1 for r in results if r["stratum"] == "model_high_uncited"))
            ),
            "semantic_hard_negative_positive_rate": (
                sum(positive(r["gold_label"], include_borderline) for r in results
                    if r["stratum"] == "semantic_hard_negative")
                / max(1, sum(1 for r in results if r["stratum"] == "semantic_hard_negative"))
            ),
            "rule_clear_negative_positive_rate": (
                sum(positive(r["gold_label"], include_borderline) for r in results
                    if r["stratum"] == "rule_clear_negative")
                / max(1, sum(1 for r in results if r["stratum"] == "rule_clear_negative"))
            ),
        }

    return {
        "model": MODEL,
        "n": len(results),
        "label_counts": dict(counts),
        "by_stratum": {k: dict(v) for k, v in by_stratum.items()},
        "weak_citation_label_vs_gold": {k: dict(v) for k, v in weak_vs_gold.items()},
        "sample_level_diagnostics": diagnostics,
        "note": (
            "The audit sample is risk/model stratified, not population-proportional. "
            "Use it to validate label semantics and error modes; do not report as a population precision without reweighting."
        ),
    }


def main() -> None:
    if not OFOX:
        print("请先 export OFOX_API_KEY=...", file=sys.stderr)
        sys.exit(1)
    items = load_items()
    print(f"审计 {len(items)} 条 S2c/F eligibility 样本，MODEL-A={MODEL}")
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, res in enumerate(ex.map(verify_one, items)):
            results[i] = res
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(items)}", flush=True)
    summary = summarize(results)
    out = {**summary, "verdicts": results}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "note"}, ensure_ascii=False, indent=2))
    print(f"写入 {OUT}")


if __name__ == "__main__":
    main()
