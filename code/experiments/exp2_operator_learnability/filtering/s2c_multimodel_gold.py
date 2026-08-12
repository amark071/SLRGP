#!/usr/bin/env python3
"""
Blind multi-model eligibility gold-proxy annotation for S2c/F.

Models never receive citation membership, weak labels, learned scores, retrieved
rank or sampling stratum. Three independent flagship models label each item.
Consensus policy:
  - binary admission: eligible|borderline = ADMIT, all other classes = EXCLUDE;
  - >=2 identical binary votes -> consensus;
  - 1:1:1 / unavailable vote -> UNCERTAIN, excluded from primary gold-set metrics
    and retained in an uncertainty sensitivity analysis.

This is an LLM-consensus proxy, explicitly not a human gold standard.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "slrgp"))
from llm_client import LLMClient  # noqa: E402

HERE = Path(__file__).resolve().parent
MODEL_PANEL = {
    "gpt55": "openai/gpt-5.5",
    "gemini31pro": "google/gemini-3.1-pro-preview",
    "sonnet5": "anthropic/claude-sonnet-5",
}
OFOX = os.environ.get("OFOX_API_KEY", "")

PROMPT = """You are independently annotating an eligibility decision for a literature-review system.

You must judge only whether this candidate is eligible to enter the target review's evidence space. You are NOT being asked whether the target review cited it, whether it is highly ranked, or how it was retrieved. Do not infer any hidden label.

ADMIT (eligible): clearly within scope and usable scholarly evidence.
ADMIT (borderline): plausibly usable but peripheral or requiring reviewer discretion.
EXCLUDE (out_of_scope): unrelated except for superficial vocabulary overlap.
EXCLUDE (wrong_document_type): editorial, erratum, call, book review, etc.
EXCLUDE (post_cutoff_or_temporal_issue): temporally inadmissible.
EXCLUDE (insufficient_metadata): impossible to judge from available record.

Target review:
title: {review_title}
abstract: {review_abstract}
year: {review_year}
discipline: {review_discipline}

Candidate:
title: {candidate_title}
abstract: {candidate_abstract}
source: {source}
tier: {tier}
venue: {venue}
year difference (review year − candidate year): {year_diff}
same discipline: {same_discipline}

Return only valid JSON:
{{"label":"eligible|borderline|out_of_scope|wrong_document_type|post_cutoff_or_temporal_issue|insufficient_metadata",
"confidence":0.0,"reason":"one concise sentence"}}"""

VALID = {
    "eligible", "borderline", "out_of_scope", "wrong_document_type",
    "post_cutoff_or_temporal_issue", "insufficient_metadata",
}


def load_items(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def annotate(item: dict, model_name: str, model_id: str) -> dict:
    review = item["review"]
    candidate = item["candidate"]
    meta = item["metadata"]
    client = LLMClient(
        base_url="https://api.ofox.ai/v1",
        model=model_id,
        api_key=OFOX,
        send_thinking_kwarg=False,
        default_timeout=90,
    )
    message = [{
        "role": "user",
        "content": PROMPT.format(
            review_title=review.get("title", ""),
            review_abstract=review.get("abstract", ""),
            review_year=review.get("year", ""),
            review_discipline=review.get("discipline", ""),
            candidate_title=candidate.get("title", ""),
            candidate_abstract=candidate.get("abstract_preview", ""),
            source=meta.get("source", ""),
            tier=meta.get("tier", ""),
            venue=meta.get("venue", ""),
            year_diff=meta.get("year_diff", ""),
            same_discipline=meta.get("same_discipline", ""),
        ),
    }]
    try:
        response = client.chat_json(message, max_tokens=500, temperature=0.0)
        label = str(response.get("label", "")).lower().strip()
        if label not in VALID:
            label = "error"
        return {
            "model": model_name,
            "label": label,
            "binary": "admit" if label in {"eligible", "borderline"} else "exclude",
            "confidence": max(0.0, min(1.0, float(response.get("confidence", 0.0)))),
            "reason": response.get("reason", ""),
        }
    except Exception as e:
        return {"model": model_name, "label": "error", "binary": "error",
                "confidence": 0.0, "reason": f"{type(e).__name__}: {e}"}


def consensus(votes: list[dict]) -> dict:
    binary = [v["binary"] for v in votes if v["binary"] in {"admit", "exclude"}]
    counts = Counter(binary)
    if not counts or len(binary) < 2 or (len(counts) == 2 and counts["admit"] == counts["exclude"]):
        return {"binary": "uncertain", "fine_label": "uncertain"}
    winner, n = counts.most_common(1)[0]
    if n < 2:
        return {"binary": "uncertain", "fine_label": "uncertain"}
    fine = [v["label"] for v in votes if v["binary"] == winner]
    return {"binary": winner, "fine_label": Counter(fine).most_common(1)[0][0]}


def save_checkpoint(path: str, items: list[dict], votes_by_id: dict[str, dict]) -> None:
    """Atomically persist every completed call so interrupted runs resume exactly."""
    payload = {"items": items, "votes_by_id": votes_by_id}
    fd, tmp = tempfile.mkstemp(prefix=".s2c_checkpoint_", suffix=".json", dir=str(Path(path).parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    args = ap.parse_args()
    if not OFOX:
        raise SystemExit("OFOX_API_KEY missing")

    source_items = load_items(args.input)
    checkpoint = f"{args.output}.checkpoint.json"
    votes_by_id = {}
    if os.path.exists(checkpoint):
        prior = json.load(open(checkpoint, encoding="utf-8"))
        items = prior["items"]
        votes_by_id = prior.get("votes_by_id", {})
        if [x["sample_id"] for x in items] != [x["sample_id"] for x in source_items]:
            raise RuntimeError("checkpoint does not match input items")
        print(f"Resuming checkpoint: {sum(len(v) for v in votes_by_id.values())} completed calls")
    else:
        items = source_items

    jobs = []
    for i, item in enumerate(items):
        done_models = set(votes_by_id.get(item["sample_id"], {}))
        for key, model in MODEL_PANEL.items():
            if key not in done_models:
                jobs.append((i, key, model))
    total = len(items) * len(MODEL_PANEL)
    print(f"Annotating {len(jobs)} remaining calls ({total - len(jobs)}/{total} already checkpointed)")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(annotate, items[i], key, model): (i, key) for i, key, model in jobs}
        for done, fut in enumerate(as_completed(futures), 1):
            i, key = futures[fut]
            votes_by_id.setdefault(items[i]["sample_id"], {})[key] = fut.result()
            if done % args.checkpoint_every == 0 or done == len(jobs):
                save_checkpoint(checkpoint, items, votes_by_id)
            if done % 50 == 0 or done == len(jobs):
                n_done = sum(len(v) for v in votes_by_id.values())
                print(f"  {n_done}/{total}", flush=True)

    out_items = []
    for i, item in enumerate(items):
        votes = sorted(votes_by_id.get(item["sample_id"], {}).values(), key=lambda x: x["model"])
        # Preserve only annotation-relevant content in output: weak labels/sampling
        # strata are retained for later reweighting, never sent in model prompts.
        out_items.append({
            **item,
            "panel_votes": votes,
            "consensus": consensus(votes),
        })

    pairs = []
    for r in out_items:
        votes = [v for v in r["panel_votes"] if v["binary"] in {"admit", "exclude"}]
        for a in range(len(votes)):
            for b in range(a + 1, len(votes)):
                pairs.append(int(votes[a]["binary"] == votes[b]["binary"]))
    binary_agreement = sum(pairs) / len(pairs) if pairs else None
    summary = {
        "models": MODEL_PANEL,
        "n_items": len(out_items),
        "consensus_counts": dict(Counter(r["consensus"]["binary"] for r in out_items)),
        "pairwise_binary_agreement": binary_agreement,
        "note": "Blind multi-model consensus proxy; not human gold standard.",
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "items": out_items}, f, ensure_ascii=False, indent=2)
    if os.path.exists(checkpoint):
        os.unlink(checkpoint)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
