#!/usr/bin/env python3
"""Blind Instrument-J scoring for S5a length/depth outputs."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path


JUDGES = {
    "gpt55": "openai/gpt-5.5",
    "gemini31pro": "google/gemini-3.1-pro-preview",
    "sonnet5": "anthropic/claude-sonnet-5",
}
DIMENSIONS = (
    "organizational_quality",
    "critical_synthesis",
    "global_coherence",
    "citation_plausibility",
)
API_URL = "https://api.ofox.ai/v1/chat/completions"
MAX_TOKENS = 4000

RUBRIC = """You are an expert academic reviewer. Evaluate the literature review below strictly on its own merits.
You do not know how it was produced and must not infer its provenance.

Score each dimension on an integer 1-5 scale (1=poor, 3=acceptable, 5=excellent). For each dimension provide one
concise concrete rationale of at most 35 words, based only on the supplied text.
1. organizational_quality: coherent hierarchy and logical progression of sections.
2. critical_synthesis: meaningful comparison, contrast, and synthesis across works.
3. global_coherence: consistent terminology and narrative without contradictions or non-sequiturs.
4. citation_plausibility: citations plausibly support their adjacent claims and are not decorative or mismatched.

Return ONLY one JSON object with exactly these keys:
{"organizational_quality":{"score":1,"rationale":"..."}, "critical_synthesis":{"score":1,"rationale":"..."},
 "global_coherence":{"score":1,"rationale":"..."}, "citation_plausibility":{"score":1,"rationale":"..."}}

REVIEW:
---
{text}
---
"""


def parse_json(raw: str) -> dict:
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and all(key in value for key in DIMENSIONS):
            return value
    raise ValueError(f"No valid judge JSON: {raw[:500]!r}")


def score(api_key: str, model: str, text: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": RUBRIC.replace("{text}", text)}],
        "temperature": 0.2,
        "max_tokens": MAX_TOKENS,
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    error: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=360) as response:
                body = json.loads(response.read())
            raw = body["choices"][0]["message"]["content"]
            parsed = parse_json(raw)
            return {
                f"{dimension}_score": int(parsed[dimension]["score"])
                for dimension in DIMENSIONS
            } | {
                f"{dimension}_rationale": str(parsed[dimension].get("rationale", ""))
                for dimension in DIMENSIONS
            } | {"usage": body.get("usage", {})}
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, ValueError, json.JSONDecodeError) as exc:
            error = exc
            time.sleep(min(45, 2 ** (attempt + 2)))
    raise RuntimeError(repr(error))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    api_key = os.environ.get("OFOX_API_KEY", "")
    if not api_key:
        raise SystemExit("OFOX_API_KEY missing")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    out_dir = args.out_dir or args.root / "blind_judging"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for row in audit["rows"]:
        topic_dir = args.root / row["tier"] / row["topic"].replace(" ", "_").replace("/", "_") / row["arm"]
        survey = topic_dir / "survey.md"
        text = survey.read_text(encoding="utf-8")
        records.append({
            "topic": row["topic"],
            "tier": row["tier"],
            "arm": row["arm"],
            "primary_eligible": row["primary_eligible"],
            "words": row["words"],
            "text": text,
        })
    random.Random(20260713).shuffle(records)
    blind_map = {
        f"review_{index:03d}": {
            "topic": record["topic"],
            "tier": record["tier"],
            "arm": record["arm"],
            "primary_eligible": record["primary_eligible"],
            "words": record["words"],
            "sha256": hashlib.sha256(record["text"].encode()).hexdigest(),
        }
        for index, record in enumerate(records, 1)
    }
    (out_dir / "blind_mapping.json").write_text(json.dumps(blind_map, ensure_ascii=False, indent=2), encoding="utf-8")
    checkpoint = out_dir / "blind_scores.jsonl"
    done = set()
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not row.get("error"):
                done.add((row["blind_id"], row["judge"]))
    tasks = [
        (f"review_{index:03d}", record, judge, model)
        for index, record in enumerate(records, 1)
        for judge, model in JUDGES.items()
        if (f"review_{index:03d}", judge) not in done
    ]
    print(f"S5a blind scoring: {len(tasks)} calls remaining", flush=True)

    def run(task: tuple[str, dict, str, str]) -> dict:
        blind_id, record, judge_tag, model = task
        started = time.time()
        try:
            result = score(api_key, model, record["text"])
            result["error"] = ""
        except Exception as exc:
            result = {"error": repr(exc)}
        result.update({
            "blind_id": blind_id,
            "judge": judge_tag,
            "model": model,
            "elapsed_sec": round(time.time() - started, 1),
            "n_words": len(record["text"].split()),
        })
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        for row in executor.map(run, tasks):
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{row['blind_id']} {row['judge']} error={bool(row['error'])}", flush=True)


if __name__ == "__main__":
    main()
