#!/usr/bin/env python3
"""Checkpointed, order-reversed blind pairwise judging for native S3."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


API_URL = "https://api.ofox.ai/v1/chat/completions"
JUDGES = {
    "gpt55": "openai/gpt-5.5",
    "gemini31pro": "google/gemini-3.1-pro-preview",
    "sonnet5": "anthropic/claude-sonnet-5",
}
BASELINES = ("autosurvey", "surveyforge", "surveygen")
DIMENSIONS = (
    "organizational_quality",
    "critical_synthesis",
    "global_coherence",
    "citation_plausibility",
)
CHOICES = {"A", "B", "tie"}

RUBRIC = """You are conducting a blinded comparison of two academic literature reviews about the same registered topic.
Do not infer how either review was produced. Judge only the supplied texts. Length alone is neither a benefit nor a
penalty; reward useful coverage only when it remains organized, synthetic, coherent, and citation-grounded.

Registered scope:
{scope}

For each dimension choose A, B, or tie:
1. organizational_quality: coherent hierarchy, useful sectioning, and logical progression.
2. critical_synthesis: comparison, contrast, qualification, and integration across cited works.
3. global_coherence: consistent concepts and narrative without contradictions, prompt leakage, or non-sequiturs.
4. citation_plausibility: citations plausibly support adjacent claims and are not decorative or mismatched.

Then choose the better review overall (A, B, or tie), weighing all four dimensions equally. Return ONLY:
{{"overall_preference":"A|B|tie",
  "organizational_quality":"A|B|tie",
  "critical_synthesis":"A|B|tie",
  "global_coherence":"A|B|tie",
  "citation_plausibility":"A|B|tie",
  "rationale":"at most 80 words citing concrete textual evidence"}}

REVIEW A
========
{review_a}

REVIEW B
========
{review_b}
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
        required = ("overall_preference", *DIMENSIONS)
        if isinstance(value, dict) and all(value.get(key) in CHOICES for key in required):
            return value
    raise ValueError(f"No valid preference JSON: {raw[:500]!r}")


def judge(api_key: str, model: str, prompt: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        # Gemini 3.1 Pro uses part of this budget for internal reasoning; a
        # 1,200-token cap can truncate the short JSON before its final fields.
        "max_tokens": 8000,
    }
    error: Exception | None = None
    for attempt in range(4):
        request = urllib.request.Request(
            API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                body = json.loads(response.read())
            parsed = parse_json(body["choices"][0]["message"]["content"])
            return {**parsed, "usage": body.get("usage") or {}}
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            KeyError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            error = exc
            time.sleep(min(90, 5 * (2**attempt)))
    raise RuntimeError(repr(error))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--topics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args()
    api_key = os.environ.get("OFOX_API_KEY", "")
    if not api_key:
        raise SystemExit("OFOX_API_KEY missing")

    topics = json.loads(args.topics.read_text(encoding="utf-8"))["topics"]
    out_dir = args.out_dir or args.root / "blind_preference"
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    mapping = {}
    for topic in topics:
        texts = {
            system: (
                args.root / "systems" / system / topic["topic_id"] / "survey.md"
            ).read_text(encoding="utf-8")
            for system in ("slrgp", *BASELINES)
        }
        for baseline in BASELINES:
            for repeat in (1, 2):
                order = ("slrgp", baseline) if repeat == 1 else (baseline, "slrgp")
                digest = hashlib.sha256(
                    f"{topic['topic_id']}|{baseline}|{repeat}|20260716".encode()
                ).hexdigest()[:16]
                blind_id = f"pair_{digest}"
                mapping[blind_id] = {
                    "topic_id": topic["topic_id"],
                    "baseline": baseline,
                    "repeat": repeat,
                    "system_a": order[0],
                    "system_b": order[1],
                    "sha256_a": hashlib.sha256(texts[order[0]].encode()).hexdigest(),
                    "sha256_b": hashlib.sha256(texts[order[1]].encode()).hexdigest(),
                }
                prompt = RUBRIC.format(
                    scope=topic["scope"],
                    review_a=texts[order[0]],
                    review_b=texts[order[1]],
                )
                for judge_tag, model in JUDGES.items():
                    tasks.append((blind_id, judge_tag, model, prompt))

    mapping_path = out_dir / "blind_mapping.json"
    if mapping_path.is_file():
        if json.loads(mapping_path.read_text(encoding="utf-8")) != mapping:
            raise SystemExit("Existing blind mapping differs from frozen task mapping")
    else:
        mapping_path.write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    checkpoint = out_dir / "preference_scores.jsonl"
    done = set()
    if checkpoint.is_file():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not row.get("error"):
                done.add((row["blind_id"], row["judge"]))
    remaining = [
        task for task in tasks if (task[0], task[1]) not in done
    ]
    lock = threading.Lock()

    def run(task: tuple[str, str, str, str]) -> dict:
        blind_id, judge_tag, model, prompt = task
        started = time.monotonic()
        try:
            result = judge(api_key, model, prompt)
            result["error"] = ""
        except Exception as exc:
            result = {"error": repr(exc)}
        return {
            "blind_id": blind_id,
            "judge": judge_tag,
            "model": model,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            **result,
        }

    print(f"blind preference calls remaining={len(remaining)}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        for row in executor.map(run, remaining):
            with lock, checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                row["blind_id"],
                row["judge"],
                f"error={bool(row['error'])}",
                flush=True,
            )


if __name__ == "__main__":
    main()
