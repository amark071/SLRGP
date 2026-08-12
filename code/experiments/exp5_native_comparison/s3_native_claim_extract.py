#!/usr/bin/env python3
"""System-blind, checkpointed extraction of cited substantive claims."""
from __future__ import annotations

import argparse
import concurrent.futures
import difflib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


API_URL = "https://api.ofox.ai/v1/chat/completions"
SYSTEMS = ("slrgp", "autosurvey", "surveyforge", "surveygen")
MODEL = "openai/gpt-5.4-mini"
POSITIONS = {"front", "middle", "rear"}

PROMPT = """Extract exactly 12 substantive, externally checkable claims that have adjacent explicit citations from
the blinded literature review below. Use four claims from the front third, four from the middle third, and four from
the rear third of the review body. Within each third, include both single-source and multi-source claims when the text
permits. Do not extract headings, generic transitions, author opinions, or uncited claims.

For every item:
- quote must be an exact contiguous quotation from the review, at most 100 words;
- citation_markers must copy every adjacent bracketed citation marker exactly;
- position must be front, middle, or rear;
- citation_cardinality must be single or multiple.

Mandatory final check: return exactly 12 items, and every item must have at
least one non-empty citation_markers entry. Replace any uncited candidate
before responding.

Return ONLY JSON:
{{"claims":[{{"claim_id":"C01","quote":"...","citation_markers":["[...]"],
"position":"front","citation_cardinality":"single"}}]}}

REVIEW
======
{text}
"""


def normalize(value: str) -> str:
    # Preserve contiguous wording while ignoring Markdown emphasis and punctuation
    # that models commonly omit when copying otherwise exact review text.
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def canonical_quote(quote: str, text: str) -> str | None:
    quote_tokens = normalize(quote).split()
    source_matches = list(re.finditer(r"[a-z0-9]+", text.lower()))
    source_tokens = [match.group(0) for match in source_matches]
    if len(quote_tokens) < 6 or len(source_tokens) < len(quote_tokens):
        return None

    anchor_size = min(10, len(quote_tokens))
    anchor_starts = sorted(
        {
            0,
            max(0, len(quote_tokens) // 2 - anchor_size // 2),
            len(quote_tokens) - anchor_size,
        }
    )
    best: tuple[float, int, int] | None = None
    for quote_anchor_start in anchor_starts:
        anchor = quote_tokens[
            quote_anchor_start : quote_anchor_start + anchor_size
        ]
        first = anchor[0]
        for source_anchor_start, token in enumerate(source_tokens):
            if token != first:
                continue
            if (
                source_tokens[
                    source_anchor_start : source_anchor_start + anchor_size
                ]
                != anchor
            ):
                continue
            source_start = source_anchor_start - quote_anchor_start
            if source_start < 0:
                continue
            for length_delta in range(-8, 9):
                source_end = source_start + len(quote_tokens) + length_delta
                if source_end <= source_start or source_end > len(source_tokens):
                    continue
                candidate = source_tokens[source_start:source_end]
                ratio = difflib.SequenceMatcher(
                    None, quote_tokens, candidate, autojunk=False
                ).ratio()
                if best is None or ratio > best[0]:
                    best = (ratio, source_start, source_end)
    if best is None or best[0] < 0.82:
        return None
    _, source_start, source_end = best
    return text[
        source_matches[source_start].start() : source_matches[source_end - 1].end()
    ]


def parse(raw: str, text: str) -> list[dict]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        claims = value.get("claims") if isinstance(value, dict) else None
        if not isinstance(claims, list) or len(claims) != 12:
            continue
        if any(
            item.get("position") not in POSITIONS
            or not item.get("citation_markers")
            for item in claims
        ):
            continue
        canonical = [
            canonical_quote(str(item.get("quote") or ""), text) for item in claims
        ]
        if any(quote is None for quote in canonical):
            continue
        for item, quote in zip(claims, canonical):
            item["quote"] = quote
        return claims
    raise ValueError(f"No valid 12-claim extraction: {raw[:500]!r}")


def call(api_key: str, text: str) -> tuple[list[dict], dict]:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT.format(text=text)}],
        "temperature": 0.1,
        "max_tokens": 5000,
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
            return parse(body["choices"][0]["message"]["content"], text), (
                body.get("usage") or {}
            )
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
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args()
    api_key = os.environ.get("OFOX_API_KEY", "")
    if not api_key:
        raise SystemExit("OFOX_API_KEY missing")
    topics = json.loads(args.topics.read_text(encoding="utf-8"))["topics"]
    out_dir = args.root / "claim_support" / "extracted_claims"
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for topic in topics:
        for system in SYSTEMS:
            output = out_dir / f"{topic['topic_id']}__{system}.json"
            if output.is_file():
                continue
            text = (
                args.root / "systems" / system / topic["topic_id"] / "survey.md"
            ).read_text(encoding="utf-8")
            tasks.append((topic["topic_id"], system, text, output))

    def run(task: tuple[str, str, str, Path]) -> tuple[Path, dict]:
        topic_id, system, text, output = task
        claims, usage = call(api_key, text)
        return output, {
            "topic_id": topic_id,
            "system": system,
            "extractor_model": MODEL,
            "claims": claims,
            "usage": usage,
        }

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = {executor.submit(run, task): task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                output, payload = future.result()
                output.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(task[0], task[1], "ok", flush=True)
            except Exception as exc:
                print(task[0], task[1], f"error={exc!r}", flush=True)


if __name__ == "__main__":
    main()
