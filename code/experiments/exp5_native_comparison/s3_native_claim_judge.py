#!/usr/bin/env python3
"""Independent batched claim-support judging with model adjudication."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


API_URL = "https://api.ofox.ai/v1/chat/completions"
PRIMARY_JUDGES = {
    "gpt55": "openai/gpt-5.5",
    "gemini31pro": "google/gemini-3.1-pro-preview",
}
ADJUDICATOR = ("sonnet5", "anthropic/claude-sonnet-5")
LABELS = {
    "supported",
    "partially_supported",
    "unsupported",
    "source_inaccessible_or_insufficient",
}

PROMPT = """Assess whether each quoted claim is supported by its own cited source packet. Do not use outside
knowledge and do not reward plausibility. Label:
- supported: the accessible cited sources directly support the full material claim;
- partially_supported: they support a material part but not the full scope, strength, or qualification;
- unsupported: accessible sources fail to support or contradict the material claim;
- source_inaccessible_or_insufficient: the packet lacks enough source content for a support judgment.

Return ONLY JSON with one result per claim:
{{"results":[{{"claim_id":"C01","label":"supported|partially_supported|unsupported|source_inaccessible_or_insufficient",
"rationale":"at most 35 words grounded in the packet"}}]}}

CLAIM PACKET
============
{packet}
"""


def parse(raw: str, expected_ids: set[str]) -> list[dict]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        rows = value.get("results") if isinstance(value, dict) else None
        if (
            isinstance(rows, list)
            and {str(row.get("claim_id")) for row in rows} == expected_ids
            and all(row.get("label") in LABELS for row in rows)
        ):
            return rows
    raise ValueError(f"No valid claim-support JSON: {raw[:500]!r}")


def call(api_key: str, model: str, claims: list[dict]) -> tuple[list[dict], dict]:
    expected = {str(row["claim_id"]) for row in claims}
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": PROMPT.format(
                    packet=json.dumps(claims, ensure_ascii=False)
                ),
            }
        ],
        "temperature": 0,
        # Leave sufficient room for reasoning-capable providers to return all
        # 12 required labels after their hidden deliberation.
        "max_tokens": 12000,
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
            return parse(body["choices"][0]["message"]["content"], expected), (
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


def read_successes(path: Path) -> dict[tuple[str, str, str], dict]:
    values = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not row.get("error"):
            values[(row["topic_id"], row["system"], row["judge"])] = row
    return values


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
    claim_root = args.root / "claim_support"
    checkpoint = claim_root / "claim_judgments.jsonl"
    successes = read_successes(checkpoint)
    tasks = []
    packets = {}
    for topic in topics:
        for system in ("slrgp", "autosurvey", "surveyforge", "surveygen"):
            packet = json.loads(
                (
                    claim_root
                    / "source_packets"
                    / f"{topic['topic_id']}__{system}.json"
                ).read_text(encoding="utf-8")
            )
            packets[(topic["topic_id"], system)] = packet["claims"]
            for judge, model in PRIMARY_JUDGES.items():
                if (topic["topic_id"], system, judge) not in successes:
                    tasks.append((topic["topic_id"], system, judge, model, packet["claims"]))

    def run(task: tuple[str, str, str, str, list[dict]]) -> dict:
        topic_id, system, judge, model, claims = task
        try:
            results, usage = call(api_key, model, claims)
            return {
                "topic_id": topic_id,
                "system": system,
                "judge": judge,
                "model": model,
                "results": results,
                "usage": usage,
                "error": "",
            }
        except Exception as exc:
            return {
                "topic_id": topic_id,
                "system": system,
                "judge": judge,
                "model": model,
                "error": repr(exc),
            }

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        for row in executor.map(run, tasks):
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(row["topic_id"], row["system"], row["judge"], not row["error"], flush=True)

    successes = read_successes(checkpoint)
    adjudication_tasks = []
    for key, claims in packets.items():
        first = successes.get((*key, "gpt55"))
        second = successes.get((*key, "gemini31pro"))
        if not first or not second:
            continue
        labels_first = {row["claim_id"]: row["label"] for row in first["results"]}
        labels_second = {row["claim_id"]: row["label"] for row in second["results"]}
        disagreements = [
            claim
            for claim in claims
            if labels_first[claim["claim_id"]] != labels_second[claim["claim_id"]]
        ]
        if disagreements and (*key, ADJUDICATOR[0]) not in successes:
            adjudication_tasks.append((*key, ADJUDICATOR[0], ADJUDICATOR[1], disagreements))
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        for row in executor.map(run, adjudication_tasks):
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                row["topic_id"],
                row["system"],
                row["judge"],
                not row["error"],
                flush=True,
            )


if __name__ == "__main__":
    main()
