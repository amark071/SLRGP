#!/usr/bin/env python3
"""Checkpointed Semantic Scholar availability gate for SurveyGen topics."""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


def atomic_write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def query_first_page(title: str, timeout: int) -> tuple[int, list[dict]]:
    params = urllib.parse.urlencode(
        {
            "query": title,
            "fields": "externalIds,title,abstract,publicationDate",
            "limit": 100,
            "offset": 0,
            "year": "2000-2023",
        }
    )
    request = urllib.request.Request(
        f"{API_URL}?{params}", headers={"User-Agent": "SLRGP-S3-preflight/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
        return int(response.status), list(body.get("data") or [])
    except urllib.error.HTTPError as exc:
        return int(exc.code), []
    except (urllib.error.URLError, TimeoutError):
        return 0, []


def valid_count(rows: list[dict], cutoff: str) -> int:
    return sum(
        bool((row.get("externalIds") or {}).get("DOI"))
        and bool(row.get("abstract"))
        and bool(row.get("publicationDate"))
        and str(row["publicationDate"]) <= cutoff
        for row in rows
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--interval-seconds", type=int, default=65)
    parser.add_argument("--timeout-seconds", type=int, default=45)
    args = parser.parse_args()

    manifest = json.loads(args.topics.read_text(encoding="utf-8"))
    topics = manifest["topics"]
    state = {
        "system": "surveygen",
        "native_retrieval_service": "Semantic Scholar",
        "minimum_pre_cutoff_candidates": args.minimum,
        "publication_cutoff": manifest["publication_cutoff"],
        "topics": {},
    }
    if args.output.is_file():
        state = json.loads(args.output.read_text(encoding="utf-8"))

    for round_index in range(1, args.rounds + 1):
        pending = [
            topic
            for topic in topics
            if int(
                (state["topics"].get(topic["topic_id"]) or {}).get("valid_count") or 0
            )
            < args.minimum
        ]
        if not pending:
            break
        for topic in pending:
            status, rows = query_first_page(topic["title"], args.timeout_seconds)
            count = valid_count(rows, manifest["publication_cutoff"])
            previous = state["topics"].get(topic["topic_id"]) or {}
            state["topics"][topic["topic_id"]] = {
                "title": topic["title"],
                "valid_count": max(count, int(previous.get("valid_count") or 0)),
                "returned": len(rows),
                "last_http_status": status,
                "attempts": int(previous.get("attempts") or 0) + 1,
                "passed": max(count, int(previous.get("valid_count") or 0))
                >= args.minimum,
                "last_round": round_index,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(args.output, state)
            print(
                topic["topic_id"],
                status,
                state["topics"][topic["topic_id"]]["valid_count"],
                flush=True,
            )
            time.sleep(args.interval_seconds)

    failed = [
        topic["topic_id"]
        for topic in topics
        if not (state["topics"].get(topic["topic_id"]) or {}).get("passed")
    ]
    state["complete"] = not failed
    state["failed_topics"] = failed
    atomic_write(args.output, state)
    if failed:
        raise SystemExit(f"Retrieval gate incomplete: {failed}")


if __name__ == "__main__":
    main()
