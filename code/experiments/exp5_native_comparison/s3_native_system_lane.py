#!/usr/bin/env python3
"""Run one system's assigned topics serially inside a parallel system lane."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from s3_native_common import sha256_file
from s3_native_runner import (
    INFRA_ROOT,
    SERVER_ROOT,
    build_wrappers,
    load_topics,
    make_spec,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--topics", type=Path, default=INFRA_ROOT / "topics_s3_native12.json")
    parser.add_argument("--results-root", type=Path, default=SERVER_ROOT / "work/results/s3_native")
    parser.add_argument("--topic-ids", default="")
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument("--budget-usd", type=float, default=16.0)
    args = parser.parse_args()

    manifest = json.loads(args.topics.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_after_all_system_retrieval_gate":
        raise SystemExit("Confirmatory topic manifest is not frozen")
    topics = load_topics(args.topics)
    if args.topic_ids:
        requested = {value.strip() for value in args.topic_ids.split(",") if value.strip()}
        topics = [topic for topic in topics if topic.topic_id in requested]
    wrappers = build_wrappers()
    if args.system not in wrappers:
        raise SystemExit(f"Unknown system: {args.system}")
    wrapper = wrappers[args.system]
    run_root = args.results_root / args.run_id
    frozen_protocol_path = run_root / "frozen_protocol.json"
    if frozen_protocol_path.is_file():
        protocol_hash = json.loads(
            frozen_protocol_path.read_text(encoding="utf-8")
        )["protocol_sha256"]
    else:
        protocol_path = SERVER_ROOT / "S3_NATIVE_END_TO_END_PROTOCOL.md"
        protocol_hash = (
            sha256_file(protocol_path) if protocol_path.is_file() else "remote-sync"
        )

    for topic in topics:
        print(f"[S3-lane:{args.system}] START {topic.topic_id}", flush=True)
        spec = make_spec(
            run_id=args.run_id,
            system_id=args.system,
            topic=topic,
            protocol_hash=protocol_hash,
            timeout_seconds=args.timeout_seconds,
            budget_usd=args.budget_usd,
        )
        run_dir = wrapper.run(run_root, spec, resume=True)
        status = json.loads(run_dir.status_path.read_text(encoding="utf-8"))
        print(
            f"[S3-lane:{args.system}] END {topic.topic_id}: {status.get('status')}",
            flush=True,
        )
        if status.get("status") != "ok":
            raise SystemExit(
                f"Lane stopped after {args.system}/{topic.topic_id}: "
                f"{status.get('status')}"
            )


if __name__ == "__main__":
    main()
