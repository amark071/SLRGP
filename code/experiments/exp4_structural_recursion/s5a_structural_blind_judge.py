#!/usr/bin/env python3
"""One-batch blind Instrument-J evaluation for the six-topic S5a structural test."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import time
from pathlib import Path

from s5a_blind_judge import JUDGES, score


TOPICS = (
    "3D Gaussian Splatting",
    "Graph Neural Networks",
    "Multimodal Large Language Models",
    "Retrieval-Augmented Generation for Large Language Models",
    "Federated Learning",
    "Efficient Inference for Large Language Models",
)
ARMS = (
    "structural_lotcf_recursive",
    "naive_recursive_chunking",
    "fixed_outline_no_reentry",
    "flat_single_pass",
)
OLD_ARM_MAP = {"naive_recursive_chunking": "recursive_slrgp"}


def safe_name(text: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def old_text(old_root: Path, topic: str, arm: str) -> str:
    old_arm = OLD_ARM_MAP.get(arm, arm)
    path = old_root / "long" / safe_name(topic) / old_arm / "survey.md"
    return path.read_text(encoding="utf-8")


def structural_text(qual_root: Path, quant_root: Path, topic: str) -> str:
    root = qual_root if topic == "Retrieval-Augmented Generation for Large Language Models" else quant_root
    path = root / safe_name(topic) / "structural_lotcf_recursive" / "survey.md"
    return path.read_text(encoding="utf-8")


def latest_successes(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[(row["blind_id"], row["judge"])] = row
    return {key for key, row in latest.items() if not row.get("error")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--structural-qual-root", type=Path, required=True)
    parser.add_argument("--structural-quant-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    api_key = os.environ.get("OFOX_API_KEY", "")
    if not api_key:
        raise SystemExit("OFOX_API_KEY missing")

    records = []
    for topic in TOPICS:
        for arm in ARMS:
            text = (
                structural_text(args.structural_qual_root, args.structural_quant_root, topic)
                if arm == "structural_lotcf_recursive"
                else old_text(args.old_root, topic, arm)
            )
            records.append({"topic": topic, "tier": "long", "arm": arm, "text": text})
    random.Random(20260714).shuffle(records)
    mapping = {
        f"review_{index:03d}": {
            "topic": record["topic"],
            "tier": record["tier"],
            "arm": record["arm"],
            "sha256": hashlib.sha256(record["text"].encode()).hexdigest(),
            "n_words": len(record["text"].split()),
        }
        for index, record in enumerate(records, 1)
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "blind_mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    checkpoint = args.out_dir / "blind_scores.jsonl"
    done = latest_successes(checkpoint)
    tasks = [
        (f"review_{idx:03d}", record, judge, model)
        for idx, record in enumerate(records, 1)
        for judge, model in JUDGES.items()
        if (f"review_{idx:03d}", judge) not in done
    ]
    print(f"S5a structural blind scoring: {len(tasks)} calls remaining", flush=True)

    def run(task: tuple[str, dict, str, str]) -> dict:
        blind_id, record, judge_tag, model = task
        started = time.time()
        try:
            result = score(api_key, model, record["text"])
            result["error"] = ""
        except Exception as exc:  # noqa: BLE001 - retained for checkpointed retries
            result = {"error": repr(exc)}
        result.update({
            "blind_id": blind_id,
            "judge": judge_tag,
            "model": model,
            "elapsed_sec": round(time.time() - started, 1),
            "n_words": len(record["text"].split()),
        })
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for row in executor.map(run, tasks):
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{row['blind_id']} {row['judge']} error={bool(row['error'])}", flush=True)


if __name__ == "__main__":
    main()
