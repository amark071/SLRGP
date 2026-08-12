#!/usr/bin/env python3
"""Run retrieval-only availability gates through each system's native index."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SERVER_ROOT = SCRIPT_DIR.parents[2]
# Baseline checkouts and corpus assets are not shipped; rebuild per
# code/data_rebuild/README.md and point S3_NATIVE_WORK_ROOT at the result.
WORK_ROOT = Path(os.environ.get("S3_NATIVE_WORK_ROOT", SERVER_ROOT / "work/s3_native"))
BASELINES_ROOT = WORK_ROOT / "baselines"
ASSETS_ROOT = WORK_ROOT / "assets"


def atomic_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def digest_ids(ids: list[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(set(ids))).encode("utf-8")
    ).hexdigest()


def load_native_database(system: str):
    if system == "autosurvey":
        source = BASELINES_ROOT / "autosurvey"
        database_dir = ASSETS_ROOT / "autosurvey/database"
        embedding_model = ASSETS_ROOT / "autosurvey/models/nomic-embed-text-v1"
    elif system == "surveyforge":
        source = BASELINES_ROOT / "surveyforge/code"
        database_dir = ASSETS_ROOT / "surveyforge/database/database"
        embedding_model = ASSETS_ROOT / "surveyforge/models/gte-large-en-v1.5"
    else:
        raise ValueError(system)
    os.chdir(source)
    sys.path.insert(0, str(source))
    from src.database import database

    return database(str(database_dir), str(embedding_model))


def retrieve_local_database(index, topic: dict, requested: int) -> list[str]:
    return [
        str(value)
        for value in index.get_ids_from_query(topic["title"], requested)
        if value
    ]


def load_slrgp_index():
    sys.path.insert(0, str(SERVER_ROOT / "code"))
    sys.path.insert(0, str(SCRIPT_DIR))
    from s3_slrgp_full_deployment import LearnedHybridLIndex

    return LearnedHybridLIndex(device="cuda")


def retrieve_slrgp(index, topic: dict, requested: int) -> list[str]:
    cutoff_year = int(topic["publication_cutoff"][:4])
    papers = index.search(
        topic["title"],
        top_k_each=max(3000, requested * 10),
        top_n=requested * 3,
    )
    return [
        str(paper.doc_id)
        for paper in papers
        if paper.doc_id and paper.year and int(paper.year) <= cutoff_year
    ][:requested]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system", choices=("autosurvey", "surveyforge", "slrgp"), required=True
    )
    parser.add_argument("--topics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum", type=int, default=30)
    parser.add_argument("--requested", type=int, default=120)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.topics.read_text(encoding="utf-8"))
    os.environ["S3_PUBLICATION_CUTOFF"] = manifest["publication_cutoff"]
    os.environ.setdefault("S3_EMBEDDING_DEVICE", "cuda")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    details_path = args.output.with_name(args.output.stem + "_manifest.json")
    details = {
        "schema_version": "1.0",
        "system": args.system,
        "retrieval_mode": "released_native_index_direct_query",
        "publication_cutoff": manifest["publication_cutoff"],
        "minimum_pre_cutoff_candidates": args.minimum,
        "requested": args.requested,
        "topics": {},
    }
    if args.resume and details_path.is_file():
        details = json.loads(details_path.read_text(encoding="utf-8"))

    index = load_slrgp_index() if args.system == "slrgp" else load_native_database(
        args.system
    )
    for topic in manifest["topics"]:
        topic_id = topic["topic_id"]
        previous = details["topics"].get(topic_id) or {}
        if args.resume and previous.get("passed"):
            continue
        ids = (
            retrieve_slrgp(index, topic, args.requested)
            if args.system == "slrgp"
            else retrieve_local_database(index, topic, args.requested)
        )
        details["topics"][topic_id] = {
            "title": topic["title"],
            "candidate_count": len(ids),
            "passed": len(ids) >= args.minimum,
            "candidate_ids_sha256": digest_ids(ids),
            "candidate_ids": ids,
        }
        atomic_write(details_path, details)
        atomic_write(
            args.output,
            {
                key: int(row["candidate_count"])
                for key, row in details["topics"].items()
            },
        )
        print(topic_id, len(ids), flush=True)

    failed = [
        topic["topic_id"]
        for topic in manifest["topics"]
        if not (details["topics"].get(topic["topic_id"]) or {}).get("passed")
    ]
    details["complete"] = not failed
    details["failed_topics"] = failed
    atomic_write(details_path, details)
    if failed:
        raise SystemExit(f"{args.system} retrieval gate failed: {failed}")


if __name__ == "__main__":
    main()
