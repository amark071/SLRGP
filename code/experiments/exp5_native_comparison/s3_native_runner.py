#!/usr/bin/env python3
"""Preflight and execute the S3 native-flow pilot matrix."""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

from s3_native_common import (
    RunSpec,
    TopicIntent,
    atomic_write_json,
    sha256_file,
    utc_timestamp,
)
from system_wrapper.arise_wrapper import AriseWrapper
from system_wrapper.autosurvey_wrapper import AutoSurveyWrapper
from system_wrapper.slrgp_wrapper import SLRGPWrapper
from system_wrapper.surveyforge_wrapper import SurveyForgeWrapper
from system_wrapper.surveygen_wrapper import SurveyGenWrapper


COMMON_ALIAS = "s3-common-backbone"
COMMON_PROVIDER_MODEL = "anthropic/claude-sonnet-4.6"
SCRIPT_DIR = Path(__file__).resolve().parent
SERVER_ROOT = SCRIPT_DIR.parents[2]
INFRA_ROOT = SCRIPT_DIR
# Baseline system checkouts, their corpus assets, and per-system Python
# environments are NOT shipped with this package (third-party code and large
# model/database files). Rebuild them per code/data_rebuild/README.md and
# point S3_NATIVE_WORK_ROOT at the resulting directory.
WORK_ROOT = Path(os.environ.get("S3_NATIVE_WORK_ROOT", SERVER_ROOT / "work/s3_native"))
BASELINES_ROOT = WORK_ROOT / "baselines"
ASSETS_ROOT = WORK_ROOT / "assets"
ENVS_ROOT = WORK_ROOT / "envs"
CHAT_BASE = "http://127.0.0.1:18080/v1"
CHAT_COMPLETIONS = CHAT_BASE + "/chat/completions"


def load_topics(path: Path) -> list[TopicIntent]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [TopicIntent.from_dict(value) for value in payload["topics"]]


def build_wrappers() -> dict[str, object]:
    surveyforge_source = BASELINES_ROOT / "surveyforge/code"
    slrgp_runner = INFRA_ROOT / "s3_slrgp_full_deployment.py"
    slrgp_python = Path(os.environ.get("PYTHON", sys.executable))
    return {
        "slrgp": SLRGPWrapper(
            SERVER_ROOT,
            runner_path=slrgp_runner,
            python_executable=slrgp_python,
            chat_base_url=CHAT_BASE,
            model=COMMON_PROVIDER_MODEL,
        ),
        "autosurvey": AutoSurveyWrapper(
            BASELINES_ROOT / "autosurvey",
            database_dir=ASSETS_ROOT / "autosurvey/database",
            embedding_model=str(
                ASSETS_ROOT / "autosurvey/models/nomic-embed-text-v1"
            ),
            python_executable=ENVS_ROOT / "autosurvey/bin/python",
            chat_url=CHAT_COMPLETIONS,
            model=COMMON_ALIAS,
        ),
        "surveyforge": SurveyForgeWrapper(
            surveyforge_source,
            database_dir=ASSETS_ROOT / "surveyforge/database/database",
            outline_dir=ASSETS_ROOT / "surveyforge/outlines",
            embedding_model=ASSETS_ROOT / "surveyforge/models/gte-large-en-v1.5",
            python_executable=ENVS_ROOT / "surveyforge/bin/python",
            chat_url=CHAT_COMPLETIONS,
            model=COMMON_ALIAS,
        ),
        "surveygen": SurveyGenWrapper(
            BASELINES_ROOT / "surveygen",
            driver_path=INFRA_ROOT / "surveygen_native_driver.py",
            bge_model=ASSETS_ROOT / "surveygen/models/bge-large-en",
            python_executable=ENVS_ROOT / "surveygen/bin/python",
            chat_base_url=CHAT_BASE,
            model=COMMON_ALIAS,
        ),
        "arise": AriseWrapper(
            BASELINES_ROOT / "arise",
            driver_path=INFRA_ROOT / "arise_native_driver.py",
            cutoff_filter_path=INFRA_ROOT / "arise_cutoff_filter.py",
            python_executable=ENVS_ROOT / "arise/bin/python",
            chat_base_url=CHAT_BASE,
            model="openai/" + COMMON_ALIAS,
            max_refinement_rounds=2,
        ),
    }


def source_revisions() -> dict[str, str]:
    return {
        "slrgp": "local-" + sha256_file(
            INFRA_ROOT / "s3_slrgp_full_deployment.py"
        )[:16],
        "autosurvey": "5e8f389f3d51b29bad16dc6ae75db3e8a45a3b65",
        "surveyforge": "9114a0b7895a0f7eb614938d9bc0c956cf25245b",
        "surveygen": "7c8fd1e7325730f858a0a42acc9884ce5d5de55f",
        "arise": "5db26bbd64bb9e865dd64b17c75dc2cb26e0e150",
    }


def make_spec(
    *,
    run_id: str,
    system_id: str,
    topic: TopicIntent,
    protocol_hash: str,
    timeout_seconds: int,
    budget_usd: float,
) -> RunSpec:
    return RunSpec(
        run_id=run_id,
        system_id=system_id,
        source_revision=source_revisions()[system_id],
        model_policy=f"common_backbone:{COMMON_PROVIDER_MODEL}",
        topic=topic,
        budget_usd=budget_usd,
        timeout_seconds=timeout_seconds,
        protocol_hash=protocol_hash,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--topics",
        type=Path,
        default=INFRA_ROOT / "pilot_topic_candidates.json",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=SERVER_ROOT / "work/results/s3_native",
    )
    parser.add_argument(
        "--core-reference-sets",
        type=Path,
        default=INFRA_ROOT / "core_reference_sets_s3_native12.json",
    )
    parser.add_argument(
        "--systems",
        default="slrgp,autosurvey,surveyforge,surveygen",
    )
    parser.add_argument(
        "--topic-ids",
        default="",
        help="Optional comma-separated subset for checkpointed reruns.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument(
        "--budget-usd",
        type=float,
        default=16.0,
        help="Frozen per-system/topic ceiling derived from the two-topic pilot.",
    )
    args = parser.parse_args()

    run_root = args.results_root / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    topics = load_topics(args.topics)
    if any(topic.analysis_role == "confirmatory_primary" for topic in topics):
        topic_manifest = json.loads(args.topics.read_text(encoding="utf-8"))
        if topic_manifest.get("status") != "frozen_after_all_system_retrieval_gate":
            raise SystemExit("Confirmatory topic manifest has not passed its retrieval gate")
        if not args.core_reference_sets.is_file():
            raise SystemExit("Confirmatory generation requires frozen core-reference sets")
    if args.topic_ids:
        requested_topics = {
            value.strip() for value in args.topic_ids.split(",") if value.strip()
        }
        topics = [topic for topic in topics if topic.topic_id in requested_topics]
        missing_topics = requested_topics - {topic.topic_id for topic in topics}
        if missing_topics:
            raise SystemExit(f"Unknown topic IDs: {sorted(missing_topics)}")
    wrappers = build_wrappers()
    selected = [value.strip() for value in args.systems.split(",") if value.strip()]
    unknown = sorted(set(selected) - set(wrappers))
    if unknown:
        raise SystemExit(f"Unknown executable systems: {unknown}")
    protocol_path = SERVER_ROOT / "S3_NATIVE_END_TO_END_PROTOCOL.md"
    protocol_hash = sha256_file(protocol_path) if protocol_path.is_file() else "not-shipped"
    shutil.copy2(args.topics, run_root / "topics_s3_native12.json")
    if args.core_reference_sets.is_file():
        shutil.copy2(
            args.core_reference_sets,
            run_root / "core_reference_sets_s3_native12.json",
        )
    atomic_write_json(
        run_root / "frozen_protocol.json",
        {
            "generated_at": utc_timestamp(),
            "protocol_path": str(protocol_path),
            "protocol_sha256": protocol_hash,
            "common_backbone": COMMON_PROVIDER_MODEL,
            "selected_systems": selected,
            "per_system_topic_budget_usd": args.budget_usd,
            "timeout_seconds": args.timeout_seconds,
            "order_rule": "topic-seed deterministic shuffle",
            "publication_cutoff": topics[0].publication_cutoff,
            "core_reference_sets_sha256": (
                sha256_file(args.core_reference_sets)
                if args.core_reference_sets.is_file()
                else None
            ),
        },
    )

    disposition = {
        "generated_at": utc_timestamp(),
        "common_backbone": COMMON_PROVIDER_MODEL,
        "publication_cutoff": topics[0].publication_cutoff,
        "executable_primary_pilot": selected,
        "preflight_exclusions": {
            "itersurvey": (
                "The released approximately 680K-paper database and builder are absent. "
                "Using AutoSurvey's corpus would violate the registered no-borrowed-retrieval rule."
            ),
            "surveyg": (
                "The released workflow mixes a hard-required Semantic Scholar key, direct Gemini SDK "
                "generation, and date-relative crawling. Enforcing the common backbone and exact cutoff "
                "would require replacing core retrieval/writing paths rather than a thin adapter."
            ),
            "arise": (
                "Repeated native retrieval preflights stalled in released CrewAI tool execution and "
                "the released citation parser collapsed multiple citations into one record. Replacing "
                "that stage with direct Serper retrieval would change the core system treatment."
            ),
        },
    }
    atomic_write_json(run_root / "preflight_disposition.json", disposition)

    preflight_rows = []
    for system_id in selected:
        wrapper = wrappers[system_id]
        for topic in topics:
            spec = make_spec(
                run_id=args.run_id,
                system_id=system_id,
                topic=topic,
                protocol_hash=protocol_hash,
                timeout_seconds=args.timeout_seconds,
                budget_usd=args.budget_usd,
            )
            report = wrapper.preflight(spec)
            preflight_rows.append(
                {
                    "system_id": system_id,
                    "topic_id": topic.topic_id,
                    **asdict(report),
                }
            )
    atomic_write_json(run_root / "preflight_matrix.json", preflight_rows)
    for system_id in selected:
        system_root = run_root / "systems" / system_id
        system_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            system_root / "source_manifest.json",
            {
                "system_id": system_id,
                "source_revision": source_revisions()[system_id],
                "common_backbone": COMMON_PROVIDER_MODEL,
                "preflight_reports": [
                    row for row in preflight_rows if row["system_id"] == system_id
                ],
            },
        )
    failures = [row for row in preflight_rows if not row["eligible"]]
    print(
        f"[S3-native] preflight={len(preflight_rows) - len(failures)}/"
        f"{len(preflight_rows)} eligible",
        flush=True,
    )
    if failures:
        for row in failures:
            print(
                f"[S3-native] ineligible {row['system_id']} {row['topic_id']}: "
                f"{row['issues']}",
                flush=True,
            )
        if args.mode == "run":
            raise SystemExit("Preflight failures block pilot generation")
    if args.mode == "preflight":
        return

    for topic in topics:
        topic_order = list(selected)
        random.Random(topic.seed).shuffle(topic_order)
        for system_id in topic_order:
            print(f"[S3-native] START {topic.topic_id} / {system_id}", flush=True)
            spec = make_spec(
                run_id=args.run_id,
                system_id=system_id,
                topic=topic,
                protocol_hash=protocol_hash,
                timeout_seconds=args.timeout_seconds,
                budget_usd=args.budget_usd,
            )
            run_dir = wrappers[system_id].run(run_root, spec, resume=True)
            status = json.loads(run_dir.status_path.read_text(encoding="utf-8"))
            print(
                f"[S3-native] END {topic.topic_id} / {system_id}: "
                f"{status.get('status')}",
                flush=True,
            )
            if status.get("status") != "ok":
                raise SystemExit(
                    f"Pilot stopped after {system_id}/{topic.topic_id}: "
                    f"{status.get('status')}"
                )


if __name__ == "__main__":
    main()
