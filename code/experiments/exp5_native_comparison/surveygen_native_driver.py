#!/usr/bin/env python3
"""Execute the released QUAL-SG scripts in their documented order."""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


def run(
    script: Path,
    *,
    cwd: Path,
    env: dict[str, str],
    expected_output: Path,
) -> None:
    if expected_output.is_file() and expected_output.stat().st_size > 0:
        print(f"[resume] {script.name}: {expected_output.name} already present")
        return
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=cwd,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"{script.name} failed with exit code {completed.returncode}")
    if not expected_output.is_file() or expected_output.stat().st_size == 0:
        raise SystemExit(f"{script.name} did not produce {expected_output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--publication-cutoff", required=True)
    parser.add_argument("--max-references", type=int, required=True)
    parser.add_argument("--max-candidates", type=int, default=120)
    parser.add_argument("--target-words", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--bge-model", type=Path, required=True)
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    topic_csv = output_dir / "topic.csv"
    with topic_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["title", "year", "keywords", "max_refs"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "title": args.topic,
                "year": args.publication_cutoff[:4],
                "keywords": args.topic,
                "max_refs": args.max_references,
            }
        )

    retrieved_json = output_dir / f"{args.topic}.json"
    augmented_json = output_dir / "final_ref_set.json"
    ranked_csv = output_dir / "paper_after_reranking.csv"
    outline_json = output_dir / "survey_outline.json"
    sections_dir = output_dir / "sections"
    content_json = output_dir / "survey_content.json"

    env = dict(os.environ)
    env.update(
        {
            "S3_TOPIC_CSV": str(topic_csv),
            "S3_OUTPUT_DIR": str(output_dir),
            "S3_RETRIEVED_JSON": str(retrieved_json),
            "S3_AUGMENTED_JSON": str(augmented_json),
            "S3_RANKED_CSV": str(ranked_csv),
            "S3_OUTLINE_JSON": str(outline_json),
            "S3_SECTIONS_DIR": str(sections_dir),
            "S3_CONTENT_JSON": str(content_json),
            "S3_PUBLICATION_CUTOFF": args.publication_cutoff,
            "S3_MAX_CANDIDATES": str(args.max_candidates),
            "S3_TARGET_WORDS": str(args.target_words),
            "S3_MODEL": args.model,
            "S3_API_BASE_URL": args.api_base_url,
            "S3_API_KEY": "relay-managed",
            "S3_BGE_MODEL": str(args.bge_model.resolve()),
        }
    )
    code_dir = source_dir / "Code"
    stages = [
        ("paper_retrieval.py", retrieved_json),
        ("quality_signal_augmentation.py", augmented_json),
        ("paper_reranking.py", ranked_csv),
        ("outline_generation.py", outline_json),
        ("survey_generation.py", content_json),
    ]
    for script_name, expected_output in stages:
        run(
            code_dir / script_name,
            cwd=output_dir,
            env=env,
            expected_output=expected_output,
        )
    if not content_json.is_file():
        raise SystemExit("SurveyGen content bundle was not produced")


if __name__ == "__main__":
    main()
