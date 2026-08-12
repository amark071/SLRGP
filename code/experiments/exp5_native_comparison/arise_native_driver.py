#!/usr/bin/env python3
"""Execute the released ARISE stages with cutoff and stage checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_stage(
    script: Path,
    *,
    cwd: Path,
    env: dict[str, str],
    done_dir: Path,
    timeout_seconds: int,
) -> None:
    marker = done_dir / f"{script.name}.done"
    if marker.is_file():
        print(f"[resume] {script.name}")
        return
    try:
        completed = subprocess.run(
            [sys.executable, script.name],
            cwd=cwd,
            env=env,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(
            f"{script.name} exceeded stage timeout {timeout_seconds}s"
        ) from exc
    if completed.returncode != 0:
        raise SystemExit(f"{script.name} failed with exit code {completed.returncode}")
    marker.write_text("ok\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoff-filter", type=Path, required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--publication-cutoff", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--paper-count", type=int, default=15)
    parser.add_argument("--max-refinement-rounds", type=int, default=2)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    work_dir = output_dir / "work"
    done_dir = output_dir / "stage_checkpoints"
    done_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        args.source_dir.resolve() / "ARISE_Source_Code",
        work_dir,
        dirs_exist_ok=True,
    )
    with (work_dir / "topics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["topic"])
        writer.writerow([args.topic])

    env = dict(os.environ)
    env.update(
        {
            "OPENAI_API_KEY": "relay-managed",
            "OPENAI_API_BASE": args.api_base_url,
            "OPENAI_BASE_URL": args.api_base_url,
            "GEMINI_API_KEY": "relay-managed",
            "ANTHROPIC_API_KEY": "relay-managed",
            "MODEL": args.model,
            "S3_COMMON_MODEL": args.model,
            "S3_PUBLICATION_CUTOFF": args.publication_cutoff,
            "PAPER_COUNT": str(args.paper_count),
            "JOURNAL": "acm",
            "TEMPERATURE": "0",
            "S3_ARISE_MAX_ROUNDS": str(args.max_refinement_rounds),
        }
    )
    if not env.get("SERPER_API_KEY"):
        raise SystemExit("SERPER_API_KEY missing")

    citation_script = work_dir / "1_citation.py"
    run_stage(
        citation_script,
        cwd=work_dir,
        env=env,
        done_dir=done_dir,
        timeout_seconds=900,
    )
    cutoff_marker = done_dir / "citation_cutoff.done"
    if not cutoff_marker.is_file():
        filtered = work_dir / "results.filtered.csv"
        completed = subprocess.run(
            [
                sys.executable,
                str(args.cutoff_filter.resolve()),
                "--input",
                str(work_dir / "results.csv"),
                "--output",
                str(filtered),
                "--audit",
                str(output_dir / "cutoff_audit.json"),
                "--cutoff",
                args.publication_cutoff,
                "--minimum-kept",
                "5",
            ],
            cwd=work_dir,
            env=env,
            check=False,
        )
        if completed.returncode != 0:
            citation_marker = done_dir / "1_citation.py.done"
            citation_marker.unlink(missing_ok=True)
            raise SystemExit("ARISE citation cutoff gate failed")
        filtered.replace(work_dir / "results.csv")
        cutoff_marker.write_text("ok\n", encoding="utf-8")

    for script_name in [
        "2deduplicated.py",
        "3_summary.py",
        "4merge.py",
        "5writing-Copy1.py",
        "6formatting.py",
        "arl_refinement.py",
    ]:
        timeout_seconds = 3600 if script_name == "arl_refinement.py" else 1800
        run_stage(
            work_dir / script_name,
            cwd=work_dir,
            env=env,
            done_dir=done_dir,
            timeout_seconds=timeout_seconds,
        )

    candidates = sorted((work_dir / "output").glob("final_round*_paper.tex"))
    final_tex = candidates[-1] if candidates else work_dir / "survey_paper.tex"
    if not final_tex.is_file():
        raise SystemExit("ARISE final LaTeX output missing")
    (output_dir / "final_output.json").write_text(
        json.dumps(
            {
                "final_tex": str(final_tex.relative_to(output_dir)),
                "citation_csv": str((work_dir / "citation.csv").relative_to(output_dir)),
                "completed_markdown": str(
                    (work_dir / "completed_research_paper.md").relative_to(output_dir)
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
