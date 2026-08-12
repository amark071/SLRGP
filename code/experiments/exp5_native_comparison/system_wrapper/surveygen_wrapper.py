#!/usr/bin/env python3
"""Thin native wrapper for the released SurveyGen/QUAL-SG workflow."""
from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from s3_native_common import RunSpec, utc_timestamp
from system_wrapper.base import (
    NativeSystemWrapper,
    NormalizedArtifacts,
    PreflightReport,
)


class SurveyGenWrapper(NativeSystemWrapper):
    system_id = "surveygen"
    preserve_partial_raw = True

    def __init__(
        self,
        source_dir: Path,
        *,
        driver_path: Path,
        bge_model: Path,
        python_executable: Path,
        chat_base_url: str,
        model: str,
    ):
        super().__init__(source_dir)
        self.driver_path = driver_path
        self.bge_model = bge_model
        self.python_executable = python_executable
        self.chat_base_url = chat_base_url
        self.model = model

    def preflight(self, spec: RunSpec) -> PreflightReport:
        issues: list[str] = []
        status_lines: list[str] = []
        source_clean = False
        try:
            status_lines = subprocess.run(
                ["git", "-C", str(self.source_dir), "status", "--short"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            allowed_changes = {
                "Code/paper_retrieval.py",
                "Code/quality_signal_augmentation.py",
                "Code/paper_reranking.py",
                "Code/outline_generation.py",
                "Code/survey_generation.py",
            }
            changed_paths = {line[3:].strip() for line in status_lines if len(line) >= 4}
            source_clean = all(
                path in allowed_changes or "/__pycache__/" in f"/{path}"
                for path in changed_paths
            )
        except (OSError, subprocess.CalledProcessError):
            issues.append("git_status_unavailable")
        if not source_clean:
            issues.append("source_has_non_allowlisted_changes")
        scripts = [
            "paper_retrieval.py",
            "quality_signal_augmentation.py",
            "paper_reranking.py",
            "outline_generation.py",
            "survey_generation.py",
        ]
        missing_scripts = [
            name for name in scripts if not (self.source_dir / "Code" / name).is_file()
        ]
        if missing_scripts:
            issues.append("released_pipeline_scripts_missing")
        if not self.driver_path.is_file():
            issues.append("execution_adapter_missing")
        if not self.python_executable.is_file():
            issues.append("python_environment_missing")
        if not self.bge_model.is_dir():
            issues.append("bge_model_missing")
        return PreflightReport(
            system_id=self.system_id,
            checked_at=utc_timestamp(),
            source_revision=spec.source_revision,
            source_clean=source_clean,
            entrypoint_present=not missing_scripts and self.driver_path.is_file(),
            dependencies_present=self.python_executable.is_file(),
            native_topic_to_survey=True,
            common_cutoff_supported=True,
            usage_logging_supported=True,
            eligible=not issues,
            issues=issues,
            details={
                "missing_scripts": missing_scripts,
                "driver_path": str(self.driver_path),
                "bge_model": str(self.bge_model),
                "chat_base_url": self.chat_base_url,
                "retrieval_services": ["Semantic Scholar", "OpenAlex"],
                "git_status_lines": status_lines,
            },
        )

    def build_command(self, spec: RunSpec, raw_dir: Path) -> list[str]:
        return [
            str(self.python_executable),
            str(self.driver_path),
            "--source-dir",
            str(self.source_dir),
            "--output-dir",
            str(raw_dir),
            "--topic",
            spec.topic.title,
            "--publication-cutoff",
            spec.topic.publication_cutoff,
            "--max-references",
            str(spec.topic.target_references_max),
            "--max-candidates",
            "120",
            "--target-words",
            str(spec.topic.target_words),
            "--model",
            self.model,
            "--api-base-url",
            self.chat_base_url,
            "--bge-model",
            str(self.bge_model),
        ]

    def normalize_output(self, spec: RunSpec, raw_dir: Path) -> NormalizedArtifacts:
        content_path = raw_dir / "survey_content.json"
        ranked_path = raw_dir / "paper_after_reranking.csv"
        if not content_path.is_file() or not ranked_path.is_file():
            raise FileNotFoundError("SurveyGen content or ranked-reference output missing")
        payload = json.loads(content_path.read_text(encoding="utf-8"))
        references = load_selected_references(ranked_path)
        survey = render_markdown(payload, references)
        usage_events = load_jsonl(raw_dir / "usage_events.jsonl")
        return NormalizedArtifacts(
            survey_text=survey,
            references=references,
            retrieval_manifest={
                "system": self.system_id,
                "publication_cutoff": spec.topic.publication_cutoff,
                "candidate_file": f"{spec.topic.title}.json",
                "ranked_file": ranked_path.name,
                "final_reference_ids": [
                    row.get("doi") or row.get("title") for row in references
                ],
            },
            stage_log={
                "system": self.system_id,
                "native_stages": [
                    "Semantic Scholar retrieval",
                    "OpenAlex quality-signal augmentation",
                    "quality-aware reranking",
                    "outline generation",
                    "subsection generation",
                ],
                "raw_output_file": content_path.name,
            },
            usage=aggregate_usage(usage_events),
            meta={
                "system": self.system_id,
                "topic": spec.topic.title,
                "source_revision": spec.source_revision,
                "model_alias": self.model,
                "normalized_at": utc_timestamp(),
                "raw_output": content_path.name,
                "deterministic_renderer": "surveygen_bundle_to_markdown_v1",
            },
        )

    def environment(self, spec: RunSpec, raw_dir: Path) -> dict[str, str]:
        env = super().environment(spec, raw_dir)
        env.update(
            {
                "S3_EMBEDDING_DEVICE": os.environ.get("S3_EMBEDDING_DEVICE", "cuda"),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            }
        )
        return env


def load_selected_references(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            dict(row)
            for row in csv.DictReader(handle)
            if row.get("tag") == "selected"
        ]


def render_markdown(payload: dict[str, Any], references: list[dict[str, Any]]) -> str:
    lines = [f"# {payload.get('title') or 'Survey'}"]
    current_section = None
    for item in payload.get("subsections") or []:
        section = str(item.get("section_title") or "").strip()
        subsection = str(item.get("subsection_title") or "").strip()
        content = str(item.get("content") or "").strip()
        if section and section != current_section:
            lines.extend(["", f"## {section}"])
            current_section = section
        if subsection:
            lines.extend(["", f"### {subsection}"])
        if content:
            lines.extend(["", content])
    lines.extend(["", "## References"])
    for index, row in enumerate(references, 1):
        title = (row.get("title") or "").strip()
        doi = (row.get("doi") or "").strip()
        suffix = f" https://doi.org/{doi}" if doi else ""
        lines.extend(["", f"[{index}] {title}.{suffix}".rstrip()])
    return "\n".join(lines).strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def aggregate_usage(events: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "n_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "provider_cost_usd": None,
        "provider_cost_observed": False,
        "events_file_present": bool(events),
    }
    for event in events:
        usage = event.get("usage") or {}
        totals["n_calls"] += 1
        totals["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        totals["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        totals["total_tokens"] += int(usage.get("total_tokens") or 0)
    return totals
