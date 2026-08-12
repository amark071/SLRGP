#!/usr/bin/env python3
"""Thin native wrapper for the released AutoSurvey workflow."""
from __future__ import annotations

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


class AutoSurveyWrapper(NativeSystemWrapper):
    system_id = "autosurvey"

    def __init__(
        self,
        source_dir: Path,
        *,
        database_dir: Path,
        embedding_model: str,
        python_executable: Path,
        chat_url: str,
        model: str,
    ):
        super().__init__(source_dir)
        self.database_dir = database_dir
        self.embedding_model = embedding_model
        self.python_executable = python_executable
        self.chat_url = chat_url
        self.model = model

    def preflight(self, spec: RunSpec) -> PreflightReport:
        issues = []
        source_clean = False
        status_lines: list[str] = []
        try:
            status_lines = subprocess.run(
                ["git", "-C", str(self.source_dir), "status", "--short"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            allowed_changes = {"src/model.py", "src/database.py"}
            changed_paths = {line[3:].strip() for line in status_lines if len(line) >= 4}
            source_clean = all(
                path in allowed_changes or "/__pycache__/" in f"/{path}"
                for path in changed_paths
            )
        except (OSError, subprocess.CalledProcessError):
            issues.append("git_status_unavailable")
        if not source_clean:
            issues.append("source_has_non_allowlisted_changes")
        entrypoint_present = (self.source_dir / "main.py").is_file()
        if not entrypoint_present:
            issues.append("main_py_missing")
        dependency_present = (self.source_dir / "requirements.txt").is_file()
        if not dependency_present:
            issues.append("requirements_missing")
        if not self.python_executable.is_file():
            issues.append("python_environment_missing")
        required_database_files = (
            "arxiv_paper_db.json",
            "faiss_paper_title_embeddings.bin",
            "faiss_paper_abs_embeddings.bin",
            "arxivid_to_index_abs.json",
        )
        missing_database = [
            name for name in required_database_files if not (self.database_dir / name).is_file()
        ]
        if missing_database:
            issues.append("database_incomplete")
        return PreflightReport(
            system_id=self.system_id,
            checked_at=utc_timestamp(),
            source_revision=spec.source_revision,
            source_clean=source_clean,
            entrypoint_present=entrypoint_present,
            dependencies_present=dependency_present and self.python_executable.is_file(),
            native_topic_to_survey=True,
            common_cutoff_supported=True,
            usage_logging_supported=True,
            eligible=not issues,
            issues=issues,
            details={
                "database_dir": str(self.database_dir),
                "missing_database_files": missing_database,
                "embedding_model": self.embedding_model,
                "chat_url": self.chat_url,
                "git_status_lines": status_lines,
            },
        )

    def build_command(self, spec: RunSpec, raw_dir: Path) -> list[str]:
        return [
            str(self.python_executable),
            "main.py",
            "--topic",
            spec.topic.title,
            "--gpu",
            "0",
            "--saving_path",
            str(raw_dir),
            "--model",
            self.model,
            "--section_num",
            "6",
            "--subsection_len",
            "55",
            "--rag_num",
            "40",
            "--outline_reference_num",
            "120",
            "--db_path",
            str(self.database_dir),
            "--embedding_model",
            self.embedding_model,
            "--api_url",
            self.chat_url,
            "--api_key",
            "relay-managed",
        ]

    def normalize_output(self, spec: RunSpec, raw_dir: Path) -> NormalizedArtifacts:
        expected = raw_dir / f"{spec.topic.title}.json"
        candidates = [expected] if expected.is_file() else sorted(raw_dir.glob("*.json"))
        candidates = [path for path in candidates if path.name != "usage.json"]
        if not candidates:
            raise FileNotFoundError("AutoSurvey JSON output missing")
        payload = json.loads(candidates[0].read_text(encoding="utf-8"))
        survey = str(payload.get("survey") or "").strip()
        if not survey:
            raise ValueError("AutoSurvey output has no survey text")
        references = normalize_references(payload.get("reference"))
        usage_events = []
        usage_log = raw_dir / "usage_events.jsonl"
        if usage_log.is_file():
            for line in usage_log.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    usage_events.append(json.loads(line))
        usage = aggregate_usage(usage_events)
        return NormalizedArtifacts(
            survey_text=survey,
            references=references,
            retrieval_manifest={
                "system": self.system_id,
                "publication_cutoff": spec.topic.publication_cutoff,
                "final_reference_ids": [
                    row.get("id") or row.get("arxiv_id") or row.get("title")
                    for row in references
                ],
                "note": "The released output exposes final selected references; candidate-level logging is added by an allowlisted instrumentation patch before pilot eligibility.",
            },
            stage_log={
                "system": self.system_id,
                "native_stages": [
                    "database retrieval",
                    "outline generation",
                    "subsection writing",
                    "refinement",
                ],
                "raw_output_file": candidates[0].name,
            },
            usage=usage,
            meta={
                "system": self.system_id,
                "topic": spec.topic.title,
                "source_revision": spec.source_revision,
                "model": self.model,
                "normalized_at": utc_timestamp(),
                "raw_output": candidates[0].name,
            },
        )

    def environment(self, spec: RunSpec, raw_dir: Path) -> dict[str, str]:
        env = super().environment(spec, raw_dir)
        env["S3_API_MAX_THREADS"] = os.environ.get("S3_API_MAX_THREADS", "3")
        return env


def normalize_references(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        rows = []
        for key, item in value.items():
            row = dict(item) if isinstance(item, dict) else {"value": item}
            row.setdefault("source_key", str(key))
            rows.append(row)
        return rows
    if isinstance(value, list):
        return [dict(item) if isinstance(item, dict) else {"value": item} for item in value]
    return []


def aggregate_usage(events: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "n_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "provider_cost_usd": None,
        "provider_cost_observed": False,
    }
    for event in events:
        usage = event.get("usage") or {}
        totals["n_calls"] += 1
        totals["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        totals["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        totals["total_tokens"] += int(usage.get("total_tokens") or 0)
        if event.get("cost_usd") is not None:
            if totals["provider_cost_usd"] is None:
                totals["provider_cost_usd"] = 0.0
            totals["provider_cost_usd"] += float(event["cost_usd"])
            totals["provider_cost_observed"] = True
    totals["events_file_present"] = bool(events)
    return totals
