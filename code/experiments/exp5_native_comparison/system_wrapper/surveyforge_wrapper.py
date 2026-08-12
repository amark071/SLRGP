#!/usr/bin/env python3
"""Thin native wrapper for the released SurveyForge workflow."""
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


class SurveyForgeWrapper(NativeSystemWrapper):
    system_id = "surveyforge"

    def __init__(
        self,
        source_dir: Path,
        *,
        database_dir: Path,
        outline_dir: Path,
        embedding_model: Path,
        python_executable: Path,
        chat_url: str,
        model: str,
    ):
        super().__init__(source_dir)
        self.database_dir = database_dir
        self.outline_dir = outline_dir
        self.embedding_model = embedding_model
        self.python_executable = python_executable
        self.chat_url = chat_url
        self.model = model

    def preflight(self, spec: RunSpec) -> PreflightReport:
        issues: list[str] = []
        status_lines: list[str] = []
        source_clean = False
        try:
            status_lines = subprocess.run(
                ["git", "-C", str(self.source_dir.parent), "status", "--short"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            allowed_changes = {
                "code/src/model.py",
                "code/src/database.py",
                "code/src/rag.py",
                "code/src/agents/outline_writer.py",
                "code/src/agents/writer.py",
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

        entrypoint_present = (self.source_dir / "main.py").is_file()
        dependency_present = (self.source_dir / "requirement.txt").is_file()
        if not entrypoint_present:
            issues.append("main_py_missing")
        if not dependency_present:
            issues.append("requirement_manifest_missing")
        if not self.python_executable.is_file():
            issues.append("python_environment_missing")

        database_files = (
            "arxiv_paper_db_with_cc.json",
            "surveys_arxiv_paper_db.json",
            "faiss_paper_title_embeddings_FROM_2012_0101_TO_240926.bin",
            "faiss_paper_title_abs_embeddings_FROM_2012_0101_TO_240926.bin",
            "faiss_survey_title_embeddings_FROM_1501_TO_2409_gte.bin",
            "faiss_survey_title_abs_embeddings_FROM_1501_TO_2409_gte.bin",
            "arxivid_to_index_abs.json",
            "surveys_arxivid_to_index_abs.json",
        )
        missing_database = [
            name for name in database_files if not (self.database_dir / name).is_file()
        ]
        if missing_database:
            issues.append("database_incomplete")
        missing_outline_dirs = [
            name
            for name in ("Final_outline", "Final_outline_First")
            if not (self.outline_dir / name).is_dir()
        ]
        if missing_outline_dirs:
            issues.append("survey_outline_database_incomplete")
        if not self.embedding_model.is_dir():
            issues.append("embedding_model_missing")
        cutoff_supported = spec.topic.publication_cutoff <= "2024-09-26"
        if not cutoff_supported:
            issues.append("topic_cutoff_differs_from_released_corpus_cutoff")

        return PreflightReport(
            system_id=self.system_id,
            checked_at=utc_timestamp(),
            source_revision=spec.source_revision,
            source_clean=source_clean,
            entrypoint_present=entrypoint_present,
            dependencies_present=dependency_present and self.python_executable.is_file(),
            native_topic_to_survey=True,
            common_cutoff_supported=cutoff_supported,
            usage_logging_supported=True,
            eligible=not issues,
            issues=issues,
            details={
                "database_dir": str(self.database_dir),
                "missing_database_files": missing_database,
                "outline_dir": str(self.outline_dir),
                "missing_outline_dirs": missing_outline_dirs,
                "embedding_model": str(self.embedding_model),
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
            "--saving_path",
            str(raw_dir),
            "--model",
            self.model,
            "--section_num",
            "6",
            "--subsection_len",
            "70",
            "--outline_reference_num",
            "300",
            "--rag_num",
            "60",
            "--rag_max_out",
            "45",
            "--db_path",
            str(self.database_dir),
            "--survey_outline_path",
            str(self.outline_dir),
            "--embedding_model",
            str(self.embedding_model),
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
            raise FileNotFoundError("SurveyForge JSON output missing")
        payload = json.loads(candidates[0].read_text(encoding="utf-8"))
        survey = str(payload.get("survey") or "").strip()
        if not survey:
            raise ValueError("SurveyForge output has no survey text")
        references = normalize_references(payload.get("reference"))
        usage_events = load_jsonl(raw_dir / "usage_events.jsonl")
        return NormalizedArtifacts(
            survey_text=survey,
            references=references,
            retrieval_manifest={
                "system": self.system_id,
                "publication_cutoff": spec.topic.publication_cutoff,
                "corpus_release_cutoff": "2024-09-26",
                "effective_cutoff": spec.topic.publication_cutoff,
                "final_reference_ids": [
                    row.get("id") or row.get("arxiv_id") for row in references
                ],
            },
            stage_log={
                "system": self.system_id,
                "native_stages": [
                    "paper and prior-survey retrieval",
                    "rough outline generation",
                    "outline merging",
                    "suboutline generation",
                    "subsection writing and citation checking",
                    "local coherence refinement",
                ],
                "raw_output_file": candidates[0].name,
            },
            usage=aggregate_usage(usage_events),
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
        env.update(
            {
                "S3_API_MAX_THREADS": os.environ.get("S3_API_MAX_THREADS", "3"),
                "S3_SECTION_THREADS": os.environ.get("S3_SECTION_THREADS", "2"),
                "S3_EMBEDDING_DEVICE": os.environ.get("S3_EMBEDDING_DEVICE", "cuda"),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            }
        )
        return env


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def normalize_references(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [
            {"citation_number": str(number), "arxiv_id": str(arxiv_id)}
            for number, arxiv_id in value.items()
        ]
    if isinstance(value, list):
        return [dict(item) if isinstance(item, dict) else {"id": item} for item in value]
    return []


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
