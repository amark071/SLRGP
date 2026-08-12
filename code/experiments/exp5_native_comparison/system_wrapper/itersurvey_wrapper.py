#!/usr/bin/env python3
"""Thin native wrapper for the released IterSurvey workflow."""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from s3_native_common import RunSpec, utc_timestamp
from system_wrapper.base import (
    NativeSystemWrapper,
    NormalizedArtifacts,
    PreflightReport,
)


ARXIV_PATTERN = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)")


class IterSurveyWrapper(NativeSystemWrapper):
    system_id = "itersurvey"

    def __init__(
        self,
        source_dir: Path,
        *,
        database_dir: Path,
        embedding_model: Path,
        python_executable: Path,
        chat_base_url: str,
        model: str,
    ):
        super().__init__(source_dir)
        self.database_dir = database_dir
        self.embedding_model = embedding_model
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
            allowed_changes = {"main.py", "src/model.py", "src/database.py"}
            changed_paths = {line[3:].strip() for line in status_lines if len(line) >= 4}
            source_clean = changed_paths.issubset(allowed_changes)
        except (OSError, subprocess.CalledProcessError):
            issues.append("git_status_unavailable")
        if not source_clean:
            issues.append("source_has_non_allowlisted_changes")

        entrypoint_present = (self.source_dir / "main.py").is_file()
        dependency_present = (self.source_dir / "requirements.txt").is_file()
        if not entrypoint_present:
            issues.append("main_py_missing")
        if not dependency_present:
            issues.append("requirements_missing")
        if not self.python_executable.is_file():
            issues.append("python_environment_missing")
        database_files = (
            "arxiv_paper_db.json",
            "faiss_paper_title_embeddings.bin",
            "faiss_paper_abs_embeddings.bin",
            "arxivid_to_index_abs.json",
        )
        missing_database = [
            name for name in database_files if not (self.database_dir / name).is_file()
        ]
        if missing_database:
            issues.append("database_incomplete")
        if not self.embedding_model.is_dir():
            issues.append("embedding_model_missing")
        cutoff_supported = spec.topic.publication_cutoff <= "2024-09-26"
        if not cutoff_supported:
            issues.append("topic_cutoff_differs_from_frozen_corpus")
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
                "embedding_model": str(self.embedding_model),
                "chat_base_url": self.chat_base_url,
                "full_text_mode": False,
                "git_status_lines": status_lines,
            },
        )

    def build_command(self, spec: RunSpec, raw_dir: Path) -> list[str]:
        cutoff_yymm = (
            spec.topic.publication_cutoff[2:4]
            + spec.topic.publication_cutoff[5:7]
        )
        return [
            str(self.python_executable),
            "main.py",
            "--topic",
            spec.topic.title,
            "--description",
            spec.topic.scope,
            "--saving_path",
            str(raw_dir),
            "--model",
            self.model,
            "--vision_model",
            self.model,
            "--api_url",
            self.chat_base_url,
            "--vision_api_url",
            self.chat_base_url,
            "--api_key",
            "relay-managed",
            "--vision_api_key",
            "relay-managed",
            "--section_num",
            "6",
            "--outline_reference_num",
            "300",
            "--subsection_len",
            "450",
            "--rag_num",
            str(spec.topic.target_references_max),
            "--end_time",
            cutoff_yymm,
            "--db_path",
            str(self.database_dir),
            "--embedding_model",
            str(self.embedding_model),
            "--use_abs",
            "True",
            "--mineru_port",
            "18999",
        ]

    def normalize_output(self, spec: RunSpec, raw_dir: Path) -> NormalizedArtifacts:
        candidates = [
            path
            for path in raw_dir.rglob("main.md")
            if path.parent.name != "markdown"
        ]
        if not candidates:
            candidates = list(raw_dir.rglob("main.md"))
        if not candidates:
            raise FileNotFoundError("IterSurvey main.md output missing")
        output = max(candidates, key=lambda path: path.stat().st_mtime)
        survey = output.read_text(encoding="utf-8").strip()
        if not survey:
            raise ValueError("IterSurvey output has no survey text")
        ids = list(dict.fromkeys(ARXIV_PATTERN.findall(survey)))
        references = [{"arxiv_id": value} for value in ids]
        usage_events = load_jsonl(raw_dir / "usage_events.jsonl")
        return NormalizedArtifacts(
            survey_text=survey,
            references=references,
            retrieval_manifest={
                "system": self.system_id,
                "publication_cutoff": spec.topic.publication_cutoff,
                "native_end_time": (
                    spec.topic.publication_cutoff[2:4]
                    + spec.topic.publication_cutoff[5:7]
                ),
                "final_reference_ids": ids,
                "input_mode": "released abstract mode",
            },
            stage_log={
                "system": self.system_id,
                "native_stages": [
                    "recurrent outline generation",
                    "paper-card extraction",
                    "section drafting",
                    "three review-refine rounds",
                    "title and abstract generation",
                    "figure/table generation",
                    "markdown generation",
                ],
                "raw_output_file": str(output.relative_to(raw_dir)),
            },
            usage=aggregate_usage(usage_events),
            meta={
                "system": self.system_id,
                "topic": spec.topic.title,
                "source_revision": spec.source_revision,
                "model_alias": self.model,
                "normalized_at": utc_timestamp(),
                "raw_output": str(output.relative_to(raw_dir)),
            },
        )

    def environment(self, spec: RunSpec, raw_dir: Path) -> dict[str, str]:
        env = super().environment(spec, raw_dir)
        env.update(
            {
                "S3_API_MAX_THREADS": os.environ.get("S3_API_MAX_THREADS", "3"),
                "S3_SECTION_THREADS": os.environ.get("S3_SECTION_THREADS", "2"),
                "S3_EMBEDDING_DEVICE": os.environ.get("S3_EMBEDDING_DEVICE", "cuda"),
                "S3_MODEL_CACHE_DIR": str(raw_dir / "model_cache"),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "SWANLAB_MODE": "local",
            }
        )
        return env


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
        "events_file_present": bool(events),
    }
    for event in events:
        usage = event.get("usage") or {}
        totals["n_calls"] += 1
        totals["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        totals["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        totals["total_tokens"] += int(usage.get("total_tokens") or 0)
    return totals
