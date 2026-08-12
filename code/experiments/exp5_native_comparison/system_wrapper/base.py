#!/usr/bin/env python3
"""Typed wrapper boundary around released native survey-generation systems."""
from __future__ import annotations

import abc
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from s3_native_common import (
    RunDirectory,
    RunSpec,
    atomic_write_json,
    atomic_write_text,
    run_logged_command,
    sha256_file,
    utc_timestamp,
)


@dataclass(frozen=True)
class PreflightReport:
    system_id: str
    checked_at: str
    source_revision: str
    source_clean: bool
    entrypoint_present: bool
    dependencies_present: bool
    native_topic_to_survey: bool
    common_cutoff_supported: bool
    usage_logging_supported: bool
    eligible: bool
    issues: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedArtifacts:
    survey_text: str
    references: list[dict[str, Any]]
    retrieval_manifest: dict[str, Any]
    stage_log: dict[str, Any]
    usage: dict[str, Any]
    meta: dict[str, Any]


class NativeSystemWrapper(abc.ABC):
    """A thin adapter that must not replace released system logic."""

    system_id: str
    preserve_partial_raw = False

    def __init__(self, source_dir: Path):
        self.source_dir = source_dir

    @abc.abstractmethod
    def preflight(self, spec: RunSpec) -> PreflightReport:
        """Check native-flow eligibility without generating a review."""

    @abc.abstractmethod
    def build_command(self, spec: RunSpec, raw_dir: Path) -> list[str]:
        """Return the released workflow's entry command."""

    @abc.abstractmethod
    def normalize_output(self, spec: RunSpec, raw_dir: Path) -> NormalizedArtifacts:
        """Map released outputs into the common artifact contract without editing text."""

    def environment(self, spec: RunSpec, raw_dir: Path) -> dict[str, str]:
        env = dict(os.environ)
        env.update(
            {
                "S3_RUN_ID": spec.run_id,
                "S3_SYSTEM_ID": spec.system_id,
                "S3_TOPIC_ID": spec.topic.topic_id,
                "S3_PUBLICATION_CUTOFF": spec.topic.publication_cutoff,
                "S3_TARGET_WORDS": str(spec.topic.target_words),
                "S3_BUDGET_USD": str(spec.budget_usd),
                "S3_RAW_OUTPUT_DIR": str(raw_dir),
                "S3_USAGE_LOG": str(raw_dir / "usage_events.jsonl"),
            }
        )
        return env

    def run(self, results_root: Path, spec: RunSpec, resume: bool = True) -> RunDirectory:
        run_dir = RunDirectory(results_root, spec)
        if resume and run_dir.completed_and_valid():
            run_dir.event("resume_skip_valid")
            return run_dir
        run_dir.initialize()
        preflight = self.preflight(spec)
        atomic_write_json(run_dir.root / "preflight.json", asdict(preflight))
        if not preflight.eligible:
            run_dir.event("preflight_failed", issues=preflight.issues)
            run_dir.write_status("ineligible", issues=preflight.issues)
            return run_dir

        raw_dir = run_dir.root / "raw"
        if raw_dir.exists() and not (resume and self.preserve_partial_raw):
            shutil.rmtree(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        command = self.build_command(spec, raw_dir)
        run_dir.event("generation_started")
        run_dir.write_status("running", started_at=utc_timestamp())
        result = run_logged_command(
            command,
            cwd=self.source_dir,
            log_path=run_dir.log_path,
            timeout_seconds=spec.timeout_seconds,
            env=self.environment(spec, raw_dir),
        )
        if result.timed_out:
            run_dir.event("generation_timeout", elapsed_seconds=result.elapsed_seconds)
            run_dir.write_status("timeout", elapsed_seconds=result.elapsed_seconds)
            return run_dir
        if result.returncode != 0:
            run_dir.event(
                "generation_failed",
                returncode=result.returncode,
                elapsed_seconds=result.elapsed_seconds,
            )
            run_dir.write_status(
                "failed",
                returncode=result.returncode,
                elapsed_seconds=result.elapsed_seconds,
            )
            return run_dir

        try:
            normalized = self.normalize_output(spec, raw_dir)
            word_total = len(normalized.survey_text.split())
            reference_total = len(normalized.references)
            self.write_normalized(run_dir, normalized)
        except Exception as exc:
            run_dir.event("normalization_failed", error=type(exc).__name__)
            run_dir.write_status(
                "failed",
                failure_stage="normalization",
                error=f"{type(exc).__name__}: {exc}",
            )
            return run_dir
        run_dir.finalize_success(
            elapsed_seconds=result.elapsed_seconds,
            word_count=word_total,
            n_references=reference_total,
            length_compliant=spec.topic.min_words <= word_total <= spec.topic.max_words,
            reference_compliant=(
                spec.topic.target_references_min
                <= reference_total
                <= spec.topic.target_references_max
            ),
        )
        return run_dir

    @staticmethod
    def write_normalized(run_dir: RunDirectory, value: NormalizedArtifacts) -> None:
        atomic_write_text(run_dir.root / "survey.md", value.survey_text.rstrip() + "\n")
        atomic_write_json(run_dir.root / "references.json", value.references)
        atomic_write_json(run_dir.root / "retrieval_manifest.json", value.retrieval_manifest)
        atomic_write_json(run_dir.root / "trace_or_stage_log.json", value.stage_log)
        atomic_write_json(run_dir.root / "usage.json", value.usage)
        atomic_write_json(run_dir.root / "meta.json", value.meta)


def load_json_if_present(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def source_file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
