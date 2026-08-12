#!/usr/bin/env python3
"""Native wrapper for the best validated SLRGP deployment."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from s3_native_common import RunSpec, safe_name, sha256_file, utc_timestamp
from system_wrapper.base import NativeSystemWrapper, NormalizedArtifacts, PreflightReport


class SLRGPWrapper(NativeSystemWrapper):
    system_id = "slrgp"

    def __init__(
        self,
        source_dir: Path,
        *,
        runner_path: Path,
        python_executable: Path,
        chat_base_url: str,
        model: str,
    ):
        super().__init__(source_dir)
        self.runner_path = runner_path
        self.python_executable = python_executable
        self.chat_base_url = chat_base_url
        self.model = model

    def preflight(self, spec: RunSpec) -> PreflightReport:
        issues: list[str] = []
        entrypoint_present = self.runner_path.is_file()
        if not entrypoint_present:
            issues.append("slrgp_runner_missing")
        dependencies_present = self.python_executable.is_file()
        if not dependencies_present:
            issues.append("python_environment_missing")
        source_files = [
            self.runner_path,
            self.source_dir / "code/slrgp/pipeline_real.py",
            self.source_dir / "code/slrgp/control.py",
        ]
        missing_source_files = [str(path) for path in source_files if not path.is_file()]
        source_clean = not missing_source_files
        if missing_source_files:
            issues.append("slrgp_source_files_missing")
        return PreflightReport(
            system_id=self.system_id,
            checked_at=utc_timestamp(),
            source_revision=spec.source_revision,
            source_clean=source_clean,
            entrypoint_present=entrypoint_present,
            dependencies_present=dependencies_present,
            native_topic_to_survey=True,
            common_cutoff_supported=True,
            usage_logging_supported=True,
            eligible=not issues,
            issues=issues,
            details={
                "runner": str(self.runner_path),
                "model": self.model,
                "chat_base_url": self.chat_base_url,
                "source_artifacts": {
                    str(path.relative_to(self.source_dir)): sha256_file(path)
                    for path in source_files
                    if path.is_file()
                },
                "missing_source_files": missing_source_files,
                "instantiation": "E_default-L_validated-F_default-R_learned-O_learned",
            },
        )

    def build_command(self, spec: RunSpec, raw_dir: Path) -> list[str]:
        cutoff_year = int(spec.topic.publication_cutoff[:4])
        return [
            str(self.python_executable),
            str(self.runner_path),
            "--topic",
            spec.topic.title,
            "--out-dir",
            str(raw_dir),
            "--base-url",
            self.chat_base_url,
            "--model",
            self.model,
            "--top-n",
            "300",
            "--min-abs-len",
            "100",
            "--d-max",
            "2",
            "--theta-leaf",
            "8",
            "--target-leaf-words",
            "280",
            "--target-words",
            str(spec.topic.target_words),
            "--min-words",
            str(spec.topic.min_words),
            "--max-words",
            str(spec.topic.max_words),
            "--min-references",
            str(spec.topic.target_references_min),
            "--max-references",
            str(spec.topic.target_references_max),
            "--evidence-cards",
            "120",
            "--publication-cutoff-year",
            str(cutoff_year),
        ]

    def normalize_output(self, spec: RunSpec, raw_dir: Path) -> NormalizedArtifacts:
        topic_dir = raw_dir / safe_name(spec.topic.title)
        survey_path = topic_dir / "survey.md"
        if not survey_path.is_file():
            raise FileNotFoundError(f"SLRGP survey missing: {survey_path}")
        survey = survey_path.read_text(encoding="utf-8").strip()
        meta = load_json(topic_dir / "meta.json", {})
        if not meta.get("ok"):
            raise ValueError(f"SLRGP native run failed: {meta.get('error', 'unknown')}")
        refs_value = load_json(topic_dir / "ref.json", {})
        references = [
            {"source_key": str(key), **(value if isinstance(value, dict) else {"value": value})}
            for key, value in refs_value.items()
        ]
        evidence = load_json(topic_dir / "evidence_package.json", [])
        leaf_evidence = load_json(topic_dir / "leaf_evidence_provenance.json", {})
        trace = load_json(topic_dir / "trace.json", {})
        summary = load_json(raw_dir / "_run_summary.json", {})
        usage = summary.get("llm_usage") or meta.get("llm_usage_so_far") or {}
        usage.setdefault("provider_cost_usd", None)
        usage.setdefault("provider_cost_observed", False)
        return NormalizedArtifacts(
            survey_text=survey,
            references=references,
            retrieval_manifest={
                "system": self.system_id,
                "publication_cutoff": spec.topic.publication_cutoff,
                "candidate_count": meta.get("n_candidates"),
                "evidence_cards": evidence,
                "leaf_evidence": leaf_evidence,
                "final_reference_ids": [row.get("doc_id") for row in references],
            },
            stage_log=trace,
            usage=usage,
            meta={
                **meta,
                "system": self.system_id,
                "source_revision": spec.source_revision,
                "model": self.model,
                "normalized_at": utc_timestamp(),
            },
        )


def load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))
