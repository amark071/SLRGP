#!/usr/bin/env python3
"""Thin native wrapper for the released ARISE workflow."""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from s3_native_common import RunSpec, utc_timestamp
from system_wrapper.base import (
    NativeSystemWrapper,
    NormalizedArtifacts,
    PreflightReport,
)


class AriseWrapper(NativeSystemWrapper):
    system_id = "arise"
    preserve_partial_raw = True

    def __init__(
        self,
        source_dir: Path,
        *,
        driver_path: Path,
        cutoff_filter_path: Path,
        python_executable: Path,
        chat_base_url: str,
        model: str,
        max_refinement_rounds: int = 2,
    ):
        super().__init__(source_dir)
        self.driver_path = driver_path
        self.cutoff_filter_path = cutoff_filter_path
        self.python_executable = python_executable
        self.chat_base_url = chat_base_url
        self.model = model
        self.max_refinement_rounds = max_refinement_rounds

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
                f"ARISE_Source_Code/{name}"
                for name in (
                    "1_citation.py",
                    "2deduplicated.py",
                    "3_summary.py",
                    "4merge.py",
                    "5writing-Copy1.py",
                    "6formatting.py",
                    "arl_refinement.py",
                )
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
        scripts_dir = self.source_dir / "ARISE_Source_Code"
        required_scripts = (
            "1_citation.py",
            "2deduplicated.py",
            "3_summary.py",
            "4merge.py",
            "5writing-Copy1.py",
            "6formatting.py",
            "arl_refinement.py",
        )
        missing_scripts = [
            name for name in required_scripts if not (scripts_dir / name).is_file()
        ]
        if missing_scripts:
            issues.append("released_pipeline_scripts_missing")
        if not self.driver_path.is_file() or not self.cutoff_filter_path.is_file():
            issues.append("execution_or_cutoff_adapter_missing")
        if not self.python_executable.is_file():
            issues.append("python_environment_missing")
        if not os.environ.get("SERPER_API_KEY"):
            issues.append("serper_api_key_missing")
        if shutil.which("pdflatex") is None or shutil.which("bibtex") is None:
            issues.append("latex_toolchain_missing")
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
                "cutoff_filter_path": str(self.cutoff_filter_path),
                "serper_configured": bool(os.environ.get("SERPER_API_KEY")),
                "chat_base_url": self.chat_base_url,
                "max_refinement_rounds": self.max_refinement_rounds,
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
            "--cutoff-filter",
            str(self.cutoff_filter_path),
            "--topic",
            spec.topic.title,
            "--publication-cutoff",
            spec.topic.publication_cutoff,
            "--model",
            self.model,
            "--api-base-url",
            self.chat_base_url,
            "--paper-count",
            "15",
            "--max-refinement-rounds",
            str(self.max_refinement_rounds),
        ]

    def normalize_output(self, spec: RunSpec, raw_dir: Path) -> NormalizedArtifacts:
        final_meta = json.loads((raw_dir / "final_output.json").read_text(encoding="utf-8"))
        tex_path = raw_dir / final_meta["final_tex"]
        citation_path = raw_dir / final_meta["citation_csv"]
        if not tex_path.is_file() or not citation_path.is_file():
            raise FileNotFoundError("ARISE final output or citation manifest missing")
        survey = latex_to_markdown(tex_path.read_text(encoding="utf-8", errors="replace"))
        references = load_citations(citation_path)
        cutoff_audit = json.loads(
            (raw_dir / "cutoff_audit.json").read_text(encoding="utf-8")
        )
        return NormalizedArtifacts(
            survey_text=survey,
            references=references,
            retrieval_manifest={
                "system": self.system_id,
                "publication_cutoff": spec.topic.publication_cutoff,
                "cutoff_audit": cutoff_audit,
                "final_reference_ids": [
                    row.get("citation") for row in references
                ],
            },
            stage_log={
                "system": self.system_id,
                "native_stages": [
                    "agentic journal and paper search",
                    "citation deduplication",
                    "paper-content summarization",
                    "outline merging",
                    "survey writing",
                    "citation formatting",
                    "three-role rubric review and iterative refinement",
                ],
                "stage_checkpoints": sorted(
                    path.name
                    for path in (raw_dir / "stage_checkpoints").glob("*.done")
                ),
                "raw_output_file": final_meta["final_tex"],
            },
            usage={
                "source": "central_relay_ledger",
                "note": "Per-call usage is reconciled from the serial pilot interval.",
            },
            meta={
                "system": self.system_id,
                "topic": spec.topic.title,
                "source_revision": spec.source_revision,
                "model_alias": self.model,
                "normalized_at": utc_timestamp(),
                "raw_output": final_meta["final_tex"],
                "max_refinement_rounds": self.max_refinement_rounds,
                "deterministic_renderer": "latex_to_markdown_v1",
            },
        )

    def environment(self, spec: RunSpec, raw_dir: Path) -> dict[str, str]:
        env = super().environment(spec, raw_dir)
        env.update(
            {
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "LITELLM_LOG": "ERROR",
            }
        )
        return env


def load_citations(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {"index": row.get("Index"), "citation": row.get("Citation")}
            for row in csv.DictReader(handle)
        ]


def latex_to_markdown(value: str) -> str:
    value = re.sub(
        r"\\(?:section|subsection|subsubsection)\*?\{([^{}]*)\}",
        lambda match: "\n\n" + {"section": "##", "subsection": "###"}.get(
            match.group(0).split("{", 1)[0].lstrip("\\").rstrip("*"), "####"
        )
        + " "
        + match.group(1)
        + "\n",
        value,
    )
    value = re.sub(r"\\cite[tp]?\{([^{}]+)\}", r"[\1]", value)
    value = re.sub(r"\\(?:textbf|textit|emph)\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"(?s)\\begin\{(?:document|abstract)\}|\\end\{(?:document|abstract)\}", "", value)
    value = re.sub(r"(?m)^\\(?:documentclass|usepackage|bibliography|bibliographystyle).*$", "", value)
    value = re.sub(r"(?m)^%.*$", "", value)
    value = re.sub(r"\\(?:label|ref)\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?", "", value)
    value = value.replace("\\&", "&").replace("\\%", "%").replace("\\_", "_")
    value = re.sub(r"[{}]", "", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()
