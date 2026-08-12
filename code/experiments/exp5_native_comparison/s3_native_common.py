#!/usr/bin/env python3
"""Shared contracts for native-flow S3 system wrappers.

This module deliberately contains no system-specific generation logic. Each
wrapper must invoke the released workflow and adapt only inputs, endpoints,
logging and output capture allowed by the frozen protocol.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "1.0"
TERMINAL_STATUSES = {"ok", "failed", "budget_exceeded", "timeout", "ineligible"}
SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|authorization)([=:\s]+)(\S+)"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
)


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def redact(value: str) -> str:
    redacted = value
    for pattern in SECRET_PATTERNS:
        if pattern.groups >= 3:
            redacted = pattern.sub(r"\1\2<REDACTED>", redacted)
        else:
            redacted = pattern.sub("<REDACTED>", redacted)
    return redacted


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@dataclass(frozen=True)
class TopicIntent:
    topic_id: str
    title: str
    scope: str
    target_audience: str
    language: str
    publication_cutoff: str
    target_words: int
    min_words: int
    max_words: int
    target_references_min: int
    target_references_max: int
    seed: int
    analysis_role: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TopicIntent":
        return cls(**{field: value[field] for field in cls.__dataclass_fields__})

    def frozen_hash(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    system_id: str
    source_revision: str
    model_policy: str
    topic: TopicIntent
    budget_usd: float
    timeout_seconds: int
    protocol_hash: str

    def frozen_hash(self) -> str:
        return sha256_json(
            {
                **asdict(self),
                "topic": asdict(self.topic),
            }
        )


class RunDirectory:
    """Atomic status and artifact contract for one system-topic run."""

    REQUIRED_ARTIFACTS = (
        "run_spec.json",
        "preflight.json",
        "survey.md",
        "references.json",
        "retrieval_manifest.json",
        "trace_or_stage_log.json",
        "usage.json",
        "meta.json",
    )

    def __init__(self, root: Path, spec: RunSpec):
        self.root = root / "systems" / spec.system_id / safe_name(spec.topic.topic_id)
        self.spec = spec
        self.status_path = self.root / "status.json"
        self.events_path = self.root / "events.jsonl"
        self.log_path = self.root / "runner.log"

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.root / "run_spec.json", self.serialized_spec())
        self.event("initialized", run_spec_hash=self.spec.frozen_hash())
        self.write_status("pending")

    def serialized_spec(self) -> dict[str, Any]:
        value = asdict(self.spec)
        value["topic"] = asdict(self.spec.topic)
        value["run_spec_hash"] = self.spec.frozen_hash()
        return value

    def event(self, event: str, **fields: Any) -> None:
        append_jsonl(
            self.events_path,
            {
                "at": utc_timestamp(),
                "event": event,
                **fields,
            },
        )

    def write_status(self, status: str, **fields: Any) -> None:
        if status not in TERMINAL_STATUSES | {"pending", "running"}:
            raise ValueError(f"Unknown S3 status: {status}")
        atomic_write_json(
            self.status_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": status,
                "updated_at": utc_timestamp(),
                "run_spec_hash": self.spec.frozen_hash(),
                **fields,
            },
        )

    def completed_and_valid(self) -> bool:
        if not self.status_path.exists():
            return False
        status = json.loads(self.status_path.read_text(encoding="utf-8"))
        if status.get("status") != "ok":
            return False
        if status.get("run_spec_hash") != self.spec.frozen_hash():
            return False
        manifest_path = self.root / "artifact_manifest.json"
        if not manifest_path.is_file():
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if status.get("artifact_manifest_sha256") != sha256_json(manifest):
            return False
        files = manifest.get("files") or {}
        for name in self.REQUIRED_ARTIFACTS:
            path = self.root / name
            record = files.get(name) or {}
            if not path.is_file() or record.get("sha256") != sha256_file(path):
                return False
        return True

    def artifact_manifest(self) -> dict[str, Any]:
        files = {}
        for name in self.REQUIRED_ARTIFACTS:
            path = self.root / name
            if path.is_file():
                files[name] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": utc_timestamp(),
            "run_spec_hash": self.spec.frozen_hash(),
            "files": files,
        }

    def finalize_success(self, **fields: Any) -> None:
        manifest = self.artifact_manifest()
        atomic_write_json(self.root / "artifact_manifest.json", manifest)
        self.event("completed", artifact_count=len(manifest["files"]))
        self.write_status("ok", artifact_manifest_sha256=sha256_json(manifest), **fields)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    elapsed_seconds: float
    timed_out: bool


def run_logged_command(
    command: Iterable[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout_seconds: int,
    env: dict[str, str] | None = None,
) -> CommandResult:
    """Run one released-system command without persisting environment secrets."""

    argv = [str(value) for value in command]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timed_out = False
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{utc_timestamp()}] command={redact(' '.join(argv))}\n")
        log.flush()
        process = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                returncode = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait()
        elapsed = time.monotonic() - started
        log.write(
            f"[{utc_timestamp()}] returncode={returncode} "
            f"elapsed_seconds={elapsed:.3f} timed_out={timed_out}\n"
        )
    return CommandResult(
        returncode=returncode,
        elapsed_seconds=elapsed,
        timed_out=timed_out,
    )
