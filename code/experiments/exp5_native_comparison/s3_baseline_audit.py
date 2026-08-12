#!/usr/bin/env python3
"""Audit provenance and executable surface of cloned S3 baseline repositories."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from s3_native_common import atomic_write_json, sha256_file, utc_timestamp


REPOSITORY_DIRS = {
    "autosurvey": "autosurvey",
    "surveyforge": "surveyforge",
    "itersurvey": "itersurvey",
    "surveygen": "surveygen",
    "surveyg": "surveyg",
    "arise": "arise",
    "lira": "lira",
    "autosurvey2": "autosurvey2",
}
ENTRYPOINT_NAMES = {
    "main.py",
    "run.py",
    "run.sh",
    "pipeline.py",
    "paper_retrieval.py",
    "survey_generation.py",
    "outline_generation.py",
}
DEPENDENCY_NAMES = {
    "requirements.txt",
    "environment.yml",
    "environment.yaml",
    "pyproject.toml",
    "setup.py",
    "Pipfile",
    "Dockerfile",
}
LICENSE_PREFIXES = ("license", "licence", "copying")


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def relative_matches(repo: Path, predicate) -> list[str]:
    return sorted(str(path.relative_to(repo)) for path in repo.rglob("*") if path.is_file() and predicate(path))


def audit_repo(system_id: str, repo: Path) -> dict[str, Any]:
    if not (repo / ".git").is_dir():
        return {
            "system_id": system_id,
            "repo": str(repo),
            "present": False,
            "eligible_for_source_freeze": False,
            "issues": ["missing_git_checkout"],
        }
    issues = []
    try:
        commit = git(repo, "rev-parse", "HEAD")
        branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
        status = git(repo, "status", "--short")
        remotes = git(repo, "remote", "-v").splitlines()
    except subprocess.CalledProcessError as exc:
        return {
            "system_id": system_id,
            "repo": str(repo),
            "present": True,
            "eligible_for_source_freeze": False,
            "issues": [f"git_error:{exc.returncode}"],
        }
    if status:
        issues.append("dirty_worktree")
    entrypoints = relative_matches(repo, lambda p: p.name in ENTRYPOINT_NAMES)
    dependencies = relative_matches(repo, lambda p: p.name in DEPENDENCY_NAMES)
    licenses = relative_matches(repo, lambda p: p.name.lower().startswith(LICENSE_PREFIXES))
    readmes = relative_matches(repo, lambda p: p.name.lower().startswith("readme"))
    if not entrypoints:
        issues.append("no_known_entrypoint")
    if not dependencies:
        issues.append("no_dependency_manifest")
    if not licenses:
        issues.append("no_license_file")
    manifest_files = sorted(set(dependencies + licenses + readmes))
    hashes = {name: sha256_file(repo / name) for name in manifest_files}
    return {
        "system_id": system_id,
        "repo": str(repo),
        "present": True,
        "commit": commit,
        "branch": branch,
        "remotes": remotes,
        "dirty": bool(status),
        "status_lines": status.splitlines(),
        "entrypoints": entrypoints,
        "dependency_manifests": dependencies,
        "license_files": licenses,
        "readmes": readmes,
        "manifest_sha256": hashes,
        "eligible_for_source_freeze": not any(
            issue in {"dirty_worktree", "no_known_entrypoint", "no_dependency_manifest"}
            for issue in issues
        ),
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("work/exp5_native_comparison/baselines"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        audit_repo(system_id, args.repo_root / dirname)
        for system_id, dirname in REPOSITORY_DIRS.items()
    ]
    report = {
        "schema_version": "1.0",
        "audited_at": utc_timestamp(),
        "repo_root": str(args.repo_root),
        "systems": rows,
        "n_present": sum(row["present"] for row in rows),
        "n_source_freeze_ready": sum(row["eligible_for_source_freeze"] for row in rows),
    }
    atomic_write_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
