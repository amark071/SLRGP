#!/usr/bin/env python3
"""Create deterministic front/middle/rear excerpts for length sensitivity."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SYSTEMS = ("slrgp", "autosurvey", "surveyforge", "surveygen")


def excerpt(text: str, target: int) -> str:
    tokens = list(re.finditer(r"\S+", text))
    if len(tokens) <= target:
        return text.rstrip() + "\n"
    per_window = target // 3
    starts = (
        0,
        max(0, len(tokens) // 2 - per_window // 2),
        len(tokens) - per_window,
    )
    windows = []
    for start in starts:
        first = tokens[start].start()
        last = tokens[start + per_window - 1].end()
        windows.append(text[first:last].strip())
    return (
        "# Registered common-window excerpt\n\n"
        + windows[0]
        + "\n\n[... deterministic omitted span ...]\n\n"
        + windows[1]
        + "\n\n[... deterministic omitted span ...]\n\n"
        + windows[2]
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--topics", type=Path, required=True)
    parser.add_argument("--target-words", type=int, default=4000)
    args = parser.parse_args()
    topics = json.loads(args.topics.read_text(encoding="utf-8"))["topics"]
    out_root = args.root / "common_window"
    manifest = []
    for topic in topics:
        for system in SYSTEMS:
            source = (
                args.root / "systems" / system / topic["topic_id"] / "survey.md"
            )
            text = source.read_text(encoding="utf-8")
            output = out_root / "systems" / system / topic["topic_id"] / "survey.md"
            output.parent.mkdir(parents=True, exist_ok=True)
            value = excerpt(text, args.target_words)
            output.write_text(value, encoding="utf-8")
            manifest.append(
                {
                    "topic_id": topic["topic_id"],
                    "system": system,
                    "source_words": len(text.split()),
                    "excerpt_words": len(value.split()),
                    "rule": "full text when <= target; otherwise equal front/middle/rear word windows",
                }
            )
    (out_root / "excerpt_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
