#!/usr/bin/env python3
"""Progress summary for S4b Qwen robustness runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--topics-file", required=True, type=Path)
    parser.add_argument("--arms", required=True)
    parser.add_argument("--write-json", type=Path, default=None)
    args = parser.parse_args()

    topics = [line.strip() for line in args.topics_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    rows = []
    for topic in topics:
        tdir = args.out_dir / topic.replace(" ", "_").replace("/", "_")
        bundle_ok = (tdir / "frozen_bundle.json").exists()
        for arm in arms:
            meta_path = tdir / arm / "meta.json"
            ok = False
            words = None
            error = ""
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    ok = bool(meta.get("ok"))
                    words = meta.get("word_count")
                    error = str(meta.get("error", ""))
                except Exception as exc:
                    error = repr(exc)
            rows.append({"topic": topic, "arm": arm, "bundle": bundle_ok, "ok": ok, "words": words, "error": error})

    summary = {
        "out_dir": str(args.out_dir),
        "n_expected": len(topics) * len(arms),
        "n_ok": sum(1 for row in rows if row["ok"]),
        "n_meta": sum(1 for row in rows if (args.out_dir / row["topic"].replace(" ", "_").replace("/", "_") / row["arm"] / "meta.json").exists()),
        "rows": rows,
    }
    if args.write_json:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"n_ok": summary["n_ok"], "n_expected": summary["n_expected"], "n_meta": summary["n_meta"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
