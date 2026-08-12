#!/usr/bin/env python3
"""Create a fixed, seed-recorded stratified subsample for cost-bounded F training."""
import argparse
import json
from collections import Counter

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--per-stratum", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.input, encoding="utf-8") if line.strip()]
    rng = np.random.default_rng(args.seed)
    by = {}
    for r in rows:
        by.setdefault(r["stratum"], []).append(r)
    chosen = []
    for stratum in sorted(by):
        source = by[stratum]
        n = min(args.per_stratum, len(source))
        chosen.extend(source[i] for i in rng.choice(len(source), size=n, replace=False))
    rng.shuffle(chosen)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in chosen:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({
        "n": len(chosen), "per_stratum_requested": args.per_stratum,
        "seed": args.seed, "by_stratum": Counter(r["stratum"] for r in chosen),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
