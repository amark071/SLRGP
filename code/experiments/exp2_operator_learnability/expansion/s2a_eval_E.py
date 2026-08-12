#!/usr/bin/env python3
"""Evaluate S2a/E retrieval expansions against the frozen S2b title+abstract baseline."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from stats_utils import paired_bootstrap_ci, paired_test  # noqa: E402

BASE_DIR = Path(os.environ.get("S2B_UNION_DIR", "work/exp2_operator_learnability/ranking/union_L"))
E_DIR = Path(os.environ.get("S2A_OUT_DIR", "data/exp2_operator_learnability/expansion"))
OUT_DIR = Path(os.environ.get("S2A_RESULT_DIR", "work/exp2_operator_learnability/expansion"))
PRIMARY = 300


def load_p() -> dict[tuple[str, str], int]:
    raw = json.load(open(BASE_DIR / "P_of.json", encoding="utf-8"))
    return {(a, x["split"]): x["P"] for a, x in raw.items()}


def per_review(df: pd.DataFrame, split: str, p_of: dict, depth: int) -> dict[str, float]:
    groups = {a: g for a, g in df.groupby("arxiv_id")}
    out = {}
    for (aid, sp), den in p_of.items():
        if sp != split or not den:
            continue
        g = groups.get(aid)
        hit = int(g.nlargest(depth, "rrf_score")["label"].sum()) if g is not None else 0
        out[aid] = hit / den
    return out


def summary(per: dict[str, float]) -> float:
    return float(np.mean(list(per.values()))) if per else 0.0


def eval_variant(base: pd.DataFrame, expanded: pd.DataFrame, split: str, p_of: dict, n_terms: int) -> dict:
    out = {"n_terms": n_terms, "depths": {}, "retrieval_ceiling": {}}
    for depth in [100, 300, 500, 1000]:
        b = per_review(base, split, p_of, depth)
        e = per_review(expanded, split, p_of, depth)
        aids = sorted(set(b) & set(e))
        diffs = [e[a] - b[a] for a in aids]
        out["depths"][str(depth)] = {
            "baseline_macro": summary(b), "expanded_macro": summary(e),
            "mean_diff": float(np.mean(diffs)),
        }
        if depth == PRIMARY:
            ci = paired_bootstrap_ci(diffs, n_boot=10000)
            pt = paired_test(diffs)
            out["primary_paired_expanded_vs_baseline"] = {
                "n_reviews": len(aids), "mean_diff": ci.point_estimate,
                "ci_low": ci.ci_low, "ci_high": ci.ci_high, "ci_method": ci.method,
                "wilcoxon_p": pt.wilcoxon_p, "permutation_p": pt.permutation_p,
                "rank_biserial": pt.rank_biserial,
            }
    for name, df in [("baseline", base), ("expanded", expanded)]:
        pooled = {a: int(g["label"].sum()) for a, g in df.groupby("arxiv_id")}
        vals = [pooled.get(a, 0) / den for (a, sp), den in p_of.items() if sp == split and den]
        out["retrieval_ceiling"][name] = float(np.mean(vals)) if vals else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--term-counts", default="3,6,10")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = [int(x) for x in args.term_counts.split(",")]
    p_of = load_p()
    base = pd.read_parquet(BASE_DIR / f"{args.split}.parquet")
    report = {"split": args.split, "primary": "macro Coverage@300", "variants": {}}
    for n in counts:
        expanded = pd.read_parquet(E_DIR / f"{args.split}_terms{n}.parquet")
        report["variants"][str(n)] = eval_variant(base, expanded, args.split, p_of, n)

    if args.split == "val":
        best = max(counts, key=lambda n: report["variants"][str(n)]["depths"]["300"]["expanded_macro"])
        report["selection_rule"] = "maximize validation expanded macro Coverage@300; tie -> fewer terms"
        report["selected_term_count"] = best
        with open(E_DIR / "frozen_hyperparameters.json", "w", encoding="utf-8") as f:
            json.dump({"selected_term_count": best, "selection_split": "val",
                       "criterion": "expanded macro Coverage@300"}, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / f"s2a_eval_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
