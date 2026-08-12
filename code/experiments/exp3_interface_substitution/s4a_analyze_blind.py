#!/usr/bin/env python3
"""Analyze blind Instrument-J scores for matched S4a."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path


DIMENSIONS = (
    "organizational_quality",
    "critical_synthesis",
    "global_coherence",
    "citation_plausibility",
)
CONTRASTS = (
    ("O semantic vs rank slab", "intact", "o_rank_slab_matched"),
    ("Recursion vs flat", "intact", "flat_no_reentry"),
    ("Guarded V vs unguarded stress", "v_guarded_stress", "v_unguarded_stress"),
)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(diffs: list[float], n_boot: int = 10000, seed: int = 20260713) -> list[float]:
    if not diffs:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    vals = []
    n = len(diffs)
    for _ in range(n_boot):
        vals.append(mean([diffs[rng.randrange(n)] for _ in range(n)]))
    vals.sort()
    return [vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot)]]


def exact_signflip_p(diffs: list[float]) -> float:
    nz = [d for d in diffs if abs(d) > 1e-12]
    if not nz:
        return 1.0
    obs = abs(mean(nz))
    ge = 0
    total = 0
    for signs in itertools.product((-1, 1), repeat=len(nz)):
        total += 1
        val = abs(mean([s * abs(d) for s, d in zip(signs, nz)]))
        if val >= obs - 1e-12:
            ge += 1
    return ge / total


def effect_sign_biserial(diffs: list[float]) -> float:
    nz = [d for d in diffs if abs(d) > 1e-12]
    if not nz:
        return 0.0
    return (sum(d > 0 for d in nz) - sum(d < 0 for d in nz)) / len(nz)


def fmt(x: float) -> str:
    if math.isnan(x):
        return "NA"
    return f"{x:.3f}"


def add_bh_qvalues(rows: list[dict]) -> None:
    families = defaultdict(list)
    for index, row in enumerate(rows):
        families[row["endpoint"]].append((index, row["p_signflip"]))
    for items in families.values():
        ordered = sorted(items, key=lambda item: item[1])
        m = len(ordered)
        qs = [1.0] * m
        running = 1.0
        for rank_from_end, (idx, p) in enumerate(reversed(ordered), 1):
            rank = m - rank_from_end + 1
            running = min(running, p * m / rank)
            qs[rank - 1] = running
        for (idx, _), q in zip(ordered, qs):
            rows[idx]["q_bh_by_endpoint"] = min(1.0, q)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    blind_dir = args.root / "blind_judging"
    mapping = json.loads((blind_dir / "blind_mapping.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (blind_dir / "blind_scores.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    row_by_key = {(r["topic"], r["arm"]): r for r in audit["rows"]}

    scored = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("error"):
            continue
        meta = mapping[row["blind_id"]]
        key = (meta["topic"], meta["arm"])
        for dim in DIMENSIONS:
            scored[key][dim].append(float(row[f"{dim}_score"]))

    panel = {}
    for key, dims in scored.items():
        dim_means = {dim: mean(vals) for dim, vals in dims.items()}
        panel[key] = dim_means | {"overall": mean(list(dim_means.values()))}

    results = []
    for label, arm_a, arm_b in CONTRASTS:
        for endpoint in ("overall",) + DIMENSIONS:
            diffs = []
            topic_rows = []
            for topic in audit["summary"]["topics"]:
                a_meta = row_by_key.get((topic, arm_a))
                b_meta = row_by_key.get((topic, arm_b))
                if not a_meta or not b_meta:
                    continue
                if not (a_meta["primary_eligible"] and b_meta["primary_eligible"]):
                    continue
                a = panel[(topic, arm_a)][endpoint]
                b = panel[(topic, arm_b)][endpoint]
                diffs.append(a - b)
                topic_rows.append({"topic": topic, "a": a, "b": b, "diff": a - b})
            wins = sum(d > 1e-12 for d in diffs)
            losses = sum(d < -1e-12 for d in diffs)
            ties = len(diffs) - wins - losses
            results.append({
                "contrast": label,
                "arm_a": arm_a,
                "arm_b": arm_b,
                "endpoint": endpoint,
                "n": len(diffs),
                "mean_a": mean([r["a"] for r in topic_rows]),
                "mean_b": mean([r["b"] for r in topic_rows]),
                "mean_diff": mean(diffs),
                "ci95": bootstrap_ci(diffs),
                "p_signflip": exact_signflip_p(diffs),
                "sign_biserial": effect_sign_biserial(diffs),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "topics": topic_rows,
            })
    add_bh_qvalues(results)

    output = {
        "root": str(args.root),
        "n_scores": len(rows),
        "n_valid_scores": sum(1 for row in rows if not row.get("error")),
        "dimensions": DIMENSIONS,
        "contrasts": results,
    }
    out_json = args.out_json or args.root / "S4A_BLIND_STATS.json"
    out_md = args.out_md or args.root / "S4A_BLIND_STATS.md"
    out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# S4a matched blind statistics",
        "",
        "Positive delta means arm A scored higher than arm B. Scores are topic-level panel means over three blind judges.",
        "",
        "| Contrast | Endpoint | n | A mean | B mean | Δ | 95% bootstrap CI | exact p | BH q | sign-biserial | W/T/L |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        ci = row["ci95"]
        lines.append(
            f"| {row['contrast']} | {row['endpoint']} | {row['n']} | {fmt(row['mean_a'])} | {fmt(row['mean_b'])} | "
            f"{fmt(row['mean_diff'])} | [{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(row['p_signflip'])} | "
            f"{fmt(row['q_bh_by_endpoint'])} | {fmt(row['sign_biserial'])} | {row['wins']}/{row['ties']}/{row['losses']} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
