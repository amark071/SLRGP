#!/usr/bin/env python3
"""Paired analysis of six-topic long-form structural SLRGP blind scores."""
from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

from s5a_blind_judge import DIMENSIONS


TOPICS = (
    "3D Gaussian Splatting",
    "Graph Neural Networks",
    "Multimodal Large Language Models",
    "Retrieval-Augmented Generation for Large Language Models",
    "Federated Learning",
    "Efficient Inference for Large Language Models",
)
ENDPOINTS = ("overall",) + DIMENSIONS
CONTRASTS = (
    ("structural vs naive chunking", "structural_lotcf_recursive", "naive_recursive_chunking"),
    ("structural vs fixed outline", "structural_lotcf_recursive", "fixed_outline_no_reentry"),
    ("structural vs flat", "structural_lotcf_recursive", "flat_single_pass"),
)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def bootstrap_ci(diffs: list[float], n_boot: int = 20_000, seed: int = 20260714) -> list[float]:
    rng = random.Random(seed)
    samples = sorted(mean([diffs[rng.randrange(len(diffs))] for _ in diffs]) for _ in range(n_boot))
    return [samples[int(0.025 * n_boot)], samples[int(0.975 * n_boot)]]


def signflip_p(diffs: list[float]) -> float:
    nonzero = [value for value in diffs if abs(value) > 1e-12]
    if not nonzero:
        return 1.0
    observed = abs(mean(nonzero))
    values = [
        abs(mean([sign * abs(value) for sign, value in zip(signs, nonzero)]))
        for signs in itertools.product((-1, 1), repeat=len(nonzero))
    ]
    return sum(value >= observed - 1e-12 for value in values) / len(values)


def sign_biserial(diffs: list[float]) -> float:
    nonzero = [value for value in diffs if abs(value) > 1e-12]
    return 0.0 if not nonzero else (sum(value > 0 for value in nonzero) - sum(value < 0 for value in nonzero)) / len(nonzero)


def bh_fdr(rows: list[dict]) -> None:
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["p_signflip"])
    n = len(rows)
    running = 1.0
    for rank, (index, row) in reversed(list(enumerate(ordered, 1))):
        adjusted = min(running, row["p_signflip"] * n / rank)
        rows[index]["p_bh_fdr"] = adjusted
        running = adjusted


def load_panel(blind_dir: Path) -> tuple[dict, dict]:
    mapping = json.loads((blind_dir / "blind_mapping.json").read_text(encoding="utf-8"))
    latest = {}
    raw_rows = 0
    for line in (blind_dir / "blind_scores.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            raw_rows += 1
            row = json.loads(line)
            latest[(row["blind_id"], row["judge"])] = row
    grouped = defaultdict(lambda: defaultdict(list))
    for row in latest.values():
        if row.get("error"):
            continue
        meta = mapping[row["blind_id"]]
        key = (meta["topic"], meta["arm"])
        for dim in DIMENSIONS:
            grouped[key][dim].append(float(row[f"{dim}_score"]))
    panel = {
        key: {dim: mean(values) for dim, values in dims.items()} | {"overall": mean([mean(values) for values in dims.values()])}
        for key, dims in grouped.items()
    }
    return panel, {"raw_attempts": raw_rows, "latest_records": len(latest), "valid_latest": sum(not row.get("error") for row in latest.values())}


def fmt(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if audit["n_primary_eligible"] != len(TOPICS):
        raise SystemExit("Structural design audit is not fully eligible.")
    panel, scoring = load_panel(args.blind_dir)
    rows = []
    for label, arm_a, arm_b in CONTRASTS:
        for endpoint in ENDPOINTS:
            topic_rows = []
            for topic in TOPICS:
                a = panel[(topic, arm_a)][endpoint]
                b = panel[(topic, arm_b)][endpoint]
                topic_rows.append({"topic": topic, "structural": a, "control": b, "diff": a - b})
            diffs = [row["diff"] for row in topic_rows]
            rows.append({
                "contrast": label,
                "arm_a": arm_a,
                "arm_b": arm_b,
                "endpoint": endpoint,
                "n": len(diffs),
                "mean_structural": mean([row["structural"] for row in topic_rows]),
                "mean_control": mean([row["control"] for row in topic_rows]),
                "mean_diff": mean(diffs),
                "ci95": bootstrap_ci(diffs),
                "p_signflip": signflip_p(diffs),
                "sign_biserial": sign_biserial(diffs),
                "wins": sum(value > 1e-12 for value in diffs),
                "ties": sum(abs(value) <= 1e-12 for value in diffs),
                "losses": sum(value < -1e-12 for value in diffs),
                "topics": topic_rows,
            })
    bh_fdr(rows)
    output = {"scoring": scoring, "audit": audit, "contrasts": rows}
    args.out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# S5a Structural-Recursion Long-Form Analysis",
        "",
        "All estimates use a same-batch, randomized, blinded three-model Instrument-J panel over six topics. Positive delta favors structural LOTCF-LR recursion.",
        "",
        f"- Design audit: {audit['n_primary_eligible']}/{audit['n_topics']} topics primary eligible.",
        f"- Blind scores: {scoring['valid_latest']}/{scoring['latest_records']} valid latest records ({scoring['raw_attempts']} raw attempts).",
        "",
        "| Contrast | Endpoint | n | Structural | Control | Delta | 95% CI | Exact p | BH-FDR p | Sign-biserial | W/T/L |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ci = row["ci95"]
        lines.append(
            f"| {row['contrast']} | {row['endpoint']} | {row['n']} | {fmt(row['mean_structural'])} | "
            f"{fmt(row['mean_control'])} | {fmt(row['mean_diff'])} | [{fmt(ci[0])}, {fmt(ci[1])}] | "
            f"{fmt(row['p_signflip'])} | {fmt(row['p_bh_fdr'])} | {fmt(row['sign_biserial'])} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} |"
        )
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
