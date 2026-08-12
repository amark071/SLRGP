#!/usr/bin/env python3
"""Analyze S5a length/depth blind scores."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path


DIMENSIONS = ("organizational_quality", "critical_synthesis", "global_coherence", "citation_plausibility")
ENDPOINTS = ("overall",) + DIMENSIONS
CONTRASTS = (
    ("recursive vs flat", "recursive_slrgp", "flat_single_pass"),
    ("recursive vs fixed outline", "recursive_slrgp", "fixed_outline_no_reentry"),
)
TIER_ORDER = ("short", "medium", "long")


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(diffs: list[float], n_boot: int = 10000, seed: int = 20260713) -> list[float]:
    if not diffs:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    n = len(diffs)
    vals = sorted(mean([diffs[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
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


def sign_biserial(diffs: list[float]) -> float:
    nz = [d for d in diffs if abs(d) > 1e-12]
    if not nz:
        return 0.0
    return (sum(d > 0 for d in nz) - sum(d < 0 for d in nz)) / len(nz)


def fmt(x: float) -> str:
    return "NA" if math.isnan(x) else f"{x:.3f}"


def load_panel(root: Path) -> tuple[dict, dict]:
    blind_dir = root / "blind_judging"
    mapping = json.loads((blind_dir / "blind_mapping.json").read_text(encoding="utf-8"))
    score_rows = [json.loads(line) for line in (blind_dir / "blind_scores.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    latest_rows = {}
    for row in score_rows:
        latest_rows[(row["blind_id"], row["judge"])] = row
    scored = defaultdict(lambda: defaultdict(list))
    for row in latest_rows.values():
        if row.get("error"):
            continue
        meta = mapping[row["blind_id"]]
        key = (meta["topic"], meta["tier"], meta["arm"])
        for dim in DIMENSIONS:
            scored[key][dim].append(float(row[f"{dim}_score"]))
    panel = {}
    for key, dims in scored.items():
        dim_means = {dim: mean(vals) for dim, vals in dims.items()}
        panel[key] = dim_means | {"overall": mean(list(dim_means.values()))}
    return panel, {
        "n_attempt_rows": len(score_rows),
        "n_latest_scores": len(latest_rows),
        "n_valid_latest_scores": sum(1 for row in latest_rows.values() if not row.get("error")),
        "n_latest_errors": sum(1 for row in latest_rows.values() if row.get("error")),
    }


def paired_contrasts(audit: dict, panel: dict, eligibility: str) -> list[dict]:
    rows_by_key = {(row["topic"], row["tier"], row["arm"]): row for row in audit["rows"]}
    topics = audit["summary"]["topics"]
    tiers = list(audit["summary"]["tiers"].keys())
    out = []
    for tier in tiers:
        for label, arm_a, arm_b in CONTRASTS:
            for endpoint in ENDPOINTS:
                diffs = []
                topic_rows = []
                for topic in topics:
                    a_meta = rows_by_key.get((topic, tier, arm_a))
                    b_meta = rows_by_key.get((topic, tier, arm_b))
                    if not a_meta or not b_meta:
                        continue
                    if eligibility == "primary" and not (a_meta["primary_eligible"] and b_meta["primary_eligible"]):
                        continue
                    if eligibility == "all_ok" and not (a_meta["ok"] and b_meta["ok"]):
                        continue
                    if (topic, tier, arm_a) not in panel or (topic, tier, arm_b) not in panel:
                        continue
                    a = panel[(topic, tier, arm_a)][endpoint]
                    b = panel[(topic, tier, arm_b)][endpoint]
                    diffs.append(a - b)
                    topic_rows.append({"topic": topic, "a": a, "b": b, "diff": a - b})
                wins = sum(d > 1e-12 for d in diffs)
                losses = sum(d < -1e-12 for d in diffs)
                ties = len(diffs) - wins - losses
                out.append({
                    "eligibility": eligibility,
                    "tier": tier,
                    "contrast": label,
                    "arm_a": arm_a,
                    "arm_b": arm_b,
                    "endpoint": endpoint,
                    "n": len(diffs),
                    "mean_a": mean([row["a"] for row in topic_rows]),
                    "mean_b": mean([row["b"] for row in topic_rows]),
                    "mean_diff": mean(diffs),
                    "ci95": bootstrap_ci(diffs),
                    "p_signflip": exact_signflip_p(diffs),
                    "sign_biserial": sign_biserial(diffs),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "topics": topic_rows,
                })
    return out


def degradation(audit: dict, panel: dict, eligibility: str) -> list[dict]:
    rows_by_key = {(row["topic"], row["tier"], row["arm"]): row for row in audit["rows"]}
    out = []
    for arm in audit["summary"]["arms"]:
        for endpoint in ENDPOINTS:
            topic_slopes = []
            for topic in audit["summary"]["topics"]:
                vals = []
                usable = True
                for tier in TIER_ORDER:
                    meta = rows_by_key.get((topic, tier, arm))
                    if not meta:
                        usable = False
                        break
                    if eligibility == "primary" and not meta["primary_eligible"]:
                        usable = False
                        break
                    if eligibility == "all_ok" and not meta["ok"]:
                        usable = False
                        break
                    if (topic, tier, arm) not in panel:
                        usable = False
                        break
                    vals.append(panel[(topic, tier, arm)][endpoint])
                if usable and len(vals) == 3:
                    topic_slopes.append({"topic": topic, "short": vals[0], "medium": vals[1], "long": vals[2], "long_minus_short": vals[2] - vals[0]})
            diffs = [row["long_minus_short"] for row in topic_slopes]
            out.append({
                "eligibility": eligibility,
                "arm": arm,
                "endpoint": endpoint,
                "n": len(diffs),
                "mean_long_minus_short": mean(diffs),
                "ci95": bootstrap_ci(diffs),
                "topics": topic_slopes,
            })
    return out


def write_md(output: dict, out_md: Path) -> None:
    lines = [
        "# S5a Length/Depth Analysis",
        "",
        "Positive contrast deltas mean recursive SLRGP scored higher than the control arm. Degradation is long minus short; less negative is better length robustness.",
        "",
        "## Audit",
        "",
        f"- Outputs: {output['audit']['n_ok']}/{output['audit']['n_rows']} ok.",
        f"- Primary eligible: {output['audit']['n_primary_eligible']}/{output['audit']['n_rows']}.",
        f"- Blind scores: {output['blind_scores']['n_valid_latest_scores']}/{output['blind_scores']['n_latest_scores']} valid latest records "
        f"({output['blind_scores']['n_attempt_rows']} raw attempts).",
        "",
        "## Paired Quality Contrasts",
        "",
        "| Eligibility | Tier | Contrast | Endpoint | n | recursive mean | control mean | delta | 95% CI | exact p | sign-biserial | W/T/L |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in output["contrasts"]:
        ci = row["ci95"]
        lines.append(
            f"| {row['eligibility']} | {row['tier']} | {row['contrast']} | {row['endpoint']} | {row['n']} | "
            f"{fmt(row['mean_a'])} | {fmt(row['mean_b'])} | {fmt(row['mean_diff'])} | "
            f"[{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(row['p_signflip'])} | {fmt(row['sign_biserial'])} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} |"
        )
    lines += [
        "",
        "## Length Degradation",
        "",
        "| Eligibility | Arm | Endpoint | n | long-short | 95% CI |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in output["degradation"]:
        ci = row["ci95"]
        lines.append(f"| {row['eligibility']} | {row['arm']} | {row['endpoint']} | {row['n']} | {fmt(row['mean_long_minus_short'])} | [{fmt(ci[0])}, {fmt(ci[1])}] |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    panel, score_summary = load_panel(args.root)
    output = {
        "root": str(args.root),
        "audit": {key: audit[key] for key in ("n_rows", "n_ok", "n_primary_eligible", "issues")},
        "blind_scores": score_summary,
        "contrasts": paired_contrasts(audit, panel, "primary") + paired_contrasts(audit, panel, "all_ok"),
        "degradation": degradation(audit, panel, "primary") + degradation(audit, panel, "all_ok"),
    }
    out_json = args.out_json or args.root / "S5A_LENGTH_STATS.json"
    out_md = args.out_md or args.root / "S5A_LENGTH_STATS.md"
    out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(output, out_md)
    print(f"wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
