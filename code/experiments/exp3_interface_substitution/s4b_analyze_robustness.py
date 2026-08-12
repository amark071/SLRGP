#!/usr/bin/env python3
"""Analyze S4b Qwen robustness outputs.

The script always summarizes generation/audit eligibility. If blind Instrument-J
scores are present under ``blind_judging/``, it also computes paired quality
contrasts for the two targeted robustness checks.
"""
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


def sign_biserial(diffs: list[float]) -> float:
    nz = [d for d in diffs if abs(d) > 1e-12]
    if not nz:
        return 0.0
    return (sum(d > 0 for d in nz) - sum(d < 0 for d in nz)) / len(nz)


def fmt(x: float) -> str:
    return "NA" if math.isnan(x) else f"{x:.3f}"


def summarize_audit(audit: dict) -> dict:
    rows = audit["rows"]
    by_arm = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)
    pairability = []
    topics = sorted({row["topic"] for row in rows})
    row_by_key = {(row["topic"], row["arm"]): row for row in rows}
    for label, arm_a, arm_b in CONTRASTS:
        primary_topics = []
        all_ok_topics = []
        for topic in topics:
            a = row_by_key.get((topic, arm_a))
            b = row_by_key.get((topic, arm_b))
            if not a or not b:
                continue
            if a.get("ok") and b.get("ok"):
                all_ok_topics.append(topic)
            if a.get("primary_eligible") and b.get("primary_eligible"):
                primary_topics.append(topic)
        pairability.append({
            "contrast": label,
            "arm_a": arm_a,
            "arm_b": arm_b,
            "primary_n": len(primary_topics),
            "primary_topics": primary_topics,
            "all_ok_n": len(all_ok_topics),
            "all_ok_topics": all_ok_topics,
        })
    return {
        "n_rows": len(rows),
        "n_ok": sum(1 for row in rows if row.get("ok")),
        "n_primary_eligible": sum(1 for row in rows if row.get("primary_eligible")),
        "issues": audit.get("issues", []),
        "arms": {
            arm: {
                "n": len(arm_rows),
                "n_primary_eligible": sum(1 for row in arm_rows if row.get("primary_eligible")),
                "mean_words": mean([float(row.get("words") or 0) for row in arm_rows]),
                "mean_llm_calls": mean([float(row.get("llm_calls") or 0) for row in arm_rows]),
                "mean_completion_tokens": mean([float(row.get("completion_tokens") or 0) for row in arm_rows]),
            }
            for arm, arm_rows in sorted(by_arm.items())
        },
        "pairability": pairability,
    }


def load_panel_scores(root: Path) -> dict | None:
    blind_dir = root / "blind_judging"
    mapping_path = blind_dir / "blind_mapping.json"
    scores_path = blind_dir / "blind_scores.jsonl"
    if not (mapping_path.exists() and scores_path.exists()):
        return None
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in scores_path.read_text(encoding="utf-8").splitlines() if line.strip()]
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
    return {"panel": panel, "n_scores": len(rows), "n_valid_scores": sum(1 for row in rows if not row.get("error"))}


def quality_contrasts(audit: dict, panel_data: dict, eligibility: str) -> list[dict]:
    panel = panel_data["panel"]
    row_by_key = {(row["topic"], row["arm"]): row for row in audit["rows"]}
    topics = sorted({row["topic"] for row in audit["rows"]})
    results = []
    for label, arm_a, arm_b in CONTRASTS:
        for endpoint in ("overall",) + DIMENSIONS:
            diffs = []
            topic_rows = []
            for topic in topics:
                a_meta = row_by_key.get((topic, arm_a))
                b_meta = row_by_key.get((topic, arm_b))
                if not a_meta or not b_meta:
                    continue
                if eligibility == "primary":
                    if not (a_meta.get("primary_eligible") and b_meta.get("primary_eligible")):
                        continue
                elif not (a_meta.get("ok") and b_meta.get("ok")):
                    continue
                if (topic, arm_a) not in panel or (topic, arm_b) not in panel:
                    continue
                a = panel[(topic, arm_a)][endpoint]
                b = panel[(topic, arm_b)][endpoint]
                diffs.append(a - b)
                topic_rows.append({"topic": topic, "a": a, "b": b, "diff": a - b})
            wins = sum(d > 1e-12 for d in diffs)
            losses = sum(d < -1e-12 for d in diffs)
            ties = len(diffs) - wins - losses
            results.append({
                "eligibility": eligibility,
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
    return results


def write_markdown(output: dict, out_md: Path) -> None:
    audit = output["audit_summary"]
    lines = [
        "# S4b Qwen Robustness Analysis",
        "",
        "S4b is a targeted open-weight reproducibility check. It asks whether the S4a ordering of the two strongest contrasts is preserved when generation is moved from the commercial main backbone to self-hosted Qwen3-32B executed through a 4-bit bitsandbytes service.",
        "",
        "## Generation And Audit",
        "",
        f"- Outputs completed: {audit['n_ok']}/{audit['n_rows']}.",
        f"- Primary design-gate eligible rows: {audit['n_primary_eligible']}/{audit['n_rows']}.",
        f"- Audit issues: {len(audit['issues'])}.",
        "",
        "| Arm | n | primary eligible | mean words | mean LLM calls | mean completion tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, row in audit["arms"].items():
        lines.append(
            f"| {arm} | {row['n']} | {row['n_primary_eligible']} | {fmt(row['mean_words'])} | "
            f"{fmt(row['mean_llm_calls'])} | {fmt(row['mean_completion_tokens'])} |"
        )
    lines += ["", "## Pairability", ""]
    for row in audit["pairability"]:
        lines.append(
            f"- {row['contrast']}: primary n={row['primary_n']} ({', '.join(row['primary_topics']) or 'none'}); "
            f"all-ok n={row['all_ok_n']}."
        )
    if audit["issues"]:
        lines += ["", "## Design-Gate Issues", ""]
        for issue in audit["issues"]:
            lines.append(f"- {issue['topic']} / {issue['arm']}: {issue['issue']} (words={issue.get('words')}).")
    if output.get("quality"):
        lines += [
            "",
            "## Blind Quality Contrasts",
            "",
            "Positive delta means arm A scored higher than arm B. Primary uses design-gate eligible pairs; all-ok is a sensitivity analysis over all successfully generated pairs.",
            "",
            "| Eligibility | Contrast | Endpoint | n | A mean | B mean | delta | 95% bootstrap CI | exact p | sign-biserial | W/T/L |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in output["quality"]:
            ci = row["ci95"]
            lines.append(
                f"| {row['eligibility']} | {row['contrast']} | {row['endpoint']} | {row['n']} | "
                f"{fmt(row['mean_a'])} | {fmt(row['mean_b'])} | {fmt(row['mean_diff'])} | "
                f"[{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(row['p_signflip'])} | "
                f"{fmt(row['sign_biserial'])} | {row['wins']}/{row['ties']}/{row['losses']} |"
            )
    else:
        lines += [
            "",
            "## Blind Quality Contrasts",
            "",
            "Blind Instrument-J scores are not present yet. Run `s4a_blind_judge.py` on this S4b root, then rerun this analysis script to fill the quality-contrast table.",
        ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    output = {
        "root": str(args.root),
        "audit_path": str(args.audit),
        "audit_summary": summarize_audit(audit),
    }
    panel_data = load_panel_scores(args.root)
    if panel_data:
        output["blind_scores"] = {k: v for k, v in panel_data.items() if k != "panel"}
        output["quality"] = quality_contrasts(audit, panel_data, "primary") + quality_contrasts(audit, panel_data, "all_ok")
    out_json = args.out_json or args.root / "S4B_ROBUSTNESS_STATS.json"
    out_md = args.out_md or args.root / "S4B_ROBUSTNESS_STATS.md"
    out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(output, out_md)
    print(f"wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
