#!/usr/bin/env python3
"""Unblind native-S3 preference calls and compute topic-level estimates."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path


DIMENSIONS = (
    "organizational_quality",
    "critical_synthesis",
    "global_coherence",
    "citation_plausibility",
)
MEASURES = ("overall_preference", *DIMENSIONS)
JUDGES = ("gpt55", "gemini31pro", "sonnet5")
RNG = random.Random(20260716)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def bootstrap_ci(values: list[float], n_boot: int = 10_000) -> list[float]:
    means = [
        sum(RNG.choice(values) for _ in values) / len(values) for _ in range(n_boot)
    ]
    return [percentile(means, 0.025), percentile(means, 0.975)]


def exact_signflip_p(margins: list[float]) -> float:
    observed = abs(sum(margins) / len(margins))
    null = (
        abs(
            sum(sign * value for sign, value in zip(signs, margins))
            / len(margins)
        )
        for signs in itertools.product((-1, 1), repeat=len(margins))
    )
    values = list(null)
    return sum(value >= observed - 1e-12 for value in values) / len(values)


def slrgp_value(choice: str, system_a: str) -> float:
    if choice == "tie":
        return 0.5
    selected = system_a if choice == "A" else "slrgp" if system_a != "slrgp" else ""
    return 1.0 if selected == "slrgp" else 0.0


def summarize(values_by_topic: dict[str, list[float]]) -> dict:
    topic_scores = {
        topic: sum(values) / len(values)
        for topic, values in sorted(values_by_topic.items())
    }
    margins = [value - 0.5 for value in topic_scores.values()]
    return {
        "n_topics": len(topic_scores),
        "mean_preference_rate": sum(topic_scores.values()) / len(topic_scores),
        "mean_margin_from_tie": sum(margins) / len(margins),
        "paired_bootstrap_95ci_margin": bootstrap_ci(margins),
        "exact_signflip_p_two_sided": exact_signflip_p(margins),
        "topic_win_tie_loss": [
            sum(value > 0.5 for value in topic_scores.values()),
            sum(value == 0.5 for value in topic_scores.values()),
            sum(value < 0.5 for value in topic_scores.values()),
        ],
        "topic_scores": topic_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    judge_dir = args.root / "blind_preference"
    mapping = json.loads(
        (judge_dir / "blind_mapping.json").read_text(encoding="utf-8")
    )
    latest = {}
    for line in (judge_dir / "preference_scores.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        row = json.loads(line)
        if not row.get("error"):
            latest[(row["blind_id"], row["judge"])] = row
    expected = len(mapping) * len(JUDGES)
    if len(latest) != expected:
        raise SystemExit(f"Need {expected} successful calls, found {len(latest)}")

    baselines = sorted({value["baseline"] for value in mapping.values()})
    result = {"n_calls": len(latest), "comparisons": {}}
    for baseline in baselines:
        comparison = {}
        for measure in MEASURES:
            pooled: dict[str, list[float]] = {}
            by_judge = {judge: {} for judge in JUDGES}
            by_repeat = {1: {}, 2: {}}
            leave_one_out = {}
            for (blind_id, judge), row in latest.items():
                info = mapping[blind_id]
                if info["baseline"] != baseline:
                    continue
                value = slrgp_value(row[measure], info["system_a"])
                pooled.setdefault(info["topic_id"], []).append(value)
                by_judge[judge].setdefault(info["topic_id"], []).append(value)
                by_repeat[info["repeat"]].setdefault(info["topic_id"], []).append(
                    value
                )
            for omitted in JUDGES:
                values: dict[str, list[float]] = {}
                for (blind_id, judge), row in latest.items():
                    info = mapping[blind_id]
                    if info["baseline"] != baseline or judge == omitted:
                        continue
                    values.setdefault(info["topic_id"], []).append(
                        slrgp_value(row[measure], info["system_a"])
                    )
                leave_one_out[omitted] = summarize(values)
            comparison[measure] = {
                "pooled": summarize(pooled),
                "by_judge": {
                    judge: summarize(values) for judge, values in by_judge.items()
                },
                "by_repeat": {
                    str(repeat): summarize(values)
                    for repeat, values in by_repeat.items()
                },
                "leave_one_judge_out": leave_one_out,
            }
        result["comparisons"][baseline] = comparison

    analysis_dir = args.root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "preference_stats.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# S3 native-flow blind preference statistics",
        "",
        f"- Successful judge calls: {len(latest)}.",
        "- Unit of inference: topic; ties contribute 0.5.",
        "",
    ]
    for baseline, comparison in result["comparisons"].items():
        stat = comparison["overall_preference"]["pooled"]
        wins, ties, losses = stat["topic_win_tie_loss"]
        low, high = stat["paired_bootstrap_95ci_margin"]
        lines.append(
            f"- SLRGP vs {baseline}: preference={stat['mean_preference_rate']:.3f}; "
            f"margin={stat['mean_margin_from_tie']:+.3f}, 95% CI "
            f"[{low:+.3f}, {high:+.3f}], p={stat['exact_signflip_p_two_sided']:.4f}, "
            f"W/T/L={wins}/{ties}/{losses}."
        )
    (analysis_dir / "S3_PREFERENCE_STATS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
