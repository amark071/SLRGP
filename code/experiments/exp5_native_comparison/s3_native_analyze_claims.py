#!/usr/bin/env python3
"""Finalize claim labels and compute clustered support-rate contrasts."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path


BASELINES = ("autosurvey", "surveyforge", "surveygen")
PRIMARY = ("gpt55", "gemini31pro")
ADJUDICATOR = "sonnet5"
VALUE = {
    "supported": 1.0,
    "partially_supported": 0.5,
    "unsupported": 0.0,
}
INACCESSIBLE = "source_inaccessible_or_insufficient"
RNG = random.Random(20260716)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def exact_signflip_p(diffs: list[float]) -> float:
    nonzero = [value for value in diffs if value != 0]
    if not nonzero:
        return 1.0
    observed = abs(sum(nonzero) / len(nonzero))
    null = [
        abs(
            sum(sign * value for sign, value in zip(signs, nonzero))
            / len(nonzero)
        )
        for signs in itertools.product((-1, 1), repeat=len(nonzero))
    ]
    return sum(value >= observed - 1e-12 for value in null) / len(null)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def hierarchical_ci(
    by_topic_system: dict[tuple[str, str], list[float]],
    topics: list[str],
    baseline: str,
    n_boot: int = 10_000,
) -> list[float]:
    estimates = []
    for _ in range(n_boot):
        sampled_topics = [RNG.choice(topics) for _ in topics]
        topic_diffs = []
        for topic in sampled_topics:
            left = by_topic_system[(topic, "slrgp")]
            right = by_topic_system[(topic, baseline)]
            left_sample = [RNG.choice(left) for _ in left]
            right_sample = [RNG.choice(right) for _ in right]
            topic_diffs.append(mean(left_sample) - mean(right_sample))
        estimates.append(mean(topic_diffs))
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def bh_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted = {}
    running = 1.0
    total = len(ordered)
    for rank_from_end, (name, value) in enumerate(reversed(ordered), 1):
        rank = total - rank_from_end + 1
        running = min(running, value * total / rank)
        adjusted[name] = min(1.0, running)
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.root / "claim_support" / "claim_judgments.jsonl"
    latest = {}
    for line in checkpoint.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not row.get("error"):
            latest[(row["topic_id"], row["system"], row["judge"])] = row
    keys = sorted({(topic, system) for topic, system, _judge in latest})
    finalized = []
    for topic, system in keys:
        first = latest.get((topic, system, PRIMARY[0]))
        second = latest.get((topic, system, PRIMARY[1]))
        if not first or not second:
            raise SystemExit(f"Missing primary claim judgments: {topic}/{system}")
        left = {row["claim_id"]: row for row in first["results"]}
        right = {row["claim_id"]: row for row in second["results"]}
        adjudication = latest.get((topic, system, ADJUDICATOR))
        adjudicated = (
            {row["claim_id"]: row for row in adjudication["results"]}
            if adjudication
            else {}
        )
        for claim_id in sorted(left):
            if left[claim_id]["label"] == right[claim_id]["label"]:
                label = left[claim_id]["label"]
                source = "agreement"
            else:
                if claim_id not in adjudicated:
                    raise SystemExit(f"Missing adjudication: {topic}/{system}/{claim_id}")
                label = adjudicated[claim_id]["label"]
                source = "adjudicated"
            finalized.append(
                {
                    "topic_id": topic,
                    "system": system,
                    "claim_id": claim_id,
                    "label": label,
                    "finalization": source,
                }
            )

    accessible: dict[tuple[str, str], list[float]] = {}
    conservative: dict[tuple[str, str], list[float]] = {}
    inaccessible_counts = {}
    for row in finalized:
        key = (row["topic_id"], row["system"])
        conservative.setdefault(key, []).append(VALUE.get(row["label"], 0.0))
        inaccessible_counts[key] = inaccessible_counts.get(key, 0) + (
            row["label"] == INACCESSIBLE
        )
        if row["label"] != INACCESSIBLE:
            accessible.setdefault(key, []).append(VALUE[row["label"]])
    topics = sorted({topic for topic, _system in conservative})
    comparisons = {}
    p_values = {}
    for baseline in BASELINES:
        topic_diffs = [
            mean(accessible[(topic, "slrgp")])
            - mean(accessible[(topic, baseline)])
            for topic in topics
        ]
        conservative_diffs = [
            mean(conservative[(topic, "slrgp")])
            - mean(conservative[(topic, baseline)])
            for topic in topics
        ]
        p_value = exact_signflip_p(topic_diffs)
        p_values[f"claim_support:{baseline}"] = p_value
        comparisons[baseline] = {
            "mean_topic_paired_difference": mean(topic_diffs),
            "hierarchical_bootstrap_95ci": hierarchical_ci(
                accessible, topics, baseline
            ),
            "exact_signflip_p_two_sided": p_value,
            "topic_win_tie_loss": [
                sum(value > 0 for value in topic_diffs),
                sum(value == 0 for value in topic_diffs),
                sum(value < 0 for value in topic_diffs),
            ],
            "conservative_inaccessible_as_unsupported_difference": mean(
                conservative_diffs
            ),
            "topic_differences": dict(zip(topics, topic_diffs)),
        }

    preference_path = args.root / "analysis" / "preference_stats.json"
    if preference_path.is_file():
        preference = json.loads(preference_path.read_text(encoding="utf-8"))
        for baseline in BASELINES:
            p_values[f"preference:{baseline}"] = preference["comparisons"][baseline][
                "overall_preference"
            ]["pooled"]["exact_signflip_p_two_sided"]
    adjusted = bh_adjust(p_values)
    for baseline in BASELINES:
        comparisons[baseline]["bh_fdr_q_across_six_primary_tests"] = adjusted.get(
            f"claim_support:{baseline}"
        )
    payload = {
        "n_finalized_claims": len(finalized),
        "finalized_claims": finalized,
        "comparisons": comparisons,
        "inaccessible_counts": {
            f"{topic}|{system}": count
            for (topic, system), count in sorted(inaccessible_counts.items())
        },
        "primary_family_raw_p": p_values,
        "primary_family_bh_fdr_q": adjusted,
    }
    analysis_dir = args.root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "claim_support_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
