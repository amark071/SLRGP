"""
分析仪器 J 全量探针结果（instrument_j.py 的输出），产出：
  - 逐维度 Krippendorff's alpha（跨裁判，含两次重复）
  - within-judge repeat 方差（同一 judge 两次打分的差异）
  - 每个 (variant) 相对 control 的配对差值 + BCa bootstrap CI（按 doc x judge x repeat 配对）
  - 三个探针的通过/未通过判定（判定标准见论文 Methods「Instrument validation」）
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stats_utils import krippendorff_alpha_ordinal, paired_bootstrap_ci, paired_test  # noqa: E402

DIMENSIONS = ["organizational_quality", "critical_synthesis", "global_coherence", "citation_plausibility"]
_PKG_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
SCORES_JSON = os.environ.get(
    "PHASE0_SCORES",
    os.path.join(_PKG_ROOT, "data", "instrument_validation", "scores", "phase0_full_scores.json"),
)


def load_rows():
    with open(SCORES_JSON, encoding="utf-8") as f:
        return json.load(f)


def krippendorff_by_dimension(rows):
    print("\n=== Krippendorff's alpha (ordinal, across judges, both repeats pooled as separate raters-by-repeat) ===")
    judges = sorted({r["judge"] for r in rows if "error" not in r})
    units = sorted({(r["doc_id"], r["variant"]) for r in rows if "error" not in r})
    # 把 (judge, repeat) 当作独立的"评分员"，共 3 judges x 2 repeats = 6 行
    raters = sorted({(r["judge"], r["repeat"]) for r in rows if "error" not in r})
    for dim in DIMENSIONS:
        matrix = []
        for rater in raters:
            row_scores = []
            for unit in units:
                match = [r for r in rows if "error" not in r and (r["judge"], r["repeat"]) == rater
                         and (r["doc_id"], r["variant"]) == unit]
                row_scores.append(match[0].get(f"{dim}_score") if match else None)
            matrix.append(row_scores)
        alpha = krippendorff_alpha_ordinal(matrix)
        n_scores = sum(1 for row in matrix for v in row if v is not None)
        print(f"  {dim:28s} alpha={alpha:+.3f}  n_scores={n_scores}  n_units={len(units)}  n_raters={len(raters)}")


def within_judge_repeat_variance(rows):
    print("\n=== Within-judge repeat variance (rep1 vs rep2, same judge+doc+variant) ===")
    for dim in DIMENSIONS:
        diffs = []
        by_key = defaultdict(dict)
        for r in rows:
            if "error" in r:
                continue
            key = (r["judge"], r["doc_id"], r["variant"])
            by_key[key][r["repeat"]] = r.get(f"{dim}_score")
        for key, reps in by_key.items():
            if 1 in reps and 2 in reps and reps[1] is not None and reps[2] is not None:
                diffs.append(abs(reps[1] - reps[2]))
        if diffs:
            import statistics
            print(f"  {dim:28s} mean|rep1-rep2|={statistics.mean(diffs):.3f}  "
                  f"exact-match-rate={sum(1 for d in diffs if d == 0) / len(diffs):.2f}  n_pairs={len(diffs)}")


def condition_means(rows, dim):
    by_variant = defaultdict(list)
    for r in rows:
        if "error" in r:
            continue
        v = r.get(f"{dim}_score")
        if v is not None:
            by_variant[r["variant"]].append(v)
    return {k: (sum(v) / len(v), len(v)) for k, v in by_variant.items()}


def paired_vs_control(rows, dim, variant):
    """按 (doc_id, judge, repeat) 配对：该变体分数 - control 分数。"""
    by_key = {}
    for r in rows:
        if "error" in r:
            continue
        key = (r["doc_id"], r["judge"], r["repeat"])
        by_key.setdefault(key, {})[r["variant"]] = r.get(f"{dim}_score")
    diffs = []
    for key, d in by_key.items():
        if variant in d and "control" in d and d[variant] is not None and d["control"] is not None:
            diffs.append(d[variant] - d["control"])
    return diffs


def probe_report(rows):
    print("\n=== Probe validation (criteria: manuscript Methods, Instrument validation) ===")

    print("\n--- (i) Structure probe: control vs structure_shuffled ---")
    for dim in ["organizational_quality", "global_coherence"]:
        diffs = paired_vs_control(rows, dim, "structure_shuffled")
        if len(diffs) >= 2:
            ci = paired_bootstrap_ci(diffs)
            pt = paired_test(diffs)
            verdict = "SEPARATES (control > shuffled)" if ci.ci_high < 0 else (
                "directionally correct but CI includes 0" if sum(diffs) < 0 else "WRONG DIRECTION / no effect")
            print(f"  {dim}: mean_diff(shuffled-control)={ci.point_estimate:+.3f} {ci} | {pt} -> {verdict}")

    print("\n--- (ii) Citation-corruption probe: control vs {shuffled, deleted, replaced} ---")
    for variant in ["citation_shuffled", "citation_deleted", "citation_replaced"]:
        diffs = paired_vs_control(rows, "citation_plausibility", variant)
        if len(diffs) >= 2:
            ci = paired_bootstrap_ci(diffs)
            pt = paired_test(diffs)
            verdict = "SEPARATES (control > corrupted)" if ci.ci_high < 0 else (
                "directionally correct but CI includes 0" if sum(diffs) < 0 else "WRONG DIRECTION / no effect")
            print(f"  citation_plausibility vs {variant}: mean_diff={ci.point_estimate:+.3f} {ci} | {pt} -> {verdict}")

    print("\n--- (iii) Length-sensitivity probe: control vs {shortened, lengthened_redundant} ---")
    for variant in ["shortened", "lengthened_redundant"]:
        for dim in ["organizational_quality", "global_coherence", "citation_plausibility"]:
            diffs = paired_vs_control(rows, dim, variant)
            if len(diffs) >= 2:
                ci = paired_bootstrap_ci(diffs)
                print(f"  {dim} vs {variant}: mean_diff={ci.point_estimate:+.3f} {ci}")
    lengthened_org = paired_vs_control(rows, "organizational_quality", "lengthened_redundant")
    if lengthened_org:
        rewarded = sum(1 for d in lengthened_org if d > 0)
        print(f"\n  lengthened_redundant rewarded on organizational_quality in {rewarded}/{len(lengthened_org)} "
              f"(doc,judge,repeat) triples (expect: not systematically rewarded)")

    print("\n--- (iv) Critical-synthesis probe: control vs synthesis_flattened (lexical marker substitution) ---")
    diffs = paired_vs_control(rows, "critical_synthesis", "synthesis_flattened")
    if len(diffs) >= 2:
        ci = paired_bootstrap_ci(diffs)
        pt = paired_test(diffs)
        verdict = "SEPARATES (control > flattened)" if ci.ci_high < 0 else (
            "directionally correct but CI includes 0" if sum(diffs) < 0 else "NOT DETECTED / no effect")
        print(f"  critical_synthesis vs synthesis_flattened: mean_diff={ci.point_estimate:+.3f} {ci} | {pt} -> {verdict}")
    # 判别效度：语篇替换不应显著影响与"批判性综合"无关的维度
    for dim in ["organizational_quality", "citation_plausibility"]:
        d2 = paired_vs_control(rows, dim, "synthesis_flattened")
        if len(d2) >= 2:
            ci2 = paired_bootstrap_ci(d2)
            print(f"  (discriminant check) {dim} vs synthesis_flattened: mean_diff={ci2.point_estimate:+.3f} {ci2} "
                  f"(expect CI to include 0 — flattening should not move unrelated dimensions)")


def main():
    rows = load_rows()
    n_errors = sum(1 for r in rows if "error" in r)
    print(f"Loaded {len(rows)} rows ({n_errors} errors) from {SCORES_JSON}")
    for dim in DIMENSIONS:
        means = condition_means(rows, dim)
        print(f"\n=== {dim}: mean by condition ===")
        for v, (m, n) in sorted(means.items(), key=lambda kv: -kv[1][0]):
            print(f"  {v:22s} mean={m:.3f}  n={n}")
    krippendorff_by_dimension(rows)
    within_judge_repeat_variance(rows)
    probe_report(rows)


if __name__ == "__main__":
    main()
