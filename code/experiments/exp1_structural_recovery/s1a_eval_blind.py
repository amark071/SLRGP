#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
# 输入为冻结的确认性数据(含 layer1 汇总), 输出写入 work/;可用环境变量覆盖。
DEFAULT_IN = WORKSPACE / "data" / "exp1_structural_recovery" / "confirmatory"
DEFAULT_RES = WORKSPACE / "work" / "exp1_structural_recovery"
IN = Path(os.environ.get("S1A_IN_DIR", DEFAULT_IN))
RES = Path(os.environ.get("S1A_RESULTS_DIR", DEFAULT_RES))
RES.mkdir(parents=True, exist_ok=True)


def load(path: Path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def ci(vals, iters=10000):
    if not vals:
        return [None, None]
    rng = random.Random(20260710)
    outs = []
    for _ in range(iters):
        outs.append(mean([vals[rng.randrange(len(vals))] for __ in vals]))
    outs.sort()
    return [outs[int(0.025 * iters)], outs[int(0.975 * iters) - 1]]


def perm_p(ds, iters=10000):
    if not ds:
        return None
    obs = abs(mean(ds))
    rng = random.Random(7)
    ge = 0
    for _ in range(iters):
        v = abs(mean([d if rng.random() < 0.5 else -d for d in ds]))
        if v >= obs:
            ge += 1
    return (ge + 1) / (iters + 1)


def auc(labels, scores):
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    npos = sum(labels)
    nneg = len(labels) - npos
    if not npos or not nneg:
        return None
    rank = sum(i for i, (_, y) in enumerate(pairs, 1) if y)
    return (rank - npos * (npos + 1) / 2) / (npos * nneg)


def summarize_pairs(diffs, fds, labels, scores):
    return {
        "n_pairs_scored": len(diffs),
        "mean_expressibility_diff_auth_minus_synth": mean(diffs),
        "diff_bootstrap_ci": ci(diffs),
        "permutation_p_two_sided": perm_p(diffs),
        "pairwise_accuracy_auth_gt_synth": mean([1 if d > 0 else 0.5 if d == 0 else 0 for d in diffs]),
        "roc_auc_auth_vs_synth": auc(labels, scores),
        "mean_faithfulness_diff": mean(fds),
        "faithfulness_ci": ci(fds),
    }


def eval_pass(pid):
    ann_path = IN / f"s1a_annotations_blind_pass{pid}_dedup.jsonl"
    if not ann_path.exists():
        ann_path = IN / f"s1a_annotations_blind_pass{pid}.jsonl"
    ann = {
        r["item_id"]: r
        for r in load(ann_path)
        if isinstance(r.get("expressibility_score"), (int, float)) and r.get("blind_prompt") is True
    }
    pairs = load(IN / "s1a_pairs.jsonl")
    items = {row["item_id"]: row for row in load(IN / "s1a_items.jsonl")}
    out = {"pass_id": pid, "n_annotations_valid": len(ann), "blind_prompt": True, "tiers": {}}
    for tier, key in [("tier1", "tier1_item_id"), ("tier2", "tier2_item_id")]:
        diffs = []
        fds = []
        labels = []
        scores = []
        per = defaultdict(list)
        per_length = defaultdict(list)
        per_discipline_length = defaultdict(list)
        counts = Counter()
        for p in pairs:
            authentic = ann.get(p["authentic_item_id"])
            synthetic = ann.get(p[key])
            if not authentic or not synthetic:
                continue
            d = float(authentic["expressibility_score"]) - float(synthetic["expressibility_score"])
            diffs.append(d)
            fds.append(float(authentic.get("faithfulness_rating") or 0) - float(synthetic.get("faithfulness_rating") or 0))
            labels += [1, 0]
            scores += [float(authentic["expressibility_score"]), float(synthetic["expressibility_score"])]
            acc = 1 if d > 0 else 0.5 if d == 0 else 0
            per[p.get("discipline")].append(acc)
            counts[p.get("discipline")] += 1
            length_bucket = items.get(p["authentic_item_id"], {}).get("review_length_bucket", "unknown")
            per_length[length_bucket].append((d, float(authentic.get("faithfulness_rating") or 0) - float(synthetic.get("faithfulness_rating") or 0), float(authentic["expressibility_score"]), float(synthetic["expressibility_score"])))
            per_discipline_length[(p.get("discipline"), length_bucket)].append((d, float(authentic.get("faithfulness_rating") or 0) - float(synthetic.get("faithfulness_rating") or 0), float(authentic["expressibility_score"]), float(synthetic["expressibility_score"])))
        summary = summarize_pairs(diffs, fds, labels, scores)
        summary.update({
            "per_discipline_pairwise_accuracy": {k: mean(v) for k, v in sorted(per.items())},
            "discipline_pair_counts": dict(counts),
            "by_review_length_bucket": {
                bucket: summarize_pairs(
                    [row[0] for row in values],
                    [row[1] for row in values],
                    [1, 0] * len(values),
                    [score for row in values for score in row[2:]],
                )
                for bucket, values in sorted(per_length.items())
            },
            "by_discipline_review_length": {
                f"{discipline}::{bucket}": summarize_pairs(
                    [row[0] for row in values],
                    [row[1] for row in values],
                    [1, 0] * len(values),
                    [score for row in values for score in row[2:]],
                )
                for (discipline, bucket), values in sorted(per_discipline_length.items())
            },
            "review_length_definition": "short/medium/long are frozen tertiles of parsed section count, not body-word-count tiers.",
            "bootstrap_iterations": 10000,
            "permutation_iterations": 10000,
        })
        out["tiers"][tier] = summary
    return out


def main():
    final = {
        "layer1": json.load(open(IN / "s1a_layer1_summary.json", encoding="utf-8"))
        if (IN / "s1a_layer1_summary.json").exists() else None,
        "negative_construction": json.load(open(IN / "s1a_negative_construction_summary.json", encoding="utf-8"))
        if (IN / "s1a_negative_construction_summary.json").exists() else None,
        "annotation_evaluation": [],
        "evaluation_note": "Uses blind MODEL-A annotations with opaque prompt IDs; no human annotation is used.",
    }
    for pid in [1, 2]:
        # 优先使用去重后的盲评标注(*_dedup.jsonl),未去重版本亦可
        if (IN / f"s1a_annotations_blind_pass{pid}_dedup.jsonl").exists() or \
                (IN / f"s1a_annotations_blind_pass{pid}.jsonl").exists():
            final["annotation_evaluation"].append(eval_pass(pid))
    (RES / "s1a_eval_summary_blind.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False, indent=2)[:6000])


if __name__ == "__main__":
    main()
