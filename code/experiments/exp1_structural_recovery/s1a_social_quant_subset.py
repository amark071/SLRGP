#!/usr/bin/env python3
"""Social-science (quantitative) S1a Layer-2 analysis.

Extracts the economics + quantitative-finance subset already present in the frozen
confirmatory corpus. These reviews were parsed deterministically from arXiv LaTeX,
carry the same held-out nested-node endpoint (depth 0/1/2) as the main claim, and
were blind-scored in the same MODEL-A pass. No new data acquisition or scoring is
performed; the endpoint and statistics are identical to s1a_eval_blind.py.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

IN = Path(__file__).resolve().parents[3] / "data" / "exp1_structural_recovery" / "confirmatory"
SOCIAL_QUANT = {"economics", "quantitative_finance"}


def load(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def ci(vals, iters=10000):
    if not vals:
        return [None, None]
    rng = random.Random(20260710)
    outs = [mean([vals[rng.randrange(len(vals))] for __ in vals]) for _ in range(iters)]
    outs.sort()
    return [outs[int(0.025 * iters)], outs[int(0.975 * iters) - 1]]


def perm_p(ds, iters=10000):
    if not ds:
        return None
    obs = abs(mean(ds))
    rng = random.Random(7)
    ge = sum(1 for _ in range(iters) if abs(mean([d if rng.random() < 0.5 else -d for d in ds])) >= obs)
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


def evaluate(pairs, ann, items, disc_filter=None):
    out = {"tiers": {}}
    sel = [p for p in pairs if (disc_filter is None or p.get("discipline") in disc_filter)]
    for tier, key in [("tier1", "tier1_item_id"), ("tier2", "tier2_item_id")]:
        diffs, fds, labels, scores = [], [], [], []
        per = defaultdict(list)
        per_length = defaultdict(list)
        counts = Counter()
        for p in sel:
            a = ann.get(p["authentic_item_id"])
            s = ann.get(p[key])
            if not a or not s:
                continue
            d = float(a["expressibility_score"]) - float(s["expressibility_score"])
            fd = float(a.get("faithfulness_rating") or 0) - float(s.get("faithfulness_rating") or 0)
            diffs.append(d)
            fds.append(fd)
            labels += [1, 0]
            scores += [float(a["expressibility_score"]), float(s["expressibility_score"])]
            per[p.get("discipline")].append(1 if d > 0 else 0.5 if d == 0 else 0)
            counts[p.get("discipline")] += 1
            lb = items.get(p["authentic_item_id"], {}).get("review_length_bucket", "unknown")
            per_length[lb].append((d, fd, float(a["expressibility_score"]), float(s["expressibility_score"])))
        summary = summarize_pairs(diffs, fds, labels, scores)
        summary["per_discipline_pairwise_accuracy"] = {k: mean(v) for k, v in sorted(per.items())}
        summary["discipline_pair_counts"] = dict(counts)
        summary["by_review_length_bucket"] = {
            b: summarize_pairs([r[0] for r in v], [r[1] for r in v], [1, 0] * len(v), [x for r in v for x in r[2:]])
            for b, v in sorted(per_length.items())
        }
        out["tiers"][tier] = summary
    return out


def main():
    ann_path = IN / "s1a_annotations_blind_pass1_dedup.jsonl"
    ann = {
        r["item_id"]: r
        for r in load(ann_path)
        if isinstance(r.get("expressibility_score"), (int, float)) and r.get("blind_prompt") is True
    }
    pairs = load(IN / "s1a_pairs.jsonl")
    items = {row["item_id"]: row for row in load(IN / "s1a_items.jsonl")}

    result = {
        "endpoint": "held-out nested-node blind MODEL-A expressibility, identical to main confirmatory claim",
        "primary": "tier2 hard matched synthetic (depth- and child-count-matched counterfactual regroupings)",
        "source": "arXiv LaTeX deterministic parse; frozen blind pass1 dedup annotations; no new scoring",
        "full_corpus_sanity_check": evaluate(pairs, ann, items, None),
        "social_quantitative_economics_qfin": evaluate(pairs, ann, items, SOCIAL_QUANT),
        "economics_only": evaluate(pairs, ann, items, {"economics"}),
        "quantitative_finance_only": evaluate(pairs, ann, items, {"quantitative_finance"}),
    }
    out_path = Path(__file__).resolve().parents[3] / "work" / "results" / "s1a_social" / "s1a_social_quant_subset.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    def line(tag, node):
        t2 = node["tiers"]["tier2"]
        print(f"{tag:28s} n={t2['n_pairs_scored']:4d}  diff={t2['mean_expressibility_diff_auth_minus_synth']:+.3f}  "
              f"CI={[round(x,3) for x in t2['diff_bootstrap_ci']]}  AUC={t2['roc_auc_auth_vs_synth']:.3f}  "
              f"acc={t2['pairwise_accuracy_auth_gt_synth']:.3f}  p={t2['permutation_p_two_sided']:.4f}")

    print("TIER2 (primary) endpoints:")
    line("full corpus (all 11 disc)", result["full_corpus_sanity_check"])
    line("social quant (econ+qfin)", result["social_quantitative_economics_qfin"])
    line("economics only", result["economics_only"])
    line("quant_finance only", result["quantitative_finance_only"])
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
