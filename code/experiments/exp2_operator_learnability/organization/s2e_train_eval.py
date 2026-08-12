#!/usr/bin/env python3
"""
S2e/O Phase 4 — held-out schema learning evaluation.

Inputs:
  - nodes_train.jsonl / nodes_heldout.jsonl from deterministic LaTeX parse trees
  - consensus_labels.jsonl from the two-pass MODEL-A annotation audit

Outputs:
  - s2e_learning_report.json
  - heldout_predictions.jsonl

The evaluation unit is the organization node, but confidence intervals are
review-level bootstraps so siblings from the same review are not treated as
independent documents for uncertainty estimates.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline

_PKG_ROOT = Path(__file__).resolve().parents[4]
IN_DIR = Path(os.environ.get(
    "S2E_OUT_DIR",
    _PKG_ROOT / "data/exp2_operator_learnability/organization/confirmatory",
))
OUT_DIR = Path(os.environ.get("S2E_WORK_DIR", _PKG_ROOT / "work/exp2_organization"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
AXES = ["facet", "relation"]


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_nodes() -> dict[str, dict]:
    nodes = {}
    for split in ["train", "heldout"]:
        for n in load_jsonl(IN_DIR / f"nodes_{split}.jsonl"):
            nodes[f"{n['arxiv_id']}::{n['node_path']}"] = n
    return nodes


def load_dataset() -> list[dict]:
    nodes = load_nodes()
    rows = []
    # 若 s2e_agreement_consensus.py 已在本轮重新生成共识标签，优先读工作副本，
    # 否则读随包的冻结标签。
    consensus_path = OUT_DIR / "consensus_labels.jsonl"
    if not consensus_path.exists():
        consensus_path = IN_DIR / "consensus_labels.jsonl"
    for lab in load_jsonl(consensus_path):
        n = nodes[lab["node_id"]]
        row = {**n, **lab}
        rows.append(row)
    return rows


def text_features(r: dict) -> str:
    child_titles = " ; ".join(r.get("child_titles") or [])
    return (
        f"node_title: {r.get('node_title', '')}\n"
        f"parent_title: {r.get('parent_title', '')}\n"
        f"children: {child_titles}\n"
        f"depth_token: DEPTH_{r.get('node_depth')}\n"
        f"length_token: LEN_{r.get('review_length_bucket')}\n"
        f"discipline_token: DISC_{r.get('discipline', '')}\n"
        f"n_children_token: NCH_{min(int(r.get('n_children', 0)), 8)}"
    )


class MajorityBaseline:
    def __init__(self, keys: list[str] | None = None):
        self.keys = keys or []
        self.lookup = {}
        self.global_label = None

    def fit(self, rows: list[dict], axis: str) -> "MajorityBaseline":
        self.global_label = Counter(r[axis] for r in rows).most_common(1)[0][0]
        buckets = defaultdict(list)
        for r in rows:
            key = tuple(str(r[k]) for k in self.keys)
            buckets[key].append(r[axis])
        self.lookup = {k: Counter(v).most_common(1)[0][0] for k, v in buckets.items()}
        return self

    def predict(self, rows: list[dict]) -> list[str]:
        out = []
        for r in rows:
            key = tuple(str(r[k]) for k in self.keys)
            out.append(self.lookup.get(key, self.global_label))
        return out


def make_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0, n_jobs=1)),
    ])


def metric_bundle(y_true: list[str], y_pred: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def review_level_ci(rows: list[dict], y_true: list[str], pred_a: list[str], pred_b: list[str],
                    n_boot: int = 10000, seed: int = 42) -> dict:
    by_review = defaultdict(list)
    for i, r in enumerate(rows):
        by_review[r["arxiv_id"]].append(i)
    reviews = sorted(by_review)
    diffs = []
    for rid in reviews:
        idx = by_review[rid]
        acc_a = np.mean([pred_a[i] == y_true[i] for i in idx])
        acc_b = np.mean([pred_b[i] == y_true[i] for i in idx])
        diffs.append(acc_a - acc_b)
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    n = len(diffs)
    for b in range(n_boot):
        boot[b] = diffs[rng.integers(0, n, size=n)].mean()
    return {
        "n_reviews": int(n),
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
    }


def evaluate_axis(train: list[dict], heldout: list[dict], axis: str) -> tuple[dict, dict[str, list[str]]]:
    y_train = [r[axis] for r in train]
    y_test = [r[axis] for r in heldout]
    x_train = [text_features(r) for r in train]
    x_test = [text_features(r) for r in heldout]

    predictors = {
        "global_majority": MajorityBaseline([]).fit(train, axis),
        "depth_majority": MajorityBaseline(["node_depth"]).fit(train, axis),
        "depth_length_majority": MajorityBaseline(["node_depth", "review_length_bucket"]).fit(train, axis),
    }
    predictions = {name: pred.predict(heldout) for name, pred in predictors.items()}

    model = make_model()
    model.fit(x_train, y_train)
    predictions["tfidf_logreg"] = model.predict(x_test).tolist()

    metrics = {name: metric_bundle(y_test, pred) for name, pred in predictions.items()}
    best_baseline = max([k for k in metrics if k != "tfidf_logreg"], key=lambda k: metrics[k]["accuracy"])
    report = {
        "label_distribution_train": dict(Counter(y_train)),
        "label_distribution_heldout": dict(Counter(y_test)),
        "metrics": metrics,
        "best_baseline_by_accuracy": best_baseline,
        "tfidf_logreg_minus_best_baseline_accuracy_ci": review_level_ci(
            heldout, y_test, predictions["tfidf_logreg"], predictions[best_baseline]
        ),
    }
    return report, predictions


def stratified_accuracy(rows: list[dict], y_true: list[str], y_pred: list[str], key: str) -> dict:
    buckets = defaultdict(list)
    for i, r in enumerate(rows):
        buckets[str(r[key])].append(i)
    return {
        b: {
            "n_nodes": len(idx),
            "accuracy": float(np.mean([y_pred[i] == y_true[i] for i in idx])),
        }
        for b, idx in sorted(buckets.items())
    }


def main() -> None:
    rows = load_dataset()
    train = [r for r in rows if r["split"] == "train"]
    heldout = [r for r in rows if r["split"] == "heldout"]
    report = {
        "n_train_nodes": len(train),
        "n_heldout_nodes": len(heldout),
        "n_train_reviews": len({r["arxiv_id"] for r in train}),
        "n_heldout_reviews": len({r["arxiv_id"] for r in heldout}),
        "axes": {},
        "stratified_tfidf_logreg_accuracy": {},
    }
    all_predictions = {}
    for axis in AXES:
        axis_report, preds = evaluate_axis(train, heldout, axis)
        report["axes"][axis] = axis_report
        all_predictions[axis] = preds
        y_true = [r[axis] for r in heldout]
        y_pred = preds["tfidf_logreg"]
        report["stratified_tfidf_logreg_accuracy"][axis] = {
            "by_depth": stratified_accuracy(heldout, y_true, y_pred, "node_depth"),
            "by_review_length_bucket": stratified_accuracy(heldout, y_true, y_pred, "review_length_bucket"),
        }

    with open(OUT_DIR / "heldout_predictions.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(heldout):
            out = {
                "node_id": r["node_id"],
                "arxiv_id": r["arxiv_id"],
                "node_path": r["node_path"],
                "node_depth": r["node_depth"],
                "review_length_bucket": r["review_length_bucket"],
                "gold_facet": r["facet"],
                "gold_relation": r["relation"],
            }
            for axis in AXES:
                for name, pred in all_predictions[axis].items():
                    out[f"pred_{axis}_{name}"] = pred[i]
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    with open(OUT_DIR / "s2e_learning_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({
        "n_train_nodes": report["n_train_nodes"],
        "n_heldout_nodes": report["n_heldout_nodes"],
        "facet_metrics": report["axes"]["facet"]["metrics"],
        "facet_ci": report["axes"]["facet"]["tfidf_logreg_minus_best_baseline_accuracy_ci"],
        "relation_metrics": report["axes"]["relation"]["metrics"],
        "relation_ci": report["axes"]["relation"]["tfidf_logreg_minus_best_baseline_accuracy_ci"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
