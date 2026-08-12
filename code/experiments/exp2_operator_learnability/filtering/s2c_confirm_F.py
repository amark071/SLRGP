#!/usr/bin/env python3
"""
S2c/F confirmatory evaluation using blind three-model consensus eligibility proxy.

The target is an admission filter, not a citation ranker:
  input: RRF top-500 from S2b's honest retrieval union
  output: thresholded variable-size admissible set
  primary: pool reduction at validation-selected >=95% gold-proxy ADMIT recall
  confirmatory test: blind consensus labels; uncertain consensus excluded from main
  sensitivity: uncertain=ADMIT / uncertain=EXCLUDE bounds.

No citation status or learned score is used as a model feature. Citation coverage is
reported only as a downstream safety diagnostic.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

UNION_DIR = os.environ.get("S2B_UNION_DIR", "work/exp2_operator_learnability/ranking/union_L")
CORPUS_DB = os.environ.get("S2C_CORPUS_DB", "data/common/unified_corpus/unified_corpus.db")
GOLD_DIR = os.environ.get("S2C_GOLD_DIR", os.path.join(os.path.dirname(__file__), "work"))
OUT_DIR = os.environ.get("S2C_CONFIRM_OUT", "work/exp2_operator_learnability/filtering_confirm")
INPUT_DEPTH = 500
GOLD_RECALL_TARGET = 0.95
SEED = 42
FEATURES = [
    "bm25_score", "bm25_rank_recip", "dense_score", "dense_rank_recip",
    "rrf_score", "year_diff", "same_discipline", "tier_ord", "has_tier",
    "has_doi", "has_venue", "abstract_len", "title_len",
    "source_is_stem", "source_is_social",
]
TIER_ORDER = {"T1": 1, "T2": 2, "T3": 3}


def load_consensus(split: str) -> pd.DataFrame:
    # Cost-bounded stratified train panel; validation/test use their full panels.
    name = "s2c_multimodel_train_300.json" if split == "train" else f"s2c_multimodel_{split}.json"
    p = os.path.join(GOLD_DIR, name)
    d = json.load(open(p, encoding="utf-8"))
    rows = []
    for x in d["items"]:
        rows.append({
            "arxiv_id": x["arxiv_id"], "doc_id": x["doc_id"],
            "gold_binary": x["consensus"]["binary"],
            "stratum": x["stratum"],
        })
    return pd.DataFrame(rows)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    conn = sqlite3.connect(CORPUS_DB)
    ids = df["doc_id"].unique().tolist()
    meta = {}
    for i in range(0, len(ids), 900):
        c = ids[i:i + 900]
        ph = ",".join("?" for _ in c)
        for d, source, title, abstract, venue, doi, tier in conn.execute(
            f"SELECT doc_id,source,title,abstract,venue,doi,tier FROM papers WHERE doc_id IN ({ph})", c
        ):
            abstract, title = abstract or "", title or ""
            meta[d] = {
                "tier_ord": TIER_ORDER.get(tier or "", 4), "has_tier": int(bool(tier)),
                "has_doi": int(bool(doi)), "has_venue": int(bool(venue)),
                "abstract_len": len(abstract.split()), "title_len": len(title.split()),
                "source_is_stem": int((source or "").lower() == "stem"),
                "source_is_social": int("social" in (source or "").lower()),
            }
    m = pd.DataFrame.from_dict(meta, orient="index")
    m.index.name = "doc_id"
    out = df.merge(m.reset_index(), on="doc_id", how="left")
    for c in FEATURES:
        if c not in out:
            out[c] = 0
        out[c] = out[c].fillna(0)
    return out


def load_pool(split: str) -> pd.DataFrame:
    d = pd.read_parquet(os.path.join(UNION_DIR, f"{split}.parquet"))
    d["bm25_rank_recip"] = np.where(d["bm25_rank"] >= 0, 1 / (d["bm25_rank"] + 1), 0.0)
    d["dense_rank_recip"] = np.where(d["dense_rank"] >= 0, 1 / (d["dense_rank"] + 1), 0.0)
    d = d.sort_values(["arxiv_id", "rrf_score"], ascending=[True, False]).groupby(
        "arxiv_id", group_keys=False
    ).head(INPUT_DEPTH).copy()
    return enrich(d)


def hard_rule(df: pd.DataFrame) -> np.ndarray:
    return (
        (df["abstract_len"] >= 50)
        & (df["tier_ord"] <= 3)
        & (df["year_diff"] >= 0)
    ).astype(float).values


def metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    keep = score >= threshold
    pos = y == 1
    neg = y == 0
    recall = (keep & pos).sum() / max(1, pos.sum())
    precision = (keep & pos).sum() / max(1, keep.sum())
    specificity = ((~keep) & neg).sum() / max(1, neg.sum())
    return {
        "gold_recall": float(recall), "gold_precision": float(precision),
        "gold_specificity": float(specificity), "gold_pool_kept_rate": float(keep.mean()),
        "gold_pool_reduction": float(1 - keep.mean()), "n_decided": int(len(y)),
    }


def choose_threshold(y: np.ndarray, score: np.ndarray) -> float:
    candidates = np.unique(np.quantile(score, np.linspace(0, 1, 1001)))
    valid = [t for t in candidates if metrics(y, score, t)["gold_recall"] >= GOLD_RECALL_TARGET]
    return float(max(valid)) if valid else float(np.min(score) - 1e-12)


def citation_safety(pool: pd.DataFrame, score: np.ndarray, threshold: float) -> dict:
    pool = pool.copy()
    pool["_keep"] = score >= threshold
    per = []
    for _, g in pool.groupby("arxiv_id"):
        positives = int(g["label"].sum())
        if positives:
            per.append(int(g.loc[g["_keep"], "label"].sum()) / positives)
    return {"input_citation_retention_macro": float(np.mean(per)) if per else 0.0,
            "input_citation_retention_micro": float(pool.loc[pool["_keep"], "label"].sum() / max(1, pool["label"].sum())),
            "mean_admitted_pool_size": float(pool.groupby("arxiv_id")["_keep"].sum().mean())}


def main() -> None:
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    os.makedirs(OUT_DIR, exist_ok=True)
    pools = {s: load_pool(s) for s in ["train", "val", "test"]}
    gold = {s: load_consensus(s) for s in ["train", "val", "test"]}
    labeled = {s: pools[s].merge(gold[s], on=["arxiv_id", "doc_id"], how="inner") for s in pools}

    # Primary analysis excludes panel uncertainty. Sensitivity is evaluated later.
    decided = {s: d[d["gold_binary"].isin(["admit", "exclude"])].copy() for s, d in labeled.items()}
    for s, d in decided.items():
        d["y"] = (d["gold_binary"] == "admit").astype(int)

    scaler = StandardScaler().fit(decided["train"][FEATURES])
    linear = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED)
    linear.fit(scaler.transform(decided["train"][FEATURES]), decided["train"]["y"])

    grid = []
    best = None
    for leaves in [7, 15, 31]:
        for min_leaf in [20, 40, 80]:
            model = LGBMClassifier(
                n_estimators=300, learning_rate=0.03, num_leaves=leaves,
                min_child_samples=min_leaf, class_weight="balanced",
                random_state=SEED, n_jobs=int(os.environ.get("S2C_NUM_THREADS", "32")),
                verbosity=-1,
            )
            model.fit(decided["train"][FEATURES], decided["train"]["y"])
            s = model.predict_proba(decided["val"][FEATURES])[:, 1]
            th = choose_threshold(decided["val"]["y"].values, s)
            ev = metrics(decided["val"]["y"].values, s, th)
            grid.append({"leaves": leaves, "min_leaf": min_leaf, "threshold": th, **ev})
            key = (ev["gold_pool_reduction"], ev["gold_precision"])
            if best is None or key > best["key"]:
                best = {"model": model, "threshold": th, "key": key, "val": ev,
                        "params": {"leaves": leaves, "min_leaf": min_leaf}}
    assert best is not None

    # Score every input candidate with the gold-trained filter.
    for s in pools:
        pools[s]["gold_F_score"] = best["model"].predict_proba(pools[s][FEATURES])[:, 1]
        # Add full-pool scores to sampled gold rows for gold-set evaluation.
        scored = pools[s][["arxiv_id", "doc_id", "gold_F_score"]]
        decided[s] = decided[s].merge(scored, on=["arxiv_id", "doc_id"], how="left")

    systems = {}
    for name, score_col in {
        "gold_trained_lgb_F": "gold_F_score",
        "hard_rule_F": "_hard_rule",
        "rrf_threshold": "rrf_score",
    }.items():
        for s in ["val", "test"]:
            pool = pools[s].copy()
            if name == "hard_rule_F":
                pool["_hard_rule"] = hard_rule(pool)
            lab = decided[s]
            score = lab[score_col].values if score_col in lab else (
                hard_rule(lab) if name == "hard_rule_F" else lab["rrf_score"].values
            )
            th = best["threshold"] if name == "gold_trained_lgb_F" else choose_threshold(
                decided["val"]["y"].values,
                (hard_rule(decided["val"]) if name == "hard_rule_F" else decided["val"]["rrf_score"].values),
            )
            systems.setdefault(name, {})[s] = {
                "gold_metrics": metrics(lab["y"].values, score, th),
                "citation_safety": citation_safety(pool, pool[score_col].values if score_col in pool else hard_rule(pool), th),
                "threshold": th,
            }

    # Test AUC for the learned filter, decided consensus only.
    test_scores = decided["test"]["gold_F_score"].values
    test_y = decided["test"]["y"].values
    systems["gold_trained_lgb_F"]["test"]["gold_auc"] = float(roc_auc_score(test_y, test_scores))
    # Per-item test table enables post-freeze paired bootstrap / sensitivity reporting.
    test_table = decided["test"][["arxiv_id", "doc_id", "stratum", "gold_binary", "y",
                                  "gold_F_score", "rrf_score"]].copy()
    test_table["keep_gold_F"] = test_table["gold_F_score"] >= systems["gold_trained_lgb_F"]["test"]["threshold"]
    test_table["keep_rrf_threshold"] = test_table["rrf_score"] >= systems["rrf_threshold"]["test"]["threshold"]
    test_table.to_json(
        os.path.join(OUT_DIR, "s2c_confirm_F_test_items.json"),
        orient="records", force_ascii=False, indent=2,
    )

    out = {
        "protocol": "blind three-model consensus proxy; not human gold",
        "input_depth": INPUT_DEPTH, "gold_recall_target": GOLD_RECALL_TARGET,
        "n_labeled": {s: {"all": int(len(labeled[s])), "decided": int(len(decided[s])),
                         "uncertain": int((labeled[s]["gold_binary"] == "uncertain").sum())} for s in labeled},
        "features": FEATURES, "grid": grid, "best": {"params": best["params"], "val": best["val"]},
        "systems": systems,
        "feature_importance": dict(zip(FEATURES, [int(x) for x in best["model"].feature_importances_])),
    }
    with open(os.path.join(OUT_DIR, "s2c_confirm_F.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out["best"], ensure_ascii=False, indent=2))
    print(json.dumps(out["systems"]["gold_trained_lgb_F"]["test"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
