#!/usr/bin/env python3
"""
S2c (Learning F) eligibility filtering / evidence-space admission.

This script deliberately treats F as a filter, not as another ranking endpoint:

  L: retrieve a broad pool (here: RRF top-500 from the honest S2b union).
  F: decide which located papers are admissible enough to enter the evidence space.
  R: prioritize among admissible papers (evaluated later as a cascade diagnostic).

Primary weak-supervision endpoint:
  Can a learned F reduce candidate-pool noise at the fixed downstream budget while
  preserving true-citation coverage better than simple rule/score filters?

Important label caveat:
  "uncited" is an observed negative, not proof of irrelevance. Therefore weak labels
  are used for model training and citation-retention diagnostics only. The script also
  emits a stratified audit sample for MODEL-A / human eligibility labelling.

Inputs:
  S2B_UNION_DIR=work/exp2_operator_learnability/ranking/union_L
  CORPUS_DB=data/common/unified_corpus/unified_corpus.db

Outputs:
  S2C_OUT_DIR=data/exp2_operator_learnability/filtering
    s2c_eval_F.json
    s2c_gold_audit_sample.jsonl
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
import glob
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from stats_utils import paired_bootstrap_ci, paired_test  # noqa: E402

UNION_DIR = os.environ.get("S2B_UNION_DIR", "work/exp2_operator_learnability/ranking/union_L")
CORPUS_DB = os.environ.get("S2C_CORPUS_DB", "data/common/unified_corpus/unified_corpus.db")
CAND_DIR = os.environ.get("S2C_CAND_DIR", "work/exp2_operator_learnability/ranking/candidates")
OUT_DIR = os.environ.get("S2C_OUT_DIR", "data/exp2_operator_learnability/filtering")

INPUT_DEPTH = int(os.environ.get("S2C_INPUT_DEPTH", "500"))
OUTPUT_BUDGET = int(os.environ.get("S2C_OUTPUT_BUDGET", "300"))
RETENTION_TARGET = float(os.environ.get("S2C_RETENTION_TARGET", "0.95"))
SEED = int(os.environ.get("S2C_SEED", "42"))
NUM_THREADS = int(os.environ.get("S2C_NUM_THREADS", "48"))

TIER_ORDER = {"T1": 1, "T2": 2, "T3": 3}
TEXT_PREVIEW_CHARS = 900

FEATURES_MAIN = [
    # L-derived semantic/retrieval boundary signals, not labels.
    "bm25_score",
    "bm25_rank_recip",
    "dense_score",
    "dense_rank_recip",
    "rrf_score",
    "year_diff",
    "same_discipline",
    # F-native qualification / metadata signals (manuscript Methods, operator F).
    "tier_ord",
    "has_tier",
    "has_doi",
    "has_venue",
    "abstract_len",
    "title_len",
    "source_is_stem",
    "source_is_social",
]

FEATURES_RULE = ["tier_ord", "has_tier", "has_abstract", "abstract_len", "year_diff", "same_discipline"]


@dataclass
class SystemEval:
    micro_coverage: float
    macro_coverage: float
    mean_pool_size: float
    median_pool_size: float
    candidate_reduction_vs_input: float
    observed_precision: float
    input_relative_retention_micro: float
    input_relative_retention_macro: float


def load_base() -> tuple[dict[str, pd.DataFrame], dict[tuple[str, str], int]]:
    dfs = {}
    for split in ["train", "val", "test"]:
        df = pd.read_parquet(os.path.join(UNION_DIR, f"{split}.parquet"))
        df["bm25_rank_recip"] = np.where(df["bm25_rank"] >= 0, 1.0 / (df["bm25_rank"] + 1), 0.0)
        df["dense_rank_recip"] = np.where(df["dense_rank"] >= 0, 1.0 / (df["dense_rank"] + 1), 0.0)
        dfs[split] = df
    raw = json.load(open(os.path.join(UNION_DIR, "P_of.json"), encoding="utf-8"))
    p_of = {(aid, v["split"]): int(v["P"]) for aid, v in raw.items()}
    return dfs, p_of


def enrich_metadata(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    doc_ids = sorted(set(pd.concat([d["doc_id"] for d in dfs.values()], ignore_index=True)))
    conn = sqlite3.connect(CORPUS_DB)
    meta = {}
    for i in range(0, len(doc_ids), 900):
        chunk = doc_ids[i:i + 900]
        ph = ",".join("?" for _ in chunk)
        for row in conn.execute(
            f"SELECT doc_id, source, title, abstract, venue, doi, tier FROM papers WHERE doc_id IN ({ph})",
            chunk,
        ):
            doc_id, source, title, abstract, venue, doi, tier = row
            title = title or ""
            abstract = abstract or ""
            meta[doc_id] = {
                "source": source or "",
                "title": title,
                "abstract": abstract,
                "venue": venue or "",
                "doi": doi or "",
                "tier": tier or "",
                "tier_ord": TIER_ORDER.get(tier or "", 4),
                "has_tier": int(bool(tier)),
                "has_doi": int(bool(doi)),
                "has_venue": int(bool(venue)),
                "has_abstract": int(bool(abstract.strip())),
                "abstract_len": len(abstract.split()),
                "title_len": len(title.split()),
                "source_is_stem": int((source or "").lower() == "stem"),
                "source_is_social": int("social" in (source or "").lower()),
            }
    for split, df in dfs.items():
        mdf = pd.DataFrame.from_dict(meta, orient="index")
        mdf.index.name = "doc_id"
        out = df.merge(mdf.reset_index(), on="doc_id", how="left")
        for col in ["tier_ord", "has_tier", "has_doi", "has_venue", "has_abstract",
                    "abstract_len", "title_len", "source_is_stem", "source_is_social"]:
            out[col] = out[col].fillna(0 if col != "tier_ord" else 4)
        out["title"] = out["title"].fillna("")
        out["abstract"] = out["abstract"].fillna("")
        dfs[split] = out
    return dfs


def load_review_context() -> dict[str, dict]:
    """Review title/abstract are required for gold eligibility annotation."""
    conn = sqlite3.connect(CORPUS_DB)
    out = {}
    for jf in glob.glob(os.path.join(CAND_DIR, "*", "*", "*.json")):
        rec = json.load(open(jf, encoding="utf-8"))
        aid = rec["arxiv_id"]
        rdid = rec.get("review_doc_id")
        row = conn.execute("SELECT title, abstract, year, discipline FROM papers WHERE doc_id=?", (rdid,)).fetchone()
        if not row:
            continue
        title, abstract, year, discipline = row
        out[aid] = {
            "review_doc_id": rdid,
            "review_title": title or "",
            "review_abstract": abstract or "",
            "review_year": year,
            "review_discipline": discipline or rec.get("discipline", ""),
        }
    return out


def top_input(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["arxiv_id", "rrf_score"], ascending=[True, False])
        .groupby("arxiv_id", group_keys=False)
        .head(INPUT_DEPTH)
        .copy()
    )


def add_rule_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Deterministic F: quality gate over tier / year / abstract availability.
    # These are intentionally conservative and separately reported from learned semantic F.
    out["hard_rule_pass"] = (
        (out["has_abstract"] == 1)
        & (out["abstract_len"] >= 50)
        & (out["tier_ord"] <= 3)
        & (out["year_diff"] >= 0)
    ).astype(int)
    out["metadata_rule_score"] = (
        out["hard_rule_pass"] * 10.0
        + (4 - out["tier_ord"].clip(1, 4)) * 0.4
        + np.log1p(out["abstract_len"].clip(lower=0)) * 0.05
        + out["same_discipline"] * 0.2
        + out["has_doi"] * 0.1
    )
    return out


def train_models(dfs: dict[str, pd.DataFrame]) -> tuple[object, object, dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb

    tr = dfs["train"]
    va = dfs["val"]

    scaler = StandardScaler().fit(tr[FEATURES_MAIN].fillna(0).values)
    logit = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    logit.fit(scaler.transform(tr[FEATURES_MAIN].fillna(0).values), tr["label"].values)

    groups = tr.groupby("arxiv_id").size().values
    ds = lgb.Dataset(tr[FEATURES_MAIN], label=tr["label"], group=groups)
    grid = []
    best = None
    for leaves in [15, 31, 63]:
        for lr in [0.03, 0.05, 0.1]:
            for min_leaf in [50, 100, 200]:
                params = {
                    "objective": "binary",
                    "metric": "binary_logloss",
                    "num_leaves": leaves,
                    "learning_rate": lr,
                    "min_data_in_leaf": min_leaf,
                    "feature_fraction": 0.8,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 5,
                    "num_threads": NUM_THREADS,
                    "seed": SEED,
                    "verbosity": -1,
                }
                booster = lgb.train(params, ds, num_boost_round=300)
                va["_tmp_lgb"] = booster.predict(va[FEATURES_MAIN])
                ev = eval_budgeted(va, "val", "_tmp_lgb", OUTPUT_BUDGET)
                score = ev["input_relative_retention_macro"]
                grid.append({"params": [leaves, lr, min_leaf], "val_retention_macro": score})
                if best is None or score > best["score"]:
                    best = {"score": score, "params": [leaves, lr, min_leaf], "booster": booster}
    assert best is not None
    meta = {
        "features": FEATURES_MAIN,
        "logistic": {"class_weight": "balanced"},
        "lgb_grid": grid,
        "lgb_best": {"params": best["params"], "val_retention_macro": best["score"]},
        "lgb_feature_importance": dict(
            zip(FEATURES_MAIN, [int(x) for x in best["booster"].feature_importance()])
        ),
    }
    return (scaler, logit), best["booster"], meta


def single_feature_probe(dfs: dict[str, pd.DataFrame], best_params: list) -> dict:
    """Diagnostic, not an automatic abort.

    For F, a strong single retrieval feature is not leakage by itself because L's
    ordering is a legitimate upstream input. But if one feature nearly matches the
    full model, the manuscript must not over-claim semantic eligibility learning.
    """
    import lightgbm as lgb

    tr = dfs["train"]
    va = dfs["val"]
    full = eval_budgeted(va, "val", "lgb_F", OUTPUT_BUDGET)
    per_feature = {}
    for feat in FEATURES_MAIN:
        ds = lgb.Dataset(tr[[feat]], label=tr["label"], group=tr.groupby("arxiv_id").size().values)
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": best_params[0],
            "learning_rate": best_params[1],
            "min_data_in_leaf": best_params[2],
            "num_threads": NUM_THREADS,
            "seed": SEED,
            "verbosity": -1,
        }
        booster = lgb.train(params, ds, num_boost_round=200)
        va[f"_sf_{feat}"] = booster.predict(va[[feat]])
        ev = eval_budgeted(va, "val", f"_sf_{feat}", OUTPUT_BUDGET)
        per_feature[feat] = {
            "val_micro_coverage": ev["micro_coverage"],
            "val_macro_coverage": ev["macro_coverage"],
            "val_input_relative_retention_macro": ev["input_relative_retention_macro"],
        }
    threshold = 0.95 * full["macro_coverage"]
    flagged = {k: v for k, v in per_feature.items() if v["val_macro_coverage"] >= threshold}
    return {
        "full_val_budgeted": full,
        "threshold_95pct_full_macro_coverage": threshold,
        "per_feature": per_feature,
        "flagged_near_full": flagged,
        "interpretation": (
            "Near-full single retrieval features indicate that weak citation-supervised F "
            "is largely admission scoring over L signals; gold eligibility audit is required "
            "before claiming semantic qualification learning."
        ),
    }


def score_models(dfs: dict[str, pd.DataFrame], logistic_bundle: object, booster: object) -> None:
    scaler, logit = logistic_bundle
    for split, df in dfs.items():
        x = df[FEATURES_MAIN].fillna(0).values
        df["logistic_F"] = logit.predict_proba(scaler.transform(x))[:, 1]
        df["lgb_F"] = booster.predict(df[FEATURES_MAIN])
        # Hybrid F: symbolic quality first, learned semantic/metadata score second.
        df["hybrid_F"] = df["hard_rule_pass"] * 0.15 + df["lgb_F"]
        dfs[split] = df


def _selected_by_score(df: pd.DataFrame, score_col: str, budget: int) -> dict[str, pd.DataFrame]:
    return {
        aid: g.nlargest(budget, score_col).copy()
        for aid, g in df.groupby("arxiv_id")
    }


def _input_groups(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {aid: g for aid, g in df.groupby("arxiv_id")}


def _compute_eval(selected: dict[str, pd.DataFrame], base: dict[str, pd.DataFrame],
                  split: str, p_of: dict[tuple[str, str], int]) -> SystemEval:
    total_hit = total_den = total_sel = total_input = 0
    per_cov = []
    per_rel = []
    pool_sizes = []
    for (aid, s), p in p_of.items():
        if s != split or p == 0:
            continue
        inp = base.get(aid)
        sel = selected.get(aid)
        input_hit = int(inp["label"].sum()) if inp is not None else 0
        hit = int(sel["label"].sum()) if sel is not None else 0
        size = len(sel) if sel is not None else 0
        in_size = len(inp) if inp is not None else 0
        total_hit += hit
        total_den += p
        total_sel += size
        total_input += in_size
        per_cov.append(hit / p)
        if input_hit > 0:
            per_rel.append(hit / input_hit)
        else:
            per_rel.append(1.0 if hit == 0 else 0.0)
        pool_sizes.append(size)
    observed_precision = total_hit / total_sel if total_sel else 0.0
    input_hits = sum(int(g["label"].sum()) for g in base.values())
    return SystemEval(
        micro_coverage=total_hit / total_den if total_den else 0.0,
        macro_coverage=float(np.mean(per_cov)) if per_cov else 0.0,
        mean_pool_size=float(np.mean(pool_sizes)) if pool_sizes else 0.0,
        median_pool_size=float(np.median(pool_sizes)) if pool_sizes else 0.0,
        candidate_reduction_vs_input=1.0 - (total_sel / total_input) if total_input else 0.0,
        observed_precision=observed_precision,
        input_relative_retention_micro=total_hit / input_hits if input_hits else 0.0,
        input_relative_retention_macro=float(np.mean(per_rel)) if per_rel else 0.0,
    )


def eval_budgeted(df: pd.DataFrame, split: str, score_col: str, budget: int,
                  p_of: dict[tuple[str, str], int] | None = None) -> dict:
    # During validation-grid selection p_of can be omitted; use in-pool positives as denominator.
    if p_of is None:
        p_of = {(aid, split): max(1, int(g["label"].sum())) for aid, g in df.groupby("arxiv_id")}
    base = _input_groups(df)
    selected = _selected_by_score(df, score_col, budget)
    return asdict(_compute_eval(selected, base, split, p_of))


def eval_threshold(df: pd.DataFrame, split: str, score_col: str, threshold: float,
                   p_of: dict[tuple[str, str], int]) -> dict:
    base = _input_groups(df)
    selected = {aid: g[g[score_col] >= threshold].copy() for aid, g in df.groupby("arxiv_id")}
    return asdict(_compute_eval(selected, base, split, p_of))


def per_review_retention(df: pd.DataFrame, split: str, score_col: str, budget: int,
                         p_of: dict[tuple[str, str], int]) -> dict[str, float]:
    base = _input_groups(df)
    selected = _selected_by_score(df, score_col, budget)
    out = {}
    for (aid, s), p in p_of.items():
        if s != split or p == 0:
            continue
        hit = int(selected.get(aid, pd.DataFrame({"label": []}))["label"].sum()) if aid in selected else 0
        out[aid] = hit / p
    return out


def choose_threshold_for_retention(df: pd.DataFrame, split: str, score_col: str,
                                   p_of: dict[tuple[str, str], int]) -> tuple[float, dict]:
    # Search score quantiles; choose the highest threshold that retains target input positives.
    scores = df[score_col].values
    qs = np.linspace(0.01, 0.99, 99)
    thresholds = sorted(set(float(np.quantile(scores, q)) for q in qs))
    best = None
    for th in thresholds:
        ev = eval_threshold(df, split, score_col, th, p_of)
        if ev["input_relative_retention_micro"] >= RETENTION_TARGET:
            if best is None or ev["candidate_reduction_vs_input"] > best[1]["candidate_reduction_vs_input"]:
                best = (th, ev)
    if best is None:
        th = float(np.min(scores) - 1e-12)
        return th, eval_threshold(df, split, score_col, th, p_of)
    return best


def paired_contrast(df: pd.DataFrame, split: str, score_a: str, score_b: str,
                    p_of: dict[tuple[str, str], int]) -> dict:
    pa = per_review_retention(df, split, score_a, OUTPUT_BUDGET, p_of)
    pb = per_review_retention(df, split, score_b, OUTPUT_BUDGET, p_of)
    aids = sorted(set(pa) & set(pb))
    diffs = [pa[a] - pb[a] for a in aids]
    ci = paired_bootstrap_ci(diffs)
    pt = paired_test(diffs)
    return {
        "n": len(aids),
        "mean_diff": ci.point_estimate,
        "ci_low": ci.ci_low,
        "ci_high": ci.ci_high,
        "ci_method": ci.method,
        "wilcoxon_p": pt.wilcoxon_p,
        "permutation_p": pt.permutation_p,
        "rank_biserial": pt.rank_biserial,
    }


def make_audit_sample(df: pd.DataFrame, out_path: str, review_context: dict[str, dict],
                      n_total: int = 500) -> dict:
    rng = np.random.default_rng(SEED)
    work = df.copy()
    work["stratum"] = "ambiguous_midscore"
    work.loc[(work["label"] == 1), "stratum"] = "citation_positive"
    work.loc[(work["hard_rule_pass"] == 0) & (work["label"] == 0), "stratum"] = "rule_clear_negative"
    work.loc[(work["label"] == 0) & (work["rrf_score"] >= work["rrf_score"].quantile(0.90)), "stratum"] = "semantic_hard_negative"
    work.loc[(work["label"] == 0) & (work["lgb_F"] >= work["lgb_F"].quantile(0.90)), "stratum"] = "model_high_uncited"
    # Fixed proportions, scaled per split. Sampling strata are written to disk for
    # later population reweighting but never exposed to blind annotator prompts.
    proportions = {
        "citation_positive": 0.20,
        "rule_clear_negative": 0.20,
        "semantic_hard_negative": 0.24,
        "model_high_uncited": 0.24,
        "ambiguous_midscore": 0.12,
    }
    quotas = {k: round(v * n_total) for k, v in proportions.items()}
    quotas["ambiguous_midscore"] += n_total - sum(quotas.values())
    samples = []
    for stratum, quota in quotas.items():
        sub = work[work["stratum"] == stratum]
        if len(sub) == 0:
            continue
        take = min(quota, len(sub))
        idx = rng.choice(sub.index.values, size=take, replace=False)
        samples.append(work.loc[idx])
    samp = pd.concat(samples, ignore_index=True).head(n_total)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in samp.iterrows():
            item = {
                "sample_id": f"s2c_gold_{i:04d}",
                "arxiv_id": row["arxiv_id"],
                "doc_id": row["doc_id"],
                "stratum": row["stratum"],
                "weak_label_cited": int(row["label"]),
                "hard_rule_pass": int(row["hard_rule_pass"]),
                "rrf_score": float(row["rrf_score"]),
                "lgb_F": float(row["lgb_F"]),
                "metadata": {
                    "source": row.get("source", ""),
                    "tier": row.get("tier", ""),
                    "venue": row.get("venue", ""),
                    "doi": row.get("doi", ""),
                    "year_diff": float(row.get("year_diff", math.nan)),
                    "same_discipline": int(row.get("same_discipline", 0)),
                    "abstract_len": int(row.get("abstract_len", 0)),
                },
                "review": {
                    "title": review_context.get(row["arxiv_id"], {}).get("review_title", ""),
                    "abstract": (
                        review_context.get(row["arxiv_id"], {}).get("review_abstract", "") or ""
                    )[:TEXT_PREVIEW_CHARS],
                    "year": review_context.get(row["arxiv_id"], {}).get("review_year"),
                    "discipline": review_context.get(row["arxiv_id"], {}).get("review_discipline", ""),
                },
                "candidate": {
                    "title": row.get("title", ""),
                    "abstract_preview": (row.get("abstract", "") or "")[:TEXT_PREVIEW_CHARS],
                },
                "annotation_task": {
                    "question": "Is this candidate eligible for consideration in the target review evidence space, ignoring whether the review actually cited it?",
                    "labels": ["eligible", "borderline", "out_of_scope", "wrong_document_type", "post_cutoff_or_temporal_issue", "insufficient_metadata"],
                    "notes": "Uncited does not imply ineligible; judge scope, document type, metadata adequacy, and temporal admissibility.",
                },
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return {"n": int(len(samp)), "by_stratum": samp["stratum"].value_counts().to_dict()}


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    dfs, p_of = load_base()
    review_context = load_review_context()
    dfs = enrich_metadata(dfs)
    for split in ["train", "val", "test"]:
        dfs[split] = add_rule_scores(top_input(dfs[split]))
        print(f"[{split}] input rows={len(dfs[split])} reviews={dfs[split]['arxiv_id'].nunique()}")

    logistic_bundle, booster, model_meta = train_models(dfs)
    score_models(dfs, logistic_bundle, booster)

    systems = {
        "rrf_top300_noF": "rrf_score",
        "metadata_rule_F": "metadata_rule_score",
        "logistic_F": "logistic_F",
        "lgb_F": "lgb_F",
        "hybrid_symbolic_lgb_F": "hybrid_F",
    }

    res = {
        "task": "S2c Learning F",
        "input_depth": INPUT_DEPTH,
        "output_budget": OUTPUT_BUDGET,
        "retention_target": RETENTION_TARGET,
        "label_caveat": "citation labels are weak admissibility evidence; uncited candidates are observed negatives, not asserted irrelevant",
        "features_main": FEATURES_MAIN,
        "features_rule": FEATURES_RULE,
        "model_meta": model_meta,
        "systems_budgeted": {},
        "systems_retention_threshold": {},
        "paired_budgeted_vs_rrf": {},
    }
    res["single_feature_probe"] = single_feature_probe(dfs, model_meta["lgb_best"]["params"])

    for name, score_col in systems.items():
        res["systems_budgeted"][name] = {
            split: eval_budgeted(dfs[split], split, score_col, OUTPUT_BUDGET, p_of)
            for split in ["val", "test"]
        }
        th, val_ev = choose_threshold_for_retention(dfs["val"], "val", score_col, p_of)
        res["systems_retention_threshold"][name] = {
            "threshold_selected_on_val": th,
            "val": val_ev,
            "test": eval_threshold(dfs["test"], "test", score_col, th, p_of),
        }
        if name != "rrf_top300_noF":
            res["paired_budgeted_vs_rrf"][name] = paired_contrast(
                dfs["test"], "test", score_col, "rrf_score", p_of
            )
        print(
            f"[{name}] budgeted test cov={res['systems_budgeted'][name]['test']['micro_coverage']:.4f} "
            f"macro={res['systems_budgeted'][name]['test']['macro_coverage']:.4f} "
            f"precision={res['systems_budgeted'][name]['test']['observed_precision']:.4f}"
        )

    # Label-permutation sentinel for the strongest learned family (LGB): retrain on permuted labels.
    import lightgbm as lgb
    tr = dfs["train"].copy()
    rng = np.random.default_rng(SEED)
    tr["label"] = tr.groupby("arxiv_id")["label"].transform(lambda s: rng.permutation(s.values))
    ds = lgb.Dataset(tr[FEATURES_MAIN], label=tr["label"], group=tr.groupby("arxiv_id").size().values)
    best_params = model_meta["lgb_best"]["params"]
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": best_params[0],
        "learning_rate": best_params[1],
        "min_data_in_leaf": best_params[2],
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "num_threads": NUM_THREADS,
        "seed": SEED,
        "verbosity": -1,
    }
    pbooster = lgb.train(params, ds, num_boost_round=300)
    dfs["test"]["lgb_permuted_F"] = pbooster.predict(dfs["test"][FEATURES_MAIN])
    res["systems_budgeted"]["lgb_permuted_sentinel"] = {
        "test": eval_budgeted(dfs["test"], "test", "lgb_permuted_F", OUTPUT_BUDGET, p_of)
    }

    # Multi-model gold-proxy samples. Train / val / test never share reviews.
    # Each split is annotated independently; the blind panel does not see citation
    # labels, learned scores or sampling stratum.
    audit_specs = {"train": 800, "val": 250, "test": 500}
    res["gold_proxy_samples"] = {}
    for split, n in audit_specs.items():
        sample_path = os.path.join(OUT_DIR, f"s2c_gold_proxy_{split}.jsonl")
        res["gold_proxy_samples"][split] = make_audit_sample(
            dfs[split], sample_path, review_context, n_total=n
        )

    out_path = os.path.join(OUT_DIR, "s2c_eval_F.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"写入 {out_path}")
    print(f"写入多模型 gold-proxy 样本: {audit_specs}")


if __name__ == "__main__":
    main()
