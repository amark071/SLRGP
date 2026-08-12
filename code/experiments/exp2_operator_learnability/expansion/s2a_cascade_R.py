#!/usr/bin/env python3
"""Secondary S2a cascade: score E-expanded L pools with frozen S2d R."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from stats_utils import paired_bootstrap_ci, paired_test  # noqa: E402

CAND_DIR = "work/exp2_operator_learnability/ranking/candidates"
CACHE_DB = "work/exp2_operator_learnability/ranking/openalex_cache.db"
CORPUS_DB = "data/common/unified_corpus/unified_corpus.db"
S2D_FEAT_DIR = Path(os.environ.get("S2D_FEAT_DIR", "data/exp2_operator_learnability/ranking/features_s2d_tight"))
S2D_MODEL = os.environ.get("S2D_MODEL", "models/s2d_tight/lgb_main.txt")
S2A_OUT_DIR = Path(os.environ.get("S2A_OUT_DIR", "data/exp2_operator_learnability/expansion"))
S2A_RESULT_DIR = Path(os.environ.get("S2A_RESULT_DIR", "work/exp2_operator_learnability/expansion"))
PROD_DEPTH = 300
TIER_ORD = {"T1": 4, "T2": 3, "T3": 2, "PREPRINT": 1, "UNRANKED": 0}


def normalize_name(n: str | None) -> str:
    n = (n or "").lower().strip()
    n = re.sub(r"[^a-z ]", "", n)
    return re.sub(r"\s+", " ", n).strip()


def flat_author_names(authors_json_field) -> set[str]:
    try:
        items = json.loads(authors_json_field or "[]")
    except Exception:
        return set()
    names = set()
    for item in items:
        for part in str(item).split(","):
            nn = normalize_name(part)
            if len(nn) >= 4:
                names.add(nn)
    return names


def load_openalex_caches():
    conn = sqlite3.connect(CACHE_DB)
    works = {}
    for doc_id, found, cbc, aids_json in conn.execute(
        "SELECT doc_id, found, cited_by_count_openalex, author_ids FROM works_cache"
    ):
        works[doc_id] = {
            "found": bool(found),
            "cited_by_count_openalex": cbc,
            "author_ids": json.loads(aids_json) if aids_json else [],
        }
    conn.close()
    return works


def dcg(labels, k):
    return sum(l / np.log2(i + 2) for i, l in enumerate(labels[:k]))


def per_review_metrics(df: pd.DataFrame, score: str) -> dict[str, dict]:
    out = {}
    for aid, sub in df.groupby("arxiv_id"):
        npos = int(sub["label"].sum())
        if npos == 0:
            out[aid] = {"nDCG@10": 0.0, "Recall@50": 0.0, "MRR": 0.0, "npos": 0}
            continue
        order = np.argsort(-sub[score].values, kind="stable")
        ls = sub["label"].values[order]
        idcg = dcg(np.sort(sub["label"].values)[::-1], 10)
        first = np.argmax(ls == 1) + 1 if (ls == 1).any() else 0
        out[aid] = {
            "nDCG@10": dcg(ls, 10) / idcg if idcg > 0 else 0.0,
            "Recall@50": float(ls[:50].sum() / npos),
            "MRR": float(1.0 / first) if first else 0.0,
            "npos": npos,
        }
    return out


def load_review_lookup() -> dict[str, dict]:
    out = {}
    for p in glob.glob(os.path.join(CAND_DIR, "*", "*", "*.json")):
        r = json.load(open(p, encoding="utf-8"))
        out[r["arxiv_id"]] = r
    return out


def batch_meta(conn: sqlite3.Connection, ids: set[str]) -> dict:
    ids = list(ids)
    out = {}
    for i in range(0, len(ids), 800):
        chunk = ids[i:i + 800]
        ph = ",".join("?" for _ in chunk)
        for row in conn.execute(
            f"SELECT doc_id, year, tier, cited_by_count, discipline, abstract, title, authors "
            f"FROM papers WHERE doc_id IN ({ph})", chunk
        ):
            out[row[0]] = row
    return out


def build_expanded_features(split: str, n_terms: int, main_features: list[str]) -> pd.DataFrame:
    raw = pd.read_parquet(S2A_OUT_DIR / f"{split}_terms{n_terms}.parquet")
    raw = raw.sort_values(["arxiv_id", "rrf_score"], ascending=[True, False])
    raw = raw.groupby("arxiv_id", group_keys=False).head(PROD_DEPTH).copy()
    reviews = load_review_lookup()
    conn = sqlite3.connect(CORPUS_DB)
    ids = set(raw["doc_id"]) | {reviews[a]["review_doc_id"] for a in raw["arxiv_id"].unique() if a in reviews}
    meta = batch_meta(conn, ids)
    conn.close()
    works_cache = load_openalex_caches()
    rows = []
    for _, row in raw.iterrows():
        aid, did = row["arxiv_id"], row["doc_id"]
        rec = reviews[aid]
        cm = meta.get(did)
        rm = meta.get(rec["review_doc_id"])
        if cm is None:
            continue
        _, cyear, tier, _cbc, cdisc, abstract, title, authors_json = cm
        r_author_ids = set(works_cache.get(rec["review_doc_id"], {}).get("author_ids", []))
        c_author_ids = set(works_cache.get(did, {}).get("author_ids", []))
        r_author_names = flat_author_names(rm[7] if rm else None)
        c_author_names = flat_author_names(authors_json)
        if r_author_ids and c_author_ids:
            is_self = int(len(r_author_ids & c_author_ids) > 0)
            self_reliable = 1
        else:
            is_self = int(len(r_author_names & c_author_names) > 0)
            self_reliable = 0
        review_year = rec.get("review_year")
        year_diff = (review_year - cyear) if (review_year and cyear) else -1
        out = {
            "arxiv_id": aid, "doc_id": did, "label": int(row["label"]),
            "bm25_score": float(row["bm25_score"]),
            "bm25_rank_recip": 1.0 / (int(row["bm25_rank"]) + 1) if int(row["bm25_rank"]) >= 0 else 0.0,
            "dense_score": float(row["dense_score"]),
            "dense_rank_recip": 1.0 / (int(row["dense_rank"]) + 1) if int(row["dense_rank"]) >= 0 else 0.0,
            "rrf_score": float(row["rrf_score"]),
            "candidate_year": cyear if cyear else 0,
            "year_diff": year_diff,
            "year_missing": int(year_diff == -1),
            "tier_ord": TIER_ORD.get(tier, 0),
            "same_discipline": int(cdisc == rec["discipline"]),
            "abstract_len_words": len((abstract or "").split()),
            "title_len_words": len((title or "").split()),
            "is_self_citation": is_self,
            "self_citation_reliable": self_reliable,
            "n_authors": len(c_author_ids) if c_author_ids else len(c_author_names),
        }
        for f in main_features:
            out.setdefault(f, 0.0)
        rows.append(out)
    return pd.DataFrame(rows).sort_values(["arxiv_id", "rrf_score"], ascending=[True, False])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--term-count", type=int, required=True)
    args = ap.parse_args()
    S2A_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fm = json.load(open(S2D_FEAT_DIR / "feature_manifest.json", encoding="utf-8"))
    main_features = fm["main_features"]
    model = lgb.Booster(model_file=S2D_MODEL)

    baseline = pd.read_parquet(S2D_FEAT_DIR / f"{args.split}.parquet").copy()
    baseline["frozen_r"] = model.predict(baseline[main_features])
    expanded = build_expanded_features(args.split, args.term_count, main_features)
    expanded["frozen_r"] = model.predict(expanded[main_features])
    expanded.to_parquet(S2A_RESULT_DIR / f"{args.split}_terms{args.term_count}_r_features.parquet", index=False)

    bm = per_review_metrics(baseline, "frozen_r")
    em = per_review_metrics(expanded, "frozen_r")
    report = {"split": args.split, "term_count": args.term_count, "metrics": {}}
    for metric in ["nDCG@10", "Recall@50", "MRR"]:
        aids = sorted(set(bm) & set(em))
        diffs = [em[a][metric] - bm[a][metric] for a in aids]
        ci = paired_bootstrap_ci(diffs, n_boot=10000)
        pt = paired_test(diffs)
        report["metrics"][metric] = {
            "baseline_mean": float(np.mean([bm[a][metric] for a in aids])),
            "expanded_mean": float(np.mean([em[a][metric] for a in aids])),
            "mean_diff": ci.point_estimate, "ci_low": ci.ci_low, "ci_high": ci.ci_high,
            "wilcoxon_p": pt.wilcoxon_p, "permutation_p": pt.permutation_p,
            "rank_biserial": pt.rank_biserial, "n_reviews": len(aids),
        }
    with open(S2A_RESULT_DIR / f"s2a_cascade_R_{args.split}_terms{args.term_count}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
