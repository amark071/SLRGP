#!/usr/bin/env python3
"""
S2a/E — supervised conceptual expansion followed by honest retrieval.

Test-time inputs are restricted to review title, abstract, discipline and
top-50 BM25 pseudo-relevance-feedback documents. Citation-linked papers are
used only to label candidate terms in the training split. The learned term
selector is a logistic model over generic PRF/discipline statistics; it never
receives a test review's bibliography, section headings or cited-paper titles.

Run in order:
  1) --fit --split val --term-counts 3,6,10       (select term count on val)
  2) --split test --term-counts <frozen-count>    (score test once)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, ".")
sys.path.insert(0, "code/slrgp")

CAND_DIR = "work/exp2_operator_learnability/ranking/candidates"
CORPUS_DB = "data/common/unified_corpus/unified_corpus.db"
TIGHT_DIR = "data/exp2_operator_learnability/ranking/resolved_tight"
SPLIT_MANIFEST = "data/exp2_operator_learnability/ranking/features_s2d_tight/split_manifest.json"
OUT_DIR = Path(os.environ.get("S2A_OUT_DIR", "data/exp2_operator_learnability/expansion"))
TOPK_EACH = 3000
PRF_K = 50
RRF_K = 60
TOKEN_RE = re.compile(r"[a-z][a-z\-]{2,}")
STOP = {
    "the", "and", "for", "with", "from", "that", "this", "into", "using", "based",
    "study", "studies", "review", "reviews", "paper", "papers", "approach", "methods",
    "method", "results", "analysis", "data", "model", "models", "new", "also", "are",
    "was", "were", "has", "have", "can", "may", "not", "our", "their", "its",
}


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall((text or "").lower()) if t not in STOP]


def load_tight() -> dict[str, set[str]]:
    out = {}
    for p in glob.glob(os.path.join(TIGHT_DIR, "*", "*.json")):
        r = json.load(open(p, encoding="utf-8"))
        out[r["arxiv_id"]] = set(r["matched_doc_ids"])
    return out


def load_reviews(conn: sqlite3.Connection, assignments: dict[str, str]) -> list[dict]:
    reviews = []
    for p in sorted(glob.glob(os.path.join(CAND_DIR, "*", "*", "*.json"))):
        r = json.load(open(p, encoding="utf-8"))
        aid = r["arxiv_id"]
        split = assignments.get(aid)
        if not split:
            continue
        row = conn.execute(
            "SELECT title, abstract FROM papers WHERE doc_id=?", (r["review_doc_id"],)
        ).fetchone()
        if not row or not row[0]:
            continue
        reviews.append({
            "arxiv_id": aid, "split": split, "review_doc_id": r["review_doc_id"],
            "review_year": r.get("review_year"), "discipline": r["discipline"],
            "title": row[0] or "", "abstract": row[1] or "",
        })
    return reviews


def paper_text(conn: sqlite3.Connection, doc_id: str) -> str:
    row = conn.execute("SELECT title, abstract FROM papers WHERE doc_id=?", (doc_id,)).fetchone()
    return ((row[0] or "") + " " + (row[1] or "")) if row else ""


def build_discipline_salience(conn: sqlite3.Connection, reviews: list[dict],
                              tight: dict[str, set[str]]) -> dict[str, Counter]:
    """Training-only cited-paper token frequencies, normalized later by totals."""
    counts = defaultdict(Counter)
    for i, r in enumerate(reviews):
        if r["split"] != "train":
            continue
        for did in tight.get(r["arxiv_id"], set()):
            counts[r["discipline"]].update(set(tokens(paper_text(conn, did))))
        if (i + 1) % 100 == 0:
            print(f"train supervision lexicon {i + 1}/{len(reviews)}", flush=True)
    return counts


def prf_candidates(index, conn: sqlite3.Connection, query: str) -> tuple[list[tuple[str, float]], Counter]:
    hits = index._bm25_search(query, PRF_K)
    tf = Counter()
    for did, _ in hits:
        tf.update(set(tokens(paper_text(conn, did))))
    return hits, tf


def feature_row(term: str, prf_tf: Counter, salience: Counter, salience_total: int,
                query_terms: set[str]) -> list[float]:
    return [
        prf_tf[term] / PRF_K,
        np.log1p(salience[term]) / np.log1p(max(salience_total, 1)),
        float(term in query_terms),
        min(len(term), 20) / 20,
    ]


def fit_selector(index, conn, reviews, tight) -> dict:
    salience = build_discipline_salience(conn, reviews, tight)
    X, y = [], []
    rng = np.random.default_rng(42)
    train = [r for r in reviews if r["split"] == "train"]
    for i, r in enumerate(train):
        base_query = r["title"] + ". " + r["abstract"][:1000]
        _, prf_tf = prf_candidates(index, conn, base_query)
        qterms = set(tokens(base_query))
        pos_tokens = set()
        for did in tight.get(r["arxiv_id"], set()):
            pos_tokens.update(tokens(paper_text(conn, did)))
        candidates = sorted(prf_tf)
        positives = [t for t in candidates if t in pos_tokens]
        negatives = [t for t in candidates if t not in pos_tokens]
        # Balance term examples per review to avoid PRF-vocabulary size domination.
        if not positives or not negatives:
            continue
        negatives = rng.choice(negatives, size=min(len(negatives), max(10, 3 * len(positives))),
                               replace=False).tolist()
        st = salience[r["discipline"]]
        total = sum(st.values())
        for t in positives + negatives:
            X.append(feature_row(t, prf_tf, st, total, qterms))
            y.append(int(t in pos_tokens))
        if (i + 1) % 50 == 0:
            print(f"selector examples {i + 1}/{len(train)} n={len(y)}", flush=True)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(np.asarray(X), np.asarray(y))
    return {"model": model, "salience": dict(salience), "n_examples": len(y),
            "feature_names": ["prf_document_fraction", "discipline_citation_salience",
                              "in_initial_query", "token_length"]}


def select_terms(bundle: dict, r: dict, prf_tf: Counter, n_terms: int) -> list[str]:
    model = bundle["model"]
    salience = Counter(bundle["salience"].get(r["discipline"], {}))
    total = sum(salience.values())
    qterms = set(tokens(r["title"] + ". " + r["abstract"][:1000]))
    candidates = [t for t in prf_tf if t not in qterms and len(t) >= 4]
    if not candidates:
        return []
    X = np.asarray([feature_row(t, prf_tf, salience, total, qterms) for t in candidates])
    scores = model.predict_proba(X)[:, 1]
    return [t for _, t in sorted(zip(scores, candidates), reverse=True)[:n_terms]]


def retrieve_one(index, query: str, review_doc_id: str, review_year, review_disc, meta: dict) -> list[dict]:
    bm25 = index._bm25_search(query, TOPK_EACH)
    dense = index._dense_search(query, TOPK_EACH)
    br = {d: i for i, (d, _) in enumerate(bm25)}
    bs = dict(bm25)
    dr = {d: i for i, (d, _) in enumerate(dense)}
    ds = dict(dense)
    rrf = {}
    for d, rank in br.items():
        rrf[d] = rrf.get(d, 0.0) + 1 / (RRF_K + rank + 1)
    for d, rank in dr.items():
        rrf[d] = rrf.get(d, 0.0) + 1 / (RRF_K + rank + 1)
    rows = []
    for did in (set(br) | set(dr)) - {review_doc_id}:
        cy, cd = meta.get(did, (None, None))
        if review_year and cy and cy > review_year:
            continue
        rows.append({
            "doc_id": did, "bm25_rank": br.get(did, -1), "bm25_score": bs.get(did, 0.0),
            "dense_rank": dr.get(did, -1), "dense_score": ds.get(did, 0.0),
            "rrf_score": rrf.get(did, 0.0), "year_diff": (review_year - cy) if (review_year and cy) else -1,
            "same_discipline": int(cd == review_disc),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true", help="fit selector from train split only")
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--term-counts", default="3,6,10")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = [int(x) for x in args.term_counts.split(",")]

    manifest = json.load(open(SPLIT_MANIFEST, encoding="utf-8"))
    conn = sqlite3.connect(CORPUS_DB)
    reviews = load_reviews(conn, manifest["assignments"])
    tight = load_tight()
    meta = {d: (y, disc) for d, y, disc in conn.execute("SELECT doc_id, year, discipline FROM papers")}
    from slrgp.retrieval import UnifiedIndex
    index = UnifiedIndex()

    bundle_path = OUT_DIR / "term_selector.joblib"
    if args.fit:
        bundle = fit_selector(index, conn, reviews, tight)
        joblib.dump(bundle, bundle_path)
        with open(OUT_DIR / "selector_manifest.json", "w", encoding="utf-8") as f:
            json.dump({"training_split": "train", "n_examples": bundle["n_examples"],
                       "features": bundle["feature_names"], "input_policy": "title+abstract+discipline+BM25-PRF only"},
                      f, ensure_ascii=False, indent=2)
    else:
        bundle = joblib.load(bundle_path)

    rows = {n: [] for n in counts}
    expansions = {n: [] for n in counts}
    subset = [r for r in reviews if r["split"] == args.split]
    for i, r in enumerate(subset):
        base = r["title"] + ". " + r["abstract"][:1000]
        _, prf_tf = prf_candidates(index, conn, base)
        for n in counts:
            terms = select_terms(bundle, r, prf_tf, n)
            query = base + " . " + " ".join(terms)
            for cand in retrieve_one(index, query, r["review_doc_id"], r["review_year"], r["discipline"], meta):
                cand.update({"arxiv_id": r["arxiv_id"], "split": args.split,
                             "label": int(cand["doc_id"] in tight.get(r["arxiv_id"], set()))})
                rows[n].append(cand)
            expansions[n].append({"arxiv_id": r["arxiv_id"], "terms": terms, "query": query})
        if (i + 1) % 20 == 0:
            print(f"{args.split} retrieval {i + 1}/{len(subset)}", flush=True)

    for n in counts:
        pd.DataFrame(rows[n]).to_parquet(OUT_DIR / f"{args.split}_terms{n}.parquet", index=False)
        with open(OUT_DIR / f"{args.split}_terms{n}.jsonl", "w", encoding="utf-8") as f:
            for row in expansions[n]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {args.split} variants n={counts} to {OUT_DIR}")


if __name__ == "__main__":
    main()
