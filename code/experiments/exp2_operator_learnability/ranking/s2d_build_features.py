#!/usr/bin/env python3
"""
S2d 特征重建（一步覆盖 split / pool_rebuild / feature_hygiene 三项）。

关键设计：
  1) SHA-256 综述分组切分 train/val/test = 70/15/15（salt="s2d_v1"），同综述的样本不跨集合。
  2) top-300 生产接口对齐：每篇综述只保留 retrieved_by_L 候选里按 rrf 降序的前 300 —— R 只
     对 L 实际交付的候选重排。落在 top-300 外的正例（forced/漏召回）从排序数据剔除，只进覆盖诊断。
  3) 时间安全：候选 year > 综述 year 的一律先剔除（不能引用未来）。
  4) 特征卫生：删 retrieved_by_L（标签代理）与 handcrafted_score（经验公式输出，且不作为对照臂）。
     快照声望特征 cited_by_count* / *_hindex 归入 AUX 列，主模型不用（trainer 侧选择）。
     时间安全的结构特征（is_self_citation / n_authors / same_discipline / recency）留在主特征。

输出：data/exp2_operator_learnability/ranking/features_s2d/{train,val,test}.parquet + *_groups.npy
      split_manifest.json、feature_manifest.json、coverage_report.json
"""
import argparse
import glob
import json
import os
import re
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split_utils import SplitManifest  # noqa: E402

CAND_DIR = "work/exp2_operator_learnability/ranking/candidates"
CACHE_DB = "work/exp2_operator_learnability/ranking/openalex_cache.db"
CORPUS_DB = "data/common/unified_corpus/unified_corpus.db"
TIGHT_DIR = "data/exp2_operator_learnability/ranking/resolved_tight"
OUT_DIR = "data/exp2_operator_learnability/ranking/features_s2d"

PROD_DEPTH = 300
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SALT = "s2d_v1"

TIER_ORD = {"T1": 4, "T2": 3, "T3": 2, "PREPRINT": 1, "UNRANKED": 0}

MAIN_FEATURES = [
    "bm25_score", "bm25_rank_recip", "dense_score", "dense_rank_recip", "rrf_score",
    "candidate_year", "year_diff", "year_missing",
    "tier_ord", "same_discipline", "abstract_len_words", "title_len_words",
    "is_self_citation", "self_citation_reliable", "n_authors",
]
AUX_FEATURES = [
    "cited_by_count_log", "cited_by_count_openalex_log",
    "max_author_hindex", "mean_author_hindex", "author_hindex_missing",
]


def normalize_name(n):
    n = (n or "").lower().strip()
    n = re.sub(r"[^a-z ]", "", n)
    return re.sub(r"\s+", " ", n).strip()


def flat_author_names(authors_json_field):
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
    authors = {}
    for aid, found, h in conn.execute("SELECT author_id, found, h_index FROM authors_cache WHERE found=1"):
        authors[aid] = h if h is not None else 0
    conn.close()
    print(f"OpenAlex 缓存: works={len(works)}  authors(h)={len(authors)}")
    return works, authors


def author_hindex_stats(author_ids, authors_hindex):
    hs = [authors_hindex[a] for a in author_ids if a in authors_hindex]
    if not hs:
        return 0.0, 0.0, True
    return float(max(hs)), float(sum(hs) / len(hs)), False


def load_tight_labels():
    """收紧后每篇综述的正例 doc_id 集合（audited 标签）。"""
    tight = {}
    for jf in glob.glob(os.path.join(TIGHT_DIR, "*", "*.json")):
        r = json.load(open(jf, encoding="utf-8"))
        tight[r["arxiv_id"]] = set(r["matched_doc_ids"])
    return tight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tight", action="store_true", help="用 resolved_tight 收紧标签覆盖，输出 features_s2d_tight")
    args = ap.parse_args()
    global OUT_DIR
    tight_labels = None
    if args.tight:
        OUT_DIR = OUT_DIR + "_tight"
        tight_labels = load_tight_labels()
        print(f"收紧模式：载入 {len(tight_labels)} 篇收紧标签")

    os.makedirs(OUT_DIR, exist_ok=True)
    works_cache, authors_hindex = load_openalex_caches()
    conn = sqlite3.connect(CORPUS_DB)

    files = sorted(glob.glob(os.path.join(CAND_DIR, "*", "*", "*.json")))
    print(f"候选池综述文件: {len(files)}")

    # 载入全部 review 记录到内存，先建切分清单（按 arxiv_id）
    reviews = [json.load(open(jf, encoding="utf-8")) for jf in files]
    arxiv_ids = [r["arxiv_id"] for r in reviews]
    manifest = SplitManifest.build("s2d", arxiv_ids, SPLIT_RATIOS, salt=SPLIT_SALT)
    manifest.to_json(os.path.join(OUT_DIR, "split_manifest.json"))
    print(f"切分: {manifest.__dict__.get('ratios')}  桶计数="
          f"{ {b: sum(1 for v in manifest.assignments.values() if v==b) for b in SPLIT_RATIOS} }")

    # 批量取全部候选 + 综述本体的语料库元数据
    all_ids = set()
    for r in reviews:
        all_ids.add(r["review_doc_id"])
        for c in r["candidates"]:
            all_ids.add(c["doc_id"])
    meta = {}
    ids = list(all_ids)
    CH = 800
    for i in range(0, len(ids), CH):
        chunk = ids[i:i + CH]
        ph = ",".join("?" for _ in chunk)
        for row in conn.execute(
            f"SELECT doc_id, year, tier, cited_by_count, discipline, abstract, title, authors "
            f"FROM papers WHERE doc_id IN ({ph})", chunk
        ):
            meta[row[0]] = row
    print(f"语料库元数据载入: {len(meta)} 条")

    rows_by_split = {s: [] for s in SPLIT_RATIOS}
    groups_by_split = {s: [] for s in SPLIT_RATIOS}
    cov = {"per_review": [], "micro_pos_timesafe": 0, "micro_pos_in_interface": 0}

    for ri, rec in enumerate(reviews):
        arxiv_id = rec["arxiv_id"]
        split = manifest.assignments[arxiv_id]
        discipline = rec["discipline"]
        review_doc_id = rec["review_doc_id"]
        review_year = rec.get("review_year")

        tset = tight_labels.get(arxiv_id, set()) if tight_labels is not None else None

        def lab(c):
            return int(c["doc_id"] in tset) if tset is not None else c["label"]

        rmeta = meta.get(review_doc_id)
        review_author_ids = set(works_cache.get(review_doc_id, {}).get("author_ids", []))
        review_author_names = flat_author_names(rmeta[7] if rmeta else None)
        review_has_oa = len(review_author_ids) > 0

        # 时间安全过滤
        safe = []
        for c in rec["candidates"]:
            cm = meta.get(c["doc_id"])
            cyear = cm[1] if cm else None
            if review_year and cyear and cyear > review_year:
                continue
            safe.append((c, cm, cyear))

        pos_timesafe = sum(1 for c, _, _ in safe if lab(c) == 1)
        # 生产接口：retrieved_by_L 的候选按 rrf 降序取前 300
        retrieved = [t for t in safe if t[0].get("retrieved_by_L")]
        retrieved.sort(key=lambda t: -t[0].get("rrf_score", 0.0))
        interface = retrieved[:PROD_DEPTH]
        pos_in_interface = sum(1 for c, _, _ in interface if lab(c) == 1)

        cov["micro_pos_timesafe"] += pos_timesafe
        cov["micro_pos_in_interface"] += pos_in_interface
        cov["per_review"].append({
            "arxiv_id": arxiv_id, "split": split, "discipline": discipline,
            "pos_timesafe": pos_timesafe, "pos_in_interface": pos_in_interface,
            "interface_size": len(interface), "review_year": review_year,
        })

        n_rows = 0
        for c, cm, cyear in interface:
            d = c["doc_id"]
            if cm is None:
                continue
            _, _, tier, cbc, cdisc, abstract, title, authors_json = cm
            oa = works_cache.get(d, {"found": False, "cited_by_count_openalex": None, "author_ids": []})
            cand_author_ids = set(oa.get("author_ids", []))
            cand_author_names = flat_author_names(authors_json)

            if review_has_oa and cand_author_ids:
                is_self = int(len(review_author_ids & cand_author_ids) > 0)
                self_reliable = 1
            else:
                is_self = int(len(review_author_names & cand_author_names) > 0)
                self_reliable = 0
            max_h, mean_h, h_missing = author_hindex_stats(cand_author_ids, authors_hindex)
            cbc_oa = oa.get("cited_by_count_openalex")
            cbc_oa = cbc_oa if cbc_oa is not None else cbc
            year_diff = (review_year - cyear) if (review_year and cyear) else None

            rows_by_split[split].append({
                "arxiv_id": arxiv_id, "discipline": discipline, "doc_id": d, "split": split,
                "label": lab(c),
                "bm25_score": c.get("bm25_score", 0.0),
                "bm25_rank_recip": 1.0 / (c["bm25_rank"] + 1) if c.get("bm25_rank", -1) >= 0 else 0.0,
                "dense_score": c.get("dense_score", 0.0),
                "dense_rank_recip": 1.0 / (c["dense_rank"] + 1) if c.get("dense_rank", -1) >= 0 else 0.0,
                "rrf_score": c.get("rrf_score", 0.0),
                "candidate_year": cyear if cyear else 0,
                "year_diff": year_diff if year_diff is not None else -1,
                "year_missing": int(year_diff is None),
                "tier_ord": TIER_ORD.get(tier, 0),
                "same_discipline": int(cdisc == discipline),
                "abstract_len_words": len((abstract or "").split()),
                "title_len_words": len((title or "").split()),
                "is_self_citation": is_self,
                "self_citation_reliable": self_reliable,
                "n_authors": len(cand_author_ids) if cand_author_ids else len(cand_author_names),
                "cited_by_count_log": float(np.log1p(cbc)) if cbc else 0.0,
                "cited_by_count_openalex_log": float(np.log1p(cbc_oa)) if cbc_oa else 0.0,
                "max_author_hindex": max_h, "mean_author_hindex": mean_h,
                "author_hindex_missing": int(h_missing),
            })
            n_rows += 1
        groups_by_split[split].append(n_rows)
        if (ri + 1) % 200 == 0:
            print(f"进度 {ri+1}/{len(reviews)}")

    for s in SPLIT_RATIOS:
        df = pd.DataFrame(rows_by_split[s])
        df.to_parquet(os.path.join(OUT_DIR, f"{s}.parquet"), index=False)
        np.save(os.path.join(OUT_DIR, f"{s}_groups.npy"), np.array(groups_by_split[s]))
        pos = df["label"].mean() if len(df) else 0.0
        print(f"[{s}] {len(df)}行 / {len(groups_by_split[s])}组  正例占比={pos:.4f}")

    micro = cov["micro_pos_in_interface"] / cov["micro_pos_timesafe"] if cov["micro_pos_timesafe"] else 0.0
    macro = float(np.mean([p["pos_in_interface"] / p["pos_timesafe"]
                           for p in cov["per_review"] if p["pos_timesafe"] > 0]))
    cov["coverage_micro"] = micro
    cov["coverage_macro"] = macro
    cov["production_depth"] = PROD_DEPTH
    with open(os.path.join(OUT_DIR, "coverage_report.json"), "w", encoding="utf-8") as f:
        json.dump(cov, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_DIR, "feature_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"main_features": MAIN_FEATURES, "aux_features": AUX_FEATURES,
                   "dropped": ["retrieved_by_L", "handcrafted_score"],
                   "interface_depth": PROD_DEPTH, "split_ratios": SPLIT_RATIOS,
                   "split_salt": SPLIT_SALT}, f, ensure_ascii=False, indent=2)
    print(f"\ntop-{PROD_DEPTH} 接口覆盖率 micro={micro:.4f}  macro={macro:.4f}")
    print(f"输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
