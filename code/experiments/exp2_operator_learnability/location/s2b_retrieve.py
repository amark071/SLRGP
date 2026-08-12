#!/usr/bin/env python3
"""
S2b (Learning L) 诚实检索底座：对全部 1102 篇综述跑 BM25+dense 到 depth-3000，
输出完整召回并集（真实负例分布），供覆盖率诊断与学习融合评测使用。

为什么需要独立的检索底座：candidates/ 池是 R 算子的 hard-negative 训练集（每篇仅存 top-400
负例采样 + 全部正例），正例占比被人为抬高，不能用来测 L 的检索覆盖率。bm25_rank/
dense_rank 虽是真实语料库排名，但 RRF 与学习融合排名需要完整并集里所有竞争负例。

查询构造与 build_candidates 完全一致：query = title + ". " + abstract[:1000]，
故本脚本对 retrieved_by_L 集合、bm25_rank、dense_rank 应可复现（内建一致性抽查）。

输出（S2B_UNION_DIR，默认 work/exp2_operator_learnability/ranking/union_L）：
  {train,val,test}.parquet  列：arxiv_id, split, doc_id, label, bm25_rank, bm25_score,
      dense_rank, dense_score, rrf_score, year_diff, same_discipline
  P_of.json         每篇综述的时间安全收紧正例总数（覆盖率分母，含不可达）
  retrieve_summary.json
用法：python3 s2b_retrieve.py
"""
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "code/slrgp")

CAND_DIR = "work/exp2_operator_learnability/ranking/candidates"
CORPUS_DB = "data/common/unified_corpus/unified_corpus.db"
TIGHT_DIR = "data/exp2_operator_learnability/ranking/resolved_tight"
SPLIT_MANIFEST = "data/exp2_operator_learnability/ranking/features_s2d_tight/split_manifest.json"
OUT_DIR = os.environ.get("S2B_UNION_DIR", "work/exp2_operator_learnability/ranking/union_L")

TOPK_EACH = 3000
RRF_K = 60


def load_tight():
    tight = {}
    for jf in glob.glob(os.path.join(TIGHT_DIR, "*", "*.json")):
        r = json.load(open(jf, encoding="utf-8"))
        tight[r["arxiv_id"]] = set(r["matched_doc_ids"])
    return tight


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from slrgp.retrieval import UnifiedIndex

    tight = load_tight()
    split_of = json.load(open(SPLIT_MANIFEST, encoding="utf-8"))["assignments"]

    # 预载全语料 year + discipline（时间安全过滤 & same_discipline 特征）
    import sqlite3
    conn = sqlite3.connect(CORPUS_DB)
    print("预载 year/discipline ...")
    year, disc = {}, {}
    for d, y, dd in conn.execute("SELECT doc_id, year, discipline FROM papers"):
        year[d] = y
        disc[d] = dd
    print(f"  载入 {len(year)} 篇元数据")

    index = UnifiedIndex()

    files = sorted(glob.glob(os.path.join(CAND_DIR, "*", "*", "*.json")))
    print(f"综述: {len(files)}")

    rows_by_split = {"train": [], "val": [], "test": []}
    P_of = {}
    reproduce_ok = reproduce_tot = 0

    for i, jf in enumerate(files):
        rec = json.load(open(jf, encoding="utf-8"))
        aid = rec["arxiv_id"]
        split = split_of.get(aid)
        if split is None:
            continue
        ry = rec.get("review_year")
        rdisc = rec["discipline"]
        review_doc_id = rec["review_doc_id"]
        pos = tight.get(aid, set())

        # 查询：与 build_candidates 一致
        row = conn.execute("SELECT title, abstract FROM papers WHERE doc_id=?", (review_doc_id,)).fetchone()
        if not row or not row[0]:
            continue
        qtext = (row[0] or "") + ". " + (row[1] or "")[:1000]

        bm25_hits = index._bm25_search(qtext, TOPK_EACH)
        dense_hits = index._dense_search(qtext, TOPK_EACH)
        bm25_rank = {d: r for r, (d, _) in enumerate(bm25_hits)}
        bm25_score = {d: s for d, s in bm25_hits}
        dense_rank = {d: r for r, (d, _) in enumerate(dense_hits)}
        dense_score = {d: s for d, s in dense_hits}
        union = (set(bm25_rank) | set(dense_rank)) - {review_doc_id}

        rrf = {}
        for d, r_ in bm25_rank.items():
            rrf[d] = rrf.get(d, 0.0) + 1.0 / (RRF_K + r_ + 1)
        for d, r_ in dense_rank.items():
            rrf[d] = rrf.get(d, 0.0) + 1.0 / (RRF_K + r_ + 1)

        # 一致性抽查：与既有 JSON 的 retrieved_by_L 正例对比
        for c in rec["candidates"]:
            if c["label"] == 1 and c.get("retrieved_by_L"):
                reproduce_tot += 1
                if c["doc_id"] in union:
                    reproduce_ok += 1

        # 覆盖率分母 P：时间安全的收紧正例总数（不限是否召回）
        P = 0
        for d in pos:
            cy = year.get(d)
            if ry and cy and cy > ry:
                continue
            P += 1
        P_of[aid] = {"split": split, "P": P}

        # 写完整并集（时间安全过滤：剔除 year>review_year）
        for d in union:
            cy = year.get(d)
            if ry and cy and cy > ry:
                continue
            br = bm25_rank.get(d, -1)
            dr = dense_rank.get(d, -1)
            rows_by_split[split].append({
                "arxiv_id": aid, "split": split, "doc_id": d,
                "label": int(d in pos),
                "bm25_rank": br, "bm25_score": bm25_score.get(d, 0.0),
                "dense_rank": dr, "dense_score": dense_score.get(d, 0.0),
                "rrf_score": rrf.get(d, 0.0),
                "year_diff": (ry - cy) if (ry and cy) else -1,
                "same_discipline": int(disc.get(d) == rdisc),
            })

        if (i + 1) % 100 == 0:
            print(f"进度 {i+1}/{len(files)}  union_rows="
                  f"{sum(len(v) for v in rows_by_split.values())}")

    for s in ["train", "val", "test"]:
        df = pd.DataFrame(rows_by_split[s])
        df.to_parquet(os.path.join(OUT_DIR, f"{s}.parquet"), index=False)
        print(f"[{s}] {len(df)} 行 / {df['arxiv_id'].nunique() if len(df) else 0} 综述")
    with open(os.path.join(OUT_DIR, "P_of.json"), "w", encoding="utf-8") as f:
        json.dump(P_of, f)
    summary = {"topk_each": TOPK_EACH, "n_reviews": len(P_of),
               "reproduce_retrieved_pos_match": f"{reproduce_ok}/{reproduce_tot}",
               "reproduce_rate": reproduce_ok / reproduce_tot if reproduce_tot else 0.0}
    with open(os.path.join(OUT_DIR, "retrieve_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n一致性抽查（既有 retrieved 正例应在新并集内）: {reproduce_ok}/{reproduce_tot} "
          f"= {summary['reproduce_rate']:.4f}")
    print(f"输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
