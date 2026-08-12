#!/usr/bin/env python3
"""
S2b (Learning L — literature location) 阶段0 确定性诊断。

目的：在冻结的 s2b_v1 综述分组切分上，用审计后的收紧标签，把"真实引用覆盖率"
按其可恢复来源分解，为 L 的两档学习目标（重排 vs 重检索）定标：

  分母 P（每篇综述）= 时间安全的收紧真引用总数（matched_doc_ids ∩ 候选池 ∩ year<=review_year）
    —— 候选池对所有正例做过 forced-add，故收紧正例应全部在池内；不在池内者单独计数报告。

  分解（每篇）：
    P_retrieved   = 其中被任一检索器召回（bm25_rank>=0 或 dense_rank>=0）的正例数  → 检索天花板
    P_rrf@300     = 其中落在 RRF 前 300（生产接口 L 交付深度）的正例数            → 当前 L 覆盖
    gap_rank      = P_retrieved - P_rrf@300   → 仅靠更好排序/融合即可恢复（Tier-1 头room）
    gap_retrieval = P          - P_retrieved  → 只能靠更强检索器恢复（Tier-2 头room）

基线覆盖率曲线（精确口径，用语料库真实 rank）：
    BM25@k   = bm25_rank ∈ [0,k) 的正例数
    dense@k  = dense_rank ∈ [0,k) 的正例数
    RRF@k    = 按 rrf_score 降序前 k 的正例数
  深度 k ∈ {100,200,300,500,1000,2000,3000}。

只读候选 JSON + 语料库 year + 收紧标签；不调 LLM、不训练、不改数据。
用法（服务器）：python3 s2b_diag.py
"""
import glob
import json
import os
import sqlite3
from collections import defaultdict

CAND_DIR = "work/exp2_operator_learnability/ranking/candidates"
CORPUS_DB = "data/common/unified_corpus/unified_corpus.db"
TIGHT_DIR = "data/exp2_operator_learnability/ranking/resolved_tight"
SPLIT_MANIFEST = "data/exp2_operator_learnability/ranking/features_s2d_tight/split_manifest.json"
OUT = "data/exp2_operator_learnability/location/s2b_diag.json"

DEPTHS = [100, 200, 300, 500, 1000, 2000, 3000]
PROD_DEPTH = 300


def load_tight():
    tight = {}
    for jf in glob.glob(os.path.join(TIGHT_DIR, "*", "*.json")):
        r = json.load(open(jf, encoding="utf-8"))
        tight[r["arxiv_id"]] = set(r["matched_doc_ids"])
    return tight


def load_split():
    m = json.load(open(SPLIT_MANIFEST, encoding="utf-8"))
    return m["assignments"]


def main():
    tight = load_tight()
    split_of = load_split()
    conn = sqlite3.connect(CORPUS_DB)
    files = sorted(glob.glob(os.path.join(CAND_DIR, "*", "*", "*.json")))
    print(f"候选池综述文件: {len(files)}  收紧标签: {len(tight)} 篇")

    # 批量取候选 year（时间安全过滤）
    all_ids = set()
    per_review = []
    for jf in files:
        rec = json.load(open(jf, encoding="utf-8"))
        per_review.append(rec)
        for c in rec["candidates"]:
            all_ids.add(c["doc_id"])
    year = {}
    ids = list(all_ids)
    CH = 900
    for i in range(0, len(ids), CH):
        chunk = ids[i:i + CH]
        ph = ",".join("?" for _ in chunk)
        for d, y in conn.execute(f"SELECT doc_id, year FROM papers WHERE doc_id IN ({ph})", chunk):
            year[d] = y

    # 聚合器
    agg = {s: defaultdict(float) for s in ["train", "val", "test", "all"]}
    cov_micro = {s: {str(k): 0 for k in DEPTHS} for s in ["train", "val", "test", "all"]}
    cov_macro = {s: {str(k): [] for k in DEPTHS} for s in ["train", "val", "test", "all"]}
    n_rev = {s: 0 for s in ["train", "val", "test", "all"]}
    missing_pos_in_pool = 0
    tight_pos_total = 0

    for rec in per_review:
        arxiv_id = rec["arxiv_id"]
        ry = rec.get("review_year")
        split = split_of.get(arxiv_id)
        if split is None:
            continue
        pos_set = tight.get(arxiv_id, set())
        # 候选池 doc_id -> (bm25_rank, dense_rank, rrf_score, retrieved)
        cand = {}
        for c in rec["candidates"]:
            cand[c["doc_id"]] = (
                c.get("bm25_rank", -1), c.get("dense_rank", -1),
                c.get("rrf_score", 0.0), bool(c.get("retrieved_by_L")),
            )
        # 时间安全的收紧正例（且在候选池内）
        P_ids = []
        for d in pos_set:
            cy = year.get(d)
            if ry and cy and cy > ry:
                continue  # 不能引用未来
            tight_pos_total += 1
            if d not in cand:
                missing_pos_in_pool += 1
                continue
            P_ids.append(d)
        P = len(P_ids)
        if P == 0:
            continue

        # RRF 排序：仅对 retrieved_by_L 候选按 rrf 降序（与 L 部署接口一致）；
        # 未被检索/强制补入的正例不参与 RRF 名次（视为不可达），避免池小于 k 时的假覆盖。
        retrieved_sorted = sorted(
            [(d, v) for d, v in cand.items() if v[3]], key=lambda kv: -kv[1][2])
        rrf_rank_of = {d: i for i, (d, _) in enumerate(retrieved_sorted)}

        P_retrieved = sum(1 for d in P_ids if cand[d][3])
        P_rrf300 = sum(1 for d in P_ids if rrf_rank_of.get(d, 10**9) < PROD_DEPTH)
        gap_rank = P_retrieved - P_rrf300
        gap_retrieval = P - P_retrieved

        for s in (split, "all"):
            n_rev[s] += 1
            agg[s]["P"] += P
            agg[s]["P_retrieved"] += P_retrieved
            agg[s]["P_rrf300"] += P_rrf300
            agg[s]["gap_rank"] += gap_rank
            agg[s]["gap_retrieval"] += gap_retrieval
            # 覆盖率曲线
            for k in DEPTHS:
                bm = sum(1 for d in P_ids if 0 <= cand[d][0] < k)
                dn = sum(1 for d in P_ids if 0 <= cand[d][1] < k)
                rf = sum(1 for d in P_ids if rrf_rank_of.get(d, 10**9) < k)
                cov_micro[s][str(k)] += rf  # micro 主口径用 RRF（当前 L 的输入排序）
                cov_macro[s][str(k)].append(rf / P)
                agg[s][f"bm25@{k}"] += bm
                agg[s][f"dense@{k}"] += dn
                agg[s][f"rrf@{k}"] += rf

    out = {"n_reviews": n_rev, "tight_pos_time_safe_total": tight_pos_total,
           "tight_pos_missing_from_pool": missing_pos_in_pool,
           "production_depth": PROD_DEPTH, "depths": DEPTHS, "by_split": {}}
    for s in ["train", "val", "test", "all"]:
        P = agg[s]["P"]
        if P == 0:
            continue
        out["by_split"][s] = {
            "n_reviews": n_rev[s],
            "P_total": int(P),
            "retrieval_ceiling_micro": agg[s]["P_retrieved"] / P,
            "current_L_rrf300_micro": agg[s]["P_rrf300"] / P,
            "gap_rank_micro": agg[s]["gap_rank"] / P,
            "gap_retrieval_micro": agg[s]["gap_retrieval"] / P,
            "coverage_micro": {k: agg[s][f"rrf@{int(k)}"] / P for k in map(str, DEPTHS)},
            "coverage_macro": {k: (sum(v) / len(v) if v else 0.0) for k, v in cov_macro[s].items()},
            "baseline_micro_at_300": {
                "bm25": agg[s]["bm25@300"] / P,
                "dense": agg[s]["dense@300"] / P,
                "rrf": agg[s]["rrf@300"] / P,
            },
            "baseline_micro_curves": {
                "bm25": {str(k): agg[s][f"bm25@{k}"] / P for k in DEPTHS},
                "dense": {str(k): agg[s][f"dense@{k}"] / P for k in DEPTHS},
                "rrf": {str(k): agg[s][f"rrf@{k}"] / P for k in DEPTHS},
            },
        }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n=== S2b(L) 阶段0 诊断（冻结 s2b_v1 切分，收紧标签）===")
    print(f"时间安全收紧正例合计: {tight_pos_total}  不在候选池内: {missing_pos_in_pool}")
    for s in ["all", "test"]:
        b = out["by_split"][s]
        print(f"\n[{s}] 综述={b['n_reviews']}  正例总计 P={b['P_total']}")
        print(f"  检索天花板(池内召回) micro = {b['retrieval_ceiling_micro']:.4f}")
        print(f"  当前 L RRF@300      micro = {b['current_L_rrf300_micro']:.4f}")
        print(f"    → 排序可恢复 gap_rank      = {b['gap_rank_micro']:.4f}")
        print(f"    → 需重检索 gap_retrieval   = {b['gap_retrieval_micro']:.4f}")
        print(f"  基线@300 micro  bm25={b['baseline_micro_at_300']['bm25']:.4f} "
              f"dense={b['baseline_micro_at_300']['dense']:.4f} rrf={b['baseline_micro_at_300']['rrf']:.4f}")
    print(f"\n写入 {OUT}")


if __name__ == "__main__":
    main()
