#!/usr/bin/env python3
r"""
S2d 阶段0 — 确定性诊断（不改数据、不调 LLM、不依赖确认队列/ MODEL-A 决策）。

产出三样注册交付物的种子，全部从审计后标签重算：

1) top-300 生产接口下的真实引用覆盖率
   - 正例 = 综述正文实际 \cite 过、且标题精确匹配进语料库的文献（resolve_bibliography.py 已限定 used_keys）
   - 接口 = L 的部署深度：在 retrieved_by_L==True 的候选里按 rrf_score 取前 N（N=300 为生产值）
   - coverage = 落在 top-N 内的正例数 / 该综述全部正例数（micro 合计 & macro 逐综述平均）
   - 施加时间安全过滤：候选 year > 综述 year 的一律剔除（不能引用未来）

2) 候选深度曲线 N ∈ {100,200,300,400}：覆盖率随池深变化——排名-价值衰减剖面的观测面(a)

3) 通用/短标题风险统计：正例里归一化标题过短或落入通用标题集的比例，
   为 200 条分层链接审计里的"同形异义"高风险层提供依据。

用法：python3 s2d_diag.py
"""
import glob
import json
import os
import re
import sqlite3
from collections import defaultdict

CAND_DIR = "work/exp2_operator_learnability/ranking/candidates"
CORPUS_DB = "data/common/unified_corpus/unified_corpus.db"
OUT = "data/exp2_operator_learnability/ranking/s2d_diag.json"

DEPTHS = [100, 200, 300, 400]
PROD_DEPTH = 300
GENERIC_TITLE_MAXWORDS = 3  # 归一化后 <=3 词的标题视为高同形异义风险


def norm_title(t):
    t = (t or "").lower().strip()
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def load_candidate_files():
    return sorted(glob.glob(os.path.join(CAND_DIR, "*", "*", "*.json")))


def main():
    conn = sqlite3.connect(CORPUS_DB)
    files = load_candidate_files()
    print(f"候选池综述文件: {len(files)}")

    # 先收集所有正例 doc_id，批量取语料库元数据（title/year/doi/source）
    pos_ids = set()
    review_years = {}
    per_review = []  # (arxiv_id, discipline, split, review_year, [(doc_id,label,rrf,retrieved)])
    for jf in files:
        rec = json.load(open(jf, encoding="utf-8"))
        ry = rec.get("review_year")
        cands = [(c["doc_id"], c["label"], c.get("rrf_score", 0.0), bool(c.get("retrieved_by_L")))
                 for c in rec["candidates"]]
        per_review.append((rec["arxiv_id"], rec["discipline"], rec.get("split"), ry, cands))
        review_years[rec["arxiv_id"]] = ry
        for d, lab, _, _ in cands:
            if lab == 1:
                pos_ids.add(d)

    # 批量取候选 year（做时间安全过滤需要）+ 正例的 title/doi/source（做风险分层）
    all_ids = set()
    for _, _, _, _, cands in per_review:
        for d, _, _, _ in cands:
            all_ids.add(d)
    meta = {}
    ids = list(all_ids)
    CH = 800
    for i in range(0, len(ids), CH):
        chunk = ids[i:i+CH]
        ph = ",".join("?" for _ in chunk)
        for r in conn.execute(f"SELECT doc_id, year, title, doi, source, discipline FROM papers WHERE doc_id IN ({ph})", chunk):
            meta[r[0]] = {"year": r[1], "title": r[2], "doi": r[3], "source": r[4], "discipline": r[5]}

    # ---- 覆盖率（各深度）与衰减曲线 ----
    depth_micro_hit = {n: 0 for n in DEPTHS}
    depth_macro = {n: [] for n in DEPTHS}
    micro_total_pos = 0
    n_reviews_eval = 0
    forced_out_of_interface = 0  # 正例但不在 top-PROD_DEPTH（=L 漏召回的量，覆盖诊断口径）

    for arxiv_id, disc, split, ry, cands in per_review:
        # 时间安全过滤：剔除 year>review_year 的候选；review_year 缺失则不按时间过滤该篇
        safe = []
        for d, lab, rrf, retr in cands:
            cy = meta.get(d, {}).get("year")
            if ry and cy and cy > ry:
                continue
            safe.append((d, lab, rrf, retr))
        # 部署接口：retrieved_by_L 的候选按 rrf 降序
        retrieved = sorted([x for x in safe if x[3]], key=lambda x: -x[2])
        pos_total = sum(1 for _, lab, _, _ in safe if lab == 1)
        if pos_total == 0:
            continue
        n_reviews_eval += 1
        micro_total_pos += pos_total
        for n in DEPTHS:
            topn = retrieved[:n]
            hit = sum(1 for d, lab, _, _ in topn if lab == 1)
            depth_micro_hit[n] += hit
            depth_macro[n].append(hit / pos_total)
        # 生产深度下漏在接口外的正例
        top_prod_ids = set(d for d, _, _, _ in retrieved[:PROD_DEPTH])
        forced_out_of_interface += sum(1 for d, lab, _, _ in safe if lab == 1 and d not in top_prod_ids)

    coverage = {
        str(n): {
            "micro": depth_micro_hit[n] / micro_total_pos if micro_total_pos else 0.0,
            "macro": sum(depth_macro[n]) / len(depth_macro[n]) if depth_macro[n] else 0.0,
        }
        for n in DEPTHS
    }

    # ---- 通用/短标题风险 & 无 DOI 比例（正例集）----
    n_pos = len(pos_ids)
    n_generic = n_nodoi = n_preprint = 0
    norm_counts = defaultdict(int)
    for d in pos_ids:
        m = meta.get(d, {})
        nt = norm_title(m.get("title"))
        norm_counts[nt] += 1
        if nt and len(nt.split()) <= GENERIC_TITLE_MAXWORDS:
            n_generic += 1
        if not m.get("doi"):
            n_nodoi += 1
        if (m.get("source") or "").lower().startswith("arxiv"):
            n_preprint += 1
    # 正例内部归一化标题撞车（潜在同形异义链接风险）
    collide_groups = {k: v for k, v in norm_counts.items() if v > 1}

    out = {
        "n_candidate_reviews": len(per_review),
        "n_reviews_evaluated_coverage": n_reviews_eval,
        "n_unique_positive_docs": n_pos,
        "micro_total_positives_after_timefilter": micro_total_pos,
        "coverage_by_depth": coverage,
        "production_depth": PROD_DEPTH,
        "positives_outside_production_interface": forced_out_of_interface,
        "risk_strata_positive_docs": {
            "generic_or_short_title (<=%d words)" % GENERIC_TITLE_MAXWORDS: n_generic,
            "no_doi": n_nodoi,
            "preprint_source": n_preprint,
            "normalized_title_collisions_within_positives": sum(collide_groups.values()),
            "n_collision_groups": len(collide_groups),
        },
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n=== S2d 阶段0 诊断（审计后标签重算）===")
    print(f"参评综述(有正例、时间过滤后): {n_reviews_eval}")
    print(f"唯一正例文献数: {n_pos}   时间过滤后正例总计: {micro_total_pos}")
    print(f"\ntop-N 覆盖率（正例落入 L 部署接口的比例）:")
    for n in DEPTHS:
        c = coverage[str(n)]
        print(f"  N={n:<4d}  micro={c['micro']:.4f}  macro={c['macro']:.4f}")
    print(f"\n生产深度 {PROD_DEPTH} 外的正例（L 召回瓶颈量）: {forced_out_of_interface}")
    print(f"\n正例风险分层（用于 200 条审计抽样）:")
    for k, v in out["risk_strata_positive_docs"].items():
        print(f"  {k}: {v}")
    print(f"\n写入 {OUT}")


if __name__ == "__main__":
    main()
