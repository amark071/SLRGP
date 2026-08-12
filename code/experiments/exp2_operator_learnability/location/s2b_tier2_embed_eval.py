#!/usr/bin/env python3
"""
S2b (Learning L) Tier-2 阶段B+C：重嵌入全语料 + dense 检索 + 覆盖率评测。

阶段B：用两个模型对 3.66M 语料重嵌入（doc 文本 title+". "+abstract，与训练一致，无前缀）：
  - stock  = BAAI/bge-small-en-v1.5（原版；作为控制，隔离"文本管线差异"，公平对照微调）
  - ft     = 微调模型 bge_small_L_ft
  嵌入矩阵缓存到 EMB_DIR，行序与 docids 对齐；已存在则跳过（可断点续跑）。

阶段C：对 val/test 综述，各系统取 dense top-3000，算 Coverage@k：
  - dense_stock_rebuilt   原版 bge 重嵌入检索（控制）
  - dense_ft              微调 bge 检索（学习系统）
  - hybrid_ft             RRF(BM25 既有排名, dense_ft)  —— Tier-3 混合
  分母 P 用 union_L/P_of.json（时间安全收紧真引用总数，含不可达）——与 Tier-1 完全一致。
  配对统计：dense_ft vs dense_stock_rebuilt、hybrid_ft vs rrf（既有）。
  同时报告新的检索天花板（ft top-3000 能覆盖多少真引用 = 是否把 gap_retrieval 打下来）。

环境：S2B_FT_MODEL、S2B_EMB_DIR、S2B_OUT_DIR、CUDA_VISIBLE_DEVICES
用法（服务器）：python3 s2b_tier2_embed_eval.py
"""
import glob
import json
import os
import pickle
import sqlite3
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
try:
    from stats_utils import paired_bootstrap_ci, paired_test
except Exception:
    from confirmatory.stats_utils import paired_bootstrap_ci, paired_test  # type: ignore

CORPUS_DB = "data/common/unified_corpus/unified_corpus.db"
TIGHT_DIR = "data/exp2_operator_learnability/ranking/resolved_tight"
CAND_DIR = "work/exp2_operator_learnability/ranking/candidates"
UNION_DIR = os.environ.get("S2B_UNION_DIR", "work/exp2_operator_learnability/ranking/union_L")
FT_MODEL = os.environ.get("S2B_FT_MODEL", "models/bge_small_L_ft")
EMB_DIR = os.environ.get("S2B_EMB_DIR", "work/exp2_operator_learnability/location/tier2_emb")
OUT_DIR = os.environ.get("S2B_OUT_DIR", "work/exp2_operator_learnability/location")
SPLIT_MANIFEST = "data/exp2_operator_learnability/ranking/features_s2d_tight/split_manifest.json"

QPREFIX = "Represent this sentence for searching relevant passages: "
STOCK = "BAAI/bge-small-en-v1.5"
DEPTHS = [100, 300, 500, 1000, 3000]
PRIMARY = 300
DOC_CHARS = 2000
TOPK = 3000
BATCH = 1024


def doc_text(t, a):
    return ((t or "") + ". " + (a or ""))[:DOC_CHARS]


def load_tight():
    tight = {}
    for jf in glob.glob(os.path.join(TIGHT_DIR, "*", "*.json")):
        r = json.load(open(jf, encoding="utf-8"))
        tight[r["arxiv_id"]] = set(r["matched_doc_ids"])
    return tight


def encode_corpus(model_path, tag):
    """流式编码全语料，doc 顺序 = corpus 全表 doc_id 顺序，保存 emb + docids。"""
    emb_path = os.path.join(EMB_DIR, f"{tag}_emb.npy")
    ids_path = os.path.join(EMB_DIR, f"{tag}_docids.pkl")
    if os.path.exists(emb_path) and os.path.exists(ids_path):
        print(f"[{tag}] 已存在，跳过编码")
        return emb_path, ids_path
    os.makedirs(EMB_DIR, exist_ok=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_path, device="cuda")
    model.max_seq_length = 256
    conn = sqlite3.connect(CORPUS_DB)
    n = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    print(f"[{tag}] 编码 {n} 篇 ...")
    docids = []
    embs = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float32, shape=(n, 384))
    buf, buf_ids, row = [], [], 0
    cur = conn.execute("SELECT doc_id, title, abstract FROM papers")
    done = 0
    for d, t, a in cur:
        buf.append(doc_text(t, a)); buf_ids.append(d)
        if len(buf) >= BATCH:
            v = model.encode(buf, batch_size=256, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False)
            embs[row:row + len(buf)] = v
            docids.extend(buf_ids); row += len(buf); done += len(buf)
            buf, buf_ids = [], []
            if done % 102400 == 0:
                print(f"[{tag}] {done}/{n}")
    if buf:
        v = model.encode(buf, batch_size=256, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False)
        embs[row:row + len(buf)] = v
        docids.extend(buf_ids); row += len(buf)
    embs.flush()
    pickle.dump(docids, open(ids_path, "wb"))
    print(f"[{tag}] 完成 {row} 行")
    return emb_path, ids_path


def main():
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    os.makedirs(OUT_DIR, exist_ok=True)
    tight = load_tight()
    split_of = json.load(open(SPLIT_MANIFEST, encoding="utf-8"))["assignments"]
    P_raw = json.load(open(os.path.join(UNION_DIR, "P_of.json")))
    P_of = {a: v["P"] for a, v in P_raw.items()}
    split_P = {a: v["split"] for a, v in P_raw.items()}

    # 阶段B：编码
    stock_emb_p, stock_ids_p = encode_corpus(STOCK, "stock")
    ft_emb_p, ft_ids_p = encode_corpus(FT_MODEL, "ft")

    stock_emb = np.load(stock_emb_p, mmap_mode="r")
    ft_emb = np.load(ft_emb_p, mmap_mode="r")
    n = stock_emb.shape[0]
    stock_ids = pickle.load(open(stock_ids_p, "rb"))
    ft_ids = pickle.load(open(ft_ids_p, "rb"))
    stock_row = {d: i for i, d in enumerate(stock_ids)}
    ft_row = {d: i for i, d in enumerate(ft_ids)}

    conn = sqlite3.connect(CORPUS_DB)
    year = {}
    for d, y in conn.execute("SELECT doc_id, year FROM papers"):
        year[d] = y

    # 综述查询文本
    cand_files = {os.path.basename(jf)[:-5]: jf for jf in glob.glob(os.path.join(CAND_DIR, "*", "*", "*.json"))}
    eval_reviews = {s: [] for s in ["val", "test"]}
    qtext = {}
    for aid, split in split_of.items():
        if split not in ("val", "test"):
            continue
        jf = cand_files.get(aid)
        if not jf:
            continue
        rec = json.load(open(jf, encoding="utf-8"))
        row = conn.execute("SELECT title, abstract FROM papers WHERE doc_id=?", (rec["review_doc_id"],)).fetchone()
        if not row or not row[0]:
            continue
        qtext[aid] = (QPREFIX + (row[0] or "") + ". " + (row[1] or "")[:1000], rec.get("review_year"))
        eval_reviews[split].append(aid)

    qmodel_stock = SentenceTransformer(STOCK, device="cuda"); qmodel_stock.max_seq_length = 256
    qmodel_ft = SentenceTransformer(FT_MODEL, device="cuda"); qmodel_ft.max_seq_length = 256

    def retrieve(emb, row_of, qvec, ry):
        scores = emb @ qvec
        idx = np.argpartition(-scores, TOPK)[:TOPK]
        idx = idx[np.argsort(-scores[idx])]
        # 时间安全 + 返回有序 doc_id
        out = []
        for i in idx:
            d = (stock_ids if row_of is stock_row else ft_ids)[i]
            cy = year.get(d)
            if ry and cy and cy > ry:
                continue
            out.append(d)
        return out

    # 载入既有并集 rrf/bm25 排名（做 hybrid_ft 与对照 rrf 复算）
    union = {s: pd.read_parquet(os.path.join(UNION_DIR, f"{s}.parquet")) for s in ["val", "test"]}

    def cov_from_ranked(ranked_ids, pos, ry, depth):
        P = 0
        for d in pos:
            cy = year.get(d)
            if ry and cy and cy > ry:
                continue
            P += 1
        if P == 0:
            return None
        hit = sum(1 for d in ranked_ids[:depth] if d in pos)
        return hit / P

    systems = {"dense_stock_rebuilt": {}, "dense_ft": {}, "hybrid_ft": {}}
    ceiling_ft = {}
    per_review_cov = {"dense_ft": {}, "dense_stock_rebuilt": {}, "hybrid_ft": {}, "rrf": {}}

    for split in ["val", "test"]:
        micro = {sysn: {k: [0, 0] for k in DEPTHS} for sysn in systems}
        macro = {sysn: {k: [] for k in DEPTHS} for sysn in systems}
        ceil_num = ceil_den = 0
        u = union[split]
        rrf_rank = {aid: g.sort_values("rrf_score", ascending=False)["doc_id"].tolist()
                    for aid, g in u.groupby("arxiv_id")}
        bm25_rank_map = {}
        for aid, g in u.groupby("arxiv_id"):
            gg = g[g["bm25_rank"] >= 0].sort_values("bm25_rank")
            bm25_rank_map[aid] = gg["doc_id"].tolist()

        for aid in eval_reviews[split]:
            q, ry = qtext[aid]
            pos = tight.get(aid, set())
            qv_s = qmodel_stock.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
            qv_f = qmodel_ft.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
            r_stock = retrieve(stock_emb, stock_row, qv_s, ry)
            r_ft = retrieve(ft_emb, ft_row, qv_f, ry)
            # hybrid_ft: RRF(既有 bm25 排名, ft dense 排名)
            rrf_h = {}
            for rank, d in enumerate(bm25_rank_map.get(aid, [])):
                rrf_h[d] = rrf_h.get(d, 0.0) + 1.0 / (60 + rank + 1)
            for rank, d in enumerate(r_ft):
                rrf_h[d] = rrf_h.get(d, 0.0) + 1.0 / (60 + rank + 1)
            r_hyb = sorted(rrf_h, key=lambda d: -rrf_h[d])

            # ft 检索天花板：pos 落在 ft top-3000 的比例
            P = sum(1 for d in pos if not (ry and year.get(d) and year.get(d) > ry))
            if P > 0:
                ceil_num += sum(1 for d in r_ft if d in pos)
                ceil_den += P

            for sysn, ranked in [("dense_stock_rebuilt", r_stock), ("dense_ft", r_ft), ("hybrid_ft", r_hyb)]:
                for k in DEPTHS:
                    c = cov_from_ranked(ranked, pos, ry, k)
                    if c is None:
                        continue
                    hit = sum(1 for d in ranked[:k] if d in pos)
                    micro[sysn][k][0] += hit
                    micro[sysn][k][1] += P
                    macro[sysn][k].append(c)
                    if k == PRIMARY:
                        per_review_cov[sysn][aid] = c
            # rrf per-review @300（既有排名）
            c_rrf = cov_from_ranked(rrf_rank.get(aid, []), pos, ry, PRIMARY)
            if c_rrf is not None:
                per_review_cov["rrf"][aid] = c_rrf

        for sysn in systems:
            systems[sysn][split] = {str(k): {
                "micro": micro[sysn][k][0] / micro[sysn][k][1] if micro[sysn][k][1] else 0.0,
                "macro": float(np.mean(macro[sysn][k])) if macro[sysn][k] else 0.0}
                for k in DEPTHS}
        ceiling_ft[split] = ceil_num / ceil_den if ceil_den else 0.0
        print(f"[{split}] dense_ft cov@300 micro={systems['dense_ft'][split]['300']['micro']:.4f} "
              f"macro={systems['dense_ft'][split]['300']['macro']:.4f} | "
              f"stock_rebuilt={systems['dense_stock_rebuilt'][split]['300']['micro']:.4f} | "
              f"hybrid={systems['hybrid_ft'][split]['300']['micro']:.4f} | ft_ceiling@3000={ceiling_ft[split]:.4f}")

    res = {"systems": systems, "ft_retrieval_ceiling": ceiling_ft, "depths": DEPTHS, "primary": PRIMARY}
    # 配对统计（test）
    for a_sys, b_sys, name in [("dense_ft", "dense_stock_rebuilt", "ft_vs_stock"),
                               ("hybrid_ft", "rrf", "hybrid_vs_rrf"),
                               ("dense_ft", "rrf", "ft_vs_rrf")]:
        aids = sorted(set(per_review_cov[a_sys]) & set(per_review_cov[b_sys]))
        diffs = [per_review_cov[a_sys][x] - per_review_cov[b_sys][x] for x in aids]
        if len(diffs) >= 8:
            ci = paired_bootstrap_ci(diffs, n_boot=10000)
            pt = paired_test(diffs)
            res[f"paired_{name}_cov300_test"] = {
                "n": len(aids), "mean_diff": ci.point_estimate, "ci_low": ci.ci_low,
                "ci_high": ci.ci_high, "wilcoxon_p": pt.wilcoxon_p, "rank_biserial": pt.rank_biserial}
            print(f"[{name}] Δ={ci.point_estimate:.4f} CI[{ci.ci_low:.4f},{ci.ci_high:.4f}] p={pt.wilcoxon_p:.3g}")

    with open(os.path.join(OUT_DIR, "s2b_tier2_eval.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n写入 {OUT_DIR}/s2b_tier2_eval.json")


if __name__ == "__main__":
    main()
