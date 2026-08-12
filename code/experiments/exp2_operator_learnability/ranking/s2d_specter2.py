#!/usr/bin/env python3
"""
S2d 外部基线 SPECTER2（引文训练的科学文献嵌入，proximity adapter）。

对 test 集每篇综述：query = 综述 title+abstract 的 SPECTER2 向量；候选 = 候选 title+abstract 向量；
按余弦相似度排序（与其它臂同一候选池、同一查询信息边界：title+abstract）。产出 specter2_score 后
评测 nDCG@10 / Recall@50 / MRR，并与 LambdaMART 做配对统计（H2）。

依赖：adapters + transformers 4.x（本脚本跑完后需把 transformers 恢复到 5.13 以兼容 vllm）。
用法：python3 s2d_specter2.py
"""
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer  # noqa: E402
from adapters import AutoAdapterModel  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_utils import paired_bootstrap_ci, paired_test  # noqa: E402

FEAT_DIR = os.environ.get("S2D_FEAT_DIR", "data/exp2_operator_learnability/ranking/features_s2d")
CORPUS_DB = "data/common/unified_corpus/unified_corpus.db"
OUT_DIR = os.environ.get("S2D_OUT_DIR", "work/exp2_operator_learnability/ranking")
DEVICE = "cuda"
BATCH = 64


def get_texts(conn, doc_ids):
    out = {}
    ids = list(doc_ids)
    for i in range(0, len(ids), 800):
        chunk = ids[i:i + 800]
        ph = ",".join("?" for _ in chunk)
        for d, t, a in conn.execute(
            f"SELECT doc_id, title, abstract FROM papers WHERE doc_id IN ({ph})", chunk):
            out[d] = ((t or "") + tokenizer.sep_token + (a or ""))
    return out


def embed(texts):
    vecs = np.zeros((len(texts), 768), dtype=np.float32)
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        inp = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        cls = out.last_hidden_state[:, 0, :].cpu().numpy()
        vecs[i:i + len(batch)] = cls
        if (i // BATCH) % 50 == 0:
            print(f"  embed {i}/{len(texts)}", flush=True)
    return vecs


def dcg(labels, k):
    return sum(l / np.log2(j + 2) for j, l in enumerate(labels[:k]))


def per_query_ndcg(df, groups, score, k=10):
    out = []
    idx = 0
    for g in groups:
        sub = df.iloc[idx:idx + g]; idx += g
        if sub["label"].sum() == 0:
            out.append(0.0); continue
        order = np.argsort(-sub[score].values, kind="stable")
        ls = sub["label"].values[order]
        idcg = dcg(np.sort(sub["label"].values)[::-1], k)
        out.append(dcg(ls, k) / idcg if idcg > 0 else 0.0)
    return np.array(out)


def eval_by_query(df, groups, score):
    ndcgs, recalls, mrrs = [], [], []
    idx = 0
    for g in groups:
        sub = df.iloc[idx:idx + g]; idx += g
        npos = int(sub["label"].sum())
        if npos == 0:
            ndcgs.append(0.0); recalls.append(0.0); mrrs.append(0.0); continue
        order = np.argsort(-sub[score].values, kind="stable")
        ls = sub["label"].values[order]
        idcg = dcg(np.sort(sub["label"].values)[::-1], 10)
        ndcgs.append(dcg(ls, 10) / idcg if idcg > 0 else 0.0)
        recalls.append(ls[:50].sum() / npos)
        first = np.argmax(ls == 1) + 1 if (ls == 1).any() else 0
        mrrs.append(1.0 / first if first else 0.0)
    return {"nDCG@10": float(np.mean(ndcgs)), "Recall@50": float(np.mean(recalls)),
            "MRR": float(np.mean(mrrs)), "n": len(ndcgs)}


def main():
    global tokenizer, model
    print("加载 SPECTER2 ...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
    model.to(DEVICE).eval()

    test_df = pd.read_parquet(os.path.join(FEAT_DIR, "test.parquet"))
    test_g = np.load(os.path.join(FEAT_DIR, "test_groups.npy"))
    conn = sqlite3.connect(CORPUS_DB)

    cand_ids = sorted(test_df["doc_id"].unique().tolist())
    review_ids = sorted(("arxiv_" + test_df["arxiv_id"]).unique().tolist())
    print(f"候选 {len(cand_ids)}  综述 {len(review_ids)}")

    cand_text = get_texts(conn, cand_ids)
    rev_text = get_texts(conn, review_ids)
    cand_list = [c for c in cand_ids if c in cand_text]
    rev_list = [r for r in review_ids if r in rev_text]
    print("嵌入候选 ...")
    cvec = embed([cand_text[c] for c in cand_list])
    print("嵌入综述 ...")
    rvec = embed([rev_text[r] for r in rev_list])
    cidx = {c: i for i, c in enumerate(cand_list)}
    ridx = {r: i for i, r in enumerate(rev_list)}

    # 归一化做余弦
    cvec /= (np.linalg.norm(cvec, axis=1, keepdims=True) + 1e-8)
    rvec /= (np.linalg.norm(rvec, axis=1, keepdims=True) + 1e-8)

    scores = np.zeros(len(test_df), dtype=np.float32)
    for i, row in enumerate(test_df.itertuples(index=False)):
        rk = "arxiv_" + row.arxiv_id
        ci = cidx.get(row.doc_id); rj = ridx.get(rk)
        scores[i] = float(rvec[rj] @ cvec[ci]) if (ci is not None and rj is not None) else -1.0
    test_df["specter2_score"] = scores

    res = eval_by_query(test_df, test_g, "specter2_score")
    # 与 LambdaMART 配对（需要 test 已有 lgb_score；从 s2d_eval 复算不便，这里直接重载模型分）
    import lightgbm as lgb
    with open(os.path.join(FEAT_DIR, "feature_manifest.json")) as f:
        MAIN = json.load(f)["main_features"]
    booster = lgb.Booster(model_file=os.path.join(OUT_DIR, "lgb_main.txt"))
    test_df["lgb_score"] = booster.predict(test_df[MAIN])
    d = (per_query_ndcg(test_df, test_g, "lgb_score") - per_query_ndcg(test_df, test_g, "specter2_score")).tolist()
    ci = paired_bootstrap_ci(d); pt = paired_test(d)

    out = {"specter2_test": res,
           "learned_vs_specter2_ndcg10": {"ci": ci.as_dict(),
                                          "wilcoxon_p": pt.wilcoxon_p, "rank_biserial": pt.rank_biserial, "n": pt.n}}
    with open(os.path.join(OUT_DIR, "s2d_specter2.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
