#!/usr/bin/env python3
"""
S2d 标签噪声敏感性分析（把审计残余噪声传导进头条主张 learned vs RRF）。

审计测得总体分层链接精度 0.962（~3.8% 正例噪声），且噪声有结构：逐层错误率
  no_doi 0.027、clean 0.049、short_generic_title 0.143（护栏后）。
为证明 C2 结论不依赖完美标签，对 test 正例按"逐层实测错误率 × 缩放系数 c"翻为负例
（c 使总体翻转率命中目标 ε），重算 learned/RRF 逐 query nDCG@10 的配对差 Δ，200 次
Monte-Carlo。ε 三档：实测(c=1,≈0.038)、2×、3×。注册通过条件：Δ 分布 2.5 分位 > 0
（即噪声下 learned 仍稳超 RRF）。

用法：S2D_FEAT_DIR=data/exp2_operator_learnability/ranking/features_s2d_tight S2D_OUT_DIR=work/exp2_ranking \
      python3 s2d_noise_sensitivity.py
"""
import json
import os
import re
import sqlite3

import lightgbm as lgb
import numpy as np
import pandas as pd

FEAT_DIR = os.environ.get("S2D_FEAT_DIR", "data/exp2_operator_learnability/ranking/features_s2d_tight")
OUT_DIR = os.environ.get("S2D_OUT_DIR", "work/exp2_operator_learnability/ranking_tight")
CORPUS_DB = "data/common/unified_corpus/unified_corpus.db"

# 审计实测逐层错误率（1 - 分层精度；s2d_audit_combine.py / cell）
ERR = {"no_doi": 0.027, "clean": 0.049, "short_generic_title": 0.143, "no_doi|short_generic_title": 0.0}
N_MC = 200
EPS_LABELS = [("measured_1x", 1.0), ("2x", 2.0), ("3x", 3.0)]
SEED = 20260709


def norm_title(t):
    t = (t or "").lower().strip().replace("&", " and ")
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def dcg(labels, k):
    return sum(l / np.log2(i + 2) for i, l in enumerate(labels[:k]))


def stratum(doi, title):
    s = []
    if not doi:
        s.append("no_doi")
    nt = norm_title(title)
    if nt and len(nt.split()) <= 3:
        s.append("short_generic_title")
    return "|".join(sorted(s)) if s else "clean"


def main():
    with open(os.path.join(FEAT_DIR, "feature_manifest.json"), encoding="utf-8") as f:
        MAIN = json.load(f)["main_features"]
    test = pd.read_parquet(os.path.join(FEAT_DIR, "test.parquet"))
    groups = np.load(os.path.join(FEAT_DIR, "test_groups.npy"))
    booster = lgb.Booster(model_file=os.path.join(OUT_DIR, "lgb_main.txt"))
    test["lgb_score"] = booster.predict(test[MAIN], num_iteration=booster.best_iteration)

    # 逐行风险层（corpus doi/title）
    conn = sqlite3.connect(CORPUS_DB)
    ids = sorted(set(test["doc_id"].tolist()))
    meta = {}
    for i in range(0, len(ids), 800):
        chunk = ids[i:i + 800]
        ph = ",".join("?" for _ in chunk)
        for row in conn.execute(f"SELECT doc_id,doi,title FROM papers WHERE doc_id IN ({ph})", chunk):
            meta[row[0]] = (row[1], row[2])
    test["_stratum"] = [stratum(*meta.get(d, (None, None))) for d in test["doc_id"]]

    # 预切分每 query 的行索引
    qidx = []
    idx = 0
    for g in groups:
        qidx.append((idx, idx + g)); idx += g
    lgb_s = test["lgb_score"].values
    rrf_s = test["rrf_score"].values
    base_label = test["label"].values.astype(int)
    strat = test["_stratum"].values

    pos_mask = base_label == 1
    base_flip_p = np.array([ERR.get(s, 0.049) for s in strat]) * pos_mask
    # 缩放系数：使 c=1 时总体翻转率 ≈ 审计测得（各正例按其层错误率翻）
    pop_pos = pos_mask.sum()
    base_eps = base_flip_p.sum() / pop_pos
    print(f"test 正例数={pop_pos}  基线(c=1)期望翻转率={base_eps:.4f}")

    def ndcg_diff(labels):
        d = np.empty(len(qidx))
        for qi, (a, b) in enumerate(qidx):
            ls = labels[a:b]
            if ls.sum() == 0:
                d[qi] = 0.0; continue
            idcg = dcg(np.sort(ls)[::-1], 10)
            o1 = np.argsort(-lgb_s[a:b], kind="stable")
            o2 = np.argsort(-rrf_s[a:b], kind="stable")
            n1 = dcg(ls[o1], 10) / idcg if idcg > 0 else 0.0
            n2 = dcg(ls[o2], 10) / idcg if idcg > 0 else 0.0
            d[qi] = n1 - n2
        return d

    clean_diff = ndcg_diff(base_label)
    clean_mean = float(clean_diff.mean())
    print(f"干净标签 ΔnDCG@10(learned-RRF) = {clean_mean:+.4f}")

    rng = np.random.default_rng(SEED)
    out = {"clean_delta_mean": clean_mean, "n_test_queries": len(qidx),
           "test_positives": int(pop_pos), "base_eps_c1": float(base_eps),
           "per_stratum_error_used": ERR, "n_mc": N_MC, "levels": {}}
    all_pass = True
    for name, c in EPS_LABELS:
        flip_p = np.minimum(1.0, base_flip_p * c)
        eff_eps = flip_p.sum() / pop_pos
        deltas = np.empty(N_MC)
        for m in range(N_MC):
            corrupt = base_label.copy()
            draws = rng.random(len(base_label)) < flip_p
            corrupt[draws] = 0
            deltas[m] = ndcg_diff(corrupt).mean()
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        passed = bool(lo > 0)
        all_pass = all_pass and passed
        out["levels"][name] = {"scale_c": c, "effective_eps": float(eff_eps),
                               "delta_mean": float(deltas.mean()),
                               "delta_p2.5": float(lo), "delta_p97.5": float(hi),
                               "frac_draws_positive": float((deltas > 0).mean()),
                               "pass_lo_gt_0": passed}
        print(f"ε[{name}] c={c} eff={eff_eps:.4f}  Δ={deltas.mean():+.4f} "
              f"[{lo:+.4f},{hi:+.4f}]  正差占比={100*(deltas>0).mean():.1f}%  pass={passed}")

    out["registered_pass_all_levels"] = all_pass
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "s2d_noise_sensitivity.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n注册通过(全 ε 档 2.5 分位>0): {all_pass}")
    print(f"写入 {OUT_DIR}/s2d_noise_sensitivity.json")


if __name__ == "__main__":
    main()
