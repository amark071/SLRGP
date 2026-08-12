#!/usr/bin/env python3
"""
S2d 泄漏取证：坐实两个跳门的泄漏诊断是"判据过严/设错 null"还是"真泄漏"。

诊断 A（置换门）：正确的泄漏判据 = 置换标签模型是否逼近真实标签模型（若逼近=不需要真标签=泄漏）。
  跑 8 次独立"组内标签置换"重训，报其 val nDCG@10 分布（均值±sd），
  与 (a) 均匀随机打分、(b) 真实模型 val nDCG@10 对照。
  判定：置换分布应彼此接近且 ≪ 真实模型；真实模型显著高出 = 模型确实依赖真标签。

诊断 B（MRR 饱和门）：MRR=1 的 query 里首位命中是否由某个可疑特征平凡决定。
  - 消融重训：去掉 is_self_citation / 去掉 rrf_score，看 val MRR=1 占比与 nDCG 变化；
  - 首位来源：真实模型排第1的候选，落在 rrf 原始名次分布 + 自引占比。

用法：python3 s2d_forensics.py
"""
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

FEAT_DIR = "data/exp2_operator_learnability/ranking/features_s2d"
OUT = "work/exp2_operator_learnability/ranking/s2d_forensics.json"

with open(os.path.join(FEAT_DIR, "feature_manifest.json"), encoding="utf-8") as f:
    MAIN = json.load(f)["main_features"]
BEST = (15, 0.05, 100)


def load(split):
    df = pd.read_parquet(os.path.join(FEAT_DIR, f"{split}.parquet"))
    g = np.load(os.path.join(FEAT_DIR, f"{split}_groups.npy"))
    return df, g


def dcg(labels, k):
    return sum(l / np.log2(i + 2) for i, l in enumerate(labels[:k]))


def eval_ndcg_mrr(df, g, score):
    ndcgs, mrr1, nq = [], 0, 0
    idx = 0
    for gg in g:
        sub = df.iloc[idx:idx + gg]; idx += gg
        if sub["label"].sum() == 0:
            ndcgs.append(0.0); continue
        nq += 1
        order = np.argsort(-sub[score].values, kind="stable")
        ls = sub["label"].values[order]
        idcg = dcg(np.sort(sub["label"].values)[::-1], 10)
        ndcgs.append(dcg(ls, 10) / idcg if idcg > 0 else 0.0)
        if ls[0] == 1:
            mrr1 += 1
    return float(np.mean(ndcgs)), (mrr1 / nq if nq else 0.0)


def train(train_df, g, feats, params, rounds=300, val=None, vg=None):
    tr = lgb.Dataset(train_df[feats], label=train_df["label"], group=g)
    valid = [lgb.Dataset(val[feats], label=val["label"], group=vg, reference=tr)] if val is not None else None
    p = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [10],
         "num_leaves": params[0], "learning_rate": params[1], "min_data_in_leaf": params[2],
         "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5, "seed": 42, "verbose": -1}
    cb = [lgb.early_stopping(30, verbose=False)] if valid else []
    return lgb.train(p, tr, num_boost_round=rounds, valid_sets=valid, callbacks=cb)


def main():
    train_df, tg = load("train")
    val_df, vg = load("val")
    out = {}

    # 真实模型基准
    b = train(train_df, tg, MAIN, BEST, val=val_df, vg=vg)
    val_df["_s"] = b.predict(val_df[MAIN], num_iteration=b.best_iteration)
    real_ndcg, real_mrr1 = eval_ndcg_mrr(val_df, vg, "_s")

    # 均匀随机
    rnd = []
    for s in range(5):
        val_df["_r"] = np.random.default_rng(s).random(len(val_df))
        rnd.append(eval_ndcg_mrr(val_df, vg, "_r")[0])
    rnd_mean = float(np.mean(rnd))

    # 诊断A：8 次独立组内置换
    perm = []
    for s in range(8):
        pdf = train_df.copy()
        labs = pdf["label"].values.copy()
        r = np.random.default_rng(200 + s)
        idx = 0
        for gg in tg:
            sl = slice(idx, idx + gg); idx += gg
            labs[sl] = r.permutation(labs[sl])
        pdf["label"] = labs
        bb = train(pdf, tg, MAIN, BEST, rounds=300)
        val_df["_p"] = bb.predict(val_df[MAIN], num_iteration=300)
        perm.append(eval_ndcg_mrr(val_df, vg, "_p")[0])
    perm = np.array(perm)
    out["diagnosis_A_permutation"] = {
        "real_model_val_ndcg10": real_ndcg,
        "uniform_random_val_ndcg10": rnd_mean,
        "permuted_label_val_ndcg10_mean": float(perm.mean()),
        "permuted_label_val_ndcg10_sd": float(perm.std()),
        "permuted_runs": perm.tolist(),
        "real_minus_permuted": float(real_ndcg - perm.mean()),
        # 训练预算: 真实模型 rounds=300(带验证集 early stopping, 实际 best_iteration<=300);
        # 置换模型固定 300 轮(无 early stopping),对置换侧偏宽松,
        # 因此 real_minus_permuted 为保守下界。
        "train_budget": {"real_max_rounds": 300, "real_early_stopping": True,
                         "permuted_rounds": 300, "permuted_early_stopping": False},
        "verdict": "真实模型远高于置换模型 → 模型依赖真标签，非泄漏" if real_ndcg - perm.mean() > 0.15
                   else "置换模型接近真实模型 → 需排查泄漏",
    }

    # 诊断B：MRR 饱和归因
    # 消融
    feats_no_self = [f for f in MAIN if f != "is_self_citation"]
    feats_no_rrf = [f for f in MAIN if f != "rrf_score"]
    b_ns = train(train_df, tg, feats_no_self, BEST, val=val_df, vg=vg)
    val_df["_ns"] = b_ns.predict(val_df[feats_no_self], num_iteration=b_ns.best_iteration)
    ns_ndcg, ns_mrr1 = eval_ndcg_mrr(val_df, vg, "_ns")
    b_nr = train(train_df, tg, feats_no_rrf, BEST, val=val_df, vg=vg)
    val_df["_nr"] = b_nr.predict(val_df[feats_no_rrf], num_iteration=b_nr.best_iteration)
    nr_ndcg, nr_mrr1 = eval_ndcg_mrr(val_df, vg, "_nr")

    # 首位来源：真实模型排第一的候选，其在该 query 内的 rrf 名次 & 是否自引
    top1_rrf_rank_top1 = 0; top1_self = 0; top1_hit = 0; nq = 0
    idx = 0
    for gg in vg:
        sub = val_df.iloc[idx:idx + gg]; idx += gg
        if sub["label"].sum() == 0:
            continue
        nq += 1
        top1_i = np.argmax(sub["_s"].values)
        # 该候选在本 query 内 rrf 排名是否第1
        rrf_order = np.argsort(-sub["rrf_score"].values, kind="stable")
        if rrf_order[0] == top1_i:
            top1_rrf_rank_top1 += 1
        if sub["is_self_citation"].values[top1_i] == 1:
            top1_self += 1
        if sub["label"].values[top1_i] == 1:
            top1_hit += 1
    out["diagnosis_B_mrr_saturation"] = {
        "real_model_val_mrr1_frac": real_mrr1,
        "drop_is_self_citation": {"val_ndcg10": ns_ndcg, "val_mrr1_frac": ns_mrr1},
        "drop_rrf_score": {"val_ndcg10": nr_ndcg, "val_mrr1_frac": nr_mrr1},
        "among_predicted_top1": {
            "n_queries": nq,
            "frac_equal_rrf_top1": top1_rrf_rank_top1 / nq if nq else 0.0,
            "frac_self_citation": top1_self / nq if nq else 0.0,
            "frac_true_hit": top1_hit / nq if nq else 0.0,
        },
        "note": "若首位命中主要由 rrf/自引这类合法强信号驱动、且去掉后 MRR1 明显下降，则为真信号而非平凡泄漏",
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n写入 {OUT}")


if __name__ == "__main__":
    main()
