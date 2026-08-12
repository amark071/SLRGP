#!/usr/bin/env python3
"""
S2d 训练 + 评估（学习排序 R，audited re-analysis 路线）。

设计要点（对齐论文 Methods 中 R 算子可学习性的实验设计）：
  - 特征只用 MAIN（时间安全 + 无标签代理）；AUX（快照声望）单独跑辅助变体。
  - 27 组固定网格按 val nDCG@10 选参，test 一次性评测。
  - 泄漏单元测试（阈值预冻结）：标签置换塌缩 / 单特征探针 / 每 query MRR 饱和。
  - 外部基线：RRF（主对照，且是 R 部署时收到的输入序）/ BM25 / dense / 线性配对排序器。
  - 主口径：test 每篇综述接口内 nDCG@10 / Recall@50 / MRR，零正例 query 记 0；
    另报 conditional-on-retrieval（剔除零正例 query）。
  - 配对统计：学习排序 vs RRF 的逐 query nDCG@10 差值 → BCa bootstrap + Wilcoxon + 置换。

用法：python3 s2d_train_eval.py
"""
import itertools
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_utils import paired_bootstrap_ci, paired_test  # noqa: E402

FEAT_DIR = os.environ.get("S2D_FEAT_DIR", "data/exp2_operator_learnability/ranking/features_s2d")
OUT_DIR = os.environ.get("S2D_OUT_DIR", "work/exp2_operator_learnability/ranking")

with open(os.path.join(FEAT_DIR, "feature_manifest.json"), encoding="utf-8") as f:
    FM = json.load(f)
MAIN = FM["main_features"]
AUX = FM["aux_features"]

GRID = list(itertools.product([15, 31, 63], [0.02, 0.05, 0.1], [20, 50, 100]))  # num_leaves, lr, min_data
LEAK_PERMUTE_TOL = 0.02       # 标签置换后 val nDCG@10 相对随机期望的容差
LEAK_SINGLE_FEAT_FRAC = 0.90  # 任一单特征 >= 全模型 90% 则告警
LEAK_MRR_SAT_FRAC = 0.20      # val MRR==1 的 query 占比 > 20% 则告警


def load(split):
    df = pd.read_parquet(os.path.join(FEAT_DIR, f"{split}.parquet"))
    groups = np.load(os.path.join(FEAT_DIR, f"{split}_groups.npy"))
    assert groups.sum() == len(df), f"{split}: {groups.sum()} != {len(df)}"
    return df, groups


def dcg(labels, k):
    return sum(l / np.log2(i + 2) for i, l in enumerate(labels[:k]))


def eval_by_query(df, groups, score, k_ndcg=10, conditional=False):
    ndcgs, recalls, mrrs = [], [], []
    idx = 0
    for g in groups:
        sub = df.iloc[idx:idx + g]
        idx += g
        npos = int(sub["label"].sum())
        if conditional and npos == 0:
            continue
        if npos == 0:
            ndcgs.append(0.0); recalls.append(0.0); mrrs.append(0.0)
            continue
        order = np.argsort(-sub[score].values, kind="stable")
        ls = sub["label"].values[order]
        idcg = dcg(np.sort(sub["label"].values)[::-1], k_ndcg)
        ndcgs.append(dcg(ls, k_ndcg) / idcg if idcg > 0 else 0.0)
        recalls.append(ls[:50].sum() / npos)
        first = np.argmax(ls == 1) + 1 if (ls == 1).any() else 0
        mrrs.append(1.0 / first if first else 0.0)
    return {"nDCG@10": float(np.mean(ndcgs)), "Recall@50": float(np.mean(recalls)),
            "MRR": float(np.mean(mrrs)), "n": len(ndcgs)}


def per_query_ndcg(df, groups, score, k=10):
    """返回逐 query nDCG@10 列表（含零正例 query 记 0），用于配对统计。"""
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


def train_lgb(train_df, train_g, val_df, val_g, feats, params, rounds=500):
    tr = lgb.Dataset(train_df[feats], label=train_df["label"], group=train_g)
    va = lgb.Dataset(val_df[feats], label=val_df["label"], group=val_g, reference=tr)
    p = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [10],
         "num_leaves": params[0], "learning_rate": params[1], "min_data_in_leaf": params[2],
         "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
         "num_threads": int(os.environ.get("S2D_NUM_THREADS", "32")),
         "seed": 42, "verbose": -1}
    booster = lgb.train(p, tr, num_boost_round=rounds, valid_sets=[va],
                        callbacks=[lgb.early_stopping(30, verbose=False)])
    return booster


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train_df, train_g = load("train")
    val_df, val_g = load("val")
    test_df, test_g = load("test")
    print(f"train {len(train_df)}/{len(train_g)}  val {len(val_df)}/{len(val_g)}  test {len(test_df)}/{len(test_g)}")

    # ---------- 网格选参（val nDCG@10） ----------
    grid_results = []
    best = None
    for params in GRID:
        b = train_lgb(train_df, train_g, val_df, val_g, MAIN, params)
        val_df["_s"] = b.predict(val_df[MAIN], num_iteration=b.best_iteration)
        m = eval_by_query(val_df, val_g, "_s")["nDCG@10"]
        grid_results.append({"params": params, "val_ndcg10": m, "best_iter": b.best_iteration})
        if best is None or m > best["val_ndcg10"]:
            best = {"params": params, "val_ndcg10": m, "booster": b}
    print(f"最优网格: {best['params']}  val nDCG@10={best['val_ndcg10']:.4f}")
    booster = best["booster"]

    # ---------- 泄漏单元测试（在 train/val 上，test 未接触） ----------
    leak = {}
    # (i) 标签置换塌缩：组内打乱标签重训，val nDCG@10 应逼近随机期望
    rng = np.random.default_rng(0)
    rand_ndcgs = []
    for seed in range(5):
        rs = np.random.default_rng(seed).random(len(val_df))
        val_df["_rand"] = rs
        rand_ndcgs.append(eval_by_query(val_df, val_g, "_rand")["nDCG@10"])
    rand_expect = float(np.mean(rand_ndcgs))
    perm_ndcgs = []
    for seed in range(5):
        pdf = train_df.copy()
        # 组内置换标签
        idx = 0; labs = pdf["label"].values.copy()
        r2 = np.random.default_rng(100 + seed)
        for g in train_g:
            sl = slice(idx, idx + g); idx += g
            labs[sl] = r2.permutation(labs[sl])
        pdf["label"] = labs
        b = train_lgb(pdf, train_g, val_df, val_g, MAIN, best["params"], rounds=150)
        val_df["_ps"] = b.predict(val_df[MAIN], num_iteration=b.best_iteration)
        perm_ndcgs.append(eval_by_query(val_df, val_g, "_ps")["nDCG@10"])
    perm_mean = float(np.mean(perm_ndcgs))
    leak["label_permutation"] = {
        "random_expectation": rand_expect, "permuted_label_val_ndcg10": perm_mean,
        "gap": perm_mean - rand_expect, "tol": LEAK_PERMUTE_TOL,
        "pass": abs(perm_mean - rand_expect) <= LEAK_PERMUTE_TOL,
    }
    # (ii) 单特征探针
    full_val = best["val_ndcg10"]
    single = {}
    for feat in MAIN:
        b = train_lgb(train_df, train_g, val_df, val_g, [feat], best["params"], rounds=150)
        val_df["_sf"] = b.predict(val_df[[feat]], num_iteration=b.best_iteration)
        single[feat] = eval_by_query(val_df, val_g, "_sf")["nDCG@10"]
    flagged = {f: v for f, v in single.items() if v >= LEAK_SINGLE_FEAT_FRAC * full_val}
    leak["single_feature_probe"] = {"full_val_ndcg10": full_val, "per_feature": single,
                                     "threshold": LEAK_SINGLE_FEAT_FRAC * full_val,
                                     "flagged": flagged, "pass": len(flagged) == 0}
    # (iii) 每 query MRR 饱和（用最优模型在 val）
    val_df["_bs"] = booster.predict(val_df[MAIN], num_iteration=booster.best_iteration)
    idx = 0; mrr1 = 0; nq = 0
    for g in val_g:
        sub = val_df.iloc[idx:idx + g]; idx += g
        if sub["label"].sum() == 0:
            continue
        nq += 1
        order = np.argsort(-sub["_bs"].values, kind="stable")
        ls = sub["label"].values[order]
        first = np.argmax(ls == 1) + 1 if (ls == 1).any() else 0
        if first == 1:
            mrr1 += 1
    sat = mrr1 / nq if nq else 0.0
    leak["mrr_saturation"] = {"frac_val_queries_mrr1": sat, "threshold": LEAK_MRR_SAT_FRAC,
                              "pass": sat <= LEAK_MRR_SAT_FRAC}
    leak_all_pass = all(v["pass"] for v in leak.values())
    print("\n=== 泄漏单元测试 ===")
    for k, v in leak.items():
        print(f"  {k}: pass={v['pass']}  {({kk: vv for kk, vv in v.items() if kk not in ('per_feature',)})}")
    if not leak_all_pass:
        print("!! 泄漏测试未全通过——按协议应在接触 test 前排查。仍继续但结果标注告警。")

    # ---------- 线性配对基线（同特征） ----------
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train_df[MAIN].fillna(0))
    Xte = scaler.transform(test_df[MAIN].fillna(0))
    lr = LogisticRegression(max_iter=3000, class_weight="balanced")
    lr.fit(Xtr, train_df["label"])
    test_df["linear_score"] = lr.predict_proba(Xte)[:, 1]

    # ---------- test 评测：所有臂 ----------
    booster.save_model(os.path.join(OUT_DIR, "lgb_main.txt"))
    test_df["lgb_score"] = booster.predict(test_df[MAIN], num_iteration=booster.best_iteration)

    arms = {
        "rrf_fusion": "rrf_score", "bm25_only": "bm25_score", "dense_only": "dense_score",
        "linear_pairwise": "linear_score", "lambdamart_main": "lgb_score",
    }
    results = {}
    for name, col in arms.items():
        results[name] = {
            "primary": eval_by_query(test_df, test_g, col),
            "conditional_on_retrieval": eval_by_query(test_df, test_g, col, conditional=True),
        }

    # ---------- 配对统计：learned vs RRF（主假设 H1）----------
    ndcg_lgb = per_query_ndcg(test_df, test_g, "lgb_score")
    ndcg_rrf = per_query_ndcg(test_df, test_g, "rrf_score")
    diffs = (ndcg_lgb - ndcg_rrf).tolist()
    ci = paired_bootstrap_ci(diffs)
    pt = paired_test(diffs)
    paired = {"learned_vs_rrf_ndcg10": {"ci": ci.as_dict(),
                                        "test": {"n": pt.n, "wilcoxon_p": pt.wilcoxon_p,
                                                 "permutation_p": pt.permutation_p,
                                                 "rank_biserial": pt.rank_biserial}}}
    # 对 BM25/dense/linear 也给配对 CI（次要 H2/H3）
    for name, col in [("bm25_only", "bm25_score"), ("dense_only", "dense_score"),
                      ("linear_pairwise", "linear_score")]:
        d = (ndcg_lgb - per_query_ndcg(test_df, test_g, col)).tolist()
        c = paired_bootstrap_ci(d)
        paired[f"learned_vs_{name}_ndcg10"] = {"ci": c.as_dict()}

    importance = dict(sorted(zip(MAIN, booster.feature_importance("gain").tolist()),
                             key=lambda kv: -kv[1]))

    out = {
        "n_reviews": {"train": len(train_g), "val": len(val_g), "test": len(test_g)},
        "best_params": {"num_leaves": best["params"][0], "learning_rate": best["params"][1],
                        "min_data_in_leaf": best["params"][2], "best_iteration": booster.best_iteration},
        "grid": grid_results,
        "leakage_tests": leak, "leakage_all_pass": leak_all_pass,
        "test_results": results, "paired_stats": paired,
        "feature_importance_gain": importance, "features_used": MAIN,
    }
    with open(os.path.join(OUT_DIR, "s2d_eval.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n=== test 评测（主口径，零正例 query 记 0）===")
    for name in arms:
        m = results[name]["primary"]
        print(f"  {name:18s} nDCG@10={m['nDCG@10']:.4f}  Recall@50={m['Recall@50']:.4f}  MRR={m['MRR']:.4f} (n={m['n']})")
    print("\n=== 学习排序 vs RRF（H1 主假设）===")
    print(f"  ΔnDCG@10 = {ci.point_estimate:+.4f}  95%CI [{ci.ci_low:+.4f}, {ci.ci_high:+.4f}]  "
          f"Wilcoxon p={pt.wilcoxon_p:.3g}  r_rb={pt.rank_biserial:.3f}  (n={pt.n})")
    print("\n=== 特征重要性(gain) top8 ===")
    for k, v in list(importance.items())[:8]:
        print(f"  {k:26s} {v:.1f}")
    print(f"\n写入 {OUT_DIR}/s2d_eval.json")


if __name__ == "__main__":
    main()
