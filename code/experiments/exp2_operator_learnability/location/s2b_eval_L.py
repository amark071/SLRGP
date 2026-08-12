#!/usr/bin/env python3
"""
S2b (Learning L) 诚实评测：在完整召回并集（s2b_retrieve.py 产出）上做
覆盖率诊断 + Tier-1 学习融合 + 基线对比 + 配对统计 + 泄漏哨兵。

主指标 Coverage@k（真实引用覆盖率 = recall of true citations）：
  分母 P = 时间安全收紧真引用总数（含未被 depth-3000 召回的不可达正例，来自 P_of.json）。
  故任何系统的覆盖率天花板 = 检索天花板（并集内正例 / P）。

系统：
  rrf / bm25 / dense    固定检索基线（bm25/dense 用真实语料库 rank）
  linear_fusion         logistic 线性学习融合
  lambdamart            LightGBM lambdarank 学习融合（主；val 上按 Coverage@300 选网格）
泄漏哨兵：label-permutation（组内打乱标签重训，覆盖率应塌回 ~随机(k/|union|·ceiling)）。

环境：S2B_UNION_DIR、S2B_OUT_DIR、S2B_NUM_THREADS(默认48)
用法（服务器）：python3 s2b_eval_L.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
try:
    from stats_utils import paired_bootstrap_ci, paired_test
except Exception:
    from confirmatory.stats_utils import paired_bootstrap_ci, paired_test  # type: ignore

UNION_DIR = os.environ.get("S2B_UNION_DIR", "work/exp2_operator_learnability/ranking/union_L")
OUT_DIR = os.environ.get("S2B_OUT_DIR", "work/exp2_operator_learnability/location")
NUM_THREADS = int(os.environ.get("S2B_NUM_THREADS", "48"))
DEPTHS = [100, 300, 500, 1000]
PRIMARY = 300
SEED = 42

FEATURES = ["bm25_score", "bm25_rank_recip", "dense_score", "dense_rank_recip",
            "rrf_score", "year_diff", "same_discipline"]
GRID_LEAVES = [15, 31, 63]
GRID_LR = [0.05, 0.1]
GRID_NEST = [200, 400]


def load():
    dfs = {}
    for s in ["train", "val", "test"]:
        d = pd.read_parquet(os.path.join(UNION_DIR, f"{s}.parquet"))
        d["bm25_rank_recip"] = np.where(d["bm25_rank"] >= 0, 1.0 / (d["bm25_rank"] + 1), 0.0)
        d["dense_rank_recip"] = np.where(d["dense_rank"] >= 0, 1.0 / (d["dense_rank"] + 1), 0.0)
        dfs[s] = d
    P_raw = json.load(open(os.path.join(UNION_DIR, "P_of.json")))
    P_of = {(a, v["split"]): v["P"] for a, v in P_raw.items()}
    return dfs, P_of


def coverage(df, split, score_col, P_of, depth, restrict_col=None):
    sub = df
    if restrict_col is not None:
        sub = sub[sub[restrict_col] >= 0]
    groups = {a: g for a, g in sub.groupby("arxiv_id")}
    per = {}
    num = den = 0
    for (aid, s), P in P_of.items():
        if s != split or P == 0:
            continue
        g = groups.get(aid)
        hit = int(g.nlargest(depth, score_col)["label"].sum()) if g is not None else 0
        per[aid] = hit / P
        num += hit
        den += P
    return {"micro": num / den if den else 0.0,
            "macro": float(np.mean(list(per.values()))) if per else 0.0}, per


def eval_system(df, split, score_col, P_of, restrict_col=None):
    out = {}
    for k in DEPTHS:
        m, _ = coverage(df, split, score_col, P_of, k, restrict_col)
        out[str(k)] = m
    return out


def ceiling(df, split, P_of):
    """检索天花板：并集内正例 / P（micro & macro）。"""
    groups = {a: int(g["label"].sum()) for a, g in df.groupby("arxiv_id")}
    num = den = 0
    per = []
    for (aid, s), P in P_of.items():
        if s != split or P == 0:
            continue
        inpool = groups.get(aid, 0)
        num += inpool
        den += P
        per.append(inpool / P)
    return {"micro": num / den if den else 0.0, "macro": float(np.mean(per)) if per else 0.0}


def train_lgb(dfs, permute=False):
    import lightgbm as lgb
    tr = dfs["train"].sort_values("arxiv_id").copy()
    va = dfs["val"].sort_values("arxiv_id").copy()
    if permute:
        rng = np.random.default_rng(SEED)
        tr["label"] = tr.groupby("arxiv_id")["label"].transform(lambda s: rng.permutation(s.values))
    gtr = tr.groupby("arxiv_id").size().values
    ds = lgb.Dataset(tr[FEATURES].values, label=tr["label"].values, group=gtr)
    Xva = va[FEATURES].values
    best = None
    for lv in GRID_LEAVES:
        for lr in GRID_LR:
            for ne in GRID_NEST:
                params = dict(objective="lambdarank", metric="ndcg", num_leaves=lv,
                              learning_rate=lr, min_data_in_leaf=50, num_threads=NUM_THREADS,
                              label_gain=[0, 1], verbosity=-1, seed=SEED)
                m = lgb.train(params, ds, num_boost_round=ne)
                va["_s"] = m.predict(Xva)
                num = den = 0
                for _, g in va.groupby("arxiv_id"):
                    num += int(g.nlargest(PRIMARY, "_s")["label"].sum())
                    den += int(g["label"].sum())
                sc = num / den if den else 0.0
                if best is None or sc > best[0]:
                    best = (sc, (lv, lr, ne), m)
    return best[2], best[1], best[0]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dfs, P_of = load()
    alldf = pd.concat(dfs.values(), ignore_index=True)
    print(f"并集行数 train/val/test = {[len(dfs[s]) for s in ['train','val','test']]}")

    res = {"features": FEATURES, "depths": DEPTHS, "primary": PRIMARY, "systems": {}}
    # 天花板
    res["retrieval_ceiling"] = {s: ceiling(dfs[s], s, P_of) for s in ["val", "test"]}
    print(f"检索天花板 test micro={res['retrieval_ceiling']['test']['micro']:.4f} "
          f"macro={res['retrieval_ceiling']['test']['macro']:.4f}")

    # 基线
    for name, col, restrict in [("rrf", "rrf_score", None),
                                ("bm25", "bm25_rank_recip", "bm25_rank"),
                                ("dense", "dense_rank_recip", "dense_rank")]:
        res["systems"][name] = {s: eval_system(dfs[s], s, col, P_of, restrict) for s in ["val", "test"]}
        print(f"[{name}] test cov@300 micro={res['systems'][name]['test']['300']['micro']:.4f} "
              f"macro={res['systems'][name]['test']['300']['macro']:.4f}")

    # 线性学习融合
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    tr = dfs["train"]
    sc_ = StandardScaler().fit(tr[FEATURES].values)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(sc_.transform(tr[FEATURES].values), tr["label"].values)
    for s in ["val", "test"]:
        dfs[s]["logit"] = clf.decision_function(sc_.transform(dfs[s][FEATURES].values))
    res["systems"]["linear_fusion"] = {s: eval_system(dfs[s], s, "logit", P_of) for s in ["val", "test"]}
    print(f"[linear_fusion] test cov@300 micro={res['systems']['linear_fusion']['test']['300']['micro']:.4f}")

    # LambdaMART 学习融合（主）
    model, grid, vs = train_lgb(dfs, permute=False)
    for s in ["val", "test"]:
        dfs[s]["lmart"] = model.predict(dfs[s][FEATURES].values)
    res["lambdamart_grid"] = {"best": list(grid), "val_cov300": vs}
    res["systems"]["lambdamart"] = {s: eval_system(dfs[s], s, "lmart", P_of) for s in ["val", "test"]}
    res["feature_importance"] = dict(zip(FEATURES, [int(x) for x in model.feature_importance()]))
    print(f"[lambdamart] grid={grid} test cov@300 micro={res['systems']['lambdamart']['test']['300']['micro']:.4f} "
          f"macro={res['systems']['lambdamart']['test']['300']['macro']:.4f}")

    # 泄漏哨兵：打乱标签
    pmodel, _, _ = train_lgb(dfs, permute=True)
    dfs["test"]["perm"] = pmodel.predict(dfs["test"][FEATURES].values)
    res["systems"]["lambdamart_permuted"] = {"test": eval_system(dfs["test"], "test", "perm", P_of)}
    rand_expect = PRIMARY  # 随机期望：见下方计算
    print(f"[leakage permuted] test cov@300 micro={res['systems']['lambdamart_permuted']['test']['300']['micro']:.4f}")

    # 随机排序期望覆盖率@300（每篇 = min(300,|union|)/|union| · inpool/P，micro 汇总）
    num = den = 0
    for aid, g in dfs["test"].groupby("arxiv_id"):
        P = P_of.get((aid, "test"), 0)
        if P == 0:
            continue
        u = len(g)
        inpool = int(g["label"].sum())
        num += inpool * min(PRIMARY, u) / u
        den += P
    res["random_expected_cov300_test_micro"] = num / den if den else 0.0
    print(f"[random-expected] test cov@300 micro={res['random_expected_cov300_test_micro']:.4f}")

    # 配对统计：lambdamart vs rrf，@300 test
    _, per_l = coverage(dfs["test"], "test", "lmart", P_of, PRIMARY)
    _, per_r = coverage(dfs["test"], "test", "rrf_score", P_of, PRIMARY)
    aids = sorted(set(per_l) & set(per_r))
    diffs = [per_l[a] - per_r[a] for a in aids]
    ci = paired_bootstrap_ci(diffs, n_boot=10000)
    pt = paired_test(diffs)
    res["paired_lambdamart_vs_rrf_cov300_test"] = {
        "n": len(aids), "mean_diff": ci.point_estimate, "ci_low": ci.ci_low,
        "ci_high": ci.ci_high, "ci_method": ci.method,
        "wilcoxon_p": pt.wilcoxon_p, "permutation_p": pt.permutation_p, "rank_biserial": pt.rank_biserial}
    print(f"[learned vs rrf @300] Δmacro-per-review={ci.point_estimate:.4f} "
          f"CI[{ci.ci_low:.4f},{ci.ci_high:.4f}] wilcoxon_p={pt.wilcoxon_p:.3g}")

    with open(os.path.join(OUT_DIR, "s2b_eval_L.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n写入 {OUT_DIR}/s2b_eval_L.json")


if __name__ == "__main__":
    main()
