#!/usr/bin/env python3
"""
S2b (Learning L) Tier-1 学习融合 训练 + 评测。

任务：在 L 交付的候选池（retrieved_by_L，时间安全）内，用检索信号 + 轻量时间安全
元数据学习一个融合排序，把真实引用尽量排进 top-300，攻克"排序可恢复"头room
（诊断 gap_rank≈0.31 micro）。Tier-1 不做重检索，故其覆盖率天花板 = 检索天花板≈0.55。

主指标：Coverage@300（= Recall@300 的引用覆盖口径），micro（正例合计）与 macro（逐综述平均）。
  分母 P（每篇）= 时间安全的收紧真引用总数（含未被任何检索器召回者）——与 s2b_diag 完全一致，
  使 Tier-1 无法通过口径把不可达正例算作命中。

系统对比：
  固定基线   rrf / bm25 / dense（按各自分数降序）
  线性学习融合   logistic regression（pointwise，按预测概率排序）
  学习融合(主)   LightGBM LambdaMART（lambdarank 训练，val 上按 Coverage@300 选网格）

统计：learned-vs-rrf 的逐综述 Coverage@300 差值 → 配对 BCa bootstrap CI + Wilcoxon（stats_utils）。
泄漏哨兵：label-permutation（打乱训练标签重训，覆盖率应塌回基线附近）。
单特征探针：rrf 单独作为基线已报告；learned>rrf 即证明融合有增量。

环境变量：
  S2B_DEPTHS   默认 "100,300,500"
  S2B_NUM_THREADS  LightGBM 线程数（默认 48）
输出：S2B_OUT_DIR（默认 work/exp2_operator_learnability/location）/s2b_eval.json
用法（服务器）：python3 s2b_train_eval.py
"""
import glob
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
try:
    from stats_utils import paired_bootstrap_ci, paired_test
except Exception:
    from confirmatory.stats_utils import paired_bootstrap_ci, paired_test  # type: ignore

CAND_DIR = "work/exp2_operator_learnability/ranking/candidates"
CORPUS_DB = "data/common/unified_corpus/unified_corpus.db"
TIGHT_DIR = "data/exp2_operator_learnability/ranking/resolved_tight"
SPLIT_MANIFEST = "data/exp2_operator_learnability/ranking/features_s2d_tight/split_manifest.json"
OUT_DIR = os.environ.get("S2B_OUT_DIR", "work/exp2_operator_learnability/location")

DEPTHS = [int(x) for x in os.environ.get("S2B_DEPTHS", "100,300,500").split(",")]
PRIMARY_DEPTH = 300
NUM_THREADS = int(os.environ.get("S2B_NUM_THREADS", "48"))
SEED = 42

FEATURES = ["bm25_score", "bm25_rank_recip", "dense_score", "dense_rank_recip",
            "rrf_score", "year_diff", "same_discipline"]

# LightGBM lambdarank 网格（val 上按 Coverage@300 选）
GRID_LEAVES = [15, 31, 63]
GRID_LR = [0.05, 0.1]
GRID_NEST = [200, 400]


def load_tight():
    tight = {}
    for jf in glob.glob(os.path.join(TIGHT_DIR, "*", "*.json")):
        r = json.load(open(jf, encoding="utf-8"))
        tight[r["arxiv_id"]] = set(r["matched_doc_ids"])
    return tight


def build_frame():
    tight = load_tight()
    split_of = json.load(open(SPLIT_MANIFEST, encoding="utf-8"))["assignments"]
    conn = sqlite3.connect(CORPUS_DB)
    files = sorted(glob.glob(os.path.join(CAND_DIR, "*", "*", "*.json")))

    recs = [json.load(open(f, encoding="utf-8")) for f in files]
    all_ids = set()
    for r in recs:
        for c in r["candidates"]:
            all_ids.add(c["doc_id"])
    year, disc = {}, {}
    ids = list(all_ids)
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]
        ph = ",".join("?" for _ in chunk)
        for d, y, dd in conn.execute(
                f"SELECT doc_id, year, discipline FROM papers WHERE doc_id IN ({ph})", chunk):
            year[d] = y
            disc[d] = dd

    rows = []
    # 每篇综述的分母 P（时间安全收紧正例总数，含不可达），用于 coverage
    P_of = {}
    for r in recs:
        aid = r["arxiv_id"]
        split = split_of.get(aid)
        if split is None:
            continue
        ry = r.get("review_year")
        pos = tight.get(aid, set())
        rdisc = r["discipline"]
        # 分母：时间安全的收紧正例（不限是否被召回）
        P = 0
        for d in pos:
            cy = year.get(d)
            if ry and cy and cy > ry:
                continue
            P += 1
        P_of[(aid, split)] = P
        # 训练/排序行：仅 retrieved_by_L 且时间安全的候选
        for c in r["candidates"]:
            if not c.get("retrieved_by_L"):
                continue
            d = c["doc_id"]
            cy = year.get(d)
            if ry and cy and cy > ry:
                continue
            br = c.get("bm25_rank", -1)
            dr = c.get("dense_rank", -1)
            rows.append({
                "arxiv_id": aid, "split": split, "doc_id": d,
                "label": int(d in pos),
                "bm25_rank": br, "dense_rank": dr,
                "bm25_score": c.get("bm25_score", 0.0),
                "bm25_rank_recip": 1.0 / (br + 1) if br >= 0 else 0.0,
                "dense_score": c.get("dense_score", 0.0),
                "dense_rank_recip": 1.0 / (dr + 1) if dr >= 0 else 0.0,
                "rrf_score": c.get("rrf_score", 0.0),
                "year_diff": (ry - cy) if (ry and cy) else -1,
                "same_discipline": int(disc.get(d) == rdisc),
            })
    df = pd.DataFrame(rows)
    return df, P_of


def _groups(df, split, score_col, restrict_col=None):
    """返回 {aid: DataFrame(该综述可参与排序的候选)}。
    restrict_col 非空时，只保留该检索器真正召回的候选（rank>=0），得到纯检索器基线。"""
    sub = df[df["split"] == split]
    if restrict_col is not None:
        sub = sub[sub[restrict_col] >= 0]
    return {aid: g for aid, g in sub.groupby("arxiv_id")}


def coverage_per_review(df, score_col, split, P_of, depth, restrict_col=None):
    """该 split 每篇综述（P>0）的 Coverage@depth。分母含全部 P>0 综述：
    无可排序候选者 coverage=0（仍计入），避免丢掉'全不可达'综述抬高覆盖率。"""
    groups = _groups(df, split, score_col, restrict_col)
    out = {}
    for (aid, s), P in P_of.items():
        if s != split or P == 0:
            continue
        g = groups.get(aid)
        hit = int(g.nlargest(depth, score_col)["label"].sum()) if g is not None else 0
        out[aid] = hit / P
    return out


def eval_system(df, score_col, split, P_of, restrict_col=None):
    res = {}
    for k in DEPTHS:
        cov = coverage_per_review(df, score_col, split, P_of, k, restrict_col)
        vals = list(cov.values())
        groups = _groups(df, split, score_col, restrict_col)
        num = den = 0
        for (aid, s), P in P_of.items():
            if s != split or P == 0:
                continue
            g = groups.get(aid)
            hit = int(g.nlargest(k, score_col)["label"].sum()) if g is not None else 0
            num += hit
            den += P
        res[str(k)] = {"micro": num / den if den else 0.0,
                       "macro": float(np.mean(vals)) if vals else 0.0}
    return res


def train_lightgbm(df, permute=False):
    import lightgbm as lgb
    tr = df[df["split"] == "train"].copy()
    va = df[df["split"] == "val"].copy()
    if permute:
        rng = np.random.default_rng(SEED)
        # 在每篇综述内部打乱标签（保持每组正例数不变）
        tr["label"] = tr.groupby("arxiv_id")["label"].transform(
            lambda s: rng.permutation(s.values))
    tr = tr.sort_values("arxiv_id")
    va = va.sort_values("arxiv_id")
    Xtr, ytr = tr[FEATURES].values, tr["label"].values
    gtr = tr.groupby("arxiv_id").size().values
    Xva, yva = va[FEATURES].values, va["label"].values
    gva = va.groupby("arxiv_id").size().values

    ds_tr = lgb.Dataset(Xtr, label=ytr, group=gtr)
    best = None
    for leaves in GRID_LEAVES:
        for lr in GRID_LR:
            for nest in GRID_NEST:
                params = dict(objective="lambdarank", metric="ndcg",
                              num_leaves=leaves, learning_rate=lr,
                              min_data_in_leaf=50, num_threads=NUM_THREADS,
                              label_gain=[0, 1], verbosity=-1, seed=SEED)
                model = lgb.train(params, ds_tr, num_boost_round=nest)
                # val 选参：Coverage@300 的 micro 口径，分母用 val 内召回正例数（选参近似即可）
                va_tmp = va.copy()
                va_tmp["_s"] = model.predict(Xva)
                num = den = 0
                for _, g in va_tmp.groupby("arxiv_id"):
                    num += int(g.nlargest(PRIMARY_DEPTH, "_s")["label"].sum())
                    den += int(g["label"].sum())
                score = num / den if den else 0.0
                if best is None or score > best[0]:
                    best = (score, (leaves, lr, nest), model)
    return best[2], best[1], best[0]


def add_scores(df, model, col):
    df[col] = model.predict(df[FEATURES].values)
    return df


def train_logistic(df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    tr = df[df["split"] == "train"]
    sc = StandardScaler().fit(tr[FEATURES].values)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(sc.transform(tr[FEATURES].values), tr["label"].values)
    return sc, clf


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("构建特征帧 ...")
    df, P_of = build_frame()
    print(f"行数={len(df)}  train/val/test 组数="
          f"{[df[df.split==s]['arxiv_id'].nunique() for s in ['train','val','test']]}")

    result = {"features": FEATURES, "depths": DEPTHS, "primary_depth": PRIMARY_DEPTH,
              "n_rows": int(len(df)), "systems": {}}

    # 固定基线：rrf 在整个召回并集上排序；bm25/dense 限定各自真正召回的候选（纯检索器口径）
    baseline_spec = [("rrf", "rrf_score", None),
                     ("bm25", "bm25_rank_recip", "bm25_rank"),
                     ("dense", "dense_rank_recip", "dense_rank")]
    for name, col, restrict in baseline_spec:
        result["systems"][name] = {s: eval_system(df, col, s, P_of, restrict) for s in ["val", "test"]}
        print(f"[baseline {name}] test@300 micro="
              f"{result['systems'][name]['test'][str(PRIMARY_DEPTH)]['micro']:.4f}")

    # 线性学习融合
    sc, clf = train_logistic(df)
    df["logit_score"] = clf.decision_function(sc.transform(df[FEATURES].values))
    result["systems"]["linear_fusion"] = {s: eval_system(df, "logit_score", s, P_of) for s in ["val", "test"]}
    print(f"[linear_fusion] test@300 micro="
          f"{result['systems']['linear_fusion']['test'][str(PRIMARY_DEPTH)]['micro']:.4f}")

    # LambdaMART 学习融合（主）
    model, grid, valscore = train_lightgbm(df, permute=False)
    df = add_scores(df, model, "lmart_score")
    result["lambdamart_grid"] = {"best": list(grid), "val_select_score": valscore}
    result["systems"]["lambdamart"] = {s: eval_system(df, "lmart_score", s, P_of) for s in ["val", "test"]}
    print(f"[lambdamart] best grid={grid}  test@300 micro="
          f"{result['systems']['lambdamart']['test'][str(PRIMARY_DEPTH)]['micro']:.4f}")

    # 泄漏哨兵：打乱标签重训
    pmodel, _, _ = train_lightgbm(df, permute=True)
    df = add_scores(df, pmodel, "perm_score")
    result["systems"]["lambdamart_permuted"] = {s: eval_system(df, "perm_score", s, P_of) for s in ["test"]}
    print(f"[leakage: permuted-label] test@300 micro="
          f"{result['systems']['lambdamart_permuted']['test'][str(PRIMARY_DEPTH)]['micro']:.4f}")

    # 配对统计 learned(lambdamart) vs rrf，@300，test
    cov_l = coverage_per_review(df, "lmart_score", "test", P_of, PRIMARY_DEPTH)
    cov_r = coverage_per_review(df, "rrf_score", "test", P_of, PRIMARY_DEPTH)
    aids = sorted(set(cov_l) & set(cov_r))
    diffs = [cov_l[a] - cov_r[a] for a in aids]
    ci = paired_bootstrap_ci(diffs, n_boot=10000)
    pt = paired_test(diffs)
    result["paired_lambdamart_vs_rrf_at300_test"] = {
        "n_reviews": len(aids), "mean_diff": ci.point_estimate,
        "ci_low": ci.ci_low, "ci_high": ci.ci_high, "ci_method": ci.method,
        "wilcoxon_p": pt.wilcoxon_p, "permutation_p": pt.permutation_p,
        "rank_biserial": pt.rank_biserial,
    }
    print(f"\n[learned vs rrf @300 test] Δmicro-per-review mean={ci.point_estimate:.4f} "
          f"CI[{ci.ci_low:.4f},{ci.ci_high:.4f}] wilcoxon_p={pt.wilcoxon_p:.3g}")

    with open(os.path.join(OUT_DIR, "s2b_eval.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n写入 {OUT_DIR}/s2b_eval.json")


if __name__ == "__main__":
    main()
