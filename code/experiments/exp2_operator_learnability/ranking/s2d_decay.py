#!/usr/bin/env python3
"""
S2d 排名-价值衰减剖面（注册的描述性副产物，单一合并诊断；进 SI，非 C2 端点）。

同一条衰减律的两个观测面：
  (a) L 入池序（RRF rank）上的真实引用命中密度 h_rrf(k) —— 等价于覆盖率随池深的曲线；
  (b) 学习排序器输出序上的命中密度 h_learned(k)（test 综述，冻结模型，宏平均）。
两者各拟合幂律 h=a·k^-b vs 指数 h=a·e^(-bk)，AIC 比较。
并给出成本敏感的运行区间族：幂律边际价值 h(k) 与成本比 c/V 相交处 k*，随 c/V 网格变化。

用法：S2D_FEAT_DIR=data/exp2_operator_learnability/ranking/features_s2d_tight S2D_OUT_DIR=work/exp2_ranking \
      python3 s2d_decay.py
"""
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

FEAT_DIR = os.environ.get("S2D_FEAT_DIR", "data/exp2_operator_learnability/ranking/features_s2d_tight")
OUT_DIR = os.environ.get("S2D_OUT_DIR", "work/exp2_operator_learnability/ranking_tight")
DEPTH = 300


def load(split):
    df = pd.read_parquet(os.path.join(FEAT_DIR, f"{split}.parquet"))
    g = np.load(os.path.join(FEAT_DIR, f"{split}_groups.npy"))
    return df, g


def hit_density(df, groups, score_col, depth=DEPTH):
    """每 rank 位的命中密度 h(k)=P(该 rank 的候选是真实引用)，宏平均(每 query 权重相等)。
    仅统计有 >=depth 或按实际长度补齐；用每 query 在 rank k 是否命中的 0/1 平均。"""
    hit_sum = np.zeros(depth)
    cnt = np.zeros(depth)
    idx = 0
    for gsz in groups:
        sub = df.iloc[idx:idx + gsz]; idx += gsz
        if sub["label"].sum() == 0:
            continue  # 无正例 query 不进密度（否则稀释为退化 0）
        order = np.argsort(-sub[score_col].values, kind="stable")
        labs = sub["label"].values[order]
        m = min(gsz, depth)
        hit_sum[:m] += labs[:m]
        cnt[:m] += 1.0
    h = np.divide(hit_sum, cnt, out=np.zeros_like(hit_sum), where=cnt > 0)
    return h, cnt


def coverage_curve(df, groups, score_col, depth=DEPTH):
    """C(d)=宏平均(每 query top-d 内正例数 / 该 query 池内正例总数)。"""
    C = np.zeros(depth); nq = 0
    idx = 0
    for gsz in groups:
        sub = df.iloc[idx:idx + gsz]; idx += gsz
        tot = sub["label"].sum()
        if tot == 0:
            continue
        order = np.argsort(-sub[score_col].values, kind="stable")
        labs = sub["label"].values[order]
        cum = np.cumsum(labs)
        m = min(gsz, depth)
        C[:m] += cum[:m] / tot
        if m < depth:
            C[m:] += cum[m - 1] / tot  # 补齐尾部
        nq += 1
    return C / max(nq, 1), nq


def fit_models(k, h):
    """在 h>0 的点上对 log h 做 LS：幂律(log h ~ log k) vs 指数(log h ~ k)。返回 AIC 与参数。"""
    mask = h > 0
    k = k[mask]; y = np.log(h[mask])
    n = len(y)

    def ls(x):
        A = np.vstack([np.ones_like(x), x]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        rss = float(resid @ resid)
        aic = n * np.log(rss / n) + 2 * 2  # 2 参数
        return coef, rss, aic

    c_pow, rss_pow, aic_pow = ls(np.log(k))     # log h = log a - b log k
    c_exp, rss_exp, aic_exp = ls(k)             # log h = log a - b k
    return {
        "n_points": n,
        "power_law": {"a": float(np.exp(c_pow[0])), "b": float(-c_pow[1]), "rss": rss_pow, "aic": aic_pow},
        "exponential": {"a": float(np.exp(c_exp[0])), "b": float(-c_exp[1]), "rss": rss_exp, "aic": aic_exp},
        "delta_aic_pow_minus_exp": aic_pow - aic_exp,
        "preferred": "power_law" if aic_pow < aic_exp else "exponential",
    }


def operating_region(fit_pow, cost_ratios):
    """幂律边际价值 h(k)=a·k^-b；k*=argmax{k: h(k)>=c/V}=(a/(c/V))^(1/b)。"""
    a, b = fit_pow["a"], fit_pow["b"]
    out = {}
    for cr in cost_ratios:
        if b > 0 and cr > 0:
            kstar = (a / cr) ** (1.0 / b)
            out[f"{cr:.4g}"] = float(np.clip(kstar, 1, 100000))
        else:
            out[f"{cr:.4g}"] = None
    return out


def main():
    with open(os.path.join(FEAT_DIR, "feature_manifest.json"), encoding="utf-8") as f:
        MAIN = json.load(f)["main_features"]
    tr, gtr = load("train"); va, gva = load("val"); te, gte = load("test")

    # locus (a): RRF/入池序，用全部综述（检索属性，无模型选择）
    all_df = pd.concat([tr, va, te], ignore_index=True)
    all_g = np.concatenate([gtr, gva, gte])
    h_rrf, cnt_rrf = hit_density(all_df, all_g, "rrf_score")
    cov_rrf, nq_a = coverage_curve(all_df, all_g, "rrf_score")

    # locus (b): 学习排序序，仅 test（冻结模型，无乐观偏差）
    booster = lgb.Booster(model_file=os.path.join(OUT_DIR, "lgb_main.txt"))
    te = te.copy()
    te["lgb_score"] = booster.predict(te[MAIN], num_iteration=booster.best_iteration)
    h_lgb, cnt_lgb = hit_density(te, gte, "lgb_score")
    cov_lgb, nq_b = coverage_curve(te, gte, "lgb_score")
    # 同 test 上的 RRF 参照（同口径对比）
    h_rrf_te, _ = hit_density(te, gte, "rrf_score")

    k = np.arange(1, DEPTH + 1)
    fit_a = fit_models(k, h_rrf)
    fit_b = fit_models(k, h_lgb)

    cost_ratios = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    region = operating_region(fit_a["power_law"], cost_ratios)

    def curve_sample(x):
        pts = [1, 5, 10, 20, 50, 100, 150, 200, 250, 300]
        return {str(p): float(x[p - 1]) for p in pts if p <= len(x)}

    out = {
        "depth": DEPTH,
        "locus_a_pool_order": {
            "n_reviews": int(nq_a), "hit_density_sample": curve_sample(h_rrf),
            "coverage_curve_sample": curve_sample(cov_rrf), "fit": fit_a},
        "locus_b_learned_order_test": {
            "n_reviews": int(nq_b), "hit_density_sample": curve_sample(h_lgb),
            "coverage_curve_sample": curve_sample(cov_lgb), "fit": fit_b,
            "rrf_on_test_hit_density_sample": curve_sample(h_rrf_te)},
        "operating_region_kstar_by_cost_ratio": region,
        "note": "描述性诊断；成本比 c/V=单候选边际成本/单真实引用价值，运行区间随部署成本移动，不宣称唯一最优。",
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "s2d_decay.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    # 落逐 rank 曲线 CSV 便于画图
    pd.DataFrame({"rank": k, "h_rrf_all": h_rrf, "cov_rrf_all": cov_rrf,
                  "h_learned_test": h_lgb, "cov_learned_test": cov_lgb,
                  "h_rrf_test": h_rrf_te}).to_csv(os.path.join(OUT_DIR, "s2d_decay_curves.csv"), index=False)

    print("=== locus (a) L入池序(全综述, n=%d) ===" % nq_a)
    print(f"  幂律 b={fit_a['power_law']['b']:.3f} AIC={fit_a['power_law']['aic']:.1f} | "
          f"指数 b={fit_a['exponential']['b']:.4f} AIC={fit_a['exponential']['aic']:.1f} | "
          f"ΔAIC(pow-exp)={fit_a['delta_aic_pow_minus_exp']:+.1f} → {fit_a['preferred']}")
    print(f"  命中密度 h(1)={h_rrf[0]:.3f} h(10)={h_rrf[9]:.3f} h(100)={h_rrf[99]:.4f} h(300)={h_rrf[299]:.4f}")
    print(f"  覆盖 C(50)={cov_rrf[49]:.3f} C(100)={cov_rrf[99]:.3f} C(300)={cov_rrf[299]:.3f}")
    print("=== locus (b) 学习排序序(test, n=%d) ===" % nq_b)
    print(f"  幂律 b={fit_b['power_law']['b']:.3f} AIC={fit_b['power_law']['aic']:.1f} | "
          f"指数 b={fit_b['exponential']['b']:.4f} AIC={fit_b['exponential']['aic']:.1f} | "
          f"ΔAIC(pow-exp)={fit_b['delta_aic_pow_minus_exp']:+.1f} → {fit_b['preferred']}")
    print(f"  命中密度 学习 h(1)={h_lgb[0]:.3f} h(10)={h_lgb[9]:.3f} | RRF@test h(1)={h_rrf_te[0]:.3f} h(10)={h_rrf_te[9]:.3f}")
    print("=== 运行区间 k*(成本比) ===")
    for cr, ks in region.items():
        print(f"  c/V={cr}: k*={ks:.1f}" if ks else f"  c/V={cr}: n/a")
    print(f"\n写入 {OUT_DIR}/s2d_decay.json + s2d_decay_curves.csv")


if __name__ == "__main__":
    main()
