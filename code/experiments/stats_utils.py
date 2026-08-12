"""
共享统计协议（对应论文 Methods「Statistical analysis」）的可复用实现。

覆盖：
  - 配对 BCa bootstrap 置信区间（paired_bootstrap_ci；n<8 时退化为 percentile 并注明）
  - 分层 percentile bootstrap 置信区间（hierarchical_bootstrap_ci：discipline -> topic，非 BCa）
  - 配对 Wilcoxon 符号秩检验 + 置换检验（paired_test：n<=20 精确枚举，n>20 Monte Carlo）
  - 效应量：rank-biserial（配对）、Cliff's delta（非配对）
  - Benjamini-Hochberg FDR 校正
  - "comparable" 判定规则（差值 CI 落在 ±阈值内）
  - Krippendorff's alpha（有序数据）
  - SHA-256 稳定哈希（配合 split_utils 使用）

不引入除 numpy/scipy 之外的依赖。
运行本文件的自检：`python3 stats_utils.py`（需要 numpy+scipy）。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import stats as _stats

RNG_SEED_DEFAULT = 42


@dataclass
class CIResult:
    point_estimate: float
    ci_low: float
    ci_high: float
    n_boot: int
    method: str = "BCa"

    def as_dict(self) -> dict:
        return {
            "point_estimate": self.point_estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_boot": self.n_boot,
            "method": self.method,
        }

    def __repr__(self) -> str:
        return f"{self.point_estimate:.4f} [95% CI {self.ci_low:.4f}, {self.ci_high:.4f}] (n_boot={self.n_boot}, {self.method})"


def _bca_interval(data: np.ndarray, stat_fn: Callable[[np.ndarray], float],
                   boot_stats: np.ndarray, alpha: float = 0.05) -> tuple:
    """BCa (bias-corrected and accelerated) 区间。data 为原始样本(1D)，boot_stats 为 bootstrap 统计量分布。"""
    theta_hat = stat_fn(data)
    n = len(data)
    # 偏差校正 z0
    prop_less = np.mean(boot_stats < theta_hat)
    prop_less = min(max(prop_less, 1e-6), 1 - 1e-6)
    z0 = _stats.norm.ppf(prop_less)
    # 加速度 a：jackknife
    jack_stats = np.empty(n)
    idx_full = np.arange(n)
    for i in range(n):
        jack_stats[i] = stat_fn(data[idx_full != i])
    jack_mean = jack_stats.mean()
    num = np.sum((jack_mean - jack_stats) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack_stats) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0
    z_alpha_lo = _stats.norm.ppf(alpha / 2)
    z_alpha_hi = _stats.norm.ppf(1 - alpha / 2)

    def _adj(z_alpha):
        za = z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))
        return _stats.norm.cdf(za)

    lo_pct = _adj(z_alpha_lo) * 100
    hi_pct = _adj(z_alpha_hi) * 100
    lo_pct = min(max(lo_pct, 0.001), 99.999)
    hi_pct = min(max(hi_pct, 0.001), 99.999)
    ci_lo = np.percentile(boot_stats, lo_pct)
    ci_hi = np.percentile(boot_stats, hi_pct)
    return ci_lo, ci_hi


def paired_bootstrap_ci(diffs: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05,
                         seed: int = RNG_SEED_DEFAULT, stat: str = "mean") -> CIResult:
    """配对差值的 BCa bootstrap 置信区间。diffs = 每个 topic/unit 的 (system_A - system_B) 差值。

    §0.2: "Paired mean differences with BCa bootstrap confidence intervals, 10,000 resamples,
    resampling at the unit level."
    """
    data = np.asarray(diffs, dtype=float)
    data = data[~np.isnan(data)]
    n = len(data)
    if n < 2:
        raise ValueError(f"paired_bootstrap_ci 需要至少 2 个有效配对样本，得到 {n} 个")
    stat_fn = {"mean": np.mean, "median": np.median}[stat]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_stats = stat_fn(data[idx], axis=1)
    point = float(stat_fn(data))
    if stat == "mean" and n >= 8:
        ci_lo, ci_hi = _bca_interval(data, stat_fn, boot_stats, alpha)
    else:
        # n 太小时 jackknife 加速度估计不稳定，退化为 percentile bootstrap 并在结果中注明
        ci_lo, ci_hi = np.percentile(boot_stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return CIResult(point, float(ci_lo), float(ci_hi), n_boot, method="BCa" if (stat == "mean" and n >= 8) else "percentile")


def hierarchical_bootstrap_ci(nested: dict, n_boot: int = 10_000, alpha: float = 0.05,
                               seed: int = RNG_SEED_DEFAULT) -> CIResult:
    """分层 percentile bootstrap CI（如 S5b：discipline -> topic -> diff）。

    nested: {discipline_id: [diff_1, diff_2, ...]}（每个 discipline 下若干配对差值）。
    重抽样两层：先按 discipline 有放回抽样，再在被抽中的 discipline 内部有放回抽样其 topic 差值。

    本函数为分层 percentile bootstrap,不做 BCa 的 jackknife 加速度与偏置校正
    (分层重抽样下加速度项无稳定估计);需要 BCa 时使用 paired_bootstrap_ci。
    """
    disciplines = list(nested.keys())
    n_disc = len(disciplines)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        chosen_disc = rng.integers(0, n_disc, size=n_disc)
        pooled = []
        for di in chosen_disc:
            vals = nested[disciplines[di]]
            vals = np.asarray(vals, dtype=float)
            resample = rng.choice(vals, size=len(vals), replace=True)
            pooled.append(resample)
        boot_means[b] = np.concatenate(pooled).mean()
    all_vals = np.concatenate([np.asarray(v, dtype=float) for v in nested.values()])
    point = float(all_vals.mean())
    ci_lo, ci_hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return CIResult(point, float(ci_lo), float(ci_hi), n_boot, method="hierarchical-percentile")


@dataclass
class PairedTestResult:
    n: int
    wilcoxon_stat: float
    wilcoxon_p: float
    permutation_p: float
    rank_biserial: float
    n_permutations: int

    def __repr__(self) -> str:
        return (f"n={self.n} Wilcoxon W={self.wilcoxon_stat:.2f} p={self.wilcoxon_p:.4g} "
                f"perm_p={self.permutation_p:.4g} r_rb={self.rank_biserial:.3f}")


def paired_test(diffs: Sequence[float], n_permutations: int = 10_000, seed: int = RNG_SEED_DEFAULT) -> PairedTestResult:
    """配对 Wilcoxon 符号秩检验 + 置换检验 + rank-biserial 效应量。

    §0.2: "Wilcoxon signed-rank for paired ordinal/judge scores; exact permutation test
    (>=10,000 permutations) for small n; two-sided throughout."

    置换检验实现口径：n<=20 时枚举全部 2^n 个符号翻转组合（精确置换检验，
    exact permutation test）；n>20 时抽取 n_permutations 次符号翻转做 Monte Carlo
    近似（默认 10,000 次，种子固定）。返回值的 n_permutations 字段记录实际使用的
    置换次数。
    """
    data = np.asarray(diffs, dtype=float)
    data = data[data != 0]  # Wilcoxon 惯例：丢弃零差值
    n = len(data)
    if n == 0:
        raise ValueError("全部差值为零，无法检验")
    try:
        w_stat, w_p = _stats.wilcoxon(data, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    observed = np.sum(data)
    rng = np.random.default_rng(seed)
    if n <= 20:
        # n<=20 时枚举全部 2^n 个符号翻转组合做精确置换检验
        signs_grid = np.array(list(itertools.product([1, -1], repeat=n)))
        perm_sums = (signs_grid * data).sum(axis=1)
        n_perm_used = len(perm_sums)
    else:
        signs = rng.choice([1, -1], size=(n_permutations, n))
        perm_sums = (signs * data).sum(axis=1)
        n_perm_used = n_permutations
    perm_p = float(np.mean(np.abs(perm_sums) >= abs(observed) - 1e-12))

    n_pos = np.sum(data > 0)
    n_neg = np.sum(data < 0)
    r_rb = (n_pos - n_neg) / n

    return PairedTestResult(n=n, wilcoxon_stat=float(w_stat), wilcoxon_p=float(w_p),
                             permutation_p=perm_p, rank_biserial=float(r_rb), n_permutations=n_perm_used)


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """非配对效应量：Cliff's delta ∈ [-1, 1]。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    gt = sum(1 for xi in x for yi in y if xi > yi)
    lt = sum(1 for xi in x for yi in y if xi < yi)
    return (gt - lt) / (len(x) * len(y))


def bh_fdr(p_values: Sequence[float], q: float = 0.05) -> dict:
    """Benjamini-Hochberg FDR 校正。

    返回 {"adjusted_p": [...], "reject": [bool...], "q": q}，顺序与输入一致。
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    # 单调化（从大到小取累计最小值），是 BH 标准做法
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    adjusted_p = np.empty(n)
    adjusted_p[order] = adj
    reject = adjusted_p <= q
    return {"adjusted_p": adjusted_p.tolist(), "reject": reject.tolist(), "q": q}


def comparable(ci: CIResult, threshold: float = 0.25) -> bool:
    """§0.2 等价性判定：差值 95% CI 完全落在 [-threshold, threshold] 内才算 'comparable'。

    threshold 默认 0.25（1-5 判卷量表），比例型指标建议传 0.02。
    """
    return (-threshold) <= ci.ci_low and ci.ci_high <= threshold


def krippendorff_alpha_ordinal(reliability_data: Sequence[Sequence[Optional[float]]]) -> float:
    """有序数据的 Krippendorff's alpha。

    reliability_data: 形状 (n_raters, n_units)，缺失记 None/np.nan。
    使用有序度量距离 d(v1, v2) = (rank(v2) - rank(v1))^2的秩差平方（标准 ordinal metric，
    按可能取值的秩位计算，而非原始数值差），适用于 1-5 整数评分等有序量表。
    """
    data = np.array(reliability_data, dtype=float)
    values_present = sorted(set(data[~np.isnan(data)].tolist()))
    if len(values_present) < 2:
        return float("nan")
    rank_of = {v: r for r, v in enumerate(values_present)}
    n_v = len(values_present)
    # 秩边界(cumulative)，标准 ordinal metric 定义
    counts_by_value = {v: 0 for v in values_present}
    for v in data[~np.isnan(data)]:
        counts_by_value[v] += 1
    total_n = sum(counts_by_value.values())
    cum = {}
    running = 0.0
    for v in values_present:
        cum[v] = running + counts_by_value[v] / 2.0
        running += counts_by_value[v]

    def ordinal_metric(v1, v2):
        if v1 == v2:
            return 0.0
        lo, hi = (v1, v2) if v1 < v2 else (v2, v1)
        # sum of counts strictly between lo and hi, plus half of endpoints
        s = 0.0
        for v in values_present:
            if lo <= v <= hi:
                s += counts_by_value[v]
        s -= (counts_by_value[lo] + counts_by_value[hi]) / 2.0
        return s ** 2

    n_units = data.shape[1]
    # observed disagreement
    do_num, do_den = 0.0, 0.0
    for u in range(n_units):
        col = data[:, u]
        col = col[~np.isnan(col)]
        m_u = len(col)
        if m_u < 2:
            continue
        for i in range(m_u):
            for j in range(m_u):
                if i == j:
                    continue
                do_num += ordinal_metric(col[i], col[j])
        do_den += m_u * (m_u - 1)
    if do_den == 0:
        return float("nan")
    d_o = do_num / do_den

    # expected disagreement
    all_vals = data[~np.isnan(data)]
    de_num = 0.0
    for i in range(len(all_vals)):
        for j in range(len(all_vals)):
            if i == j:
                continue
            de_num += ordinal_metric(all_vals[i], all_vals[j])
    de_den = total_n * (total_n - 1)
    d_e = de_num / de_den if de_den else float("nan")

    if d_e == 0:
        return float("nan")
    return 1 - d_o / d_e


def self_check():
    rng = np.random.default_rng(0)
    diffs = rng.normal(loc=0.4, scale=1.0, size=15)
    ci = paired_bootstrap_ci(diffs)
    print("paired_bootstrap_ci:", ci)
    pt = paired_test(diffs)
    print("paired_test:", pt)
    print("comparable(threshold=0.25):", comparable(ci, 0.25))
    ps = [0.001, 0.02, 0.03, 0.04, 0.2, 0.5, 0.8]
    print("bh_fdr:", bh_fdr(ps))
    # Krippendorff alpha 自检：3 名裁判对 6 篇文档打 1-5 分，高一致 vs 低一致
    high_agree = [[3, 4, 2, 5, 1, 4], [3, 4, 2, 5, 1, 3], [4, 4, 2, 5, 2, 4]]
    low_agree = [[3, 4, 2, 5, 1, 4], [1, 2, 5, 1, 4, 2], [5, 1, 3, 2, 5, 1]]
    print("krippendorff alpha (high agreement expect >0.6):", krippendorff_alpha_ordinal(high_agree))
    print("krippendorff alpha (low agreement expect ~0):", krippendorff_alpha_ordinal(low_agree))


if __name__ == "__main__":
    self_check()
