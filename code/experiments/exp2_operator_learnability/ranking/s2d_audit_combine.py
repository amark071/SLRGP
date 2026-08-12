#!/usr/bin/env python3
"""
合并样本裁决 + 总体 cell 权重 → 总体分层加权链接精度 + CI（本地）。

- 过采样样本的整体精度是硬层加权的悲观数；
- 总体估计量 = Σ_c (N_c/N) p_c（不相交 cell），方差按分层估计（cell 内二项 + 忽略 FPC，保守）。
- 同时给出：整体样本精度（对照）、分层点估计、正态 CI、及分层 bootstrap CI。

用法：python3 s2d_audit_combine.py  （需 work/audit_verdicts_tight.json，cell_pop 内联/或改路径）
"""
import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
VERD = os.path.join(HERE, "work", "audit_verdicts_tight.json")
# 总体不相交 cell 规模（s2d_audit_popweight.py 输出）
CELL_POP = {
    ("no_doi",): 34685,
    ("clean",): 31563,
    ("short_generic_title",): 645,
    ("no_doi", "short_generic_title"): 601,
}
N_TOTAL = sum(CELL_POP.values())


def main():
    data = json.load(open(VERD, encoding="utf-8"))
    res = data["verdicts"]

    # 样本按不相交 cell 聚合
    from collections import defaultdict
    same = defaultdict(int)
    dec = defaultdict(int)
    for r in res:
        cell = tuple(sorted(r.get("strata", ["clean"])))
        if r["verdict"] == "SAME":
            same[cell] += 1; dec[cell] += 1
        elif r["verdict"] == "DIFFERENT":
            dec[cell] += 1
        # UNCERTAIN/ERROR 不计入分母（与主报告一致）

    # 把样本 cell 归并到总体 cell 键（样本可能出现总体没有的组合 → 按包含 short 与否+no_doi 与否归并）
    def canon(cell):
        s = set(cell)
        short = "short_generic_title" in s
        nodoi = "no_doi" in s
        if short and nodoi:
            return ("no_doi", "short_generic_title")
        if short:
            return ("short_generic_title",)
        if nodoi:
            return ("no_doi",)
        return ("clean",)

    csame = defaultdict(int); cdec = defaultdict(int)
    for cell in dec:
        k = canon(cell)
        csame[k] += same[cell]; cdec[k] += dec[cell]

    print("=== 分层（总体不相交 cell）===")
    print(f"{'cell':40s} {'N_pop':>7s} {'w':>6s} {'n_smp':>6s} {'same':>5s} {'p_c':>6s}")
    P = 0.0
    var = 0.0
    rows = []
    for k, Nc in CELL_POP.items():
        w = Nc / N_TOTAL
        n = cdec.get(k, 0); s = csame.get(k, 0)
        if n == 0:
            # 无样本 cell：用整体样本精度兜底（保守）
            p = sum(csame.values()) / max(1, sum(cdec.values()))
            v = p * (1 - p) / 1  # 极不确定
        else:
            p = s / n
            v = p * (1 - p) / n
        P += w * p
        var += (w ** 2) * v
        rows.append((k, Nc, w, n, s, p))
        print(f"{'|'.join(k):40s} {Nc:7d} {w:6.3f} {n:6d} {s:5d} {p:6.3f}")

    se = math.sqrt(var)
    lo, hi = P - 1.96 * se, P + 1.96 * se
    lo90 = P - 1.645 * se

    # 分层 bootstrap（cell 内重抽 + 固定权重）
    rng = random.Random(20260709)
    fallback = sum(csame.values()) / max(1, sum(cdec.values()))
    boots = []
    for _ in range(20000):
        pb = 0.0
        for k, Nc in CELL_POP.items():
            w = Nc / N_TOTAL
            n = cdec.get(k, 0); s = csame.get(k, 0)
            if n == 0:
                pb += w * fallback
            else:
                p_c = s / n
                draws = sum(1 for _ in range(n) if rng.random() < p_c)
                pb += w * (draws / n)
        boots.append(pb)
    boots.sort()
    blo = boots[int(0.025 * len(boots))]
    bhi = boots[int(0.975 * len(boots))]
    blo90 = boots[int(0.05 * len(boots))]

    # 对照：整体过采样样本精度（Wilson，两置信水平）
    tot_s = sum(csame.values()); tot_n = sum(cdec.values())
    op = tot_s / tot_n

    def wilson_lo(k, n, z):
        p = k / n
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return center - half

    ow95 = wilson_lo(tot_s, tot_n, 1.96)
    ow90 = wilson_lo(tot_s, tot_n, 1.645)

    print("\n=== 结果 ===")
    print(f"[过采样样本-悲观] 点={op:.4f} n={tot_n}  Wilson下界: 95%CI={ow95:.4f}  90%CI={ow90:.4f}")
    print(f"[总体分层加权-正确估计量] 点={P:.4f}")
    print(f"  正态下界:    95%CI={lo:.4f}  90%CI={lo90:.4f}")
    print(f"  bootstrap下界: 95%CI={blo:.4f}  90%CI={blo90:.4f}")

    print("\n=== 门槛 2×2（估计量 × 精度目标 τ，用 95%CI 下界判定）===")
    for name, lb in [("过采样样本 95%CI", ow95), ("过采样样本 90%CI", ow90),
                     ("总体分层 95%CI(bootstrap)", blo), ("总体分层 90%CI(bootstrap)", blo90)]:
        print(f"  {name:30s} 下界={lb:.4f}  vs τ=0.95→{'PASS' if lb>=0.95 else 'FAIL'}   vs τ=0.90→{'PASS' if lb>=0.90 else 'FAIL'}")

    out = {
        "oversampled_overall": {"precision": op, "wilson_lo_95": ow95, "wilson_lo_90": ow90, "n": tot_n},
        "population_stratified": {"point": P, "normal95": [lo, hi], "normal_lo_90": lo90,
                                  "bootstrap95": [blo, bhi], "bootstrap_lo_90": blo90},
        "cells": [{"cell": "|".join(k), "N_pop": Nc, "weight": Nc / N_TOTAL,
                   "n_sample": cdec.get(k, 0), "same": csame.get(k, 0),
                   "p": (csame.get(k, 0) / cdec[k] if cdec.get(k) else None)} for k, Nc in CELL_POP.items()],
        "gate_pass_population_tau95_ci95": bool(blo >= 0.95),
        "gate_pass_population_tau90_ci95": bool(blo >= 0.90),
    }
    json.dump(out, open(os.path.join(HERE, "work", "audit_popweight_result.json"), "w"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
