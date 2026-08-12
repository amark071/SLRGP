#!/usr/bin/env python3
"""
S2d 链接审计——总体分层加权精度（离线统计总体 cell 规模，无 LLM）。

审计样本对硬风险层过采样（short_generic_title 在总体仅 ~1.8%，样本占 ~21%）。
整体样本精度是"硬层加权"的悲观估计，不是总体链接精度。此脚本按总体分层权重
做分层（Horvitz–Thompson）估计：把每个正例按 strata 的 frozenset 归入不相交 cell，
统计总体各 cell 规模 N_cell，输出 cell_pop.json；本地再与 audit_verdicts_tight.json
合并得总体加权精度 + 分层方差 CI。

用法：python3 s2d_audit_popweight.py
"""
import glob
import json
import os
import re
import sqlite3
from collections import Counter

RESOLVED_DIR = "data/exp2_operator_learnability/ranking/resolved"
TIGHT_DIR = "data/exp2_operator_learnability/ranking/resolved_tight"
CORPUS_DB = "data/common/unified_corpus/unified_corpus.db"
OUT = "work/exp2_operator_learnability/ranking/cell_pop_tight.json"
MIN_POS = 5


def norm_title(t):
    t = (t or "").lower().strip().replace("&", " and ")
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def main():
    conn = sqlite3.connect(CORPUS_DB)
    tight = {}
    for jf in glob.glob(os.path.join(TIGHT_DIR, "*", "*.json")):
        r = json.load(open(jf, encoding="utf-8"))
        tight[r["arxiv_id"]] = set(r["matched_doc_ids"])

    pos = []  # (arxiv_id, res_src, doc_id)
    for jf in sorted(glob.glob(os.path.join(RESOLVED_DIR, "*", "*.json"))):
        r = json.load(open(jf, encoding="utf-8"))
        if r.get("n_matched_corpus", 0) < MIN_POS:
            continue
        ids = sorted(tight.get(r["arxiv_id"], set()))
        if len(ids) < MIN_POS:
            continue
        for d in ids:
            pos.append((r["arxiv_id"], r.get("resolution_source"), d))
    print(f"总体正例数: {len(pos)}")

    ids = sorted(set(d for _, _, d in pos))
    meta = {}
    for i in range(0, len(ids), 800):
        chunk = ids[i:i + 800]
        ph = ",".join("?" for _ in chunk)
        for row in conn.execute(
            f"SELECT doc_id,title,doi,source FROM papers WHERE doc_id IN ({ph})", chunk):
            meta[row[0]] = {"title": row[1], "doi": row[2], "source": row[3]}

    def strata_of(res_src, d):
        m = meta.get(d, {})
        nt = norm_title(m.get("title"))
        s = []
        if res_src == "bbl":
            s.append("low_reliability_bbl")
        if not m.get("doi"):
            s.append("no_doi")
        src = (m.get("source") or "").lower()
        if src.startswith("arxiv") or "arxiv" in src:
            s.append("preprint_source")
        if nt and len(nt.split()) <= 3:
            s.append("short_generic_title")
        return tuple(sorted(s)) or ("clean",)

    cell = Counter()
    for arxiv_id, res_src, d in pos:
        cell[strata_of(res_src, d)] += 1

    out = {"n_total": len(pos), "cells": {"|".join(k): v for k, v in cell.items()}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"不相交 cell 数: {len(cell)}")
    for k, v in sorted(cell.items(), key=lambda x: -x[1]):
        print(f"  {'|'.join(k):45s} {v:6d}  ({v/len(pos)*100:.2f}%)")
    print(f"\n写入 {OUT}")


if __name__ == "__main__":
    main()
