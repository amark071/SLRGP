#!/usr/bin/env python3
"""
用 CCF（中国计算机学会）推荐国际学术会议和期刊目录（2026第七版，295刊+387会议）
给 STEM 语料里还是 PENDING 的论文（主要是CS，但对eess/统计等交叉学科也有效）补分级。

思路：DOI关联的分级方案覆盖不到大量"没有DOI但确实发在顶会"的论文（CS会议论文
经常不回填DOI到arXiv元数据），但作者通常会在 arXiv 的 comments/journal-ref
自由文本字段里写"Accepted to NeurIPS 2026"这类信息。用CCF目录里387个会议+295个
期刊的简称，构造一个大的正则做词边界精确匹配（不是模糊匹配，避免误判）。

用法：
  python3 backfill_ccf_tier.py
"""
import glob
import json
import re

import pandas as pd

CCF_JSON = "data/common/tier_mapping/ccf_ranking.json"
POOL_DIR = "data/common/stem_pools"
META_DIR = "data/common/arxiv_metadata_snapshot"

RANK_TO_TIER = {"A": "T1", "B": "T2", "C": "T3"}


def load_ccf_regex():
    """按 tier 分别编译三个纯字面量交替的正则（不用命名组，避开Python re的组数上限），
    按 T1->T2->T3 优先级依次尝试，命中即返回，同名缩写以更高tier为准。"""
    d = json.load(open(CCF_JSON, encoding="utf-8"))
    by_tier = {"T1": set(), "T2": set(), "T3": set()}
    for kind in ("journals", "conferences"):
        for item in d.get(kind, []):
            tier = RANK_TO_TIER.get(item["rank"])
            abbr = (item.get("abbr") or "").strip()
            if tier and abbr and len(abbr) >= 2:
                by_tier[tier].add(abbr)
    # 去重跨tier冲突：同一缩写在多个tier出现时，只保留最高tier
    seen = set()
    regexes = {}
    for tier in ("T1", "T2", "T3"):
        abbrs = sorted(a for a in by_tier[tier] if a.lower() not in seen)
        seen |= {a.lower() for a in abbrs}
        if abbrs:
            pattern = r"\b(?:" + "|".join(re.escape(a) for a in abbrs) + r")\b"
            regexes[tier] = re.compile(pattern)
        print(f"[CCF] {tier}: {len(abbrs)} 个缩写")
    return regexes


def match_tier(regexes, text):
    if not text:
        return None
    for tier in ("T1", "T2", "T3"):
        if tier in regexes and regexes[tier].search(text):
            return tier
    return None


def load_metadata_text():
    files = sorted(glob.glob(f"{META_DIR}/train-*-of-*.parquet"))
    dfs = [pd.read_parquet(f, columns=["id", "comments", "journal-ref"]) for f in files]
    meta = pd.concat(dfs, ignore_index=True)
    meta = meta.drop_duplicates(subset="id", keep="last")
    meta["ccf_text"] = meta["comments"].fillna("") + " " + meta["journal-ref"].fillna("")
    return meta.set_index("id")["ccf_text"]


def main():
    regexes = load_ccf_regex()
    meta_text = load_metadata_text()
    print(f"[META] 元数据文本表: {len(meta_text)} 条")

    for f in sorted(glob.glob(f"{POOL_DIR}/*.parquet")):
        discipline = f.split("/")[-1].replace(".parquet", "")
        df = pd.read_parquet(f)
        pending_mask = df["tier"] == "PENDING"
        n_pending = pending_mask.sum()
        if n_pending == 0:
            print(f"[{discipline}] 无PENDING论文，跳过")
            continue

        texts = df.loc[pending_mask, "arxiv_id"].map(meta_text)
        matched_tier = texts.map(lambda t: match_tier(regexes, t) if isinstance(t, str) else None)

        n_matched = matched_tier.notna().sum()
        idx = df.index[pending_mask]
        df.loc[idx, "tier"] = matched_tier.where(matched_tier.notna(), df.loc[idx, "tier"])
        df.loc[idx[matched_tier.notna()], "tier_source"] = "ccf_catalog"

        df.to_parquet(f, index=False)
        print(f"[{discipline}] PENDING {n_pending} 篇，CCF目录命中 {n_matched} 篇 ({n_matched/n_pending:.1%})")

    print("\n全部完成")


if __name__ == "__main__":
    main()
