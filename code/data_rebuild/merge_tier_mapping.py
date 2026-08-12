#!/usr/bin/env python3
"""
合并 Scimago SJR 分级 + OpenAlex is_core/summary_stats 兜底信号，产出统一的
国际化期刊分级表 unified_tier.parquet（issn -> tier）与 unified_title_tier.parquet（期刊名 -> tier）。

优先级：
  1. SJR quartile 命中 -> 直接用（Q1->T1, Q2->T2, Q3/Q4->T3）
  2. SJR 未命中，但 OpenAlex is_core=True -> 按 h_index 分档：
       h_index >= 50 -> T1；h_index >= 15 -> T2；否则 T3
     （is_core 已经是 CWTS 认定的核心来源，即使 h_index 不高也不该判 T3 以下，故下限为 T3）
  3. 两者都未命中 -> tier = "UNKNOWN"（不臆造，留给人工兜底名单或后续抓取补全）

输入：
  data/tier_mapping/sjr_issn_tier.parquet
  data/tier_mapping/sjr_title_tier.parquet
  data/tier_mapping/openalex_sources_raw.jsonl

输出：
  data/tier_mapping/unified_tier.parquet       (issn, tier, tier_source, sjr_quartile, is_core, h_index, title)
  data/tier_mapping/unified_title_tier.parquet (title_norm, tier, tier_source, title)
"""
import json
import re

import pandas as pd

BASE = "data/common/tier_mapping"


def normalize_title(t):
    if not isinstance(t, str):
        return ""
    t = t.lower().strip()
    t = t.replace("&", " and ")
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def openalex_fallback_tier(is_core, h_index):
    if not is_core:
        return None
    h_index = h_index or 0
    if h_index >= 50:
        return "T1"
    if h_index >= 15:
        return "T2"
    return "T3"


def load_openalex_sources():
    rows = []
    with open(f"{BASE}/openalex_sources_raw.jsonl", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            issn_l = r.get("issn_l")
            issns = r.get("issn") or ([] if not issn_l else [issn_l])
            summary = r.get("summary_stats") or {}
            rows.append({
                "issns": issns,
                "title": r.get("display_name", ""),
                "is_core": bool(r.get("is_core")),
                "h_index": summary.get("h_index"),
                "works_count": r.get("works_count"),
            })
    return rows


def main():
    sjr_issn = pd.read_parquet(f"{BASE}/sjr_issn_tier.parquet").set_index("issn")
    sjr_title = pd.read_parquet(f"{BASE}/sjr_title_tier.parquet").set_index("title_norm")
    print(f"SJR ISSN映射: {len(sjr_issn)} 条；SJR 标准化期刊名映射: {len(sjr_title)} 条")

    oa_sources = load_openalex_sources()
    print(f"OpenAlex sources: {len(oa_sources)} 条")

    issn_rows = {}
    title_rows = {}

    # 先铺 SJR（优先级最高）
    for issn, r in sjr_issn.iterrows():
        issn_rows[issn] = {
            "issn": issn, "tier": r["tier"], "tier_source": "sjr",
            "sjr_quartile": r["sjr_quartile"], "is_core": None, "h_index": r["h_index"],
            "title": r["title"],
        }
    for title_norm, r in sjr_title.iterrows():
        title_rows[title_norm] = {
            "title_norm": title_norm, "tier": r["tier"], "tier_source": "sjr",
            "sjr_quartile": r["sjr_quartile"], "is_core": None, "h_index": r["h_index"],
            "title": r["title"],
        }

    # OpenAlex 兜底：只补 SJR 没覆盖到的 ISSN/期刊名
    oa_added_issn, oa_added_title = 0, 0
    for r in oa_sources:
        fallback_tier = openalex_fallback_tier(r["is_core"], r["h_index"])
        for issn in r["issns"]:
            issn = issn.upper().strip()
            if issn in issn_rows:
                continue
            if fallback_tier:
                issn_rows[issn] = {
                    "issn": issn, "tier": fallback_tier, "tier_source": "openalex_core",
                    "sjr_quartile": None, "is_core": r["is_core"], "h_index": r["h_index"],
                    "title": r["title"],
                }
                oa_added_issn += 1
        title_norm = normalize_title(r["title"])
        if title_norm and title_norm not in title_rows and fallback_tier:
            title_rows[title_norm] = {
                "title_norm": title_norm, "tier": fallback_tier, "tier_source": "openalex_core",
                "sjr_quartile": None, "is_core": r["is_core"], "h_index": r["h_index"],
                "title": r["title"],
            }
            oa_added_title += 1

    print(f"OpenAlex 兜底新增: ISSN {oa_added_issn} 条, 期刊名 {oa_added_title} 条")

    issn_df = pd.DataFrame(list(issn_rows.values()))
    title_df = pd.DataFrame(list(title_rows.values()))
    issn_df.to_parquet(f"{BASE}/unified_tier.parquet", index=False)
    title_df.to_parquet(f"{BASE}/unified_title_tier.parquet", index=False)

    print(f"\n写出 unified_tier.parquet: {len(issn_df)} 条 ISSN 映射")
    print(f"写出 unified_title_tier.parquet: {len(title_df)} 条标准化期刊名映射")
    print("\n--- Tier 分布（ISSN 口径） ---")
    print(issn_df["tier"].value_counts())
    print("\n--- Tier 来源分布（ISSN 口径） ---")
    print(issn_df["tier_source"].value_counts())


if __name__ == "__main__":
    main()
