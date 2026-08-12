#!/usr/bin/env python3
"""
从 arxiv-metadata-snapshot 全部10个分片，按 categories 字段的"主分类"（第一个列出的分类）
切分成11个目标 STEM 学科候选池。

关于 Tier（如实记录，不臆造）：
  arXiv 元数据的 journal-ref 是作者自由填写的文本（如 "Phys. Rev. D 79, 084013 (2009)"），
  doi 是可选字段；两者都不含结构化 ISSN，无法直接、可靠地 join 到 unified_tier.parquet
  （期刊缩写/全称映射本身是个需要独立处理的问题，容易出错）。
  本脚本只如实记录 has_doi / has_journal_ref，tier 一律标记 "PENDING"。
  真正的分级需要用 doi 批量查 OpenAlex（拿到 work 的 primary_location.source 再匹配 unified_tier），
  这是一个需要控制 API 配额与容错的独立步骤，本脚本不自动执行（由 merge_tier_mapping.py 完成）。

输出：data/stem_pools/<discipline>.parquet
  列：arxiv_id, title, abstract, authors, categories, primary_category,
      journal_ref, doi, has_doi, has_journal_ref, update_date, tier
"""
import glob

import pandas as pd

SRC_DIR = "data/common/arxiv_metadata_snapshot"
OUT_DIR = "data/common/stem_pools"

DISCIPLINE_PREFIXES = [
    ("computer_science", "cs"),
    ("mathematics", "math"),
    ("statistics", "stat"),
    ("condensed_matter", "cond-mat"),
    ("hep_theory", "hep-th"),
    ("astrophysics", "astro-ph"),
    ("quantum_physics", "quant-ph"),
    ("quantitative_biology", "q-bio"),
    ("quantitative_finance", "q-fin"),
    ("eess", "eess"),
    ("economics", "econ"),
]


def match_discipline(primary_cat):
    for name, prefix in DISCIPLINE_PREFIXES:
        if primary_cat == prefix or primary_cat.startswith(prefix + "."):
            return name
    return None


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(f"{SRC_DIR}/*.parquet"))
    print(f"读取 {len(files)} 个分片")

    buckets = {name: [] for name, _ in DISCIPLINE_PREFIXES}
    total, kept, skipped_short = 0, 0, 0

    for fi, f in enumerate(files, 1):
        df = pd.read_parquet(f, columns=["id", "title", "authors", "categories", "journal-ref", "doi", "abstract", "update_date"])
        total += len(df)
        for row in df.itertuples(index=False):
            abstract = row.abstract
            if not isinstance(abstract, str) or len(abstract.strip()) < 200:
                skipped_short += 1
                continue
            cats = (row.categories or "").split()
            if not cats:
                continue
            discipline = match_discipline(cats[0])
            if discipline is None:
                continue
            buckets[discipline].append({
                "arxiv_id": row.id,
                "title": row.title,
                "abstract": abstract.strip().replace("\n", " "),
                "authors": row.authors,
                "categories": row.categories,
                "primary_category": cats[0],
                "journal_ref": row._4,  # 'journal-ref' 字段(带连字符，itertuples重命名)
                "doi": row.doi,
                "has_doi": isinstance(row.doi, str) and len(row.doi) > 0,
                "has_journal_ref": isinstance(row._4, str) and len(row._4) > 0,
                "update_date": str(row.update_date),
                "tier": "PENDING",
            })
            kept += 1
        print(f"[{fi}/{len(files)}] 累计处理 {total} 行, 已入池 {kept}")

    print(f"\n总计: {total} 行, 摘要过短跳过 {skipped_short}, 入池 {kept}")
    for name, _ in DISCIPLINE_PREFIXES:
        rows = buckets[name]
        if not rows:
            print(f"  {name}: 0 篇 (跳过写出)")
            continue
        out_df = pd.DataFrame(rows).drop_duplicates(subset=["arxiv_id"])
        out_path = f"{OUT_DIR}/{name}.parquet"
        out_df.to_parquet(out_path, index=False)
        n_doi = out_df["has_doi"].sum()
        print(f"  {name}: {len(out_df)} 篇 (含doi可后续分级: {n_doi}, 占比 {n_doi/len(out_df):.1%}) -> {out_path}")


if __name__ == "__main__":
    main()
