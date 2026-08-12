#!/usr/bin/env python3
"""
用 OpenAlex 全量 S3 Parquet 快照（DuckDB httpfs 远程直查，无需下载 300GB+ 到本地）
给 11 个 STEM 学科候选池回填国际化 tier。

思路（比社科语料的ISSN/期刊名模糊匹配更精确）：
  1. 从候选池抽取全部去重 doi（948361 个，去掉 URL 前缀、小写化）
  2. 分批（每批 BATCH_FILES 个 parquet 分片）远程查询 OpenAlex works 表，
     按 doi 关联，只投影需要的叶子列（doi, primary_location.source.id/issn_l/issn/display_name/is_core,
     cited_by_count）——Parquet 嵌套结构在物理存储上是列式打平的，只读用到的叶子列，不会把整个 works
     记录都拉下来。
  3. 断点续传：已处理的分片下标记录在 checkpoint 文件里，重跑自动跳过。
  4. 用 openalex_sources_raw.jsonl 构建"OpenAlex source_id -> tier"精确映射（按 source id 直接查，
     不再需要 ISSN/期刊名模糊匹配）：source 的 ISSN 命中 SJR -> 用 SJR tier；否则 is_core+h_index 兜底。
  5. 把 tier 写回 11 个学科 parquet：有 doi 且在 OpenAlex 查到来源 -> 真实 tier；
     有 doi 但查不到 / 没有 doi -> 保持 "PENDING"（如实记录，不臆造）。

用法：
  python3 backfill_stem_tier.py            # 跑完整流程（可安全重复执行，自动跳过已完成部分）
"""
import glob
import json
import os
import re

import duckdb
import pandas as pd

POOL_DIR = "data/common/stem_pools"
WORK_DIR = "work/corpus_rebuild/tier_backfill"
TIER_DIR = "data/common/tier_mapping"
BATCH_FILES = 20
CHECKPOINT_PATH = f"{WORK_DIR}/checkpoint.json"
RESULT_PARQUET = f"{WORK_DIR}/doi_source_lookup.parquet"


def normalize_doi(doi):
    if not isinstance(doi, str) or not doi:
        return None
    d = doi.lower().strip()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
    return d or None


def stage_a_build_target_dois():
    out_path = f"{WORK_DIR}/target_dois.parquet"
    if os.path.exists(out_path):
        n = duckdb.sql(f"SELECT count(*) FROM read_parquet('{out_path}')").fetchone()[0]
        print(f"[A] target_dois 已存在: {n} 条，跳过")
        return out_path
    rows = set()
    for f in glob.glob(f"{POOL_DIR}/*.parquet"):
        df = pd.read_parquet(f, columns=["doi"])
        for d in df["doi"].dropna():
            nd = normalize_doi(d)
            if nd:
                rows.add(nd)
    out = pd.DataFrame({"doi_norm": sorted(rows)})
    out.to_parquet(out_path, index=False)
    print(f"[A] 目标doi集合: {len(out)} 条 -> {out_path}")
    return out_path


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        return json.load(open(CHECKPOINT_PATH))
    return {"done_batches": []}


def save_checkpoint(cp):
    json.dump(cp, open(CHECKPOINT_PATH, "w"))


def stage_b_query_openalex(target_doi_path):
    if os.path.exists(RESULT_PARQUET):
        n = duckdb.sql(f"SELECT count(*) FROM read_parquet('{RESULT_PARQUET}/*.parquet')").fetchone()[0] \
            if os.path.isdir(RESULT_PARQUET) else None
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")

    files = con.execute(
        "SELECT file FROM glob('s3://openalex/data/parquet/works/*/*.parquet') ORDER BY file"
    ).fetchall()
    files = [f[0] for f in files]
    print(f"[B] OpenAlex works 分片总数: {len(files)}")

    os.makedirs(f"{WORK_DIR}/batches", exist_ok=True)
    cp = load_checkpoint()
    done = set(cp["done_batches"])

    con.execute(f"CREATE TABLE target_dois AS SELECT * FROM read_parquet('{target_doi_path}')")

    n_batches = (len(files) + BATCH_FILES - 1) // BATCH_FILES
    for bi in range(n_batches):
        if bi in done:
            continue
        batch_files = files[bi * BATCH_FILES:(bi + 1) * BATCH_FILES]
        file_list_sql = ", ".join(f"'{f}'" for f in batch_files)
        out_path = f"{WORK_DIR}/batches/batch_{bi:04d}.parquet"
        query = f"""
        COPY (
            SELECT
                lower(regexp_replace(w.doi, '^https?://(dx\\.)?doi\\.org/', '')) AS doi_norm,
                w.primary_location.source.id AS source_id,
                w.primary_location.source.issn_l AS issn_l,
                w.primary_location.source.issn AS issn_list,
                w.primary_location.source.display_name AS source_display_name,
                w.primary_location.source.is_core AS is_core,
                w.cited_by_count AS cited_by_count
            FROM read_parquet([{file_list_sql}]) AS w
            INNER JOIN target_dois t
              ON lower(regexp_replace(w.doi, '^https?://(dx\\.)?doi\\.org/', '')) = t.doi_norm
            WHERE w.doi IS NOT NULL
        ) TO '{out_path}' (FORMAT PARQUET)
        """
        con.execute(query)
        done.add(bi)
        cp["done_batches"] = sorted(done)
        save_checkpoint(cp)
        if bi % 10 == 0 or bi == n_batches - 1:
            print(f"[B] 进度: {len(done)}/{n_batches} 批 ({len(done)*BATCH_FILES}/{len(files)} 分片)")

    print("[B] 全部分片处理完成，合并结果...")
    con.execute(f"""
        COPY (SELECT DISTINCT * FROM read_parquet('{WORK_DIR}/batches/*.parquet'))
        TO '{RESULT_PARQUET}' (FORMAT PARQUET)
    """)
    n = con.execute(f"SELECT count(*) FROM read_parquet('{RESULT_PARQUET}')").fetchone()[0]
    print(f"[B] 合并后命中记录数: {n} -> {RESULT_PARQUET}")


def normalize_title(t):
    if not isinstance(t, str):
        return ""
    t = t.lower().strip()
    t = t.replace("&", " and ")
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def build_source_id_tier_map():
    sjr_issn = pd.read_parquet(f"{TIER_DIR}/sjr_issn_tier.parquet").set_index("issn")["tier"].to_dict()

    def fallback_tier(is_core, h_index):
        if not is_core:
            return None
        h_index = h_index or 0
        if h_index >= 50:
            return "T1"
        if h_index >= 15:
            return "T2"
        return "T3"

    source_tier = {}
    with open(f"{TIER_DIR}/openalex_sources_raw.jsonl", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = r.get("id")
            if not sid:
                continue
            issn_l = r.get("issn_l")
            issns = r.get("issn") or ([] if not issn_l else [issn_l])
            tier, source = None, None
            for issn in issns:
                if issn in sjr_issn:
                    tier, source = sjr_issn[issn], "sjr"
                    break
            if tier is None:
                summary = r.get("summary_stats") or {}
                ft = fallback_tier(bool(r.get("is_core")), summary.get("h_index"))
                if ft:
                    tier, source = ft, "openalex_core"
            if tier:
                source_tier[sid] = (tier, source)
    print(f"[C] source_id -> tier 映射: {len(source_tier)} 条")
    return source_tier


def stage_d_write_back(source_tier):
    lookup = pd.read_parquet(RESULT_PARQUET)
    lookup["tier"] = lookup["source_id"].map(lambda x: source_tier.get(x, (None, None))[0])
    lookup["tier_source"] = lookup["source_id"].map(lambda x: source_tier.get(x, (None, None))[1])
    doi_tier = lookup.dropna(subset=["tier"]).drop_duplicates(subset=["doi_norm"]).set_index("doi_norm")

    for f in sorted(glob.glob(f"{POOL_DIR}/*.parquet")):
        discipline = os.path.basename(f).replace(".parquet", "")
        df = pd.read_parquet(f)
        df["doi_norm"] = df["doi"].map(normalize_doi)
        matched_tier = df["doi_norm"].map(doi_tier["tier"].to_dict())
        matched_source = df["doi_norm"].map(doi_tier["tier_source"].to_dict())
        matched_cited = df["doi_norm"].map(
            lookup.drop_duplicates(subset=["doi_norm"]).set_index("doi_norm")["cited_by_count"].to_dict()
        )
        df["tier"] = matched_tier.fillna("PENDING")
        df["tier_source"] = matched_source
        df["cited_by_count"] = matched_cited
        df = df.drop(columns=["doi_norm"])
        df.to_parquet(f, index=False)
        n_matched = matched_tier.notna().sum()
        print(f"[D] {discipline}: {len(df)} 篇，{n_matched} 篇成功回填tier ({n_matched/len(df):.1%})")


def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    target_doi_path = stage_a_build_target_dois()
    stage_b_query_openalex(target_doi_path)
    source_tier = build_source_id_tier_map()
    stage_d_write_back(source_tier)
    print("\n全部完成")


if __name__ == "__main__":
    main()
