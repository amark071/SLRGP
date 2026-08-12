#!/usr/bin/env python3
"""
在 CCF目录匹配之后，对剩余 PENDING 的 STEM 论文，用标题精确匹配（不依赖DOI）去
OpenAlex 全量 S3 快照里找对应的"正式发表记录"，拿到它的 source_id 再套用已有的
source_id->tier 映射（SJR优先，OpenAlex is_core+h_index兜底）。

背景：不少论文压根没在 arXiv 元数据里回填 DOI（尤其是会议论文/较老论文/非西方出版商），
但 OpenAlex 本身是按论文标题也能检索到记录的（哪怕没有DOI），所以标题精确匹配能捞回
DOI关联漏掉的一部分。

用法：
  python3 backfill_title_tier.py
"""
import glob
import json
import os
import re

import duckdb
import pandas as pd

POOL_DIR = "data/common/stem_pools"
WORK_DIR = "work/corpus_rebuild/title_backfill"
TIER_DIR = "data/common/tier_mapping"
BATCH_FILES = 20
CHECKPOINT_PATH = f"{WORK_DIR}/checkpoint.json"
RESULT_PARQUET = f"{WORK_DIR}/title_source_lookup.parquet"


def normalize_title(t):
    if not isinstance(t, str):
        return ""
    t = t.lower().strip()
    t = t.replace("&", " and ")
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def stage_a_build_target_titles():
    out_path = f"{WORK_DIR}/target_titles.parquet"
    if os.path.exists(out_path):
        n = duckdb.sql(f"SELECT count(*) FROM read_parquet('{out_path}')").fetchone()[0]
        print(f"[A] target_titles 已存在: {n} 条，跳过")
        return out_path
    rows = {}
    for f in glob.glob(f"{POOL_DIR}/*.parquet"):
        df = pd.read_parquet(f, columns=["arxiv_id", "title", "tier"])
        pending = df[df["tier"] == "PENDING"]
        for arxiv_id, title in zip(pending["arxiv_id"], pending["title"]):
            tn = normalize_title(title)
            if tn and len(tn) >= 15 and tn not in rows:  # 太短的标题误配风险高，跳过
                rows[tn] = arxiv_id
    out = pd.DataFrame({"title_norm": list(rows.keys()), "arxiv_id": list(rows.values())})
    out.to_parquet(out_path, index=False)
    print(f"[A] 目标标题集合: {len(out)} 条 -> {out_path}")
    return out_path


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        return json.load(open(CHECKPOINT_PATH))
    return {"done_batches": []}


def save_checkpoint(cp):
    json.dump(cp, open(CHECKPOINT_PATH, "w"))


SQL_NORM_TITLE = (
    "trim(regexp_replace(regexp_replace(lower(replace(coalesce({col}, ''), "
    "'&', ' and ')), '[^a-z0-9 ]', ' ', 'g'), '\\s+', ' ', 'g'))"
)


def stage_b_query_openalex(target_title_path):
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

    con.execute(f"CREATE TABLE target_titles AS SELECT * FROM read_parquet('{target_title_path}')")

    title_norm_expr = SQL_NORM_TITLE.format(col="w.title")

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
                {title_norm_expr} AS title_norm,
                w.primary_location.source.id AS source_id,
                w.primary_location.source.issn_l AS issn_l,
                w.primary_location.source.issn AS issn_list,
                w.primary_location.source.display_name AS source_display_name,
                w.primary_location.source.is_core AS is_core,
                w.cited_by_count AS cited_by_count
            FROM read_parquet([{file_list_sql}]) AS w
            INNER JOIN target_titles t
              ON {title_norm_expr} = t.title_norm
            WHERE w.title IS NOT NULL AND length(w.title) > 10
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
    lookup = lookup.dropna(subset=["tier"]).drop_duplicates(subset=["title_norm"])
    title_tier = lookup.set_index("title_norm")["tier"].to_dict()
    title_source = lookup.set_index("title_norm")["tier_source"].to_dict()
    title_cited = lookup.set_index("title_norm")["cited_by_count"].to_dict()

    for f in sorted(glob.glob(f"{POOL_DIR}/*.parquet")):
        discipline = os.path.basename(f).replace(".parquet", "")
        df = pd.read_parquet(f)
        pending_mask = df["tier"] == "PENDING"
        n_pending = pending_mask.sum()
        if n_pending == 0:
            print(f"[D] {discipline}: 无PENDING，跳过")
            continue
        idx = df.index[pending_mask]
        tnorm = df.loc[idx, "title"].map(normalize_title)
        matched_tier = tnorm.map(title_tier)
        matched_source = tnorm.map(title_source)
        matched_cited = tnorm.map(title_cited)

        n_matched = matched_tier.notna().sum()
        df.loc[idx, "tier"] = matched_tier.where(matched_tier.notna(), df.loc[idx, "tier"])
        df.loc[idx[matched_tier.notna()], "tier_source"] = matched_source[matched_tier.notna()]
        has_cited = df["cited_by_count"].notna() if "cited_by_count" in df.columns else pd.Series(False, index=df.index)
        fill_cited_idx = idx[matched_cited.notna() & ~has_cited.loc[idx]]
        if len(fill_cited_idx) and "cited_by_count" in df.columns:
            df.loc[fill_cited_idx, "cited_by_count"] = matched_cited.loc[fill_cited_idx]

        df.to_parquet(f, index=False)
        print(f"[D] {discipline}: PENDING {n_pending} 篇，标题匹配命中 {n_matched} 篇 ({n_matched/n_pending:.1%})")


def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    target_title_path = stage_a_build_target_titles()
    stage_b_query_openalex(target_title_path)
    source_tier = build_source_id_tier_map()
    stage_d_write_back(source_tier)
    print("\n全部完成")


if __name__ == "__main__":
    main()
