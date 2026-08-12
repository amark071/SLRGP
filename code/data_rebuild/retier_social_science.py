#!/usr/bin/env python3
"""
用 unified_title_tier.parquet（国际化SJR+OpenAlex分级表）给社科语料库
(papers_en.db)打 T1/T2/T3 tier 标签。

匹配方式：journal 名标准化后做精确匹配（大小写/标点/多空格归一化）。
  命中 -> 写入 T1/T2/T3
  未命中 -> 写入 "UNKNOWN"（如实记录，不臆造；覆盖率验证结果打印出来，
            未命中期刊名单写到 data/social_science/tier_unmatched_journals.csv 供人工兜底）

不修改原库，写一个新库 papers_en_retiered.db（同表结构，tier列改存文本 T1/T2/T3/UNKNOWN，
新增 tier_source 列记录判据来源），原库保留作为备份。
"""
import json
import re
import shutil
import sqlite3

import pandas as pd

DB_SRC = "data/common/social_science_corpus/papers_en.db"
DB_OUT = "data/common/social_science_corpus/papers_en_retiered.db"
TITLE_TIER_PATH = "data/common/tier_mapping/unified_title_tier.parquet"
UNMATCHED_CSV = "data/common/social_science_corpus/tier_unmatched_journals.csv"

# 人工兜底：期刊改名/带副标题导致标准化匹配失败的已知案例（首轮跑完覆盖率检查后手动补充）。
# key 是 normalize_title() 处理后的形式 -> 应重定向到的标准化期刊名(title_norm)
MANUAL_OVERRIDES_NORM = {
    "interfaces": "informs journal on applied analytics",  # 2021年期刊改名
    "international small business journal": "international small business journal researching entrepreneurship",
    "finance finance and stochastics": "finance and stochastics",  # 源文件名拼接artifact，标准化后前导逗号被去掉
}


def normalize_title(t):
    if not isinstance(t, str):
        return ""
    t = t.lower().strip()
    t = t.replace("&", " and ")
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def main():
    title_tier = pd.read_parquet(TITLE_TIER_PATH).set_index("title_norm")["tier"].to_dict()
    print(f"国际化期刊分级表: {len(title_tier)} 条")

    shutil.copy(DB_SRC, DB_OUT)
    conn = sqlite3.connect(DB_OUT)
    cur = conn.cursor()
    cur.execute("ALTER TABLE papers ADD COLUMN tier_intl TEXT")
    cur.execute("ALTER TABLE papers ADD COLUMN tier_source TEXT")
    conn.commit()

    cur.execute("SELECT doc_id, data FROM papers")
    rows = cur.fetchall()
    print(f"总papers数: {len(rows)}")

    journal_cache = {}
    unmatched_counter = {}
    n_matched, n_unmatched = 0, 0
    updates = []

    for doc_id, data_str in rows:
        try:
            data = json.loads(data_str)
        except (json.JSONDecodeError, TypeError):
            updates.append(("UNKNOWN", "parse_error", doc_id))
            n_unmatched += 1
            continue
        journal = data.get("journal", "")
        norm = journal_cache.get(journal)
        if norm is None:
            norm = normalize_title(journal)
            norm = MANUAL_OVERRIDES_NORM.get(norm, norm)
            journal_cache[journal] = norm
        tier = title_tier.get(norm)
        if tier:
            updates.append((tier, "sjr_or_openalex_title_match", doc_id))
            n_matched += 1
        else:
            updates.append(("UNKNOWN", "no_match", doc_id))
            n_unmatched += 1
            unmatched_counter[journal] = unmatched_counter.get(journal, 0) + 1

    cur.executemany("UPDATE papers SET tier_intl=?, tier_source=? WHERE doc_id=?", updates)
    conn.commit()
    conn.close()

    total = n_matched + n_unmatched
    print(f"\n命中: {n_matched} ({n_matched/total:.1%})  未命中: {n_unmatched} ({n_unmatched/total:.1%})")

    unmatched_df = pd.DataFrame(
        sorted(unmatched_counter.items(), key=lambda x: -x[1]),
        columns=["journal", "paper_count"],
    )
    unmatched_df.to_csv(UNMATCHED_CSV, index=False)
    print(f"未命中期刊清单({len(unmatched_df)}种期刊)写入 {UNMATCHED_CSV}，供人工兜底")
    print(f"\n新库: {DB_OUT}（原库 {DB_SRC} 保留不变）")


if __name__ == "__main__":
    main()
