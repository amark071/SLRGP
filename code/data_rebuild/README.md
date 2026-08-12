# Corpus rebuild / 语料重建

一键重建论文使用的检索语料与 arXiv 综述源码。**这些大语料不随包分发**(公开源 + 版权考虑),
审稿人可按本目录脚本自行重建;若只需复现实验 1–5 的统计结果,直接使用 `data/` 中已提供的
解析树、标注与特征表即可,无需重建语料。

## Quick start

```bash
cd NMI-submission            # 包根目录
export PYTHONPATH=code
bash code/data_rebuild/run_rebuild.sh
```

全程可中断重跑(断点续传)。预计:arXiv 源码约 2.5 小时(2,223 篇,4s 礼貌限速),
元数据快照约 30 分钟,候选池构建约 1 小时。

## Steps / 步骤说明

| Script | Purpose | Source |
|---|---|---|
| `download_metadata_snapshot.py` | arXiv 全量元数据快照(10 分片 parquet) | `librarian-bots/arxiv-metadata-snapshot`(HuggingFace) |
| `build_stem_pools.py` | 按 11 学科构建候选池 → `data/common/stem_pools/` | 上一步快照 |
| `fetch_arxiv_sources.py` | 按 `data/common/arxiv_review_manifest.jsonl` 下载 2,223 篇综述 LaTeX 源码 | arXiv `export.arxiv.org/e-print/` |
| `merge_tier_mapping.py` + `backfill_*.py` | 期刊/会议分级映射与回填 → `data/common/tier_mapping/` | CCF 推荐目录等公开分级表 |
| `retier_social_science.py` | 社科语料重打分级 → `data/common/social_science_corpus/` | OpenAlex(2026-06 快照) |
| `compute_stem_embeddings.py` | (可选)候选池向量编码 | BGE 系列嵌入模型 |
| `fetch_osf_preprints.py` | (可选)OSF 预印本补充 | api.osf.io |

## Notes

- 分级目录文件(如 `ccf_ranking.json`)请放入 `data/common/tier_mapping/`;CCF 目录为公开资料。
- OpenAlex 数据经其公开快照/API 获取,无需密钥;建议设置 `OPENALEX_MAILTO` 环境变量进入 polite pool。
- 社科端点原文(PDF/TEI)不随包分发;请按 `data/common/social_science_endpoint/` 清单中的
  OpenAlex work_id / OA 链接自行获取。
- 本目录脚本不含任何凭证;所需外部服务均为公开免费接口。
