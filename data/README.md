# SLRGP Data Package / 数据包说明

本目录与 `code/experiments/` 一一对应,提供论文五个实验与评审工具验证所需的全部输入数据与实验记录。
大型检索语料(arXiv 全文 LaTeX、OpenAlex 统一语料库)不随包分发,见文末「语料重建」。

## Layout / 布局

| Directory | Content / 内容 | Used by |
|---|---|---|
| `common/arxiv_review_manifest.jsonl` | 2,862 条 arXiv 综述清单(arxiv_id + discipline + parse_status;含抽样框架核算所需的失败记录)。其中 2,223 篇通过初步解析,2 篇因顶层章节少于两个(fewer than two top-level sections)被排除出分析语料,最终分析样本为 2,221 篇 | 全部实验 |
| `common/social_science_endpoint/` | 社科端点(论文 Supplementary Note 3)的候选清单、LOTCF-LR 树、标注;**不含原文 PDF/TEI**(版权),请按清单中的 OpenAlex work_id / OA 链接自行获取 | 实验 1/2 社科复核 |
| `instrument_validation/` | 评审工具 Instrument J 的 96 条盲评分数(csv/json)+ 运行日志 + 两篇源综述的 arXiv LaTeX 源码(`sources/`,arXiv:1809.00057, arXiv:2306.01660;运行 `code/experiments/instrument_validation/make_variants.py` 可重建 8 条件变体) | Methods 评审工具节 |
| `exp1_structural_recovery/lotcf_trees/` | 2,223 棵确定性解析的 LOTCF-LR 有序树(11 学科分目录;其中 2 棵因顶层章节少于两个不进入分析,实验 1 的分析语料为 2,221 篇)+ 639 条解析失败占位记录(仅含 fail_reason,供抽样框架核算) | 实验 1 |
| `exp1_structural_recovery/blind_annotations/` | 真本 vs 困难负样本的节点级盲评标注 | 实验 1 |
| `exp1_structural_recovery/confirmatory/` | 确认性盲评数据:盲评标注为单轮采集,`s1a_annotations_blind_pass1_dedup.jsonl` 为去重后全量 8,643 条,是论文数字的唯一来源;评测脚本 `s1a_eval_blind.py` 支持多轮标注文件聚合,仅存在单轮标注时按单轮计算。另含 150 条人工 faithfulness 复核与 100 条 forced-choice | 实验 1 |
| `exp2_operator_learnability/expansion/` | E 算子评测输入(3/6/10 扩展词档 parquets) | 实验 2-E |
| `exp2_operator_learnability/location/` | L 算子诊断数据;主要输入为统一语料库(见「语料重建」) | 实验 2-L |
| `exp2_operator_learnability/filtering/` | F 算子 gold 集(train/val/test jsonl) | 实验 2-F |
| `exp2_operator_learnability/ranking/` | R 算子 LTR 特征表(train/heldout parquet + groups)与引文解析标签源 | 实验 2-R |
| `exp2_operator_learnability/organization/` | O 算子图式学习标注(`schema_learning/`)与确认性数据(`confirmatory/`,含共识标签与划分清单) | 实验 2-O |
| `exp3_interface_substitution/` | 实验 3 全部生成文本、盲评与分析记录(Claude 主实验 + Qwen3-32B 稳健性) | 实验 3 |
| `exp4_structural_recursion/` | 实验 4 全部生成文本、盲评与分析记录(三长度档 × 四臂) | 实验 4 |
| `exp5_native_comparison/` | 实验 5 四系统(SLRGP/AutoSurvey/SURVEYFORGE/SurveyGen)生成全文、盲评偏好、claim-support 与分析 | 实验 5 |

## Trained models / 训练产物

训练好的模型(LambdaMART 排序器、L 算子 tier2 嵌入、E 算子词选择器)**不随包分发**。
训练代码、特征表、固定划分与随机种子均已提供,在 CPU 上分钟即可复现:
训练脚本见 `code/experiments/exp2_operator_learnability/<operator>/`,输出默认写入 `models/`。

## Corpus rebuild / 语料重建

以下大型语料来自公开源,不随包分发;按 `code/data_rebuild/`(或各脚本 docstring)重建:

- `data/common/arxiv_latex/` — 按 `common/arxiv_review_manifest.jsonl` 中的 arXiv ID 从 arXiv 批量获取源码
- `data/common/unified_corpus/` — 由 OpenAlex 快照构建(11 学科,约 366 万条)
- 一键重建脚本见 `code/data_rebuild/`(含 run_rebuild.sh 主流程);OpenAlex 版本:本研究使用 2026-06 快照

## Notes

- `work/` 目录(包根)为脚本运行时自动生成的中间产物位置,无需手工创建。
- 所有路径已改为包内相对路径;请从包根目录运行脚本,并 `export PYTHONPATH=code`。
- 本包不含任何 API 凭证;LLM 相关脚本从环境变量读取密钥。
- 冻结运行清单(preflight/frozen_protocol 等)中的文件哈希为实验执行当时版本的记录;
  随包代码仅经路径与注释层面的清理,哈希不逐字节一致,属预期差异。
