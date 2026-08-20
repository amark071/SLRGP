# SLRGP: Typed and Recursive Operators for Literature Reviewing

Code and data availability package accompanying the SLRGP manuscript.
SLRGP 论文的代码与数据可用性包。

SLRGP represents literature reviewing as transformations over a shared state
s = ⟨q, D, Γ, κ, x⟩, implemented through nine typed operators: query expansion (E),
literature location (L), eligibility filtering (F), candidate ranking (R), semantic
organization (O), guarded validation (V), evidence preparation (P), bounded writing (W)
and global reconciliation (C), with validation-triggered backtracking and recursive
descent over the organization tree.

---

## Repository layout 

```text
NMI-submission/
├── code/
│   ├── slrgp/                          Core: state, nine operators, control flow, retrieval / 九算子核心实现
│   ├── run_demo.py                     Small end-to-end demonstration entry / 端到端演示入口
│   ├── figures/                        Manuscript figure-generation scripts / 论文插图生成脚本
│   ├── requirements.txt
│   ├── data_rebuild/                 One-click corpus rebuild scripts / 语料一键重建脚本
│   └── experiments/
│       ├── split_utils.py, stats_utils.py   Shared split/statistics utilities / 共享划分与统计工具
│       ├── instrument_validation/      Evaluation-instrument (Instrument J) validation / 评审工具验证
│       ├── exp1_structural_recovery/   Experiment 1: structural recovery & hard-negative blind tests / 结构恢复与盲评
│       ├── exp2_operator_learnability/ Experiment 2: per-operator learnability / 算子可学习性
│       │   ├── expansion/  location/  filtering/  ranking/  organization/   (E / L / F / R / O)
│       ├── exp3_interface_substitution/ Experiment 3: interface-matched substitutions + open-weight robustness / 接口匹配替换
│       ├── exp4_structural_recursion/   Experiment 4: structural recursion vs length scaling / 结构递归
│       └── exp5_native_comparison/      Experiment 5: native workflow comparison / 原生系统对比
└── data/                             Data package (see data/README) / 数据包
```

## Experiment map / 实验对应表

| Paper component / 论文内容 | Code location |
|---|---|
| Evaluation-instrument validation (Instrument J) / 评审工具验证 | `code/experiments/instrument_validation/` |
| Experiment 1: Expert reviews instantiate LOTCF-LR structures / 结构恢复 | `code/experiments/exp1_structural_recovery/` |
| Experiment 2: Frontend operators are selectively learnable / 算子可学习性 | `code/experiments/exp2_operator_learnability/` |
| Experiment 3: Semantic organization and guarded validation / 接口匹配替换 | `code/experiments/exp3_interface_substitution/` |
| Experiment 4: Structural recursion vs length scaling / 结构递归 | `code/experiments/exp4_structural_recursion/` |
| Experiment 5: Native system comparison / 原生系统对比 | `code/experiments/exp5_native_comparison/` |
| Manuscript figures / 论文插图 | `code/figures/` |
| Operator contracts, guarded validation, recursion (Methods) / 九算子本体 | `code/slrgp/` |

## Requirements / 环境要求

- Python 3.10+; Linux recommended for the full pipeline.
- An OpenAI-compatible chat-completions endpoint is required for LLM-backed operators and judges.
- See `code/requirements.txt`.

## Running the demo / 运行演示

`code/run_demo.py` runs the full nine-operator pipeline over a bundled 30-paper corpus
(`data/common/sample_rl_papers.json`). All operators except deterministic retrieval are
LLM-backed, so a running OpenAI-compatible endpoint is mandatory; without one the script
fails fast with a connection error.
演示语料已随包提供，但 E/L/F/R/O/V/W 等步骤均调用 LLM，因此必须先有一个运行中的
OpenAI 兼容端点，否则脚本会直接报连接错误。

- Default endpoint: `http://localhost:8000/v1`, model `qwen3-32b` (self-hosted vLLM).
  默认连接本机 vLLM 端点。
- Self-hosted startup example / 自建端点启动示例:
  `python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-32B --port 8000`
- To use any other OpenAI-compatible endpoint, set environment variables (no code change):
  切换端点用环境变量即可:
  `SLRGP_LLM_BASE_URL` (address), `SLRGP_LLM_MODEL` (model name), `SLRGP_LLM_API_KEY`
  (Bearer key, empty for self-hosted vLLM); set `SLRGP_LLM_VLLM=0` for non-vLLM endpoints.
- Run from the package root: `PYTHONPATH=code python3 code/run_demo.py`.

## Notes on paths and credentials / 路径与凭证说明

- All paths in the code have been remapped to package-relative locations (`data/…`, `code/…`, `work/…`, ).
  Run scripts from the package root and set `export PYTHONPATH=code`.
  代码内路径已全部改为包内相对路径；请从包根目录运行脚本，并设置 `export PYTHONPATH=code`。
- All API credentials are read from environment variables; no keys are embedded in this package.
  所有 API 凭证均从环境变量读取，包内不含任何密钥。
- Commit hashes in `experiments/exp5_native_comparison/patches/*/patch_manifest.json` identify the exact
  upstream revisions of the compared baseline systems. 补丁清单中的哈希为对比基线系统的上游版本号。
- Running `python3 code/figures/make_main_figures.py` from the package root regenerates the manuscript's
  main figures (PDF/SVG/PNG/TIFF) into `./figures/` from the frozen statistics under `data/`, together with
  a provenance manifest. 主图可由冻结统计一键重生成。
