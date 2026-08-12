#!/bin/bash
# SLRGP 语料一键重建主流程(从包根目录运行: bash code/data_rebuild/run_rebuild.sh)
# 所有步骤均可安全重跑(脚本内部断点续传)。全程预计:arXiv 源码约 2.5h,元数据快照约 30min,候选池构建约 1h。
set -uo pipefail
cd "$(dirname "$0")/../.."   # 包根目录
mkdir -p logs work/corpus_rebuild

log() { echo "[$(date '+%H:%M:%S')] $1"; }
run_step() {
  local name=$1; local cmd=$2
  log "=== 开始: $name ==="
  if eval "$cmd" > "logs/rebuild_${name}.log" 2>&1; then log "=== 完成: $name ===";
  else log "!!! 失败: $name (见 logs/rebuild_${name}.log) !!!"; fi
}

# Stage 1: arXiv 元数据快照(11 学科候选池的源)
run_step "01_metadata_snapshot" "python3 code/data_rebuild/download_metadata_snapshot.py"

# Stage 2: STEM 11 学科候选池
run_step "02_stem_pools" "python3 code/data_rebuild/build_stem_pools.py"

# Stage 3: 综述 LaTeX 源码(按清单,2,223 篇)
run_step "03_arxiv_sources" "python3 code/data_rebuild/fetch_arxiv_sources.py"

# Stage 4: 期刊分级合并(公开分级目录:CCF 等,见 README)
run_step "04_merge_tier" "python3 code/data_rebuild/merge_tier_mapping.py"

# Stage 5: 分级回fill
run_step "05_backfill_stem_tier" "python3 code/data_rebuild/backfill_stem_tier.py"
run_step "06_backfill_ccf_tier" "python3 code/data_rebuild/backfill_ccf_tier.py"
run_step "07_backfill_title_tier" "python3 code/data_rebuild/backfill_title_tier.py"

# Stage 6: 社科语料(OpenAlex)与重打 tier
run_step "08_retier_social" "python3 code/data_rebuild/retier_social_science.py"

log "全部步骤结束。中间产物在 work/corpus_rebuild/,语料在 data/common/。"
