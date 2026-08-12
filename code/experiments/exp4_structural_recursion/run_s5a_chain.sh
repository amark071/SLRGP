#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON=${PYTHON:-python3}

RUN_ID=${RUN_ID:-s5a_length_$(date +%Y%m%d_%H%M%S)}
ROOT=${ROOT:-$PKG_ROOT/work/results/s5/$RUN_ID}
TOPICS=${TOPICS:-$SCRIPT_DIR/topics_s5a.txt}
TIERS=${TIERS:-short,medium,long}
ARMS=${ARMS:-recursive_slrgp,fixed_outline_no_reentry,flat_single_pass}
MODEL=${MODEL:-anthropic/claude-sonnet-4.6}
LIMIT_TOPICS=${LIMIT_TOPICS:-0}
SOURCE_ROOTS=${SOURCE_ROOTS:-$PKG_ROOT/data/exp3_interface_substitution/s4a_main8_20260711_173157,$PKG_ROOT/data/exp3_interface_substitution/s4a_matched_main8_20260713_1548,$PKG_ROOT/data/exp3_interface_substitution/s4b_qwen4_20260713_1745}
LOG=${LOG:-$ROOT/run.log}

mkdir -p "$ROOT"
echo "[S5a] run_id=$RUN_ID root=$ROOT" | tee -a "$LOG"
echo "[S5a] topics=$TOPICS tiers=$TIERS arms=$ARMS model=$MODEL limit=$LIMIT_TOPICS" | tee -a "$LOG"

"$PYTHON" "$SCRIPT_DIR/s5a_length_runner.py" \
  --topics-file "$TOPICS" \
  --source-roots "$SOURCE_ROOTS" \
  --out-dir "$ROOT" \
  --model "$MODEL" \
  --tiers "$TIERS" \
  --arms "$ARMS" \
  --limit-topics "$LIMIT_TOPICS" \
  --resume-skip-ok \
  --verbose 2>&1 | tee -a "$LOG"

"$PYTHON" "$SCRIPT_DIR/s5a_audit.py" \
  --root "$ROOT" \
  --out "$ROOT/s5a_design_audit.json" 2>&1 | tee -a "$LOG"

"$PYTHON" "$SCRIPT_DIR/s5a_blind_judge.py" \
  --root "$ROOT" \
  --audit "$ROOT/s5a_design_audit.json" \
  --out-dir "$ROOT/blind_judging" 2>&1 | tee -a "$LOG"

"$PYTHON" "$SCRIPT_DIR/s5a_analyze_length.py" \
  --root "$ROOT" \
  --audit "$ROOT/s5a_design_audit.json" 2>&1 | tee -a "$LOG"

echo "[S5a] DONE root=$ROOT" | tee -a "$LOG"
