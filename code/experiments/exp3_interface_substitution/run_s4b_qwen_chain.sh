#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-.}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CODE_DIR="$SCRIPT_DIR"
PYTHON=${PYTHON:-python3}
TOPICS=${TOPICS:-$CODE_DIR/topics_s4b_qwen4.txt}
ARMS=${ARMS:-intact,o_rank_slab_matched,v_guarded_stress,v_unguarded_stress}
RUN_ID=${RUN_ID:-s4b_qwen4_$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-$PKG_ROOT/work/results/s4/$RUN_ID}
LOGDIR=${LOGDIR:-$PKG_ROOT/work/logs/s4}
LOG="$LOGDIR/$RUN_ID.log"
PROGRESS="$LOGDIR/$RUN_ID.progress.json"
PIDFILE="$LOGDIR/$RUN_ID.pid"
VLLM_LOG="$LOGDIR/qwen3_32b_vllm.log"
VLLM_PID="$LOGDIR/qwen3_32b_vllm.pid"
BNB_LOG="$LOGDIR/qwen3_32b_bnb_server.log"
BNB_PID="$LOGDIR/qwen3_32b_bnb_server.pid"
# Local Qwen3-32B weights are not shipped with this package; point MODEL_PATH
# at a local download (see code/data_rebuild/README.md).
MODEL_PATH=${MODEL_PATH:-$ROOT/models/Qwen/Qwen3-32B}
MODEL_NAME=${MODEL_NAME:-qwen3-32b}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB:-2}
BACKEND=${BACKEND:-bnb}
BNB_GPU_MAX_MEMORY=${BNB_GPU_MAX_MEMORY:-28GiB}
BNB_CPU_MAX_MEMORY=${BNB_CPU_MAX_MEMORY:-96GiB}
PORT=${PORT:-8000}
BASE_URL=${BASE_URL:-http://127.0.0.1:$PORT/v1}
BUNDLE_SOURCE=${BUNDLE_SOURCE:-$PKG_ROOT/data/exp3_interface_substitution/s4a_main8_20260711_173157}

mkdir -p "$OUT" "$LOGDIR"
echo $$ > "$PIDFILE"

api_ready() {
  curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1
}

start_qwen_if_needed() {
  if api_ready; then
    echo "[S4b] Qwen API already ready at $BASE_URL" | tee -a "$LOG"
    return
  fi
  if [[ "$BACKEND" == "bnb" ]]; then
    if [[ -s "$BNB_PID" ]] && kill -0 "$(cat "$BNB_PID")" 2>/dev/null; then
      echo "[S4b] Qwen bitsandbytes pid exists but API not ready, waiting: $(cat "$BNB_PID")" | tee -a "$LOG"
    else
      echo "[S4b] starting Qwen3-32B bitsandbytes server on port $PORT" | tee -a "$LOG"
      CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} \
      nohup "$PYTHON" "$CODE_DIR/qwen_bnb_openai_server.py" \
        --model-path "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" \
        --gpu-max-memory "$BNB_GPU_MAX_MEMORY" \
        --cpu-max-memory "$BNB_CPU_MAX_MEMORY" \
        --host 127.0.0.1 \
        --port "$PORT" \
        > "$BNB_LOG" 2>&1 &
      echo $! > "$BNB_PID"
    fi
  else
  if [[ -s "$VLLM_PID" ]] && kill -0 "$(cat "$VLLM_PID")" 2>/dev/null; then
    echo "[S4b] Qwen pid exists but API not ready, waiting: $(cat "$VLLM_PID")" | tee -a "$LOG"
  else
    echo "[S4b] starting Qwen3-32B vLLM on port $PORT" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} nohup "$PYTHON" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_PATH" \
      --served-model-name "$MODEL_NAME" \
      --host 127.0.0.1 \
      --port "$PORT" \
      --tensor-parallel-size 2 \
      --trust-remote-code \
      --max-model-len "$MAX_MODEL_LEN" \
      --cpu-offload-gb "$CPU_OFFLOAD_GB" \
      --disable-custom-all-reduce \
      > "$VLLM_LOG" 2>&1 &
    echo $! > "$VLLM_PID"
  fi
  fi
  for i in $(seq 1 90); do
    if api_ready; then
      echo "[S4b] Qwen API ready after ${i} checks" | tee -a "$LOG"
      return
    fi
    echo "[S4b] waiting for Qwen API... $i/90" | tee -a "$LOG"
    sleep 20
  done
  echo "[S4b] ERROR: Qwen API did not become ready" | tee -a "$LOG"
  exit 2
}

write_progress() {
  "$PYTHON" "$CODE_DIR/s4b_qwen_progress.py" \
    --out-dir "$OUT" \
    --topics-file "$TOPICS" \
    --arms "$ARMS" \
    --write-json "$PROGRESS" | tee -a "$LOG"
}

echo "[S4b] run_id=$RUN_ID" | tee -a "$LOG"
echo "[S4b] out=$OUT" | tee -a "$LOG"
echo "[S4b] topics=$TOPICS arms=$ARMS" | tee -a "$LOG"
echo "[S4b] backend=$BACKEND model=$MODEL_PATH" | tee -a "$LOG"
start_qwen_if_needed

set +e
stdbuf -oL -eL "$PYTHON" "$CODE_DIR/s4a_matched_runner.py" \
  --topics-file "$TOPICS" \
  --out-dir "$OUT" \
  --bundle-source-root "$BUNDLE_SOURCE" \
  --metadata-only \
  --base-url "$BASE_URL" \
  --model "$MODEL_NAME" \
  --arms "$ARMS" \
  --total-words 1000 \
  --min-words 800 \
  --max-words 1250 \
  --root-groups 2 \
  --group-size 4 \
  --resume-skip-ok \
  >> "$LOG" 2>&1 &
RUNNER_PID=$!
echo "[S4b] runner_pid=$RUNNER_PID" | tee -a "$LOG"
while kill -0 "$RUNNER_PID" 2>/dev/null; do
  write_progress
  sleep 60
done
wait "$RUNNER_PID"
rc=$?
set -e
write_progress
if [[ "$rc" -ne 0 ]]; then
  echo "[S4b] runner failed rc=$rc; rerun the same command to resume" | tee -a "$LOG"
  exit "$rc"
fi

"$PYTHON" "$CODE_DIR/s4a_matched_audit.py" \
  --root "$OUT" \
  --arms "$ARMS" \
  --out "$OUT/s4b_qwen_audit.json" | tee -a "$LOG"
write_progress
echo "[S4b] DONE run_id=$RUN_ID out=$OUT log=$LOG progress=$PROGRESS" | tee -a "$LOG"
