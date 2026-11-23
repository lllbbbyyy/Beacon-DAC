#!/usr/bin/env bash
# Launch Beacon hierarchical exploration with automatic SQLite checkpoint resume
# on transient network disconnects while strictly preserving the wall-clock budget.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WORKLOAD="${1:-qwen3_14b_512tops}"
COMPASS_ROOT="${COMPASS_ROOT:?Please set COMPASS_ROOT=/path/to/Compass}"
COMPASS_BIN="${COMPASS_BIN:-$COMPASS_ROOT/build/compass}"
LLM_CONFIG="${LLM_CONFIG:-key.json}"
LLM_PROFILE="${LLM_PROFILE:-example-profile}"
AGENT_MODE="${AGENT_MODE:-multi}"
EVIDENCE_MODE="${EVIDENCE_MODE:-full}"
ANALYSIS_POLICY="${ANALYSIS_POLICY:-decision_first}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-small-en-v1.5}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-6}"

export CHIPLET_TUNER_EMBEDDING_DEVICE="${CHIPLET_TUNER_EMBEDDING_DEVICE:-cpu}"
export CHIPLET_TUNER_TRANSPORT_RETRY_DELAYS="${CHIPLET_TUNER_TRANSPORT_RETRY_DELAYS:-2,4,8,16,30,45}"

case "$WORKLOAD" in
  llama3_1_8b_64tops)
    SCALE=64
    BUDGET="${TIME_BUDGET_SECONDS:-3600}"
    ;;
  qwen3_14b_512tops)
    SCALE=512
    BUDGET="${TIME_BUDGET_SECONDS:-7200}"
    ;;
  mixtral_8x22b_2048tops)
    SCALE=2048
    BUDGET="${TIME_BUDGET_SECONDS:-14400}"
    ;;
  *)
    echo "Unknown workload preset: $WORKLOAD (expected llama3_1_8b_64tops | qwen3_14b_512tops | mixtral_8x22b_2048tops)" >&2
    exit 1
    ;;
esac

OUT_DIR="${OUT_DIR:-runs/${WORKLOAD}/beacon_${EVIDENCE_MODE}}"
mkdir -p "$OUT_DIR"

attempt=0
while [[ ! -f "$OUT_DIR/tuning_summary.json" ]]; do
  attempt=$((attempt + 1))
  RESUME_ARGS=()
  if [[ -f "$OUT_DIR/langgraph_checkpoints.sqlite" ]]; then
    RESUME_ARGS=(--resume)
    echo "[beacon] Resuming from SQLite checkpoint in $OUT_DIR (attempt $attempt/$MAX_ATTEMPTS)..."
  else
    echo "[beacon] Starting exploration in $OUT_DIR (attempt $attempt/$MAX_ATTEMPTS)..."
  fi

  set +e
  python3 -u main.py \
    --iterations 100000 \
    --task-type chunked_prefill \
    --agent-mode "$AGENT_MODE" \
    --analysis-policy "$ANALYSIS_POLICY" \
    --evidence-mode "$EVIDENCE_MODE" \
    --embedding-model "$EMBEDDING_MODEL" \
    --llm-max-tokens 8192 \
    --compute-scale "$SCALE" \
    --compass-root "$COMPASS_ROOT" \
    --compass-binary "$COMPASS_BIN" \
    --search-config "configs/$WORKLOAD/search_config.json" \
    --hardware "configs/$WORKLOAD/hardware.json" \
    --time-budget-seconds "$BUDGET" \
    --llm-config "$LLM_CONFIG" \
    --llm-profile "$LLM_PROFILE" \
    --output-dir "$OUT_DIR" \
    "${RESUME_ARGS[@]}"
  rc=$?
  set -e

  if [[ "$rc" -eq 0 && -f "$OUT_DIR/tuning_summary.json" ]]; then
    echo "[beacon] Completed $OUT_DIR cleanly."
    exit 0
  fi
  if [[ "$attempt" -ge "$MAX_ATTEMPTS" ]]; then
    echo "[beacon] Exhausted $MAX_ATTEMPTS attempts for $OUT_DIR (last exit code: $rc)." >&2
    exit "$rc"
  fi
  echo "[beacon] Interrupted before summary completion (rc=$rc); retrying from checkpoint in 30s..."
  sleep 30
done
