#!/usr/bin/env bash
set -euo pipefail

WORKLOAD="${1:-qwen3_14b_512tops}"
COMPASS_ROOT="${COMPASS_ROOT:?Please set COMPASS_ROOT=/path/to/Compass}"
COMPASS_BIN="${COMPASS_BIN:-$COMPASS_ROOT/build/compass}"
LLM_CONFIG="${LLM_CONFIG:-key.json}"
LLM_PROFILE="${LLM_PROFILE:-example-profile}"
EVIDENCE_MODE="${EVIDENCE_MODE:-full}"
ANALYSIS_POLICY="${ANALYSIS_POLICY:-decision_first}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-small-en-v1.5}"

case "$WORKLOAD" in
  llama3_1_8b_64tops)
    SCALE=64
    BUDGET=3600
    ;;
  qwen3_14b_512tops)
    SCALE=512
    BUDGET=7200
    ;;
  mixtral_8x22b_2048tops)
    SCALE=2048
    BUDGET=14400
    ;;
  *)
    echo "Unknown workload preset: $WORKLOAD" >&2
    exit 1
    ;;
esac

OUT_DIR="${OUT_DIR:-runs/${WORKLOAD}_beacon_${EVIDENCE_MODE}}"

python3 main.py \
  --iterations 100000 \
  --task-type chunked_prefill \
  --agent-mode multi \
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
  --output-dir "$OUT_DIR"
