#!/usr/bin/env bash
# Launch matched-budget DSE baselines (Random, HASCO-BO, Chiplet-Gym PPO, AgenticDSE-Adapted)
# from the exact same initial hardware z_0, design space Z, and wall-clock time budget T.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

METHOD="${1:-hasco_scalar}"
WORKLOAD="${2:-qwen3_14b_512tops}"
COMPASS_ROOT="${COMPASS_ROOT:?Please set COMPASS_ROOT=/path/to/Compass}"
COMPASS_BIN="${COMPASS_BIN:-$COMPASS_ROOT/build/compass}"
SEED="${SEED:-42}"

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
    echo "Unknown workload preset: $WORKLOAD" >&2
    exit 1
    ;;
esac

OUT_DIR="${OUT_DIR:-runs/${WORKLOAD}/${METHOD}}"
mkdir -p "$OUT_DIR"

EXTRA_ARGS=()
case "$METHOD" in
  random|hasco_scalar)
    ;;
  ppo)
    EXTRA_ARGS+=(--ppo-rollout-steps 8 --ppo-batch-size 4 --ppo-learning-rate 0.0003 --ppo-entropy-coefficient 0.02)
    ;;
  agentic_scalar)
    LLM_CONFIG="${LLM_CONFIG:-key.json}"
    LLM_PROFILE="${LLM_PROFILE:-example-profile}"
    EXTRA_ARGS+=(--llm-config "$LLM_CONFIG" --llm-profile "$LLM_PROFILE" --agentic-phase-evaluations 12 --agentic-warmup 5)
    ;;
  *)
    echo "Unknown baseline method: $METHOD (expected random | hasco_scalar | ppo | agentic_scalar)" >&2
    exit 1
    ;;
esac

python3 -u -m baseline.main \
  --method "$METHOD" \
  --evaluations 100000 \
  --seed "$SEED" \
  --compute-scale "$SCALE" \
  --compass-root "$COMPASS_ROOT" \
  --compass-binary "$COMPASS_BIN" \
  --search-config "configs/$WORKLOAD/search_config.json" \
  --hardware "configs/$WORKLOAD/hardware.json" \
  --time-budget-seconds "$BUDGET" \
  --output-dir "$OUT_DIR" \
  "${EXTRA_ARGS[@]}"
