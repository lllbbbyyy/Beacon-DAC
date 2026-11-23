#!/usr/bin/env bash
# Launch controlled Beacon ablations under identical initial hardware z_0 and time budget T:
#   - beacon_full   : Hierarchical System + on-demand Operator + Solution agents (default)
#   - beacon_scalar : Global (L, E, MC, J) scalars only (disables stage/cost detail tools)
#   - always_stage  : Unconditional per-round stage inspection (disables fast bypass)
#   - single_agent  : Monolithic single-agent workflow with identical tools and RAG store
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="${1:-beacon_scalar}"
WORKLOAD="${2:-qwen3_14b_512tops}"
BASE_OUT="${OUT_ROOT:-$ROOT/runs/${WORKLOAD}_ablation}"

case "$VARIANT" in
  beacon_full)
    OUT_DIR="$BASE_OUT/beacon_full" EVIDENCE_MODE=full ANALYSIS_POLICY=decision_first AGENT_MODE=multi \
      bash "$ROOT/scripts/run_beacon.sh" "$WORKLOAD"
    ;;
  beacon_scalar)
    OUT_DIR="$BASE_OUT/beacon_scalar" EVIDENCE_MODE=scalar ANALYSIS_POLICY=decision_first AGENT_MODE=multi \
      bash "$ROOT/scripts/run_beacon.sh" "$WORKLOAD"
    ;;
  always_stage)
    OUT_DIR="$BASE_OUT/always_stage" EVIDENCE_MODE=full ANALYSIS_POLICY=standard AGENT_MODE=multi \
      bash "$ROOT/scripts/run_beacon.sh" "$WORKLOAD"
    ;;
  single_agent)
    OUT_DIR="$BASE_OUT/single_agent" EVIDENCE_MODE=full ANALYSIS_POLICY=decision_first AGENT_MODE=single \
      bash "$ROOT/scripts/run_beacon.sh" "$WORKLOAD"
    ;;
  *)
    echo "Unknown ablation variant: $VARIANT (expected beacon_full | beacon_scalar | always_stage | single_agent)" >&2
    exit 1
    ;;
esac
