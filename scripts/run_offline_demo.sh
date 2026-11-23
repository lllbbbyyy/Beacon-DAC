#!/usr/bin/env bash
# Run a standalone three-agent Beacon diagnosis & proposal pass over precomputed
# evaluation artifacts in examples/sample_eval/ (requires NO simulator binary).
# If LLM_CONFIG is omitted, uses the deterministic offline MockLLMClient + hashing embeddings.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-runs/offline_demo}"
EVIDENCE_MODE="${EVIDENCE_MODE:-full}"
ANALYSIS_POLICY="${ANALYSIS_POLICY:-decision_first}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-hashing}"

EXTRA_LLM_ARGS=()
if [[ -n "${LLM_CONFIG:-}" ]]; then
  EXTRA_LLM_ARGS+=(--llm-config "$LLM_CONFIG")
  if [[ -n "${LLM_PROFILE:-}" ]]; then
    EXTRA_LLM_ARGS+=(--llm-profile "$LLM_PROFILE")
  fi
fi

python3 main.py \
  --simulator generic-files \
  --from-existing examples/sample_eval \
  --hardware examples/sample_eval/hardware.json \
  --metrics-file examples/sample_eval/search_res.csv \
  --latency-detail examples/sample_eval/search_latency_detail.json \
  --energy-detail examples/sample_eval/search_energy_detail.json \
  --mc-detail examples/sample_eval/search_mc_detail.json \
  --analysis-policy "$ANALYSIS_POLICY" \
  --evidence-mode "$EVIDENCE_MODE" \
  --embedding-model "$EMBEDDING_MODEL" \
  --output-dir "$OUT_DIR" \
  "${EXTRA_LLM_ARGS[@]}"
