# Beacon

**Beacon: Hierarchical Multi-Agent Design Space Exploration for Heterogeneous Multi-Chiplet LLM Accelerators**

Beacon is a hierarchical multi-agent framework for architecture and parallelism design space exploration (DSE) of heterogeneous multi-chiplet large language model (LLM) accelerators. It decouples fast system-level performance and manufacturing-cost ($MC$) scoping from on-demand operator-level diagnosis, combining verified reference configuration tracking, cached report query tools, and context-matched experience retrieval (RAG) to generate duplicate-free multi-parameter updates directly.

## Repository Structure

```text
.
├── chiplet_tuner/              # Core Beacon hierarchical multi-agent framework
│   ├── agents/                 # System Agent, on-demand Operator Agent, Solution Agent, ReAct loop
│   ├── core/                   # Design space Z, composite objective J, reference & coverage tracking
│   ├── llm/                    # OpenAI-compatible LLM client, prompts, and structured tracing
│   ├── pipeline/               # LangGraph exploration loop with SQLite checkpoint/resume
│   ├── rag/                    # Dual-channel context-matched RAG memory (symptom + hardware scale)
│   ├── simulators/             # Two-level evaluator interface (GenericFileAdapter & CompassAdapter)
│   ├── tools/                  # Cached simulation report query tools & duplicate-free edit tools
│   └── cli.py                  # Unified CLI entry point
├── baseline/                   # Adapted baselines (Random, HASCO-BO, Chiplet-Gym PPO, AgenticDSE-Adapted)
├── configs/                    # Representative LLM workload & initial architecture configs
│   ├── llama3_1_8b_64tops/     # Llama 3.1-8B (64 TOPS, 1 h budget)
│   ├── qwen3_14b_512tops/      # Qwen3-14B (512 TOPS, 2 h budget)
│   └── mixtral_8x22b_2048tops/ # Mixtral-8x22B (2048 TOPS, 4 h budget)
├── scripts/                    # Launch scripts for Beacon, ablations, and baselines
├── tests/                      # Unit tests for Beacon's core mechanisms
├── main.py                     # CLI entry wrapper
├── key.example.json            # Sanitized OpenAI-compatible API configuration template
└── requirements.txt            # Python dependencies
```

## Installation

```bash
conda create -n beacon python=3.12 -y
conda activate beacon
pip install -r requirements.txt
```

## Evaluator Interface

Beacon interacts with any multi-chiplet evaluator through the two-level `SimulatorAdapter` interface (`chiplet_tuner/simulators/base.py`):
1. **System-level summary metrics**: total latency $L$, total energy $E$, and manufacturing cost $MC$ (with die, I/O, and package substrate components).
2. **Optional stage-level execution records**: post-TP matrix shapes $(M, N, K)$, compute/memory/interconnect stall breakdowns, and mapped chiplet IDs (queried only when the Operator Agent is invoked on demand).

Two built-in adapters are provided:
- `generic-files`: analyzes precomputed evaluation artifacts without invoking an external simulator binary.
- `compass`: invokes an external GEMINI/Compass multi-chiplet evaluator binary (`--compass-root /path/to/Compass --compass-binary /path/to/compass`).

## Quick Verification (Unit Tests)

Run the standalone unit tests covering hierarchical fast-bypass scoping (`decision_first`), verified reference tracking, directional coverage, report query semantics, and evidence gating:

```bash
python3 -m unittest discover -s tests -v
```

## Running Beacon

1. Configure an OpenAI-compatible LLM endpoint:
   ```bash
   cp key.example.json key.json
   # Edit key.json with your endpoint URL, model name, and API key
   ```

2. Launch hierarchical multi-agent exploration (e.g., Qwen3-14B at 512 TOPS under a 2 h budget):
   ```bash
   COMPASS_ROOT=/path/to/Compass \
   LLM_CONFIG=key.json \
   LLM_PROFILE=example-profile \
   bash scripts/run_beacon.sh qwen3_14b_512tops
   ```

   To run the scalar-only ablation (`Beacon-Scalar`), pass `EVIDENCE_MODE=scalar`. To run the unconditional stage-inspection ablation (`Always-Stage`), pass `ANALYSIS_POLICY=standard`.

## License

Released under the MIT License (see `LICENSE`).
