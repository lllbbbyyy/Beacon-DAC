# Beacon

**Beacon: Hierarchical Multi-Agent Design Space Exploration for Heterogeneous Multi-Chiplet LLM Accelerators**

Beacon is a hierarchical multi-agent framework for architecture and parallelism design space exploration (DSE) of heterogeneous multi-chiplet large language model (LLM) accelerators. It decouples fast system-level performance and manufacturing-cost ($MC$) scoping from on-demand operator-level diagnosis, combining verified reference configuration tracking, cached report query tools, and context-matched experience retrieval (RAG) to generate duplicate-free multi-parameter updates directly.

---

## Key Mechanisms & Code Mapping

| Paper Component | Implementation Module | Description |
|---|---|---|
| **Design Space $\mathcal{Z}$ & Objective $J(z)$** | `chiplet_tuner/core/search_space.py`<br>`chiplet_tuner/core/objective.py` | Active chiplet count $N$, per-chiplet `(dataflow, compute_units, buffer_size)`, system bandwidths `(dram_bw, nop_bw)`, parallelism $g_{\mathrm{par}} = (\mathrm{TP}, \mathrm{MBS})$, and $J(z) = L \cdot E \cdot MC \cdot \rho(z)$ |
| **Reference Tracking & Directional Coverage** | `chiplet_tuner/pipeline/tuner.py`<br>`chiplet_tuner/core/search_coverage.py` | Anchors each round on a verified historical configuration $z_{\mathrm{base}}$, reverts immediately after regressions, and tracks tested vs. unexplored parameter directions $\mathrm{Cov}(\mathcal{H})$ |
| **System Agent & Fast Bypass (`decision_first`)** | `chiplet_tuner/agents/model_level.py`<br>`chiplet_tuner/agents/analysis_policy.py` | Evaluates global $L$, $E$, and die/IO/substrate $MC$ shares; sets `candidate_layers=[]` to bypass operator inspection on coarse system moves |
| **On-Demand Operator Agent** | `chiplet_tuner/agents/layer_level.py` | Invoked only when `candidate_layers != []` to inspect post-TP GEMM shapes $(M, N, K)$, reuse patterns, stall shares, and physical chiplet placements |
| **Solution Agent & Deduplication** | `chiplet_tuner/agents/solution.py` | Synthesizes coordinated global or per-chiplet updates validated against $\mathcal{Z}$ and deduplicated against $\mathcal{H}$ |
| **Report Query Tools & Scalar Gating** | `chiplet_tuner/tools/analysis_tools.py`<br>`chiplet_tuner/tools/evidence_policy.py` | Cached tool groups distinguishing cumulative operator work from schedule makespan, plus `--evidence-mode scalar` gating for `Beacon-Scalar` |
| **Context-Matched Dual-Channel RAG** | `chiplet_tuner/rag/vector_store.py`<br>`chiplet_tuner/rag/embeddings.py` | Stores transitions indexed by $v_{\mathrm{case}} = [\sqrt{\alpha}\,\mathrm{Emb}(d) \,\|\, \sqrt{1-\alpha}\,\mathrm{Emb}(z_{\mathrm{base}})]$ ($\alpha=0.75$) |
| **LangGraph Loop & SQLite Resume** | `chiplet_tuner/pipeline/langgraph_tuner.py`<br>`chiplet_tuner/core/search_budget.py` | Outer state graph with SQLite checkpoint `--resume` and strict wall-clock budget accounting $t_k \le T$ |
| **Two-Level Evaluator Interface** | `chiplet_tuner/simulators/base.py`<br>`chiplet_tuner/simulators/compass.py` | Simulator-agnostic `SimulatorAdapter` (`GenericFileEvaluationAdapter` & external `CompassSimulatorAdapter`); see [`docs/evaluator_interface.md`](docs/evaluator_interface.md) |
| **Matched-Budget Baselines** | `baseline/` | `Random`, `HASCO-BO` (`hasco_scalar`), `Chiplet-Gym PPO` (`ppo`), and `AgenticDSE-Adapted` (`agentic_scalar`) |

---

## Repository Structure

```text
.
├── chiplet_tuner/                  # Core Beacon hierarchical multi-agent DSE framework
├── baseline/                       # Adapted baselines & baseline unit tests
├── configs/                        # Representative LLM workloads & initial designs z_0
│   ├── llama3_1_8b_64tops/         # Llama 3.1-8B (64 TOPS, 1 h budget)
│   ├── qwen3_14b_512tops/          # Qwen3-14B (512 TOPS, 2 h budget)
│   └── mixtral_8x22b_2048tops/     # Mixtral-8x22B (2048 TOPS, 4 h budget, Top-2 router traces)
├── docs/
│   └── evaluator_interface.md      # Specification of the two-level evaluator contract
├── examples/
│   └── sample_eval/                # Precomputed multi-chiplet evaluation artifacts for offline demo
├── scripts/
│   ├── run_offline_demo.sh         # Instant standalone demo (no external simulator or API key needed)
│   ├── run_beacon.sh               # Resilient Beacon runner with automatic checkpoint resume
│   ├── run_ablation.sh             # Controlled ablations (Beacon-Scalar, Always-Stage, Single-Agent)
│   └── run_baselines.sh            # Matched-budget baseline runner
├── tests/                          # Standalone unit tests for Beacon's core mechanisms
├── main.py                         # CLI entry wrapper
├── key.example.json                # Sanitized OpenAI-compatible API config template
├── requirements.txt                # Core Beacon dependencies
└── requirements-baselines.txt      # Additional dependencies for BO / PPO / tree-ensemble baselines
```

---

## Installation

```bash
conda create -n beacon python=3.12 -y
conda activate beacon
pip install -r requirements.txt

# Optional: install surrogate / RL dependencies if running baselines
pip install -r requirements-baselines.txt
```

---

## Quick Verification (No External Simulator or API Key Required)

### 1. Run Unit Tests
Verify hierarchical `decision_first` scoping, reference state tracking, directional coverage, report timing/shape semantics, evidence gating, and baseline search loops:

```bash
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s baseline/tests -v
```

### 2. Run Standalone Offline Demo (`examples/sample_eval/`)
Run the full three-agent diagnosis, cached report tools, and RAG proposal pipeline on precomputed multi-chiplet evaluation artifacts using the deterministic offline mock LLM (or pass `LLM_CONFIG=key.json LLM_PROFILE=...` to inspect live LLM reasoning):

```bash
bash scripts/run_offline_demo.sh
```

---

## Running Iterative Multi-Chiplet Exploration

Iterative DSE invokes an external multi-chiplet evaluator binary (e.g., GEMINI/Compass compiled at `$COMPASS_ROOT/build/compass`) via `CompassSimulatorAdapter`, or any custom simulator implementing `SimulatorAdapter` (see [`docs/evaluator_interface.md`](docs/evaluator_interface.md)).

### 1. Configure LLM Endpoint
```bash
cp key.example.json key.json
# Fill in your OpenAI-compatible base URL, model name, and API key in key.json
```

### 2. Run Beacon Across Representative Workloads
```bash
# Llama 3.1-8B (64 TOPS, 1 h wall-clock budget)
COMPASS_ROOT=/path/to/Compass LLM_CONFIG=key.json LLM_PROFILE=example-profile \
  bash scripts/run_beacon.sh llama3_1_8b_64tops

# Qwen3-14B (512 TOPS, 2 h wall-clock budget)
COMPASS_ROOT=/path/to/Compass LLM_CONFIG=key.json LLM_PROFILE=example-profile \
  bash scripts/run_beacon.sh qwen3_14b_512tops

# Mixtral-8x22B (2048 TOPS, 4 h wall-clock budget)
COMPASS_ROOT=/path/to/Compass LLM_CONFIG=key.json LLM_PROFILE=example-profile \
  bash scripts/run_beacon.sh mixtral_8x22b_2048tops
```
If an API connection drops mid-run, `scripts/run_beacon.sh` automatically resumes from `<output-dir>/langgraph_checkpoints.sqlite` while preserving the cumulative wall-clock budget in `search_clock.json`.

### 3. Run Controlled Ablations & Baselines
```bash
# Controlled ablations: beacon_scalar | always_stage | single_agent
COMPASS_ROOT=/path/to/Compass LLM_CONFIG=key.json LLM_PROFILE=example-profile \
  bash scripts/run_ablation.sh beacon_scalar qwen3_14b_512tops

COMPASS_ROOT=/path/to/Compass LLM_CONFIG=key.json LLM_PROFILE=example-profile \
  bash scripts/run_ablation.sh always_stage qwen3_14b_512tops

# Matched-budget baselines: random | hasco_scalar | ppo | agentic_scalar
COMPASS_ROOT=/path/to/Compass \
  bash scripts/run_baselines.sh hasco_scalar qwen3_14b_512tops
```

---

## License

Released under the [MIT License](LICENSE).
