# Two-Level Evaluator Interface

Beacon is decoupled from internal simulator mechanics and communicates with any multi-chiplet evaluator through `SimulatorAdapter` (`chiplet_tuner/simulators/base.py`).

## 1. Candidate Input (`hardware.json`)

For each candidate design $z = (N, \{(t_i, c_i, b_i)\}_{i=1}^{N}, g_{\mathrm{hw}}, g_{\mathrm{par}}) \in \mathcal{Z}$, Beacon passes a validated dictionary specifying package topology, per-chiplet heterogeneous resources, system bandwidths, and parallelism parameters:

```json
{
  "num_chiplets": 4,
  "chip_x": 2,
  "chip_y": 2,
  "nop_bw": 64,
  "dram_bw": 32,
  "tensor_parall": 8,
  "micro_batch": 32,
  "chiplets": [
    {"type": "ws", "compute_units": 65536, "buffer_size": 8388608, "macs": "[128,512]"},
    {"type": "ws", "compute_units": 65536, "buffer_size": 8388608, "macs": "[128,512]"},
    {"type": "ws", "compute_units": 65536, "buffer_size": 8388608, "macs": "[128,512]"},
    {"type": "os", "compute_units": 65536, "buffer_size": 8388608, "macs": "[256,256]"}
  ]
}
```

## 2. Two-Level Standardized Output (`EvaluationResult`)

After inner spatial placement, tiling, and pipeline mapping simulation, the adapter returns an `EvaluationResult` containing:

### Level 1: System-Level Summary Metrics (Required)
- **`metrics`** (`search_res.csv`):
  - `latency`: end-to-end execution latency $L(z)$ (cycles)
  - `energy`: total package energy $E(z)$ (pJ)
  - `mc`: total manufacturing cost $MC(z)$
- **`detail_files["monetary_cost"]`** (`search_mc_detail.json`, optional breakdown):
  - `cost_compute`: compute-die silicon and yield cost
  - `cost_IO`: DRAM/NoP I/O die cost
  - `cost_os_overall`: organic package substrate cost
  - `cost_overall`: total package cost $MC(z)$

When `candidate_layers == []` (Fast Bypass), Beacon operates exclusively on Level-1 metrics and skips stage-level parsing entirely.

### Level 2: Stage-Level Execution Records (Queried On Demand Only)
Read by cached tools inside `AnalysisToolbox` only when the System Agent selects critical stages (`candidate_layers != []`) for the Operator Agent:
- **`detail_files["latency"]`** (`search_latency_detail.json`): JSON object keyed by physical chiplet ID (`"core0"`, `"core1"`, ...), where each list entry records:
  - `layerID`, `layerName`, `batchID`
  - `latencyBegin`, `latencyEnd`, `calcTime`, `dramTime`, `nocTime`
  - `gemm_shapes` (optional list of post-TP shapes `{"layer_id", "M", "N", "K", "rhs", "scope"}`)
- **`detail_files["energy"]`** (`search_energy_detail.json`): JSON object keyed by physical chiplet ID with `energy`, `calcEnergy`, `ubufEnergy`, `nocEnergy`, and `dramEnergy`.

See `examples/sample_eval/` for a complete working example of all five files, and `GenericFileEvaluationAdapter` in `chiplet_tuner/simulators/base.py` for a minimal reference implementation.
