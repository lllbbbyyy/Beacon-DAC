"""Definitions shared by report-consuming search controllers."""

TIMING_FIELD_DEFINITIONS = {
    "gemm_shapes": "Exact execution-graph M,N,K after TP/graph splitting, before internal buffer tiling; A[M,K] @ B[K,N]. rhs=activation for attention operands, not parameter weights. Variants remain separate across micro-batches; example_batch_ids lists at most four IDs. Missing shapes in older reports are unknown, not inferred from names.",
    "aggregation": "Layer statistics pool all records with the same layerID/name across micro-batches and chiplets; they are not one GEMM execution.",
    "latency_sum": "Sum of recorded end-minus-begin durations, including overlapping executions. Not end-to-end latency or removable delay.",
    "latency_max": "Largest single recorded duration. latency_sum/latency_max does not measure serialization or TP efficiency; unequal micro-batches also change this ratio.",
    "latency_sum_share": "latency_sum divided by model makespan; may exceed 1. Not a partition of model latency.",
    "critical_end": "Latest recorded end for the layer, not a dependency-derived critical-path length or proof that this layer is a bottleneck.",
    "critical_layer": "The last-finishing recorded layer; this tool does not reconstruct the dependency critical path.",
    "time_breakdown": "Ratios of summed compute/NoP/DRAM service estimates. Transfer overlap and cross-chiplet concurrency mean they are not fractions of model makespan.",
    "avg_core_utilization": "Sum of recorded durations divided by represented cores times timeline span; includes transfer time. Not MAC utilization or evidence of compute saturation.",
}

TIMING_REPORT_TOOLS = frozenset({
    "build_execution_timeline", "aggregate_layer_loads", "inspect_layer_details",
    "summarize_layer_rank_views", "summarize_operator_groups",
})
