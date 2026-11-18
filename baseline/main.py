from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict, Sequence

from baseline.actions import UnifiedActionSpace
from baseline.bayesian_optimization import BayesianOptimizationStrategy
from baseline.online_rl import OnlineRLStrategy
from baseline.random_search import RandomSearchStrategy
from baseline.search import BaselineSearchRunner
from chiplet_tuner.core.io import read_json
from chiplet_tuner.core.search_space import (
    BUFFER_SIZE_CANDIDATES,
    COMPUTE_UNIT_CANDIDATES,
    make_hardware_search_space,
)
from chiplet_tuner.simulators.compass import CompassSimulatorAdapter
from chiplet_tuner.simulators.compass_config import (
    COMPASS_DATASETS,
    COMPASS_SCALE_MODEL_INFO,
    COMPASS_WORKLOAD_REQ_INFO,
    write_compass_initial_hardware,
    write_compass_search_config,
)


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Random, Bayesian, PPO and AgenticDSE-adapted baselines for chiplet DSE."
    )
    parser.add_argument("--method", required=True, choices=["random", "bo", "rl", "ppo", "hasco_scalar", "agentic_scalar"])
    parser.add_argument("--evaluations", type=int, default=100, help="Simulator calls after the shared initial point.")
    parser.add_argument(
        "--time-budget-seconds", type=float, default=None,
        help="Total wall-clock budget including setup, decisions, training and evaluation. "
        "Only evaluations completed by the deadline count; --evaluations remains a safety cap.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "baseline" / "runs" / time.strftime("%Y%m%d_%H%M%S")),
    )
    parser.add_argument("--compass-root", default=str(ROOT / "Compass"))
    parser.add_argument("--compass-binary", type=Path, default=None,
                        help="Explicit Compass binary; defaults to <compass-root>/build/compass.")
    parser.add_argument("--search-config", default=None, help="Use an existing Compass search config instead of generating one.")
    parser.add_argument("--hardware", default=None, help="Use an existing initial hardware JSON instead of generating one.")

    parser.add_argument("--workload", default="chunked_prefill", choices=sorted(COMPASS_WORKLOAD_REQ_INFO))
    parser.add_argument("--dataset", default="govreport", choices=sorted(COMPASS_DATASETS))
    parser.add_argument("--compute-scale", default="64", choices=sorted(COMPASS_SCALE_MODEL_INFO))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--chunked-prefill-size", type=int, default=None)
    parser.add_argument(
        "--tensor-parall",
        "--tensor-parallel",
        dest="tensor_parall",
        type=int,
        default=None,
        help="Initial tensor-parallel shard count; remains tunable during baseline search.",
    )
    parser.add_argument(
        "--qkv-projection-mode",
        choices=["separate", "fused_tp"],
        default="fused_tp",
        help="QKV projection representation used by the generated Compass model.",
    )
    parser.add_argument(
        "--micro-batch",
        type=int,
        default=None,
        help="Initial micro-batch size; remains tunable over legal divisors of batch_size.",
    )
    parser.add_argument("--initial-num-chiplets", type=int, default=None)
    parser.add_argument("--initial-compute-units", type=int, choices=COMPUTE_UNIT_CANDIDATES, default=None)
    parser.add_argument("--initial-buffer-size", type=int, choices=BUFFER_SIZE_CANDIDATES, default=None)
    parser.add_argument("--initial-chiplet-type", choices=["balanced", "ws", "os"], default="balanced")

    parser.add_argument("--bo-warmup", type=int, default=6)
    parser.add_argument("--bo-candidate-pool", type=int, default=2048)
    parser.add_argument("--bo-max-parents", type=int, default=16)
    parser.add_argument("--bo-fit-maxiter", type=int, default=50)
    parser.add_argument("--optimizer-threads", type=int, default=1)
    parser.add_argument("--rl-update-interval", type=int, default=8)
    parser.add_argument("--rl-update-epochs", type=int, default=4)
    parser.add_argument("--ppo-rollout-steps", type=int, default=16)
    parser.add_argument("--ppo-batch-size", type=int, default=8)
    parser.add_argument("--ppo-learning-rate", type=float, default=3e-4)
    parser.add_argument("--ppo-entropy-coefficient", type=float, default=0.1)
    parser.add_argument("--llm-config", default=None)
    parser.add_argument("--llm-profile", default=None)
    parser.add_argument("--llm-reasoning-effort", default=None,
                        help="Optional provider-supported reasoning effort; match Beacon in comparisons.")
    parser.add_argument("--agentic-phase-evaluations", type=int, default=12)
    parser.add_argument("--agentic-warmup", type=int, default=5)
    parser.add_argument("--agentic-candidate-pool", type=int, default=512)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    started_at = time.perf_counter()
    args = build_parser().parse_args(argv)
    if args.optimizer_threads < 1:
        raise ValueError("optimizer-threads must be positive")
    if args.method in {"ppo", "hasco_scalar"}:
        import torch
        torch.set_num_threads(args.optimizer_threads)
    output_dir = Path(args.output_dir).resolve()
    if (output_dir / "history.jsonl").exists():
        raise FileExistsError(
            f"{output_dir / 'history.jsonl'} already exists. Use a new output directory to avoid mixing runs."
        )
    inputs_dir = output_dir / "baseline_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    compass_root = Path(args.compass_root).resolve()

    search_config_path = _prepare_search_config(args, compass_root, inputs_dir)
    search_config = read_json(search_config_path)
    model_type = str(search_config.get("model_info", {}).get("type", ""))
    search_space = make_hardware_search_space(
        task_type=args.workload,
        compute_scale=args.compute_scale,
        model_type=model_type,
        batch_size=int(search_config.get("batch_size", 128)),
    )
    hardware = _prepare_hardware(args, inputs_dir, search_config)

    action_space = UnifiedActionSpace(search_space)
    strategy = _make_strategy(args, action_space, search_space)
    simulator = CompassSimulatorAdapter(
        compass_root=compass_root,
        search_config_path=search_config_path,
        output_root=output_dir / "evaluations",
        search_space=search_space,
        binary_path=args.compass_binary,
    )
    runner = BaselineSearchRunner(
        simulator=simulator,
        strategy=strategy,
        initial_hardware=hardware,
        search_space=search_space,
        output_dir=output_dir,
        evaluations=args.evaluations,
        time_budget_seconds=args.time_budget_seconds,
        started_at=started_at,
    )
    if args.method == "ppo":
        from baseline.chiplet_gym_ppo import PPOSettings, run_ppo
        summary = run_ppo(runner, PPOSettings(
            seed=args.seed, rollout_steps=args.ppo_rollout_steps,
            batch_size=args.ppo_batch_size,
            learning_rate=args.ppo_learning_rate,
            entropy_coefficient=args.ppo_entropy_coefficient,
        ))
    else:
        summary = runner.run()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _prepare_search_config(args: argparse.Namespace, compass_root: Path, inputs_dir: Path) -> Path:
    if args.search_config:
        return Path(args.search_config).resolve()
    return write_compass_search_config(
        compass_root=compass_root,
        output_dir=inputs_dir,
        workload=args.workload,
        dataset=args.dataset,
        compute_scale=args.compute_scale,
        chunked_prefill_size=args.chunked_prefill_size,
        batch_size=args.batch_size,
        qkv_projection_mode=args.qkv_projection_mode,
    )


def _prepare_hardware(
    args: argparse.Namespace,
    inputs_dir: Path,
    search_config: Dict[str, Any],
) -> Dict[str, Any]:
    if args.hardware:
        return read_json(Path(args.hardware).resolve())
    hardware_path = write_compass_initial_hardware(
        output_dir=inputs_dir,
        task_type=args.workload,
        compute_scale=args.compute_scale,
        num_chiplets=args.initial_num_chiplets,
        compute_units=args.initial_compute_units,
        buffer_size=args.initial_buffer_size,
        chiplet_type=args.initial_chiplet_type,
        tensor_parall=args.tensor_parall,
        micro_batch=args.micro_batch,
        batch_size=int(search_config.get("batch_size", 128)),
    )
    return read_json(hardware_path)


def _make_strategy(
    args: argparse.Namespace,
    action_space: UnifiedActionSpace,
    search_space: Dict[str, Any],
) -> Any:
    if args.method == "agentic_scalar":
        from baseline.agentic_controller import AgenticController
        from baseline.agentic_search import AgenticPhaseStrategy
        from chiplet_tuner.cli import resolve_llm_config
        from chiplet_tuner.llm.clients import create_llm_client

        if not args.llm_config:
            raise ValueError("agentic_scalar requires --llm-config and a real LLM profile")
        config = resolve_llm_config(SimpleNamespace(
            llm_config=args.llm_config, llm_profile=args.llm_profile,
            llm_api_key=None, llm_base_url=None, llm_model=None,
            llm_temperature=0.2, llm_max_tokens=8192, llm_reasoning_effort=args.llm_reasoning_effort,
        ))
        controller = AgenticController(create_llm_client(config),
                                       Path(args.output_dir) / "agentic", config.model)
        return AgenticPhaseStrategy(
            search_space, controller, seed=args.seed,
            phase_evaluations=args.agentic_phase_evaluations,
            warmup=args.agentic_warmup, candidate_pool=args.agentic_candidate_pool,
        )
    if args.method == "hasco_scalar":
        from baseline.hasco_bo import HascoScalarBOStrategy
        return HascoScalarBOStrategy(
            search_space, seed=args.seed, warmup=args.bo_warmup,
            candidate_pool_size=args.bo_candidate_pool, fit_maxiter=args.bo_fit_maxiter,
        )
    if args.method == "ppo":
        return SimpleNamespace(name="chiplet_gym_derived_ppo")
    if args.method == "random":
        return RandomSearchStrategy(action_space, seed=args.seed, search_space=search_space)
    if args.method == "bo":
        return BayesianOptimizationStrategy(
            action_space,
            search_space,
            seed=args.seed,
            warmup=args.bo_warmup,
            candidate_pool_size=args.bo_candidate_pool,
            max_parents=args.bo_max_parents,
        )
    return OnlineRLStrategy(
        action_space,
        search_space,
        seed=args.seed,
        update_interval=args.rl_update_interval,
        update_epochs=args.rl_update_epochs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
