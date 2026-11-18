"""Chiplet-Gym-derived MLP PPO using Stable-Baselines3 and the common evaluator.

Retains PPO, two 64-unit tanh layers, and two-step episodes. Hardware encoding,
reward, and rollout length are adapted to the heterogeneous, costly evaluator.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from baseline.config_encoding import FullDesignCodec
from baseline.ppo_distribution import HardwarePolicy
from baseline.search import BaselineSearchRunner
from chiplet_tuner.core.io import write_json
from chiplet_tuner.simulators.base import MappingEvaluationUnavailable


@dataclass
class PPOSettings:
    seed: int = 42
    rollout_steps: int = 16
    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = 3e-4
    entropy_coefficient: float = 0.1
    episode_length: int = 2
    mapping_failure_reward: float = -1.0
    target_kl: float = 0.02

    def validate(self) -> None:
        if self.rollout_steps < 2 or self.batch_size < 2:
            raise ValueError("PPO rollout and minibatch must contain at least two transitions")
        if self.batch_size > self.rollout_steps or self.rollout_steps % self.batch_size:
            raise ValueError("PPO minibatch must divide the rollout length")
        if self.epochs < 1 or self.episode_length < 1:
            raise ValueError("PPO epochs and episode length must be positive")
        if not math.isfinite(self.mapping_failure_reward) or self.mapping_failure_reward > -1:
            raise ValueError("Mapping failure reward must be finite and at most -1")
        if not math.isfinite(self.target_kl) or self.target_kl <= 0:
            raise ValueError("target_kl must be positive and finite")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not math.isfinite(self.entropy_coefficient) or self.entropy_coefficient < 0:
            raise ValueError("entropy_coefficient must be nonnegative and finite")


def objective_reward(initial: float, candidate: float) -> float:
    """Preserve objective ordering while keeping valid rewards above failure."""
    if not (math.isfinite(initial) and math.isfinite(candidate) and initial > 0 and candidate > 0):
        raise ValueError("Objectives must be positive and finite")
    improvement = math.log(initial) - math.log(candidate)
    return improvement / (1.0 + abs(improvement))


class _SearchStopped(Exception):
    pass


class HardwareEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, runner: BaselineSearchRunner, settings: PPOSettings):
        super().__init__()
        self.runner = runner
        self.settings = settings
        self.codec = FullDesignCodec(runner.initial_hardware, runner.search_space)
        self.action_space = gym.spaces.MultiDiscrete(self.codec.nvec)
        shape = self._observation().shape
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32)
        self.episode_steps = 0

    def _observation(self) -> np.ndarray:
        record = self.runner.history[-1]
        initial = self.runner.history[0]
        ratios = [math.log(record.metrics[key] / initial.metrics[key])
                  for key in ["latency", "energy", "mc", "objective"]]
        return np.concatenate([self.codec.observation(record.hardware), ratios]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_steps = 0
        # Episodes are search episodes, not new evaluations or a reset of search history.
        return self._observation(), {}

    def step(self, action):
        runner = self.runner
        attempts = len(runner.history) + len(runner.failed_evaluations)
        if runner._budget_expired() or attempts - 1 >= runner.evaluations:
            raise _SearchStopped()
        hardware = self.codec.decode(action)
        iteration = attempts
        action_record = {"action_id": "full_design", "category": "joint_configuration",
                         "metadata": {"selection": "ppo_multidiscrete", "design_indices": action.tolist()}}
        try:
            record = runner._evaluate(hardware, iteration, runner.history[-1].iteration, action_record)
        except MappingEvaluationUnavailable as exc:
            action_record["metadata"]["mapping_failure_reward"] = self.settings.mapping_failure_reward
            failure = runner._record_failure(iteration, hardware, action_record, exc)
            self.episode_steps += 1
            # A failed attempt terminates the search episode without a fabricated
            # L/E/MC observation. The last measured design remains the state.
            return self._observation(), self.settings.mapping_failure_reward, True, False, {
                "mapping_unavailable": True, "within_time_budget": failure["within_time_budget"],
            }
        reward = objective_reward(runner.history[0].objective, record.objective)
        record.action["metadata"]["reward_bounded_log_initial_improvement"] = reward
        record.action["metadata"]["repeated_design"] = any(
            old.fingerprint == record.fingerprint for old in runner.history
        )
        runner.history.append(record)
        runner._append_record(record)
        runner._write_best()
        self.episode_steps += 1
        return self._observation(), reward, self.episode_steps >= self.settings.episode_length, False, {
            "within_time_budget": record.within_time_budget,
        }


class _DeadlineCallback(BaseCallback):
    def __init__(self, runner):
        super().__init__()
        self.runner = runner

    def _on_step(self) -> bool:
        return not self.runner._budget_expired()


def _warm_start_policy_bias(model: PPO, codec: FullDesignCodec, initial_hardware: dict[str, Any]) -> None:
    """Center initial categorical logits gently around initial_hardware while preserving full exploration."""
    action_net = getattr(model.policy, "action_net", None)
    if action_net is None or not hasattr(action_net, "bias") or action_net.bias is None:
        return
    init_action = codec.encode(initial_hardware)
    dims = [int(d) for d in codec.nvec]
    with torch.no_grad():
        bias = action_net.bias
        cursor = 0
        # 1. Chiplet count: gentle locality prior around initial count
        count_dim = dims[0]
        idx0 = int(init_action[0])
        for j in range(count_dim):
            bias[cursor + j] = -1.1 * abs(j - idx0)
        cursor += count_dim
        # 2. Per-chiplet slots: center compute and buffer mildly around initial die capacity
        init_comp = int(init_action[1])
        init_buf = int(init_action[2])
        for slot in range(codec.max_count):
            comp_dim, buf_dim, type_dim = dims[1 + 3 * slot : 4 + 3 * slot]
            c_ref = int(init_action[1 + 3 * slot]) if slot < int(initial_hardware["num_chiplets"]) else init_comp
            b_ref = int(init_action[2 + 3 * slot]) if slot < int(initial_hardware["num_chiplets"]) else init_buf
            for j in range(comp_dim):
                bias[cursor + j] = -0.75 * abs(j - c_ref)
            cursor += comp_dim
            for j in range(buf_dim):
                bias[cursor + j] = -0.5 * abs(j - b_ref)
            cursor += buf_dim
            cursor += type_dim


def run_ppo(runner: BaselineSearchRunner, settings: PPOSettings) -> dict[str, Any]:
    settings.validate()
    if runner.started_at is None:
        runner.started_at = time.perf_counter()
    runner.output_dir.mkdir(parents=True, exist_ok=True)
    runner._write_run_config()
    write_json(runner.output_dir / "ppo_config.json", {
        **vars(settings), "stable_baselines3": stable_baselines3.__version__,
        "policy_hidden_layers": [64, 64], "activation": "tanh",
        "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2,
        "value_coefficient": 0.5, "device": "cpu",
        "action_encoding": "full_configuration_multidiscrete",
        "action_probability": "chiplet_count_plus_active_slots_plus_system; unused_slots_marginalized",
        "initial_prior": "gentle_locality_around_initial_hardware",
        "reward": "r / (1 + abs(r)), r = log(initial_objective) - log(candidate_objective)",
        "duplicate_policy": "reevaluate_and_charge",
        "mapping_failure_policy": "terminal_transition_fixed_reward_keep_last_measured_state",
        "source": "https://arxiv.org/html/2406.00858v1",
    })
    initial = runner._evaluate(runner.initial_hardware, 0, None, None)
    runner.history.append(initial)
    runner._append_record(initial)
    runner._write_best()
    model = None
    if not runner._budget_expired() and runner.evaluations:
        env = HardwareEnv(runner, settings)
        model = PPO(
            HardwarePolicy, env, device="cpu", seed=settings.seed,
            n_steps=settings.rollout_steps, batch_size=settings.batch_size,
            n_epochs=settings.epochs, learning_rate=settings.learning_rate,
            ent_coef=settings.entropy_coefficient, gamma=0.99,
            gae_lambda=0.95, clip_range=0.2, vf_coef=0.5, target_kl=settings.target_kl,
            policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]},
                           "activation_fn": torch.nn.Tanh, "chiplet_counts": env.codec.counts},
        )
        _warm_start_policy_bias(model, env.codec, runner.initial_hardware)
        model.set_logger(configure(str(runner.output_dir / "training"), ["csv", "json"]))
        try:
            model.learn(total_timesteps=runner.evaluations, callback=_DeadlineCallback(runner))
        except _SearchStopped:
            pass
        finally:
            env.close()
            model.logger.dump(step=model.num_timesteps)
            model.logger.close()
        model.save(str(runner.output_dir / "ppo_policy"))
    runner.stop_reason = "time_budget" if runner._budget_expired() else "evaluation_limit"
    summary = runner._summary()
    summary.update({
        "ppo_optimizer_epochs_completed": model._n_updates if model else 0,
        "ppo_transitions_collected": model.num_timesteps if model else 0,
        "unique_designs": len({record.fingerprint for record in runner.history}),
    })
    runner._write_csv()
    write_json(runner.output_dir / "summary.json", summary)
    return summary
