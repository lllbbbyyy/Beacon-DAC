from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Set

import numpy as np

from baseline.actions import ACTION_CATEGORIES, ActionCandidate, UnifiedActionSpace
from baseline.encoding import action_vector, policy_feature, state_vector
from baseline.search import SearchRecord


@dataclass
class _PendingDecision:
    parent_iteration: int
    state: np.ndarray
    policy_features: np.ndarray
    chosen_index: int
    old_probability: float


@dataclass
class _Transition:
    state: np.ndarray
    policy_features: np.ndarray
    chosen_index: int
    old_probability: float
    reward: float


class OnlineRLStrategy:
    """Masked online actor-critic using PPO-style clipped policy updates."""

    name = "online_rl"

    def __init__(
        self,
        action_space: UnifiedActionSpace,
        search_space: Mapping[str, Any],
        seed: int = 42,
        update_interval: int = 8,
        update_epochs: int = 4,
        policy_learning_rate: float = 0.01,
        value_learning_rate: float = 0.03,
        clip_ratio: float = 0.2,
        temperature: float = 1.0,
    ) -> None:
        self.action_space = action_space
        self.search_space = dict(search_space)
        self.rng = np.random.default_rng(seed)
        self.update_interval = max(1, update_interval)
        self.update_epochs = max(1, update_epochs)
        self.policy_learning_rate = policy_learning_rate
        self.value_learning_rate = value_learning_rate
        self.clip_ratio = clip_ratio
        self.temperature = max(temperature, 1e-3)
        self.current_iteration: int | None = None
        self.policy_weights: np.ndarray | None = None
        self.value_weights: np.ndarray | None = None
        self.pending: Dict[str, _PendingDecision] = {}
        self.transitions: List[_Transition] = []

    def propose(
        self,
        history: Sequence[SearchRecord],
        forbidden_fingerprints: Set[str],
    ) -> ActionCandidate:
        parent, candidates = self._select_parent_and_candidates(history, forbidden_fingerprints)
        initial = history[0]
        best = min(history, key=lambda record: record.objective)
        state = state_vector(parent, initial, best, self.search_space)
        features = np.stack(
            [
                policy_feature(
                    state,
                    action_vector(candidate, parent.hardware, self.search_space),
                    len(ACTION_CATEGORIES),
                )
                for candidate in candidates
            ]
        )
        self._ensure_weights(state.size, features.shape[1])
        probabilities = self._probabilities(features)
        chosen_index = int(self.rng.choice(len(candidates), p=probabilities))
        chosen = candidates[chosen_index]
        chosen.metadata.update(
            {
                "selection": "masked_online_policy",
                "valid_action_count": len(candidates),
                "action_probability": float(probabilities[chosen_index]),
            }
        )
        self.pending[chosen.fingerprint] = _PendingDecision(
            parent_iteration=parent.iteration,
            state=state,
            policy_features=features,
            chosen_index=chosen_index,
            old_probability=float(probabilities[chosen_index]),
        )
        return chosen

    def observe(
        self,
        parent: SearchRecord,
        result: SearchRecord,
        proposal: ActionCandidate,
    ) -> None:
        pending = self.pending.pop(proposal.fingerprint)
        reward = math.log(parent.objective / result.objective)
        if result.action is not None:
            result.action.setdefault("metadata", {})["reward_log_objective_improvement"] = reward
        self.transitions.append(
            _Transition(
                state=pending.state,
                policy_features=pending.policy_features,
                chosen_index=pending.chosen_index,
                old_probability=pending.old_probability,
                reward=reward,
            )
        )
        self.current_iteration = result.iteration
        if len(self.transitions) >= self.update_interval:
            self._update()

    def finalize(self, history: Sequence[SearchRecord]) -> None:
        if self.transitions:
            self._update()

    def _select_parent_and_candidates(
        self,
        history: Sequence[SearchRecord],
        forbidden_fingerprints: Set[str],
    ) -> tuple[SearchRecord, List[ActionCandidate]]:
        current = history[-1] if self.current_iteration is None else _record_by_iteration(history, self.current_iteration)
        best = min(history, key=lambda record: record.objective)
        ordered = [current, best, *reversed(history)]
        visited = set()
        for parent in ordered:
            if parent.iteration in visited:
                continue
            visited.add(parent.iteration)
            candidates = self.action_space.candidates(
                parent.hardware,
                parent_iteration=parent.iteration,
                forbidden_fingerprints=forbidden_fingerprints,
            )
            if candidates:
                return parent, candidates
        raise RuntimeError("The unified action space has no unevaluated hardware candidates.")

    def _ensure_weights(self, state_size: int, policy_feature_size: int) -> None:
        if self.policy_weights is None:
            self.policy_weights = np.zeros(policy_feature_size, dtype=float)
        if self.value_weights is None:
            self.value_weights = np.zeros(state_size + 1, dtype=float)

    def _probabilities(self, features: np.ndarray) -> np.ndarray:
        assert self.policy_weights is not None
        logits = features @ self.policy_weights / self.temperature
        logits -= logits.max()
        values = np.exp(np.clip(logits, -50.0, 50.0))
        return values / values.sum()

    def _update(self) -> None:
        assert self.policy_weights is not None
        assert self.value_weights is not None
        transitions = self.transitions
        self.transitions = []
        value_features = np.stack([np.concatenate([[1.0], item.state]) for item in transitions])
        rewards = np.asarray([item.reward for item in transitions], dtype=float)
        predictions = value_features @ self.value_weights
        advantages = rewards - predictions
        if len(advantages) > 1 and float(advantages.std()) > 1e-8:
            advantages = (advantages - advantages.mean()) / advantages.std()

        for _ in range(self.update_epochs):
            order = self.rng.permutation(len(transitions))
            for index in order:
                transition = transitions[int(index)]
                probabilities = self._probabilities(transition.policy_features)
                chosen_probability = float(probabilities[transition.chosen_index])
                ratio = chosen_probability / max(transition.old_probability, 1e-12)
                advantage = float(advantages[index])
                clipped = (advantage >= 0.0 and ratio > 1.0 + self.clip_ratio) or (
                    advantage < 0.0 and ratio < 1.0 - self.clip_ratio
                )
                if not clipped:
                    expected_feature = probabilities @ transition.policy_features
                    gradient = transition.policy_features[transition.chosen_index] - expected_feature
                    self.policy_weights += self.policy_learning_rate * advantage * ratio * gradient

                value_feature = value_features[index]
                value_error = rewards[index] - float(value_feature @ self.value_weights)
                self.value_weights += self.value_learning_rate * value_error * value_feature


def _record_by_iteration(history: Sequence[SearchRecord], iteration: int) -> SearchRecord:
    for record in history:
        if record.iteration == iteration:
            return record
    raise KeyError(f"Unknown current iteration: {iteration}")
