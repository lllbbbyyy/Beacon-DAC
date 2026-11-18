"""HASCO-derived scalar hardware BO with a shared inner mapping evaluator.

Retains Sobol initialization and sequential GP acquisition from HASCO's outer
loop. Replaces its multiobjective EHVI by log-EI for the common scalar objective,
and searches a finite global/local pool of heterogeneous configurations.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence, Set

import botorch
import numpy as np
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from baseline.actions import ActionCandidate, UnifiedActionSpace, hardware_fingerprint
from baseline.config_encoding import FullDesignCodec
from baseline.search import SearchRecord
from chiplet_tuner.core.objective import hardware_compute_scale


class HascoScalarBOStrategy:
    name = "hasco_derived_scalar_bo"

    def __init__(
        self, search_space: Mapping[str, Any], seed: int = 42,
        warmup: int = 6, candidate_pool_size: int = 512, fit_maxiter: int = 50,
    ):
        if warmup < 2 or candidate_pool_size < 2 or fit_maxiter < 1:
            raise ValueError("BO requires warmup/pool >= 2 and fit_maxiter >= 1")
        self.search_space = dict(search_space)
        self.seed = seed
        self.warmup = warmup
        self.candidate_pool_size = candidate_pool_size
        self.fit_maxiter = fit_maxiter
        self.codec = None
        self.sobol = None
        self.rng = np.random.default_rng(seed)
        self.actions = UnifiedActionSpace(search_space)

    def config(self) -> dict:
        return {
            "source": "https://github.com/pku-liang/HASCO",
            "source_commit": "7a9d285e1e1df8f045579f1122543266bbde5b7b",
            "adaptation": "scalar_objective_compact_gp_global_local_pool",
            "botorch_version": botorch.__version__, "seed": self.seed,
            "warmup": self.warmup, "candidate_pool_size": self.candidate_pool_size,
            "fit_maxiter": self.fit_maxiter, "surrogate": "SingleTaskGP",
            "target": "capped_negative_log_objective", "acquisition": "LogExpectedImprovement",
            "encoding": "compact_summary_and_system_features",
        }

    def _initialize(self, history: Sequence[SearchRecord]) -> None:
        if self.codec is None:
            self.codec = FullDesignCodec(history[0].hardware, self.search_space)
            self.sobol = torch.quasirandom.SobolEngine(
                len(self.codec.nvec), scramble=True, seed=self.seed,
            )

    def _gp_features(self, hardware: Mapping[str, Any]) -> np.ndarray:
        assert self.codec is not None
        count = int(hardware["num_chiplets"])
        count_norm = self.codec.counts.index(count) / max(1, len(self.codec.counts) - 1)
        scale = max(hardware_compute_scale(hardware, self.search_space), 1e-6)
        target_raw = self.search_space.get("target_compute_scale")
        target = float(target_raw) if target_raw is not None and float(target_raw) > 0 else scale
        log_ratio = math.log2(scale / target)
        signed_scale = float(np.clip(0.5 + 0.125 * log_ratio, 0.0, 1.0))
        abs_scale = float(np.clip(abs(log_ratio) / 4.0, 0.0, 1.0))

        comp_denom = max(1, len(self.codec.compute) - 1)
        buf_denom = max(1, len(self.codec.buffers) - 1)
        comp_vals = np.array(
            [self.codec.compute.index(int(c["compute_units"])) / comp_denom for c in hardware["chiplets"]],
            dtype=np.float64,
        )
        buf_vals = np.array(
            [self.codec.buffers.index(int(c["buffer_size"])) / buf_denom for c in hardware["chiplets"]],
            dtype=np.float64,
        )
        ws_flags = np.array(
            [1.0 if str(c["type"]).lower() == "ws" else 0.0 for c in hardware["chiplets"]],
            dtype=np.float64,
        )
        half = max(1, len(ws_flags) // 2)
        sys_vals = [
            values.index(hardware[key]) / max(1, len(values) - 1)
            for key, values in self.codec.system.items()
        ]
        return np.array(
            [
                count_norm,
                signed_scale,
                abs_scale,
                float(comp_vals.mean()),
                float(comp_vals.std()),
                float(comp_vals.min()),
                float(comp_vals.max()),
                float(buf_vals.mean()),
                float(buf_vals.min()),
                float(buf_vals.max()),
                float(ws_flags.mean()),
                float(ws_flags[:half].mean()),
                *sys_vals,
            ],
            dtype=np.float64,
        )

    def _global_pool(
        self, size: int, forbidden: Set[str], max_scale_ratio: float | None = 4.0,
    ) -> list[dict]:
        result = []
        fallback = []
        seen = set(forbidden)
        target_raw = self.search_space.get("target_compute_scale")
        target = float(target_raw) if target_raw is not None and float(target_raw) > 0 else None
        for _ in range(32):
            draws = self.sobol.draw(max(8, (size - len(result)) * 4)).numpy()
            for draw in draws:
                action = np.minimum((draw * self.codec.nvec).astype(np.int64), self.codec.nvec - 1)
                hardware = self.codec.decode(action)
                fingerprint = hardware_fingerprint(hardware)
                if fingerprint in seen:
                    continue
                if target is not None and max_scale_ratio is not None:
                    scale = max(hardware_compute_scale(hardware, self.search_space), 1e-6)
                    if max(scale, target) / min(scale, target) > max_scale_ratio:
                        if len(fallback) < size:
                            fallback.append((fingerprint, hardware))
                        continue
                seen.add(fingerprint)
                result.append(hardware)
                if len(result) == size:
                    return result
        for fingerprint, hardware in fallback:
            if fingerprint not in seen:
                seen.add(fingerprint)
                result.append(hardware)
                if len(result) == size:
                    break
        return result

    def propose(self, history: Sequence[SearchRecord], forbidden_fingerprints: Set[str]) -> ActionCandidate:
        self._initialize(history)
        best = min(history, key=lambda record: record.objective)
        if len(history) <= self.warmup:
            if len(history) % 2 == 1:
                neighbors = self.actions.candidates(best.hardware, best.iteration, set(forbidden_fingerprints))
                if neighbors:
                    draw = float(self.sobol.draw(1).numpy()[0, 0])
                    picked = neighbors[min(int(draw * len(neighbors)), len(neighbors) - 1)]
                    return self._proposal(picked.hardware, best.iteration, {"selection": "sobol_warmup"})
            pool = self._global_pool(1, forbidden_fingerprints, max_scale_ratio=4.0)
            if not pool:
                pool = self._global_pool(1, forbidden_fingerprints, max_scale_ratio=None)
            if not pool:
                raise RuntimeError("No unevaluated Sobol candidate found")
            return self._proposal(pool[0], best.iteration, {"selection": "sobol_warmup"})

        pool = self._global_pool(self.candidate_pool_size // 2, forbidden_fingerprints, max_scale_ratio=4.0)
        seen = set(forbidden_fingerprints) | {hardware_fingerprint(h) for h in pool}
        local = []
        for parent in sorted(history, key=lambda record: record.objective)[:6]:
            for candidate in self.actions.candidates(parent.hardware, parent.iteration, seen):
                seen.add(candidate.fingerprint)
                local.append(candidate.hardware)
        self.rng.shuffle(local)
        pool.extend(local[:self.candidate_pool_size - len(pool)])
        if not pool:
            pool = self._global_pool(self.candidate_pool_size, forbidden_fingerprints, max_scale_ratio=None)
        if not pool:
            raise RuntimeError("No unevaluated acquisition candidate found")

        raw_objectives = np.array([r.objective for r in history], dtype=np.float64)
        capped_objectives = np.minimum(raw_objectives, 50.0 * float(raw_objectives.min()))
        train_x = torch.as_tensor(np.stack([self._gp_features(r.hardware) for r in history]), dtype=torch.double)
        train_y = -torch.log(torch.tensor(capped_objectives[:, None], dtype=torch.double))
        candidates = torch.as_tensor(np.stack([self._gp_features(h) for h in pool]), dtype=torch.double)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + len(history))
            model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": self.fit_maxiter}})
            acquisition = LogExpectedImprovement(model, best_f=train_y.max())
            with torch.no_grad():
                values = torch.cat([acquisition(chunk.unsqueeze(1)) for chunk in candidates.split(128)])
                if not torch.isfinite(values).all():
                    raise FloatingPointError("Non-finite BO acquisition values")
                index = int(values.argmax())
                posterior = model.posterior(candidates[index:index + 1])
                mean = float(posterior.mean.item())
                std = float(posterior.variance.sqrt().item())
        return self._proposal(pool[index], best.iteration, {
            "selection": "gp_log_ei", "candidate_pool_size": len(pool),
            "log_expected_improvement": float(values[index]),
            "predicted_log_objective": -mean, "predicted_log_std": std,
            "training_designs": len(history),
        })

    @staticmethod
    def _proposal(hardware, parent_iteration, metadata):
        return ActionCandidate(
            "global_bo_configuration", "joint_configuration",
            "select a full hardware/mapping configuration", hardware,
            parent_iteration, metadata,
        )

    def observe(self, parent, result, proposal) -> None:
        pass

    def finalize(self, history) -> None:
        pass
