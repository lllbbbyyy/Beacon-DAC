"""Phase-wise search engine; the controller supplies agent-selected subspaces."""
from __future__ import annotations

import numpy as np

from baseline.actions import ActionCandidate, hardware_fingerprint
from baseline.agentic_ensemble import ScalarEnsemble
from baseline.agentic_region import PhaseRegion
from baseline.config_encoding import FullDesignCodec


class AgenticPhaseStrategy:
    name = "agenticdse_adapted_scalar"

    def __init__(self, search_space, controller, seed=42, phase_evaluations=12,
                 warmup=5, candidate_pool=512, folds=5, trees=100):
        if phase_evaluations < 1 or warmup < folds or candidate_pool < 2:
            raise ValueError("Require phase_evaluations >= 1, warmup >= folds, and pool >= 2")
        self.search_space = search_space
        self.controller = controller
        self.seed = seed
        self.phase_evaluations = phase_evaluations
        self.warmup = warmup
        self.candidate_pool = candidate_pool
        self.folds = folds
        self.trees = trees
        self.rng = np.random.default_rng(seed)
        self.region = None
        self.phase = -1
        self.proposals = 0
        self.phase_changes = {}
        self.next_phase_at = 0

    def config(self):
        return {
            "source": "https://github.com/pku-pacific-lab-team/AgenticDSE",
            "source_commit": "ff2c216781d6a7a51b9f2fda34ad52f83e22dc00",
            "adaptation": "heterogeneous_full_design_scalar_objective",
            "seed": self.seed, "phase_evaluations": self.phase_evaluations,
            "warmup": self.warmup, "candidate_pool": self.candidate_pool,
            "cv_folds": self.folds, "trees": self.trees,
            "controller": self.controller.config(),
            "surrogates": ["RF", "ET", "GBRT_depth2", "GBRT_depth3"],
            "selected_members": 3, "weighting": "static_softmax_negative_mape",
            "acquisition": "scalar_expected_improvement",
        }

    def propose(self, history, forbidden_fingerprints):
        if self.region is None:
            self.region = PhaseRegion(FullDesignCodec(history[0].hardware, self.search_space))
        if self.proposals >= self.next_phase_at:
            self._new_phase(history, "scheduled")
        pool = self._pool(forbidden_fingerprints)
        if not pool:
            self._new_phase(history, "no_unevaluated_candidate_found_in_current_region")
            pool = self._pool(forbidden_fingerprints)
        if not pool:
            raise RuntimeError("Agent replanning did not provide an unevaluated design")
        compatible = [record for record in history if self.region.contains(record.hardware)]
        metadata = {
            "phase": self.phase, "phase_changes": self.phase_changes,
            "compatible_history": len(compatible), "total_history": len(history),
            "candidate_pool_size": len(pool),
        }
        index = 0
        if len(compatible) < self.warmup:
            metadata["selection"] = "phase_random_initialization"
        else:
            codec = self.region.codec
            x = np.stack([codec.observation(record.hardware) for record in compatible])
            objective = np.array([record.objective for record in compatible])
            best_in_region = float(objective.min())
            capped_objective = np.minimum(objective, 10.0 * best_in_region)
            model = ScalarEnsemble(seed=self.seed, folds=self.folds, trees=self.trees).fit(x, capped_objective)
            points = np.stack([codec.observation(hardware) for hardware in pool])
            values = model.expected_improvement(points, best_in_region)
            if not np.isfinite(values).all():
                raise FloatingPointError("Non-finite ensemble acquisition")
            index = int(np.argmax(values))
            metadata.update(selection="ensemble_ei", ensemble=model.report,
                            expected_improvement=float(values[index]))
        self.proposals += 1
        parent = min(history, key=lambda record: record.objective)
        return ActionCandidate("phase_configuration", "joint_configuration",
                               "select a configuration within the agent-defined phase",
                               pool[index], parent.iteration, metadata)

    def _new_phase(self, history, reason):
        phase = self.phase + 1
        selections = self.controller.select_region(
            phase=phase, domains={k: list(v) for k, v in self.region.domains.items()},
            current=self.region.snapshot(), history=history, reason=reason,
        )
        self.phase_changes = self.region.update(selections)
        self.phase = phase
        self.next_phase_at = self.proposals + self.phase_evaluations

    def _pool(self, forbidden_fingerprints):
        small = self.region.enumerate_small()
        if small is not None:
            self.rng.shuffle(small)
            return [h for h in small if hardware_fingerprint(h) not in forbidden_fingerprints][:self.candidate_pool]
        pool, seen = [], set(forbidden_fingerprints)
        for _ in range(10 * self.candidate_pool):
            hardware = self.region.decode_unit(self.rng.random(len(self.region.codec.nvec)))
            fingerprint = hardware_fingerprint(hardware)
            if fingerprint not in seen:
                seen.add(fingerprint)
                pool.append(hardware)
                if len(pool) >= self.candidate_pool:
                    break
        return pool

    def observe(self, parent, result, proposal):
        pass

    def finalize(self, history):
        pass
