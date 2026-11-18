"""Scalar adaptation of AgenticDSE's CV-selected surrogate ensemble.

Preserves K-fold MAPE selection and static softmax weighting. The shared scalar
objective replaces the original pair of objectives, and EI replaces HVI.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr, softmax
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import KFold


class ScalarEnsemble:
    def __init__(self, seed=42, folds=5, members=3, beta=1.0, trees=100):
        if folds < 2 or not 2 <= members <= 4 or trees < 1:
            raise ValueError("Require folds >= 2, members in [2, 4], and trees >= 1")
        if not np.isfinite(beta) or beta < 0:
            raise ValueError("Ensemble beta must be finite and nonnegative")
        self.seed, self.folds, self.members, self.beta = seed, folds, members, beta
        self.candidates = {
            "RF": RandomForestRegressor(n_estimators=trees, random_state=seed, n_jobs=1),
            "ET": ExtraTreesRegressor(n_estimators=trees, random_state=seed, n_jobs=1),
            "GBRT_depth2": GradientBoostingRegressor(
                n_estimators=trees, max_depth=2, random_state=seed),
            "GBRT_depth3": GradientBoostingRegressor(
                n_estimators=trees, max_depth=3, random_state=seed),
        }
        self.models = []
        self.weights = None
        self.report = None

    def fit(self, x, objective):
        x = np.asarray(x, dtype=float)
        objective = np.asarray(objective, dtype=float)
        if (x.ndim != 2 or objective.shape != (len(x),) or len(x) < self.folds
                or not np.isfinite(x).all() or not np.isfinite(objective).all()
                or np.any(objective <= 0)):
            raise ValueError("Need finite features, positive objectives and enough CV samples")
        # A common positive scale preserves MAPE while avoiding huge physical units.
        self.scale = float(objective[0])
        y = objective / self.scale
        if not np.isfinite(y).all():
            raise ValueError("Objective normalization overflow")
        folds = list(KFold(self.folds, shuffle=True, random_state=self.seed).split(x))
        scores = []
        for name, estimator in self.candidates.items():
            errors = []
            for train, test in folds:
                model = clone(estimator).fit(x[train], y[train])
                errors.append(float(mean_absolute_percentage_error(y[test], model.predict(x[test]))))
            scores.append({"name": name, "cv_mape": float(np.mean(errors)), "fold_mape": errors})
        scores.sort(key=lambda item: item["cv_mape"])
        selected = scores[:self.members]
        self.weights = softmax(-self.beta * np.array([item["cv_mape"] for item in selected]))
        self.models = [clone(self.candidates[item["name"]]).fit(x, y) for item in selected]
        self.report = {
            "selection": "kfold_mape", "folds": self.folds,
            "training_designs": len(x), "objective_scale": self.scale,
            "leaderboard": scores, "selected": [item["name"] for item in selected],
            "weights": self.weights.tolist(), "weighting": "static_softmax_negative_mape",
            "beta": self.beta, "uncertainty": "weighted_between_model_disagreement",
            "acquisition": "scalar_expected_improvement",
        }
        return self

    def predict(self, x):
        if not self.models:
            raise RuntimeError("Fit the ensemble before prediction")
        predictions = np.stack([model.predict(x) for model in self.models])
        mean = np.average(predictions, axis=0, weights=self.weights)
        std = np.sqrt(np.average((predictions - mean) ** 2, axis=0, weights=self.weights))
        return mean, std

    def expected_improvement(self, x, best_objective):
        if not np.isfinite(best_objective) or best_objective <= 0:
            raise ValueError("Incumbent objective must be finite and positive")
        mean, std = self.predict(x)
        improvement = best_objective / self.scale - mean
        values = np.maximum(improvement, 0)
        nonzero = std > 0
        z = improvement[nonzero] / std[nonzero]
        values[nonzero] = (improvement[nonzero] * ndtr(z)
                           + std[nonzero] * np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi))
        return np.maximum(values, 0)
