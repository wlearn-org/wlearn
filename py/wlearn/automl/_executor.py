"""Executor matching JS automl/executor.js."""

import math

import numpy as np

from ._leaderboard import Leaderboard
from ._common import now, seed_for, partial_shuffle
from ._cv import get_scorer
from ._rng import make_lcg


class Executor:
    """Evaluation engine: evaluates candidates across CV folds, applies budgets."""

    def __init__(self, folds, scoring, X, y, time_limit_ms=0, seed=42):
        """
        Args:
            folds: list of (train_indices, test_indices) np.int32 arrays
            scoring: str or callable
            X: np.ndarray feature matrix
            y: np.ndarray labels
            time_limit_ms: global time limit (0 = no limit)
            seed: base seed for reproducibility
        """
        self._folds = folds
        self._scorer_fn = get_scorer(scoring)
        self._X = X
        self._y = y
        self._time_limit_ms = time_limit_ms
        self._seed = seed
        self._start_time = now()
        self._leaderboard = Leaderboard()

    @property
    def leaderboard(self):
        return self._leaderboard

    @property
    def is_timed_out(self):
        if self._time_limit_ms <= 0:
            return False
        return (now() - self._start_time) > self._time_limit_ms

    def evaluate_candidate(self, candidate_id, cls, params, budget=None):
        """Evaluate one candidate across all CV folds.

        Args:
            candidate_id: stable identifier
            cls: estimator class with create/fit/predict/dispose
            params: hyperparameters
            budget: optional dict with 'type' and 'value'

        Returns:
            CandidateResult dict
        """
        folds = self._folds
        scores = np.zeros(len(folds), dtype=np.float64)
        t0 = now()
        total_train_used = 0

        effective_params = self._apply_rounds_budget(cls, params, budget)

        for f, (train, test) in enumerate(folds):
            # Apply subsample budget to train only
            if budget and budget.get('type') == 'subsample':
                train = self._subsample_train(train, budget['value'], candidate_id, f)

            total_train_used += len(train)

            X_train, y_train = self._X[train], self._y[train]
            X_test, y_test = self._X[test], self._y[test]

            model = cls.create(effective_params)
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                scores[f] = self._scorer_fn(y_test, preds)
            finally:
                model.dispose()

        fit_time_ms = now() - t0

        entry = self._leaderboard.add(
            model_name=candidate_id.split(':')[0],
            params=params,
            scores=scores,
            fit_time_ms=fit_time_ms,
        )

        return {
            'candidateId': candidate_id,
            'meanScore': entry['meanScore'],
            'foldScores': scores,
            'stdScore': entry['stdScore'],
            'fitTimeMs': fit_time_ms,
            'nTrainUsed': round(total_train_used / len(folds)),
            'nTest': len(folds[0][1]),
        }

    def _apply_rounds_budget(self, cls, params, budget):
        """Apply rounds budget by setting the model's rounds param."""
        if not budget or budget.get('type') != 'rounds':
            return params
        spec = None
        if hasattr(cls, 'budget_spec'):
            spec = cls.budget_spec()
        if not spec or 'roundsParam' not in spec:
            return params
        rounds_param = spec['roundsParam']
        if rounds_param in params:
            return params
        return {**params, rounds_param: budget['value']}

    def _subsample_train(self, train, fraction, candidate_id, fold_idx):
        """Subsample train indices using partial Fisher-Yates."""
        k = max(1, math.ceil(len(train) * fraction))
        if k >= len(train):
            return train
        copy = np.array(train, dtype=np.int32)
        s = seed_for(candidate_id, fold_idx, self._seed)
        rng = make_lcg(s)
        return partial_shuffle(copy, k, rng)

    def run_strategy(self, strategy):
        """Run a strategy to completion.

        Returns:
            dict with 'leaderboard'
        """
        while not strategy.is_done():
            if self.is_timed_out:
                break
            task = strategy.next()
            if task is None:
                break
            result = self.evaluate_candidate(
                task['candidateId'],
                task['cls'],
                task['params'],
                task.get('budget'),
            )
            strategy.report(result)
        return {'leaderboard': self._leaderboard}
