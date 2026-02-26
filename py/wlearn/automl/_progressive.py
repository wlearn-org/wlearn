"""Progressive search matching JS automl/progressive.js."""

import math

import numpy as np

from ..errors import ValidationError
from ._common import detect_task, scorer_greater_is_better
from ._cv import stratified_k_fold, k_fold, get_scorer
from ._executor import Executor
from ._strategy_progressive import ProgressiveStrategy


class ProgressiveSearch:
    """Probe all candidates cheaply (1 fold + subsample),
    then promote top N to full K-fold evaluation."""

    def __init__(self, models, scoring=None, cv=5, seed=42, task=None,
                 n_iter=20, max_time_ms=0, promote_count=10,
                 probe_fraction=0.5):
        if not models:
            raise ValidationError(
                'ProgressiveSearch: at least one model is required')
        self._models = models
        self._scoring = scoring
        self._cv = cv
        self._seed = seed
        self._task = task
        self._n_iter = n_iter
        self._max_time_ms = max_time_ms
        self._promote_count = promote_count
        self._probe_fraction = probe_fraction
        self._leaderboard = None
        self._best_result = None

    def fit(self, X, y):
        task = self._task or detect_task(y)
        scoring = self._scoring or (
            'accuracy' if task == 'classification' else 'r2')
        greater_is_better = scorer_greater_is_better(scoring)

        # Probe folds: 2-fold, use only first
        if task == 'classification':
            probe_folds = stratified_k_fold(y, 2, do_shuffle=True,
                                            seed=self._seed)
        else:
            probe_folds = k_fold(len(y), 2, do_shuffle=True,
                                 seed=self._seed)
        single_fold = [probe_folds[0]]

        # Full folds for promoted candidates
        if task == 'classification':
            full_folds = stratified_k_fold(y, self._cv, do_shuffle=True,
                                           seed=self._seed + 1)
        else:
            full_folds = k_fold(len(y), self._cv, do_shuffle=True,
                                seed=self._seed + 1)

        strategy = ProgressiveStrategy(
            self._models,
            n_iter=self._n_iter,
            seed=self._seed,
            promote_count=self._promote_count,
            greater_is_better=greater_is_better,
            probe_fraction=self._probe_fraction,
        )

        # Phase 1: probe with 1-fold executor
        probe_time = (int(self._max_time_ms * 0.3)
                      if self._max_time_ms > 0 else 0)
        probe_executor = Executor(
            folds=single_fold,
            scoring=scoring,
            X=X, y=y,
            time_limit_ms=probe_time,
            seed=self._seed,
        )

        while strategy.phase == 'probe' and not strategy.is_done():
            if probe_executor.is_timed_out:
                break
            cand = strategy.next()
            if cand is None:
                break
            try:
                result = probe_executor.evaluate_candidate(
                    cand['candidateId'], cand['cls'], cand['params'],
                    cand.get('budget'))
                strategy.report(result)
            except Exception:
                strategy.report({
                    'candidateId': cand['candidateId'],
                    'meanScore': -float('inf'),
                    'foldScores': np.zeros(1),
                    'stdScore': 0,
                    'fitTimeMs': 0,
                    'nTrainUsed': 0,
                    'nTest': 0,
                })

        # Phase 2: full evaluation of promoted candidates
        full_time = (int(self._max_time_ms * 0.7)
                     if self._max_time_ms > 0 else 0)
        full_executor = Executor(
            folds=full_folds,
            scoring=scoring,
            X=X, y=y,
            time_limit_ms=full_time,
            seed=self._seed,
        )

        while not strategy.is_done():
            if full_executor.is_timed_out:
                break
            cand = strategy.next()
            if cand is None:
                break
            try:
                full_executor.evaluate_candidate(
                    cand['candidateId'], cand['cls'], cand['params'],
                    cand.get('budget'))
            except Exception:
                pass

        leaderboard = full_executor.leaderboard
        if leaderboard.length == 0:
            probe_lb = probe_executor.leaderboard
            if probe_lb.length == 0:
                raise ValidationError(
                    'ProgressiveSearch: no candidates were evaluated')
            self._leaderboard = probe_lb
        else:
            self._leaderboard = leaderboard

        self._best_result = self._leaderboard.best()
        return {'leaderboard': self._leaderboard,
                'bestResult': self._best_result}

    def refit_best(self, X, y):
        if self._best_result is None:
            raise ValidationError(
                'ProgressiveSearch: must call fit() first')
        best = self._best_result
        model_spec = None
        for m in self._models:
            if m['name'] == best['modelName']:
                model_spec = m
                break
        instance = model_spec['cls'].create(best['params'])
        instance.fit(X, y)
        return instance

    @property
    def leaderboard(self):
        return self._leaderboard

    @property
    def best_result(self):
        return self._best_result
