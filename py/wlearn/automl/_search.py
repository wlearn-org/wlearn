"""Search wrappers matching JS automl/search.js and halving.js."""

import numpy as np

from ..errors import ValidationError
from ._common import detect_task, scorer_greater_is_better
from ._cv import stratified_k_fold, k_fold
from ._executor import Executor
from ._strategy_random import RandomStrategy
from ._strategy_halving import HalvingStrategy


class RandomSearch:
    """Random hyperparameter search with cross-validation."""

    def __init__(self, models, scoring=None, cv=5, seed=42, task=None,
                 n_iter=20, max_time_ms=0):
        if not models:
            raise ValidationError('RandomSearch: at least one model is required')
        self._models = models
        self._scoring = scoring
        self._cv = cv
        self._seed = seed
        self._task = task
        self._n_iter = n_iter
        self._max_time_ms = max_time_ms
        self._leaderboard = None
        self._best_result = None

    def fit(self, X, y):
        """Run the search.

        Returns:
            dict with 'leaderboard' and 'bestResult'
        """
        task = self._task or detect_task(y)
        scoring = self._scoring or ('accuracy' if task == 'classification' else 'r2')

        if task == 'classification':
            folds = stratified_k_fold(y, self._cv, do_shuffle=True, seed=self._seed)
        else:
            folds = k_fold(len(y), self._cv, do_shuffle=True, seed=self._seed)

        executor = Executor(
            folds=folds,
            scoring=scoring,
            X=X,
            y=y,
            time_limit_ms=self._max_time_ms,
            seed=self._seed,
        )

        strategy = RandomStrategy(self._models, n_iter=self._n_iter, seed=self._seed)

        result = executor.run_strategy(strategy)

        leaderboard = result['leaderboard']
        if leaderboard.length == 0:
            raise ValidationError('RandomSearch: no candidates were evaluated')

        self._leaderboard = leaderboard
        self._best_result = leaderboard.best()
        return {'leaderboard': leaderboard, 'bestResult': self._best_result}

    def refit_best(self, X, y):
        """Refit the best candidate on full data."""
        if self._best_result is None:
            raise ValidationError('RandomSearch: must call fit() first')
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


class SuccessiveHalvingSearch:
    """Successive halving search with multi-round elimination."""

    def __init__(self, models, scoring=None, cv=5, seed=42, task=None,
                 n_iter=20, max_time_ms=0, factor=3):
        if not models:
            raise ValidationError('SuccessiveHalvingSearch: at least one model is required')
        self._models = models
        self._scoring = scoring
        self._cv = cv
        self._seed = seed
        self._task = task
        self._n_iter = n_iter
        self._max_time_ms = max_time_ms
        self._factor = factor
        self._leaderboard = None
        self._best_result = None
        self._rounds = None

    def fit(self, X, y):
        """Run the search.

        Returns:
            dict with 'leaderboard', 'bestResult', 'rounds'
        """
        n = len(X)
        task = self._task or detect_task(y)
        scoring = self._scoring or ('accuracy' if task == 'classification' else 'r2')
        greater_is_better = scorer_greater_is_better(scoring)

        if task == 'classification':
            folds = stratified_k_fold(y, self._cv, do_shuffle=True, seed=self._seed)
        else:
            folds = k_fold(n, self._cv, do_shuffle=True, seed=self._seed)

        executor = Executor(
            folds=folds,
            scoring=scoring,
            X=X,
            y=y,
            time_limit_ms=self._max_time_ms,
            seed=self._seed,
        )

        strategy = HalvingStrategy(
            self._models,
            n_iter=self._n_iter,
            seed=self._seed,
            factor=self._factor,
            n_samples=n,
            greater_is_better=greater_is_better,
            cv=self._cv,
        )

        result = executor.run_strategy(strategy)

        self._leaderboard = result['leaderboard']
        self._best_result = self._leaderboard.best()
        self._rounds = strategy.rounds
        return {
            'leaderboard': self._leaderboard,
            'bestResult': self._best_result,
            'rounds': self._rounds,
        }

    def refit_best(self, X, y):
        """Refit the best candidate on full data."""
        if self._best_result is None:
            raise ValidationError('SuccessiveHalvingSearch: must call fit() first')
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

    @property
    def rounds(self):
        return self._rounds
