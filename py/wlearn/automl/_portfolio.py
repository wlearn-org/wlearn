"""Zeroshot portfolio: pre-tuned hyperparameter configs per model family.

Instead of random search over the full search space, the portfolio provides
a curated set of configs known to work well across diverse datasets. Inspired
by AutoGluon's zeroshot portfolio approach (TabRepo).

Each model family has multiple configs spanning different regularization
strengths, depths, learning rates, and structural choices to provide
ensemble diversity without runtime tuning.
"""

from copy import deepcopy

import json
from pathlib import Path


from ..errors import ValidationError
from ..task import task_params
from ..resampling import resolve_cv
from ._common import detect_task
from ._candidate import (
    create_candidate_task, normalize_model_specs,
    preprocess_choices,
)
from ._candidate_pipeline import fit_candidate
from ._executor import Executor

# ---------------------------------------------------------------------------
# Portfolio configs: task -> model_name -> list of param dicts
# ---------------------------------------------------------------------------

with (Path(__file__).with_name('portfolio.json')).open(encoding='utf-8') as _source:
    PORTFOLIO = json.load(_source)


def get_portfolio(task='classification'):
    """Return portfolio configs for the given task.

    Args:
        task: 'classification' or 'regression'

    Returns:
        dict mapping model name to list of param dicts
    """
    return PORTFOLIO.get(task, PORTFOLIO['classification'])


# ---------------------------------------------------------------------------
# PortfolioStrategy
# ---------------------------------------------------------------------------

class PortfolioStrategy:
    """Yields pre-tuned configs from the zeroshot portfolio.

    Same interface as RandomStrategy / HalvingStrategy (next/report/is_done).
    """

    def __init__(self, models, task='classification', seed=42):
        """
        Args:
            models: list of dicts with 'name', 'cls', optional 'params'
            task: 'classification' or 'regression'
            seed: unused (kept for interface consistency)
        """
        portfolio = get_portfolio(task)
        self._queue = []
        self._index = 0
        seen = {}

        for model in normalize_model_specs(models):
            portfolio_key = model.get('portfolioKey')
            if portfolio_key is None:
                portfolio_key = getattr(
                    model['cls'], 'portfolio_key',
                    getattr(model['cls'], 'portfolioKey', model['classId']))
            fixed = model.get('params') or {}

            # Package-owned warm starts precede the legacy table; callers can
            # supply an explicit portfolio without registering another family.
            configs = model.get('portfolio')
            if configs is None:
                provider = getattr(model['cls'], 'default_portfolio', None)
                configs = provider(task) if provider is not None else None
            if configs is None:
                configs = portfolio.get(portfolio_key, [{}])
            if not isinstance(configs, list) or not configs or any(not isinstance(config, dict) for config in configs):
                raise ValidationError(f'Portfolio for {model["name"]} must be a nonempty list of parameter mappings')

            for config in configs:
                params = {**config, **fixed}
                for preprocess in preprocess_choices(model):
                    self._queue.append(create_candidate_task(
                        model, params, preprocess, seen))

        self._total = len(self._queue)

    def next(self):
        """Return next candidate or None when exhausted."""
        if self._index >= self._total:
            return None
        cand = self._queue[self._index]
        self._index += 1
        return cand

    def report(self, result):
        """No-op for portfolio strategy."""
        pass

    def is_done(self):
        """True when all candidates have been yielded."""
        return self._index >= self._total


# ---------------------------------------------------------------------------
# PortfolioSearch
# ---------------------------------------------------------------------------

class PortfolioSearch:
    """Evaluate pre-tuned portfolio configs with cross-validation."""

    def __init__(self, models, scoring=None, cv=5, seed=42, task=None,
                 max_time_ms=0):
        if not models:
            raise ValidationError(
                'PortfolioSearch: at least one model is required')
        self._models = normalize_model_specs(models, 'PortfolioSearch models')
        self._scoring = scoring
        self._cv = cv
        self._seed = seed
        self._task = task
        self._max_time_ms = max_time_ms
        self._leaderboard = None
        self._best_result = None
        self._archive = None

    def fit(self, X, y):
        """Run the portfolio search.

        Returns:
            dict with 'leaderboard' and 'bestResult'
        """
        task = self._task or detect_task(y)
        models = [{**spec, 'params': task_params(spec.get('params'), task)} for spec in self._models]
        scoring = self._scoring or (
            'accuracy' if task == 'classification' else 'r2')

        folds = resolve_cv(self._cv, y, task=task, seed=self._seed)

        executor = Executor(
            folds=folds,
            scoring=scoring,
            X=X,
            y=y,
            time_limit_ms=self._max_time_ms,
            seed=self._seed,
        )

        strategy = PortfolioStrategy(models, task=task,
                                     seed=self._seed)

        result = executor.run_strategy(strategy)

        leaderboard = result['leaderboard']
        if leaderboard.length == 0:
            if executor.first_error is not None:
                raise executor.first_error
            raise ValidationError(
                'PortfolioSearch: no candidates were evaluated')

        self._leaderboard = leaderboard
        self._archive = result['archive']
        self._best_result = leaderboard.best()
        return {'leaderboard': leaderboard, 'archive': self._archive,
                'bestResult': leaderboard.best()}

    def refit_best(self, X, y):
        """Refit the best candidate on full data."""
        if self._best_result is None:
            raise ValidationError('PortfolioSearch: must call fit() first')
        best = self._best_result
        model_spec = None
        for m in self._models:
            if m['classId'] == best['candidate']['model']['classId']:
                model_spec = m
                break
        return fit_candidate(
            model_spec, best['candidate'], X, y, best['candidateId'])

    @property
    def archive(self):
        return self._archive

    @property
    def leaderboard(self):
        return self._leaderboard

    @property
    def best_result(self):
        return deepcopy(self._best_result)
