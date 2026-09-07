"""Progressive search matching JS automl/progressive.js."""

from copy import deepcopy



from ..errors import ValidationError
from ..task import task_params
from ..resampling import resolve_cv
from ._common import detect_task, scorer_greater_is_better
from ._candidate import normalize_model_specs
from ._candidate_pipeline import fit_candidate
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
        self._models = normalize_model_specs(models, 'ProgressiveSearch models')
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
        self._archive = None

    def fit(self, X, y):
        task = self._task or detect_task(y)
        models = [{**spec, 'params': task_params(spec.get('params'), task)} for spec in self._models]
        scoring = self._scoring or (
            'accuracy' if task == 'classification' else 'r2')
        greater_is_better = scorer_greater_is_better(scoring)

        # Probe the caller's first fold to preserve group boundaries.
        full_folds = resolve_cv(self._cv, y, task=task, seed=self._seed)
        single_fold = [full_folds[0]]

        strategy = ProgressiveStrategy(
            models,
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
                    cand['candidateId'], cand['candidate'],
                    cand['cls'], cand['params'],
                    cand.get('budget'))
                strategy.report(result)
            except Exception as exc:
                probe_executor.record_failure(cand, exc)
                strategy.report({
                    'candidateId': cand['candidateId'],
                    'candidate': cand['candidate'],
                    'status': 'failed', 'meanScore': None,
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
                    cand['candidateId'], cand['candidate'],
                    cand['cls'], cand['params'],
                    cand.get('budget'))
            except Exception as exc:
                full_executor.record_failure(cand, exc)

        leaderboard = full_executor.leaderboard
        if leaderboard.length == 0:
            probe_lb = probe_executor.leaderboard
            if probe_lb.length == 0:
                first_error = (
                    probe_executor.first_error or full_executor.first_error)
                if first_error is not None:
                    raise first_error
                raise ValidationError(
                    'ProgressiveSearch: no candidates were evaluated')
            self._leaderboard = probe_lb
            self._archive = probe_executor.archive
        else:
            self._leaderboard = leaderboard
            self._archive = full_executor.archive

        self._best_result = self._leaderboard.best()
        return {'leaderboard': self._leaderboard,
                'archive': self._archive,
                'bestResult': self._leaderboard.best()}

    def refit_best(self, X, y):
        if self._best_result is None:
            raise ValidationError(
                'ProgressiveSearch: must call fit() first')
        best = self._best_result
        model_spec = None
        for m in self._models:
            if m['classId'] == best['candidate']['model']['classId']:
                model_spec = m
                break
        return fit_candidate(
            model_spec, best['candidate'], X, y, best['candidateId'])

    @property
    def leaderboard(self):
        return self._leaderboard

    @property
    def best_result(self):
        return deepcopy(self._best_result)

    @property
    def archive(self):
        return self._archive
