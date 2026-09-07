"""Bayesian AutoML strategy backed by the optional wlearn-bo package."""

from copy import deepcopy

import math

from ..errors import ValidationError
from ..task import task_params
from ..resampling import resolve_cv
from ._common import detect_task
from ._candidate import (
    candidate_hash, create_candidate, create_candidate_task, normalize_model_specs,
    preprocess_choices,
)
from ._candidate_pipeline import fit_candidate
from ._executor import Executor
from ..rng import make_lcg
from ._sampler import sample_config
from ._conditions import effective_search_space


def _load_optimizer_cls():
    try:
        from wlearn_bo import BayesianOptimizer
    except ImportError as exc:
        raise ValidationError(
            'Bayesian AutoML requires wlearn-bo. Install wlearn[bo] or '
            'the wlearn-bo package, then retry strategy="bayesian".'
        ) from exc
    return BayesianOptimizer


from ._common import default_search_space


class BayesianStrategy:
    """Round-robin Bayesian suggestions, with random warmup per model."""

    optimizer_cls = None

    def __init__(self, models, n_iter=30, seed=42, acquisition_fn='ei',
                 kappa=2.0, xi=0.01, kernel='matern52', n_initial=None,
                 task=None):
        normalized = normalize_model_specs(models)
        self._models = []
        for model in normalized:
            for preprocess in preprocess_choices(model):
                variant = dict(model)
                variant['activePreprocess'] = preprocess
                variant['variantId'] = _variant_id(
                    model['classId'], preprocess)
                self._models.append(variant)
        self._n_iter = n_iter
        self._seed = seed
        self._acquisition_fn = acquisition_fn
        self._kappa = kappa
        self._xi = xi
        self._kernel = kernel
        self._n_initial = n_initial
        self._task = task
        self._optimizers = {}
        self._warmup_queues = {}
        self._warmup_counts = {}
        self._reported = {}
        self._yielded = 0
        self._total = len(self._models) * n_iter
        self._model_cycle = [m['variantId'] for m in self._models]
        self._cycle_idx = 0
        self._disposed = False
        self._seen = {}
        try:
            self._init()
        except Exception:
            self._dispose_quietly()
            raise

    def _init(self):
        rng = make_lcg(self._seed)
        optimizer_cls = self.optimizer_cls

        for model in self._models:
            effective_space = self._effective_space(model)
            n_free = len(effective_space)
            if n_free == 0:
                auto_warmup = self._n_iter
            else:
                auto_warmup = min(math.ceil(self._n_iter * 0.4),
                                  max(3, n_free + 1))
            if n_free == 0:
                n_initial = self._n_iter
            else:
                n_initial = min(
                    self._n_iter,
                    self._n_initial if self._n_initial is not None else auto_warmup,
                )

            self._warmup_counts[model['variantId']] = n_initial
            self._reported[model['variantId']] = 0

            config_rng = make_lcg(int(rng() * 0x7fffffff))
            fixed_params = model.get('params') or {}
            queue = []
            for _ in range(n_initial):
                config = sample_config(effective_space, config_rng)
                params = {**config, **fixed_params}
                queue.append(create_candidate_task(
                    model, params, model['activePreprocess'], self._seen))
            self._warmup_queues[model['variantId']] = queue

            if n_free == 0:
                continue

            if optimizer_cls is None:
                optimizer_cls = _load_optimizer_cls()

            opt_seed = int(rng() * 0x7fffffff)
            self._optimizers[model['variantId']] = optimizer_cls(
                effective_space,
                kernel=self._kernel,
                acquisition_fn=self._acquisition_fn,
                kappa=self._kappa,
                xi=self._xi,
                seed=opt_seed,
            )

    def next(self):
        if self._yielded >= self._total:
            return None

        for _ in range(len(self._model_cycle)):
            current_variant_id = self._model_cycle[self._cycle_idx]
            self._cycle_idx = (self._cycle_idx + 1) % len(self._model_cycle)

            model = next(
                m for m in self._models
                if m['variantId'] == current_variant_id)
            reported = self._reported.get(current_variant_id, 0)
            warmup_count = self._warmup_counts.get(current_variant_id, 0)
            queue = self._warmup_queues.get(current_variant_id, [])

            if queue:
                self._yielded += 1
                return queue.pop(0)

            if reported < warmup_count:
                continue

            optimizer = self._optimizers.get(current_variant_id)
            if optimizer is None:
                continue

            suggested = optimizer.suggest()
            params = {**suggested, **(model.get('params') or {})}
            self._yielded += 1
            return create_candidate_task(
                model, params, model['activePreprocess'], self._seen)

        return None

    def report(self, result):
        candidate = result.get('candidate')
        if candidate is None:
            return

        current_variant_id = _variant_id(
            candidate['model']['classId'], candidate.get('preprocess'))
        model = next((
            m for m in self._models
            if m['variantId'] == current_variant_id), None)
        optimizer = self._optimizers.get(current_variant_id)
        if model is None:
            return

        if optimizer is not None:
            params = candidate['model']['params']
            observe_params = dict(params)
            for key in (model.get('params') or {}):
                observe_params.pop(key, None)
            score = result.get('meanScore')
            if isinstance(score, (int, float)) and math.isfinite(score):
                try:
                    optimizer.observe(observe_params, float(score))
                except Exception:
                    pass

        self._reported[current_variant_id] = (
            self._reported.get(current_variant_id, 0) + 1)

    def is_done(self):
        return self._yielded >= self._total

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        first_error = None
        for opt in reversed(list(self._optimizers.values())):
            try:
                opt.dispose()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        self._optimizers.clear()
        if first_error is not None:
            raise first_error

    def _dispose_quietly(self):
        self._disposed = True
        for opt in reversed(list(self._optimizers.values())):
            try:
                opt.dispose()
            except Exception:
                pass
        self._optimizers.clear()

    def _effective_space(self, model):
        space = default_search_space(model)
        return effective_search_space(space, model.get('params') or {})


def _variant_id(class_id, preprocess):
    candidate = create_candidate({
        'displayName': 'Bayesian variant',
        'classId': class_id,
    }, {}, preprocess)
    return f'wlcv1_{candidate_hash(candidate)}'


class BayesianSearch:
    """Bayesian hyperparameter search with cross-validation."""

    def __init__(self, models, scoring=None, cv=5, seed=42, task=None,
                 n_iter=30, max_time_ms=0, acquisition_fn='ei', kappa=2.0,
                 xi=0.01, kernel='matern52', n_initial=None):
        if not models:
            raise ValidationError('BayesianSearch: at least one model is required')
        self._models = normalize_model_specs(models, 'BayesianSearch models')
        self._scoring = scoring
        self._cv = cv
        self._seed = seed
        self._task = task
        self._n_iter = n_iter
        self._max_time_ms = max_time_ms
        self._acquisition_fn = acquisition_fn
        self._kappa = kappa
        self._xi = xi
        self._kernel = kernel
        self._n_initial = n_initial
        self._leaderboard = None
        self._best_result = None
        self._archive = None

    def fit(self, X, y):
        task = self._task or detect_task(y)
        models = [{**spec, 'params': task_params(spec.get('params'), task)} for spec in self._models]
        scoring = self._scoring or ('accuracy' if task == 'classification' else 'r2')

        folds = resolve_cv(self._cv, y, task=task, seed=self._seed)

        executor = Executor(
            folds=folds,
            scoring=scoring,
            X=X,
            y=y,
            time_limit_ms=self._max_time_ms,
            seed=self._seed,
        )

        strategy = BayesianStrategy(
            models,
            n_iter=self._n_iter,
            seed=self._seed,
            acquisition_fn=self._acquisition_fn,
            kappa=self._kappa,
            xi=self._xi,
            kernel=self._kernel,
            n_initial=self._n_initial,
            task=task,
        )
        operation_error = None
        try:
            result = executor.run_strategy(strategy)
        except Exception as exc:
            operation_error = exc
            raise
        finally:
            try:
                strategy.dispose()
            except Exception:
                if operation_error is None:
                    raise

        leaderboard = result['leaderboard']
        if leaderboard.length == 0:
            if executor.first_error is not None:
                raise executor.first_error
            raise ValidationError('BayesianSearch: no candidates were evaluated')

        self._leaderboard = leaderboard
        self._archive = result['archive']
        self._best_result = leaderboard.best()
        return {
            'leaderboard': self._leaderboard,
            'archive': self._archive,
            'bestResult': self._leaderboard.best(),
        }

    def refit_best(self, X, y):
        if self._best_result is None:
            raise ValidationError('BayesianSearch: must call fit() first')
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
