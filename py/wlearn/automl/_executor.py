"""Executor matching JS automl/executor.js."""

import math

import numpy as np

from ..archive import Archive
from ..errors import ValidationError
from ._leaderboard import Leaderboard
from ._common import make_candidate_id, now, seed_for, partial_shuffle
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
        if (isinstance(seed, bool) or not isinstance(seed, int) or
                seed < 0 or seed > 0xffffffff):
            raise ValidationError(
                'Executor seed must be an unsigned 32-bit integer.')
        self._folds = folds
        self._scorer_fn = get_scorer(scoring)
        self._X = X
        self._y = y
        self._time_limit_ms = time_limit_ms
        self._seed = seed
        self._start_time = now()
        self._leaderboard = Leaderboard()
        self._metric = scoring if isinstance(scoring, str) else 'score'
        self._archive = Archive(
            id='automl',
            measures=[self._metric],
            primary_measure=self._metric,
            direction='maximize',
            metadata={
                'source': 'wlearn.automl',
                'seed': seed,
                'folds': len(folds),
            },
        )
        self._failed_seq = 0
        self._first_error = None

    @property
    def leaderboard(self):
        return self._leaderboard

    @property
    def archive(self):
        return self._archive

    @property
    def first_error(self):
        return self._first_error

    @property
    def is_timed_out(self):
        if self._time_limit_ms <= 0:
            return False
        return (now() - self._start_time) > self._time_limit_ms

    def evaluate_candidate(self, candidate_id, candidate, cls, params,
                           budget=None):
        """Evaluate one candidate across all CV folds.

        Args:
            candidate_id: stable identifier
            cls: estimator class with create/fit/predict/dispose
            params: hyperparameters
            budget: optional dict with 'type' and 'value'

        Returns:
            CandidateResult dict
        """
        if make_candidate_id(candidate) != candidate_id:
            raise ValidationError(
                'candidate_id does not match the structured candidate.')
        folds = self._folds
        scores = np.zeros(len(folds), dtype=np.float64)
        fold_seeds = np.zeros(len(folds), dtype=np.uint32)
        t0 = now()
        total_train_used = 0

        effective_params = self._apply_rounds_budget(
            cls, candidate['model']['params'], budget)

        for f, (train, test) in enumerate(folds):
            fold_seeds[f] = seed_for(candidate, f, self._seed)
            # Apply subsample budget to train only
            if budget and budget.get('type') == 'subsample':
                train = self._subsample_train(
                    train, budget['value'], candidate, f)

            total_train_used += len(train)

            X_train, y_train = self._X[train], self._y[train]
            X_test, y_test = self._X[test], self._y[test]

            model = cls.create(effective_params)
            operation_error = None
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                scores[f] = self._scorer_fn(y_test, preds)
            except Exception as error:
                operation_error = error
                raise
            finally:
                try:
                    model.dispose()
                except Exception:
                    if operation_error is None:
                        raise

        fit_time_ms = now() - t0

        entry = self._leaderboard.add(
            candidate_id=candidate_id,
            candidate=candidate,
            scores=scores,
            base_seed=self._seed,
            fold_seeds=fold_seeds,
            fit_time_ms=fit_time_ms,
        )

        self._archive.add({
            'trial_id': f"automl-{entry['id']}",
            'candidate_id': candidate_id,
            'seed': self._seed,
            'params': candidate['model']['params'],
            'budget': budget,
            'status': 'ok',
            'scores': {self._metric: entry['meanScore']},
            'primary_score': entry['meanScore'],
            'timings': {'fitTimeMs': fit_time_ms},
            'metadata': {
                'sourceCandidateId': candidate_id,
                'candidate': candidate,
                'leaderboardId': entry['id'],
                'modelName': entry['modelName'],
                'foldScores': scores.tolist(),
                'foldSeeds': [
                    {'foldId': fold_id, 'seed': int(value)}
                    for fold_id, value in enumerate(fold_seeds)
                ],
                'stdScore': entry['stdScore'],
                'nTrainUsed': round(total_train_used / len(folds)),
                'nTest': len(folds[0][1]),
            },
        })

        return {
            'candidateId': candidate_id,
            'candidate': candidate,
            'params': candidate['model']['params'],
            'meanScore': entry['meanScore'],
            'foldScores': scores,
            'baseSeed': self._seed,
            'foldSeeds': fold_seeds,
            'stdScore': entry['stdScore'],
            'fitTimeMs': fit_time_ms,
            'nTrainUsed': round(total_train_used / len(folds)),
            'nTest': len(folds[0][1]),
        }

    def record_failure(self, task, error, phase='fit'):
        if self._first_error is None:
            self._first_error = error
        candidate_id = task.get('candidateId', 'candidate') if task else 'candidate'
        seq = self._failed_seq
        self._failed_seq += 1
        return self._archive.fail({
            'trial_id': f'automl-failed-{seq}',
            'candidate_id': candidate_id,
            'seed': self._seed,
            'params': (
                task['candidate']['model']['params']
                if task and task.get('candidate') else {}),
            'budget': task.get('budget') if task else None,
            'metadata': {
                'sourceCandidateId': candidate_id,
                'candidate': task.get('candidate') if task else None,
                'foldSeeds': (
                    self._fold_seeds(task['candidate'])
                    if task and task.get('candidate') is not None else []),
                'modelName': (
                    task['candidate']['model']['displayName']
                    if task and task.get('candidate') else 'candidate'),
            },
        }, error, phase)

    def _fold_seeds(self, candidate):
        return [
            {'foldId': fold_id,
             'seed': seed_for(candidate, fold_id, self._seed)}
            for fold_id in range(len(self._folds))
        ]

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

    def _subsample_train(self, train, fraction, candidate, fold_idx):
        """Subsample train indices using partial Fisher-Yates."""
        k = max(1, math.ceil(len(train) * fraction))
        if k >= len(train):
            return train
        copy = np.array(train, dtype=np.int32)
        s = seed_for(candidate, fold_idx, self._seed)
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
            try:
                result = self.evaluate_candidate(
                    task['candidateId'],
                    task['candidate'],
                    task['cls'],
                    task['params'],
                    task.get('budget'),
                )
                strategy.report(result)
            except Exception as exc:
                self.record_failure(task, exc)
        return {'leaderboard': self._leaderboard, 'archive': self._archive}
