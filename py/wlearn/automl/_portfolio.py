"""Zeroshot portfolio: pre-tuned hyperparameter configs per model family.

Instead of random search over the full search space, the portfolio provides
a curated set of configs known to work well across diverse datasets. Inspired
by AutoGluon's zeroshot portfolio approach (TabRepo).

Each model family has multiple configs spanning different regularization
strengths, depths, learning rates, and structural choices to provide
ensemble diversity without runtime tuning.
"""

import numpy as np

from ..errors import ValidationError
from ._common import detect_task, make_candidate_id, scorer_greater_is_better
from ._cv import stratified_k_fold, k_fold, get_scorer
from ._executor import Executor

# ---------------------------------------------------------------------------
# Portfolio configs: task -> model_name -> list of param dicts
# ---------------------------------------------------------------------------

PORTFOLIO = {
    'classification': {
        'xgb': [
            # 1. Default balanced
            {'objective': 'multi:softprob', 'eta': 0.05, 'max_depth': 6,
             'numRound': 200, 'subsample': 0.8, 'colsample_bytree': 0.8,
             'min_child_weight': 1.0, 'lambda': 1.0, 'alpha': 0.0},
            # 2. Deep + slow learning rate
            {'objective': 'multi:softprob', 'eta': 0.01, 'max_depth': 10,
             'numRound': 500, 'subsample': 0.7, 'colsample_bytree': 0.65,
             'min_child_weight': 0.6, 'lambda': 0.1, 'alpha': 0.0},
            # 3. Shallow + fast learning rate
            {'objective': 'multi:softprob', 'eta': 0.1, 'max_depth': 3,
             'numRound': 100, 'subsample': 0.9, 'colsample_bytree': 0.9,
             'min_child_weight': 1.0, 'lambda': 1.0, 'alpha': 0.0},
            # 4. Heavy regularization
            {'objective': 'multi:softprob', 'eta': 0.03, 'max_depth': 7,
             'numRound': 300, 'subsample': 0.8, 'colsample_bytree': 0.7,
             'min_child_weight': 1.0, 'lambda': 5.0, 'alpha': 1.0},
            # 5. Light regularization, many rounds
            {'objective': 'multi:softprob', 'eta': 0.02, 'max_depth': 8,
             'numRound': 400, 'subsample': 0.8, 'colsample_bytree': 0.8,
             'min_child_weight': 0.8, 'lambda': 0.1, 'alpha': 0.0},
            # 6. Wide column sampling
            {'objective': 'multi:softprob', 'eta': 0.08, 'max_depth': 4,
             'numRound': 150, 'subsample': 0.85, 'colsample_bytree': 0.55,
             'min_child_weight': 1.0, 'lambda': 1.0, 'alpha': 0.1},
            # 7. Deep + narrow column sampling
            {'objective': 'multi:softprob', 'eta': 0.015, 'max_depth': 9,
             'numRound': 350, 'subsample': 0.75, 'colsample_bytree': 0.55,
             'min_child_weight': 0.9, 'lambda': 0.5, 'alpha': 0.0},
            # 8. Quick baseline
            {'objective': 'multi:softprob', 'eta': 0.3, 'max_depth': 3,
             'numRound': 50, 'subsample': 0.9, 'colsample_bytree': 0.9,
             'min_child_weight': 1.0, 'lambda': 1.0, 'alpha': 0.0},
            # 9. RF-mode (low correlation with boosting for ensemble diversity)
            {'objective': 'multi:softprob', 'num_parallel_tree': 100,
             'numRound': 1, 'subsample': 0.8, 'colsample_bynode': 0.8,
             'learning_rate': 1.0},
            # 10. RF-mode large
            {'objective': 'multi:softprob', 'num_parallel_tree': 200,
             'numRound': 1, 'subsample': 0.7, 'colsample_bynode': 0.6,
             'learning_rate': 1.0},
        ],
        'lgb': [
            # 1. Default balanced
            {'objective': 'multiclass', 'learning_rate': 0.05, 'max_depth': 6,
             'numRound': 200, 'num_leaves': 63, 'subsample': 0.8,
             'colsample_bytree': 0.8, 'min_child_weight': 1.0,
             'reg_lambda': 1.0, 'reg_alpha': 0.0},
            # 2. Deep + slow
            {'objective': 'multiclass', 'learning_rate': 0.01, 'max_depth': -1,
             'numRound': 500, 'num_leaves': 127, 'subsample': 0.7,
             'colsample_bytree': 0.65, 'reg_lambda': 0.1, 'reg_alpha': 0.0},
            # 3. Shallow + fast
            {'objective': 'multiclass', 'learning_rate': 0.1, 'max_depth': 4,
             'numRound': 100, 'num_leaves': 15, 'subsample': 0.9,
             'colsample_bytree': 0.9, 'reg_lambda': 1.0, 'reg_alpha': 0.0},
            # 4. Extra trees mode
            {'objective': 'multiclass', 'learning_rate': 0.05,
             'numRound': 200, 'num_leaves': 63, 'subsample': 0.8,
             'colsample_bytree': 0.8, 'extra_trees': True,
             'reg_lambda': 1.0, 'reg_alpha': 0.0},
            # 5. Heavy regularization
            {'objective': 'multiclass', 'learning_rate': 0.03, 'max_depth': 7,
             'numRound': 300, 'num_leaves': 63, 'subsample': 0.8,
             'colsample_bytree': 0.7, 'reg_lambda': 5.0, 'reg_alpha': 1.0},
            # 6. Large ensemble
            {'objective': 'multiclass', 'learning_rate': 0.01, 'max_depth': 8,
             'numRound': 500, 'num_leaves': 95, 'subsample': 0.75,
             'colsample_bytree': 0.55, 'reg_lambda': 0.5, 'reg_alpha': 0.0},
        ],
        'ebm': [
            # 1. Default
            {'objective': 'classification', 'learningRate': 0.01,
             'maxRounds': 500, 'maxLeaves': 3, 'maxBins': 256},
            # 2. More interactions
            {'objective': 'classification', 'learningRate': 0.01,
             'maxRounds': 500, 'maxLeaves': 4, 'maxInteractions': 15,
             'maxBins': 256},
            # 3. Fast
            {'objective': 'classification', 'learningRate': 0.05,
             'maxRounds': 300, 'maxLeaves': 3, 'maxBins': 128},
            # 4. Deep with more bins
            {'objective': 'classification', 'learningRate': 0.005,
             'maxRounds': 800, 'maxLeaves': 5, 'maxBins': 512},
        ],
        'linear': [
            # 1. L2-regularized logistic regression (primal)
            {'solver': 0, 'C': 1.0},
            # 2. L2R_LR with high C
            {'solver': 0, 'C': 10.0},
            # 3. L2-regularized logistic regression (dual)
            {'solver': 7, 'C': 1.0},
            # 4. L1-regularized logistic regression (sparse)
            {'solver': 6, 'C': 0.1},
        ],
        'svm': [
            # 1. C-SVC, RBF kernel, default
            {'svmType': 0, 'kernel': 2, 'C': 1.0, 'gamma': 0, 'probability': 1},
            # 2. C-SVC, RBF kernel, high C
            {'svmType': 0, 'kernel': 2, 'C': 10.0, 'gamma': 0.01, 'probability': 1},
            # 3. C-SVC, polynomial kernel
            {'svmType': 0, 'kernel': 1, 'C': 1.0, 'degree': 3, 'gamma': 0, 'probability': 1},
            # 4. C-SVC, linear kernel
            {'svmType': 0, 'kernel': 0, 'C': 1.0, 'probability': 1},
        ],
        'knn': [
            # 1. Default k=5
            {'k': 5, 'metric': 'l2', 'task': 'classification'},
            # 2. Larger neighborhood
            {'k': 15, 'metric': 'l2', 'task': 'classification'},
            # 3. Small neighborhood, Manhattan
            {'k': 3, 'metric': 'l1', 'task': 'classification'},
        ],
        'tsetlin': [
            # 1. Default
            {'task': 'classification', 'nClauses': 100, 'threshold': 50,
             's': 3.0, 'nEpochs': 100},
            # 2. More clauses
            {'task': 'classification', 'nClauses': 500, 'threshold': 100,
             's': 5.0, 'nEpochs': 100},
            # 3. Light
            {'task': 'classification', 'nClauses': 50, 'threshold': 25,
             's': 2.0, 'nEpochs': 60},
        ],
    },
    'regression': {
        'xgb': [
            # 1. Default balanced
            {'objective': 'reg:squarederror', 'eta': 0.05, 'max_depth': 6,
             'numRound': 200, 'subsample': 0.8, 'colsample_bytree': 0.8,
             'min_child_weight': 1.0, 'lambda': 1.0, 'alpha': 0.0},
            # 2. Deep + slow learning rate
            {'objective': 'reg:squarederror', 'eta': 0.01, 'max_depth': 10,
             'numRound': 500, 'subsample': 0.7, 'colsample_bytree': 0.65,
             'min_child_weight': 0.6, 'lambda': 0.1, 'alpha': 0.0},
            # 3. Shallow + fast learning rate
            {'objective': 'reg:squarederror', 'eta': 0.1, 'max_depth': 3,
             'numRound': 100, 'subsample': 0.9, 'colsample_bytree': 0.9,
             'min_child_weight': 1.0, 'lambda': 1.0, 'alpha': 0.0},
            # 4. Heavy regularization
            {'objective': 'reg:squarederror', 'eta': 0.03, 'max_depth': 7,
             'numRound': 300, 'subsample': 0.8, 'colsample_bytree': 0.7,
             'min_child_weight': 1.0, 'lambda': 5.0, 'alpha': 1.0},
            # 5. Light regularization, many rounds
            {'objective': 'reg:squarederror', 'eta': 0.02, 'max_depth': 8,
             'numRound': 400, 'subsample': 0.8, 'colsample_bytree': 0.8,
             'min_child_weight': 0.8, 'lambda': 0.1, 'alpha': 0.0},
            # 6. Wide column sampling
            {'objective': 'reg:squarederror', 'eta': 0.08, 'max_depth': 4,
             'numRound': 150, 'subsample': 0.85, 'colsample_bytree': 0.55,
             'min_child_weight': 1.0, 'lambda': 1.0, 'alpha': 0.1},
            # 7. Deep + narrow column sampling
            {'objective': 'reg:squarederror', 'eta': 0.015, 'max_depth': 9,
             'numRound': 350, 'subsample': 0.75, 'colsample_bytree': 0.55,
             'min_child_weight': 0.9, 'lambda': 0.5, 'alpha': 0.0},
            # 8. Quick baseline
            {'objective': 'reg:squarederror', 'eta': 0.3, 'max_depth': 3,
             'numRound': 50, 'subsample': 0.9, 'colsample_bytree': 0.9,
             'min_child_weight': 1.0, 'lambda': 1.0, 'alpha': 0.0},
            # 9. RF-mode (low correlation with boosting for ensemble diversity)
            {'objective': 'reg:squarederror', 'num_parallel_tree': 100,
             'numRound': 1, 'subsample': 0.8, 'colsample_bynode': 0.8,
             'learning_rate': 1.0},
            # 10. RF-mode large
            {'objective': 'reg:squarederror', 'num_parallel_tree': 200,
             'numRound': 1, 'subsample': 0.7, 'colsample_bynode': 0.6,
             'learning_rate': 1.0},
        ],
        'lgb': [
            # 1. Default balanced
            {'objective': 'regression', 'learning_rate': 0.05, 'max_depth': 6,
             'numRound': 200, 'num_leaves': 63, 'subsample': 0.8,
             'colsample_bytree': 0.8, 'min_child_weight': 1.0,
             'reg_lambda': 1.0, 'reg_alpha': 0.0},
            # 2. Deep + slow
            {'objective': 'regression', 'learning_rate': 0.01, 'max_depth': -1,
             'numRound': 500, 'num_leaves': 127, 'subsample': 0.7,
             'colsample_bytree': 0.65, 'reg_lambda': 0.1, 'reg_alpha': 0.0},
            # 3. Shallow + fast
            {'objective': 'regression', 'learning_rate': 0.1, 'max_depth': 4,
             'numRound': 100, 'num_leaves': 15, 'subsample': 0.9,
             'colsample_bytree': 0.9, 'reg_lambda': 1.0, 'reg_alpha': 0.0},
            # 4. Extra trees mode
            {'objective': 'regression', 'learning_rate': 0.05,
             'numRound': 200, 'num_leaves': 63, 'subsample': 0.8,
             'colsample_bytree': 0.8, 'extra_trees': True,
             'reg_lambda': 1.0, 'reg_alpha': 0.0},
            # 5. Heavy regularization
            {'objective': 'regression', 'learning_rate': 0.03, 'max_depth': 7,
             'numRound': 300, 'num_leaves': 63, 'subsample': 0.8,
             'colsample_bytree': 0.7, 'reg_lambda': 5.0, 'reg_alpha': 1.0},
            # 6. Large ensemble
            {'objective': 'regression', 'learning_rate': 0.01, 'max_depth': 8,
             'numRound': 500, 'num_leaves': 95, 'subsample': 0.75,
             'colsample_bytree': 0.55, 'reg_lambda': 0.5, 'reg_alpha': 0.0},
        ],
        'ebm': [
            # 1. Default
            {'objective': 'regression', 'learningRate': 0.01,
             'maxRounds': 500, 'maxLeaves': 3, 'maxBins': 256},
            # 2. More interactions
            {'objective': 'regression', 'learningRate': 0.01,
             'maxRounds': 500, 'maxLeaves': 4, 'maxInteractions': 15,
             'maxBins': 256},
            # 3. Fast
            {'objective': 'regression', 'learningRate': 0.05,
             'maxRounds': 300, 'maxLeaves': 3, 'maxBins': 128},
            # 4. Deep with more bins
            {'objective': 'regression', 'learningRate': 0.005,
             'maxRounds': 800, 'maxLeaves': 5, 'maxBins': 512},
        ],
        'linear': [
            # 1. L2-regularized L2-loss SVR (primal)
            {'solver': 11, 'C': 1.0},
            # 2. L2R_L2_SVR with high C
            {'solver': 11, 'C': 10.0},
            # 3. L2-regularized L2-loss SVR (dual)
            {'solver': 12, 'C': 1.0},
            # 4. L2-regularized L1-loss SVR (dual)
            {'solver': 13, 'C': 0.1},
        ],
        'svm': [
            # 1. epsilon-SVR, RBF kernel, default
            {'svmType': 3, 'kernel': 2, 'C': 1.0, 'gamma': 0},
            # 2. epsilon-SVR, RBF kernel, high C
            {'svmType': 3, 'kernel': 2, 'C': 10.0, 'gamma': 0.01},
            # 3. epsilon-SVR, polynomial kernel
            {'svmType': 3, 'kernel': 1, 'C': 1.0, 'degree': 3, 'gamma': 0},
            # 4. epsilon-SVR, linear kernel
            {'svmType': 3, 'kernel': 0, 'C': 1.0},
        ],
        'knn': [
            # 1. Default k=5
            {'k': 5, 'metric': 'l2', 'task': 'regression'},
            # 2. Larger neighborhood
            {'k': 15, 'metric': 'l2', 'task': 'regression'},
            # 3. Small neighborhood, Manhattan
            {'k': 3, 'metric': 'l1', 'task': 'regression'},
        ],
        'tsetlin': [
            # 1. Default
            {'task': 'regression', 'nClauses': 100, 'threshold': 50,
             's': 3.0, 'nEpochs': 100},
            # 2. More clauses
            {'task': 'regression', 'nClauses': 500, 'threshold': 100,
             's': 5.0, 'nEpochs': 100},
            # 3. Light
            {'task': 'regression', 'nClauses': 50, 'threshold': 25,
             's': 2.0, 'nEpochs': 60},
        ],
    },
}

# Expected config counts per model family
_EXPECTED_COUNTS = {
    'xgb': 10, 'lgb': 6, 'ebm': 4, 'linear': 4, 'svm': 4, 'knn': 3, 'tsetlin': 3,
}


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

        for model in models:
            name = model['name']
            cls = model['cls']
            fixed = model.get('params') or {}

            configs = portfolio.get(name, [{}])

            for config in configs:
                params = {**config, **fixed}
                candidate_id = make_candidate_id(name, params)
                self._queue.append({
                    'candidateId': candidate_id,
                    'cls': cls,
                    'params': params,
                })

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
        self._models = models
        self._scoring = scoring
        self._cv = cv
        self._seed = seed
        self._task = task
        self._max_time_ms = max_time_ms
        self._leaderboard = None
        self._best_result = None

    def fit(self, X, y):
        """Run the portfolio search.

        Returns:
            dict with 'leaderboard' and 'bestResult'
        """
        task = self._task or detect_task(y)
        scoring = self._scoring or (
            'accuracy' if task == 'classification' else 'r2')

        if task == 'classification':
            folds = stratified_k_fold(y, self._cv, do_shuffle=True,
                                      seed=self._seed)
        else:
            folds = k_fold(len(y), self._cv, do_shuffle=True,
                           seed=self._seed)

        executor = Executor(
            folds=folds,
            scoring=scoring,
            X=X,
            y=y,
            time_limit_ms=self._max_time_ms,
            seed=self._seed,
        )

        strategy = PortfolioStrategy(self._models, task=task,
                                     seed=self._seed)

        result = executor.run_strategy(strategy)

        leaderboard = result['leaderboard']
        if leaderboard.length == 0:
            raise ValidationError(
                'PortfolioSearch: no candidates were evaluated')

        self._leaderboard = leaderboard
        self._best_result = leaderboard.best()
        return {'leaderboard': leaderboard, 'bestResult': self._best_result}

    def refit_best(self, X, y):
        """Refit the best candidate on full data."""
        if self._best_result is None:
            raise ValidationError('PortfolioSearch: must call fit() first')
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
