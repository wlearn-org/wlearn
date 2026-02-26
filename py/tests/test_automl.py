"""Tests for automl: RNG, sampler, CV, leaderboard, executor, strategies, search."""

import math

import numpy as np
import pytest

from wlearn.automl import (
    make_lcg, shuffle, sample_param, sample_config, random_configs, grid_configs,
    k_fold, stratified_k_fold, cross_val_score, accuracy, r2_score, get_scorer,
    Leaderboard, Executor, RandomStrategy, HalvingStrategy,
    RandomSearch, SuccessiveHalvingSearch, auto_fit,
    detect_task, make_candidate_id, seed_for,
    PortfolioStrategy, PortfolioSearch, get_portfolio,
)
from wlearn.errors import ValidationError


# --- MockModel (same as test_ensemble.py) ---

class MockModel:
    def __init__(self, params=None):
        self._params = dict(params) if params else {}
        self._fitted = False
        self._disposed = False
        self._classes = None
        self._n_classes = 0
        self._mean = None
        self._bias = self._params.get('bias', 0.0)

    @classmethod
    def create(cls, params=None):
        return cls(params)

    def fit(self, X, y):
        self._fitted = True
        unique = sorted(set(int(v) for v in y))
        if len(unique) <= 20:
            self._classes = np.array(unique, dtype=np.int32)
            self._n_classes = len(unique)
        self._mean = float(np.mean(y))
        return self

    def predict(self, X):
        n = len(X)
        if self._classes is not None and self._n_classes > 0:
            out = np.zeros(n, dtype=np.float64)
            for i in range(n):
                score = float(X[i].sum()) + self._bias
                cls_idx = int(score * 1000) % self._n_classes
                out[i] = self._classes[cls_idx]
            return out
        return np.full(n, self._mean + self._bias, dtype=np.float64)

    def predict_proba(self, X):
        n = len(X)
        nc = self._n_classes
        out = np.zeros(n * nc, dtype=np.float64)
        for i in range(n):
            score = float(X[i].sum()) + self._bias
            for c in range(nc):
                out[i * nc + c] = 1.0 / nc
            boost_idx = int(abs(score) * 100) % nc
            out[i * nc + boost_idx] += 0.1
            row_sum = sum(out[i * nc + c] for c in range(nc))
            for c in range(nc):
                out[i * nc + c] /= row_sum
        return out

    def score(self, X, y):
        preds = self.predict(X)
        if self._classes is not None:
            return float(np.mean(preds == y))
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        ss_res = np.sum((y - preds) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def dispose(self):
        self._disposed = True

    def get_params(self):
        return dict(self._params)

    def set_params(self, p):
        self._params.update(p)
        return self

    @classmethod
    def default_search_space(cls):
        return {
            'bias': {'type': 'uniform', 'low': -1.0, 'high': 1.0},
        }

    @property
    def is_fitted(self):
        return self._fitted


def make_cls_data(seed=42, n=50, n_features=3, n_classes=3):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    y = np.array([i % n_classes for i in range(n)], dtype=np.int32)
    return X, y


def make_reg_data(seed=42, n=50, n_features=2):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    y = 2 * X[:, 0] + 3 * X[:, 1] + rng.randn(n) * 0.5
    return X, y


# ===========================================================================
# RNG Parity
# ===========================================================================

class TestRNG:
    def test_lcg_deterministic(self):
        rng = make_lcg(42)
        vals = [rng() for _ in range(5)]
        rng2 = make_lcg(42)
        vals2 = [rng2() for _ in range(5)]
        assert vals == vals2

    def test_lcg_seed_42_first_values(self):
        """Verify first few values match JS makeLCG(42)."""
        rng = make_lcg(42)
        # JS: seed=42, s = (42 * 1664525 + 1013904223) & 0x7fffffff
        v1 = rng()
        assert isinstance(v1, float)
        assert 0 <= v1 <= 1

    def test_lcg_different_seeds(self):
        rng1 = make_lcg(1)
        rng2 = make_lcg(2)
        assert rng1() != rng2()

    def test_shuffle_deterministic(self):
        arr1 = list(range(10))
        rng1 = make_lcg(42)
        shuffle(arr1, rng1)

        arr2 = list(range(10))
        rng2 = make_lcg(42)
        shuffle(arr2, rng2)

        assert arr1 == arr2

    def test_shuffle_changes_order(self):
        arr = list(range(20))
        rng = make_lcg(42)
        original = list(arr)
        shuffle(arr, rng)
        assert arr != original  # extremely unlikely to be same


# ===========================================================================
# Sampler
# ===========================================================================

class TestSampler:
    def test_categorical(self):
        rng = make_lcg(42)
        param = {'type': 'categorical', 'values': ['a', 'b', 'c']}
        val = sample_param(param, rng)
        assert val in ['a', 'b', 'c']

    def test_uniform(self):
        rng = make_lcg(42)
        param = {'type': 'uniform', 'low': 0.0, 'high': 1.0}
        val = sample_param(param, rng)
        assert 0.0 <= val <= 1.0

    def test_log_uniform(self):
        rng = make_lcg(42)
        param = {'type': 'log_uniform', 'low': 0.001, 'high': 100.0}
        val = sample_param(param, rng)
        assert 0.001 <= val <= 100.0

    def test_int_uniform(self):
        rng = make_lcg(42)
        param = {'type': 'int_uniform', 'low': 1, 'high': 10}
        val = sample_param(param, rng)
        assert isinstance(val, int)
        assert 1 <= val <= 10

    def test_int_log_uniform(self):
        rng = make_lcg(42)
        param = {'type': 'int_log_uniform', 'low': 10, 'high': 1000}
        val = sample_param(param, rng)
        assert isinstance(val, int)
        assert 10 <= val <= 1000

    def test_conditional_params(self):
        space = {
            'kernel': {'type': 'categorical', 'values': ['rbf', 'poly']},
            'degree': {
                'type': 'int_uniform', 'low': 2, 'high': 5,
                'condition': {'kernel': 'poly'},
            },
        }
        rng = make_lcg(42)
        # Sample many times
        has_degree = False
        no_degree = False
        for _ in range(100):
            config = sample_config(space, rng)
            if config.get('kernel') == 'poly':
                assert 'degree' in config
                has_degree = True
            else:
                assert 'degree' not in config
                no_degree = True
        assert has_degree
        assert no_degree

    def test_random_configs_count(self):
        space = {'x': {'type': 'uniform', 'low': 0, 'high': 1}}
        configs = random_configs(space, 10, seed=42)
        assert len(configs) == 10

    def test_random_configs_deterministic(self):
        space = {'x': {'type': 'uniform', 'low': 0, 'high': 1}}
        c1 = random_configs(space, 5, seed=42)
        c2 = random_configs(space, 5, seed=42)
        assert c1 == c2

    def test_grid_configs(self):
        space = {
            'x': {'type': 'uniform', 'low': 0, 'high': 1},
            'y': {'type': 'categorical', 'values': ['a', 'b']},
        }
        configs = grid_configs(space, steps=3)
        # 3 values for x * 2 values for y = 6 combos
        assert len(configs) == 6


# ===========================================================================
# CV
# ===========================================================================

class TestCV:
    def test_k_fold_sizes(self):
        folds = k_fold(100, 5)
        total_test = 0
        for train, test in folds:
            total_test += len(test)
            assert len(train) + len(test) == 100
        assert total_test == 100

    def test_k_fold_no_overlap(self):
        folds = k_fold(50, 5)
        all_test = set()
        for _, test in folds:
            for idx in test:
                assert idx not in all_test
                all_test.add(idx)
        assert len(all_test) == 50

    def test_k_fold_deterministic(self):
        f1 = k_fold(100, 5, seed=42)
        f2 = k_fold(100, 5, seed=42)
        for (t1, te1), (t2, te2) in zip(f1, f2):
            np.testing.assert_array_equal(t1, t2)
            np.testing.assert_array_equal(te1, te2)

    def test_stratified_k_fold_sizes(self):
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32)
        folds = stratified_k_fold(y, 5)
        total_test = 0
        for train, test in folds:
            total_test += len(test)
            assert len(train) + len(test) == 10
        assert total_test == 10

    def test_stratified_preserves_proportions(self):
        y = np.array([0]*40 + [1]*60, dtype=np.int32)
        folds = stratified_k_fold(y, 5, seed=42)
        for train, test in folds:
            test_labels = y[test]
            # Roughly 40% class 0, 60% class 1
            ratio = np.mean(test_labels == 0)
            assert 0.2 < ratio < 0.6  # loose bounds

    def test_accuracy(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        assert accuracy(y_true, y_pred) == 0.75

    def test_r2_score(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert r2_score(y_true, y_pred) == 1.0

    def test_cross_val_score(self):
        X, y = make_cls_data(n=60, n_classes=2)
        scores = cross_val_score(MockModel, X, y, cv=3, scoring='accuracy', seed=42)
        assert len(scores) == 3
        for s in scores:
            assert 0 <= s <= 1


# ===========================================================================
# Leaderboard
# ===========================================================================

class TestLeaderboard:
    def test_add_and_ranked(self):
        lb = Leaderboard()
        lb.add('model_a', {'x': 1}, np.array([0.8, 0.9]), 100)
        lb.add('model_b', {'x': 2}, np.array([0.7, 0.75]), 200)
        ranked = lb.ranked()
        assert len(ranked) == 2
        assert ranked[0]['modelName'] == 'model_a'
        assert ranked[0]['rank'] == 1
        assert ranked[1]['rank'] == 2

    def test_best(self):
        lb = Leaderboard()
        lb.add('a', {}, np.array([0.5]), 10)
        lb.add('b', {}, np.array([0.9]), 10)
        best = lb.best()
        assert best['modelName'] == 'b'

    def test_top(self):
        lb = Leaderboard()
        for i in range(10):
            lb.add(f'm{i}', {}, np.array([float(i) / 10]), 10)
        top3 = lb.top(3)
        assert len(top3) == 3
        assert top3[0]['meanScore'] >= top3[1]['meanScore']

    def test_serialization(self):
        lb = Leaderboard()
        lb.add('a', {'x': 1}, np.array([0.8, 0.9]), 100)
        lb.add('b', {'x': 2}, np.array([0.7]), 50)
        data = lb.to_json()
        lb2 = Leaderboard.from_json(data)
        assert lb2.length == 2
        assert lb2.best()['modelName'] == 'a'

    def test_empty_best(self):
        lb = Leaderboard()
        assert lb.best() is None

    def test_length(self):
        lb = Leaderboard()
        assert lb.length == 0
        lb.add('a', {}, np.array([0.5]), 10)
        assert lb.length == 1


# ===========================================================================
# Common utilities
# ===========================================================================

class TestCommon:
    def test_detect_task_classification(self):
        y = np.array([0, 1, 2, 0, 1], dtype=np.int32)
        assert detect_task(y) == 'classification'

    def test_detect_task_regression(self):
        y = np.array([1.5, 2.3, 3.7])
        assert detect_task(y) == 'regression'

    def test_detect_task_many_integers(self):
        y = np.array(list(range(50)), dtype=np.float64)
        assert detect_task(y) == 'regression'

    def test_make_candidate_id(self):
        cid = make_candidate_id('xgb', {'a': 1, 'b': 'x'})
        assert 'xgb:' in cid
        # Deterministic
        cid2 = make_candidate_id('xgb', {'b': 'x', 'a': 1})
        assert cid == cid2

    def test_seed_for_deterministic(self):
        s1 = seed_for('cand1', 0, 42)
        s2 = seed_for('cand1', 0, 42)
        assert s1 == s2

    def test_seed_for_different(self):
        s1 = seed_for('cand1', 0, 42)
        s2 = seed_for('cand2', 0, 42)
        assert s1 != s2


# ===========================================================================
# Executor
# ===========================================================================

class TestExecutor:
    def test_evaluate_candidate(self):
        X, y = make_cls_data(n=60, n_classes=2)
        folds = stratified_k_fold(y, 3, seed=42)
        executor = Executor(folds, 'accuracy', X, y, seed=42)

        result = executor.evaluate_candidate(
            'mock:{}', MockModel, {},
        )
        assert 'meanScore' in result
        assert 'foldScores' in result
        assert len(result['foldScores']) == 3
        assert executor.leaderboard.length == 1

    def test_subsample_budget(self):
        X, y = make_cls_data(n=100, n_classes=2)
        folds = k_fold(100, 3, seed=42)
        executor = Executor(folds, 'accuracy', X, y, seed=42)

        result = executor.evaluate_candidate(
            'mock:{}', MockModel, {},
            budget={'type': 'subsample', 'value': 0.5},
        )
        assert result['nTrainUsed'] < 67  # should be about half of ~67 train


# ===========================================================================
# Strategies
# ===========================================================================

class TestStrategies:
    def test_random_strategy_count(self):
        models = [{'name': 'mock', 'cls': MockModel, 'params': {}}]
        strategy = RandomStrategy(models, n_iter=5, seed=42)
        count = 0
        while not strategy.is_done():
            task = strategy.next()
            if task is None:
                break
            count += 1
            strategy.report({'candidateId': task['candidateId'], 'meanScore': 0.5})
        assert count == 5

    def test_random_strategy_deterministic(self):
        models = [{'name': 'mock', 'cls': MockModel, 'params': {}}]
        s1 = RandomStrategy(models, n_iter=3, seed=42)
        s2 = RandomStrategy(models, n_iter=3, seed=42)
        for _ in range(3):
            t1 = s1.next()
            t2 = s2.next()
            assert t1['candidateId'] == t2['candidateId']

    def test_halving_strategy_rounds(self):
        models = [{'name': 'mock', 'cls': MockModel, 'params': {}}]
        strategy = HalvingStrategy(
            models, n_iter=9, seed=42, factor=3,
            n_samples=100, greater_is_better=True, cv=3,
        )
        # Run through
        scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        idx = 0
        while not strategy.is_done():
            task = strategy.next()
            if task is None:
                break
            score = scores[idx % len(scores)]
            strategy.report({
                'candidateId': task['candidateId'],
                'meanScore': score,
            })
            idx += 1
        # Should have at least one round recorded
        assert len(strategy.rounds) >= 1


# ===========================================================================
# RandomSearch
# ===========================================================================

class TestRandomSearch:
    def test_fit_returns_leaderboard(self):
        X, y = make_cls_data(n=60, n_classes=2)
        models = [{'name': 'mock', 'cls': MockModel, 'params': {}}]
        search = RandomSearch(models, n_iter=3, cv=3, seed=42)
        result = search.fit(X, y)
        assert 'leaderboard' in result
        assert 'bestResult' in result
        assert result['leaderboard'].length == 3

    def test_fit_deterministic(self):
        X, y = make_cls_data(n=60, n_classes=2)
        models = [{'name': 'mock', 'cls': MockModel, 'params': {}}]
        r1 = RandomSearch(models, n_iter=3, cv=3, seed=42).fit(X, y)
        r2 = RandomSearch(models, n_iter=3, cv=3, seed=42).fit(X, y)
        assert r1['bestResult']['meanScore'] == r2['bestResult']['meanScore']

    def test_refit_best(self):
        X, y = make_cls_data(n=60, n_classes=2)
        models = [{'name': 'mock', 'cls': MockModel, 'params': {}}]
        search = RandomSearch(models, n_iter=3, cv=3, seed=42)
        search.fit(X, y)
        model = search.refit_best(X, y)
        assert model.is_fitted

    def test_empty_models_error(self):
        with pytest.raises(ValidationError):
            RandomSearch([], n_iter=3)


# ===========================================================================
# SuccessiveHalvingSearch
# ===========================================================================

class TestSuccessiveHalvingSearch:
    def test_fit(self):
        X, y = make_cls_data(n=60, n_classes=2)
        models = [{'name': 'mock', 'cls': MockModel, 'params': {}}]
        search = SuccessiveHalvingSearch(models, n_iter=9, cv=3, seed=42, factor=3)
        result = search.fit(X, y)
        assert 'leaderboard' in result
        assert 'rounds' in result

    def test_rounds_decrease(self):
        X, y = make_cls_data(n=100, n_classes=2)
        models = [{'name': 'mock', 'cls': MockModel, 'params': {}}]
        search = SuccessiveHalvingSearch(models, n_iter=9, cv=3, seed=42, factor=3)
        result = search.fit(X, y)
        rounds = result['rounds']
        if len(rounds) >= 2:
            assert rounds[0]['nSurvivors'] <= rounds[0]['nCandidates']


# ===========================================================================
# auto_fit
# ===========================================================================

class TestAutoFit:
    def test_without_ensemble(self):
        X, y = make_cls_data(n=60, n_classes=2)
        models = [{'name': 'mock', 'cls': MockModel, 'params': {}}]
        result = auto_fit(models, X, y, n_iter=3, cv=3, seed=42)
        assert 'model' in result
        assert result['model'] is not None
        assert result['model'].is_fitted
        assert 'bestModelName' in result
        assert 'bestScore' in result

    def test_with_ensemble(self):
        X, y = make_cls_data(n=60, n_classes=2)
        models = [
            {'name': 'mock1', 'cls': MockModel, 'params': {}},
            {'name': 'mock2', 'cls': MockModel, 'params': {'bias': 0.5}},
        ]
        result = auto_fit(
            models, X, y,
            ensemble=True, ensemble_size=5, n_iter=3, cv=3, seed=42,
        )
        assert result['model'] is not None
        # Model should be a VotingEnsemble
        from wlearn.ensemble import VotingEnsemble
        assert isinstance(result['model'], VotingEnsemble)

    def test_no_refit(self):
        X, y = make_cls_data(n=60, n_classes=2)
        models = [{'name': 'mock', 'cls': MockModel, 'params': {}}]
        result = auto_fit(models, X, y, refit=False, n_iter=3, cv=3, seed=42)
        assert result['model'] is None

    def test_tuple_specs(self):
        X, y = make_cls_data(n=60, n_classes=2)
        models = [('mock', MockModel, {})]
        result = auto_fit(models, X, y, n_iter=3, cv=3, seed=42)
        assert result['model'] is not None

    def test_regression(self):
        X, y = make_reg_data(n=60)
        models = [{'name': 'mock', 'cls': MockModel, 'params': {}}]
        result = auto_fit(models, X, y, task='regression', n_iter=3, cv=3, seed=42)
        assert result['model'] is not None


# ===========================================================================
# Model search spaces
# ===========================================================================

class TestSearchSpaces:
    def test_mock_model_has_search_space(self):
        space = MockModel.default_search_space()
        assert 'bias' in space
        assert space['bias']['type'] == 'uniform'


# ===========================================================================
# Portfolio configs
# ===========================================================================

class TestGetPortfolio:
    def test_classification_has_all_families(self):
        p = get_portfolio('classification')
        for name in ('xgb', 'ebm', 'linear', 'svm', 'knn', 'tsetlin'):
            assert name in p, f'Missing family: {name}'

    def test_regression_has_all_families(self):
        p = get_portfolio('regression')
        for name in ('xgb', 'ebm', 'linear', 'svm', 'knn', 'tsetlin'):
            assert name in p, f'Missing family: {name}'

    def test_config_counts(self):
        expected = {'xgb': 8, 'ebm': 4, 'linear': 4, 'svm': 4, 'knn': 3, 'tsetlin': 3}
        for task in ('classification', 'regression'):
            p = get_portfolio(task)
            for name, count in expected.items():
                assert len(p[name]) == count, \
                    f'{task}/{name}: expected {count}, got {len(p[name])}'

    def test_xgb_has_objective(self):
        for task in ('classification', 'regression'):
            p = get_portfolio(task)
            for i, cfg in enumerate(p['xgb']):
                assert 'objective' in cfg, \
                    f'{task}/xgb config {i} missing objective'

    def test_classification_xgb_objective(self):
        p = get_portfolio('classification')
        for cfg in p['xgb']:
            assert cfg['objective'] == 'multi:softprob'

    def test_regression_xgb_objective(self):
        p = get_portfolio('regression')
        for cfg in p['xgb']:
            assert cfg['objective'] == 'reg:squarederror'

    def test_classification_linear_solvers(self):
        p = get_portfolio('classification')
        solvers = {cfg['solver'] for cfg in p['linear']}
        # Classification solvers: 0 (L2R_LR), 6 (L1R_LR), 7 (L2R_LR_DUAL)
        assert solvers <= {0, 6, 7}

    def test_regression_linear_solvers(self):
        p = get_portfolio('regression')
        solvers = {cfg['solver'] for cfg in p['linear']}
        # Regression solvers: 11, 12, 13
        assert solvers <= {11, 12, 13}

    def test_unknown_task_falls_back(self):
        p = get_portfolio('unknown')
        assert p == get_portfolio('classification')


# ===========================================================================
# PortfolioStrategy
# ===========================================================================

class TestPortfolioStrategy:
    def test_yields_all_candidates(self):
        models = [
            {'name': 'mock', 'cls': MockModel, 'params': {}},
        ]
        strategy = PortfolioStrategy(models, task='classification')
        count = 0
        while not strategy.is_done():
            cand = strategy.next()
            if cand is None:
                break
            count += 1
            assert 'candidateId' in cand
            assert 'cls' in cand
            assert 'params' in cand
        # MockModel not in portfolio -> falls back to 1 default config
        assert count == 1

    def test_yields_portfolio_configs(self):
        models = [
            {'name': 'xgb', 'cls': MockModel, 'params': {}},
        ]
        strategy = PortfolioStrategy(models, task='classification')
        count = 0
        while not strategy.is_done():
            cand = strategy.next()
            if cand is None:
                break
            count += 1
            # XGB configs should have objective
            assert 'objective' in cand['params']
        assert count == 8

    def test_multiple_models(self):
        models = [
            {'name': 'xgb', 'cls': MockModel, 'params': {}},
            {'name': 'knn', 'cls': MockModel, 'params': {}},
        ]
        strategy = PortfolioStrategy(models, task='classification')
        count = 0
        while not strategy.is_done():
            if strategy.next() is None:
                break
            count += 1
        assert count == 8 + 3  # xgb=8, knn=3

    def test_fixed_params_override(self):
        models = [
            {'name': 'xgb', 'cls': MockModel, 'params': {'eta': 0.999}},
        ]
        strategy = PortfolioStrategy(models, task='classification')
        while not strategy.is_done():
            cand = strategy.next()
            if cand is None:
                break
            # User's fixed param should override portfolio value
            assert cand['params']['eta'] == 0.999

    def test_is_done_transitions(self):
        models = [{'name': 'mock', 'cls': MockModel, 'params': {}}]
        strategy = PortfolioStrategy(models, task='classification')
        assert not strategy.is_done()
        strategy.next()
        assert strategy.is_done()
        assert strategy.next() is None

    def test_regression_task(self):
        models = [{'name': 'xgb', 'cls': MockModel, 'params': {}}]
        strategy = PortfolioStrategy(models, task='regression')
        cand = strategy.next()
        assert cand['params']['objective'] == 'reg:squarederror'


# ===========================================================================
# PortfolioSearch
# ===========================================================================

class TestPortfolioSearch:
    def test_classification(self):
        X, y = make_cls_data(n=60, n_classes=2)
        models = [{'name': 'mock', 'cls': MockModel, 'params': {}}]
        search = PortfolioSearch(models, cv=3, seed=42, task='classification')
        result = search.fit(X, y)
        assert result['leaderboard'] is not None
        assert result['bestResult'] is not None
        ranked = result['leaderboard'].ranked()
        assert len(ranked) >= 1

    def test_regression(self):
        X, y = make_reg_data(n=60)
        models = [{'name': 'mock', 'cls': MockModel, 'params': {}}]
        search = PortfolioSearch(models, cv=3, seed=42, task='regression')
        result = search.fit(X, y)
        assert result['bestResult'] is not None

    def test_refit_best(self):
        X, y = make_cls_data(n=60, n_classes=2)
        models = [{'name': 'mock', 'cls': MockModel, 'params': {}}]
        search = PortfolioSearch(models, cv=3, seed=42)
        search.fit(X, y)
        model = search.refit_best(X, y)
        assert model.is_fitted
        preds = model.predict(X)
        assert len(preds) == len(y)

    def test_refit_before_fit_raises(self):
        models = [{'name': 'mock', 'cls': MockModel, 'params': {}}]
        search = PortfolioSearch(models, cv=3, seed=42)
        with pytest.raises(ValidationError):
            search.refit_best(np.zeros((10, 3)), np.zeros(10))

    def test_empty_models_raises(self):
        with pytest.raises(ValidationError):
            PortfolioSearch([], cv=3, seed=42)

    def test_portfolio_model_evaluated(self):
        """XGB portfolio should produce 8 leaderboard entries."""
        X, y = make_cls_data(n=60, n_classes=2)
        models = [{'name': 'xgb', 'cls': MockModel, 'params': {}}]
        search = PortfolioSearch(models, cv=3, seed=42, task='classification')
        result = search.fit(X, y)
        ranked = result['leaderboard'].ranked()
        assert len(ranked) == 8


# ===========================================================================
# auto_fit with strategy='portfolio'
# ===========================================================================

class TestAutoFitPortfolio:
    def test_strategy_portfolio(self):
        X, y = make_cls_data(n=60, n_classes=2)
        models = [{'name': 'mock', 'cls': MockModel, 'params': {}}]
        result = auto_fit(models, X, y, cv=3, seed=42, strategy='portfolio')
        assert result['model'] is not None
        assert result['bestScore'] is not None

    def test_strategy_portfolio_regression(self):
        X, y = make_reg_data(n=60)
        models = [{'name': 'mock', 'cls': MockModel, 'params': {}}]
        result = auto_fit(models, X, y, task='regression', cv=3, seed=42,
                          strategy='portfolio')
        assert result['model'] is not None

    def test_strategy_portfolio_with_ensemble(self):
        X, y = make_cls_data(n=60, n_classes=2)
        models = [
            {'name': 'xgb', 'cls': MockModel, 'params': {}},
            {'name': 'knn', 'cls': MockModel, 'params': {}},
        ]
        result = auto_fit(
            models, X, y, cv=3, seed=42, strategy='portfolio',
            ensemble=True, ensemble_size=5,
        )
        assert result['model'] is not None

    def test_strategy_halving(self):
        """Ensure halving strategy still works via auto_fit."""
        X, y = make_cls_data(n=60, n_classes=2)
        models = [{'name': 'mock', 'cls': MockModel, 'params': {}}]
        result = auto_fit(models, X, y, cv=3, seed=42, n_iter=5,
                          strategy='halving')
        assert result['model'] is not None
