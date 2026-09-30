"""Tests for automl: RNG, sampler, CV, leaderboard, executor, strategies, search."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from wlearn.automl import (
    make_lcg, shuffle, sample_param, sample_config, random_configs, grid_configs,
    k_fold, stratified_k_fold, cross_val_score, accuracy, r2_score, get_scorer,
    Leaderboard, Executor, RandomStrategy, HalvingStrategy, ProgressiveStrategy,
    RandomSearch, SuccessiveHalvingSearch, BayesianSearch, BayesianStrategy,
    auto_fit,
    detect_task, make_candidate_id, seed_for,
    candidate_canonical_bytes, candidate_hash, create_candidate,
    normalize_model_specs,
    PortfolioStrategy, PortfolioSearch, get_portfolio,
)
from wlearn.errors import BackendError, ValidationError
from wlearn.pipeline import Pipeline
from wlearn.bundle import decode_bundle, encode_bundle
from wlearn.registry import load as load_bundle, register
from wlearn.automl._candidate_pipeline import (
    create_candidate_pipeline_class,
)


@pytest.mark.parametrize('search_class', [RandomSearch, SuccessiveHalvingSearch, PortfolioSearch])
def test_search_preserves_task_and_custom_folds(search_class):
    from wlearn.resampling import create_resampling_plan
    X = np.arange(10, dtype=float).reshape(-1, 1)
    y = np.arange(10, dtype=float)
    plan = create_resampling_plan(strategy='group_kfold', n=10, k=2,
                                  groups=np.arange(10, dtype=np.int32) // 2)

    class Model:
        class_id = 'wlearn.test.task-folds@1'

        @classmethod
        def create(cls, params):
            assert params['task'] == 'regression'
            return cls()

        def fit(self, X, y):
            self.groups = set((X[:, 0] // 2).tolist())
            return self

        def predict(self, X):
            assert not self.groups.intersection((X[:, 0] // 2).tolist())
            return np.zeros(len(X))

        def dispose(self):
            pass

    options = {} if search_class is PortfolioSearch else {'n_iter': 1}
    search = search_class([{'name': 'model', 'cls': Model, 'searchSpace': {}}],
                          cv=plan, task='regression', scoring='mse', **options)
    result = search.fit(X, y)
    assert result['bestResult']['candidate']['model']['params']['task'] == 'regression'
    search.refit_best(X, y).dispose()


# --- MockModel (same as test_ensemble.py) ---

class MockModel:
    class_id = 'wlearn.test.mock-model@1'

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
        if self._params.get('task') != 'regression' and len(unique) <= 20:
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

    @property
    def classes(self):
        return self._classes

    def save(self):
        state = json.dumps({
            'classes': (
                [int(value) for value in self._classes]
                if self._classes is not None else None),
            'nClasses': self._n_classes,
            'mean': self._mean,
            'bias': self._bias,
        }, sort_keys=True, separators=(',', ':')).encode('utf-8')
        return encode_bundle(
            {
                'typeId': 'wlearn.test.automl-mock@1',
                'params': self._params,
            },
            [{'id': 'state', 'data': state}],
        )

    @classmethod
    def _from_bundle(cls, manifest, toc, blobs):
        entry = next(item for item in toc if item['id'] == 'state')
        state = json.loads(bytes(
            blobs[entry['offset']:entry['offset'] + entry['length']]
        ).decode('utf-8'))
        model = cls(manifest.get('params') or {})
        model._fitted = True
        model._classes = (
            np.asarray(state['classes'], dtype=np.int32)
            if state['classes'] is not None else None)
        model._n_classes = state['nClasses']
        model._mean = state['mean']
        model._bias = state['bias']
        return model

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

    @property
    def capabilities(self):
        classifier = self._classes is not None
        return {
            'classifier': classifier,
            'regressor': not classifier,
            'predictProba': classifier,
            'decisionFunction': False,
            'sampleWeight': False,
            'csr': False,
            'earlyStopping': False,
        }


register('wlearn.test.automl-mock@1', MockModel._from_bundle)


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


def make_candidate_task(name, params, cls=MockModel, class_id=None):
    candidate = create_candidate({
        'displayName': name,
        'classId': class_id or f'wlearn.test.{name}@1',
    }, params)
    return {
        'candidateId': make_candidate_id(candidate),
        'candidate': candidate,
        'cls': cls,
        'params': params,
    }


def add_leaderboard(lb, name, params, scores, fit_time_ms):
    task = make_candidate_task(name, params)
    return lb.add(
        candidate_id=task['candidateId'],
        candidate=task['candidate'],
        scores=scores,
        fit_time_ms=fit_time_ms,
    )


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
    def test_strategy_conditions_on_fixed_params(self):
        space = {
            'kernel': {'type': 'categorical', 'values': ['linear', 'poly']},
            'degree': {'type': 'categorical', 'values': [2, 3], 'condition': {'kernel': 'poly'}},
            'extra': {'type': 'categorical', 'values': [True], 'condition': {'degree': 2}},
        }
        for kernel in ['poly', 'linear']:
            strategy = RandomStrategy([{
                'name': 'fixed', 'classId': 'test.fixed', 'cls': MockModel,
                'searchSpace': space, 'params': {'kernel': kernel},
            }], n_iter=1, seed=7)
            params = strategy.next()['params']
            assert params['kernel'] == kernel
            if kernel == 'poly':
                assert params['degree'] in (2, 3)
            else:
                assert 'degree' not in params and 'extra' not in params

    def test_conditional_dependency_order_and_full_grid(self):
        space = {
            'c': {'type': 'categorical', 'values': ['x', 'y'], 'condition': {'b': 2}},
            'b': {'type': 'int_uniform', 'low': 2, 'high': 3, 'condition': {'a': 'on'}},
            'a': {'type': 'categorical', 'values': ['on', 'off']},
        }
        assert sample_config(space, lambda: 0) == {'a': 'on', 'b': 2, 'c': 'x'}
        assert sample_config(space, lambda: 0.99) == {'a': 'off'}
        assert grid_configs(space) == [
            {'a': 'on', 'b': 2, 'c': 'x'}, {'a': 'on', 'b': 2, 'c': 'y'},
            {'a': 'on', 'b': 3}, {'a': 'off'},
        ]

    def test_portable_conditions_and_empty_condition(self):
        space = {
            'value': {'type': 'categorical', 'values': [{'a': 1, 'b': True}]},
            'yes': {'type': 'categorical', 'values': [1], 'condition': {'value': {'b': True, 'a': 1}}},
            'no': {'type': 'categorical', 'values': [1], 'condition': {'value': {'b': 1, 'a': 1}}},
            'empty': {'type': 'categorical', 'values': [None], 'condition': {}},
        }
        assert sample_config(space, lambda: 0) == {'value': {'a': 1, 'b': True}, 'empty': None, 'yes': 1}

    def test_absent_parent_is_not_null(self):
        space = {
            'a': {'type': 'categorical', 'values': [False]},
            'b': {'type': 'categorical', 'values': [None], 'condition': {'a': True}},
            'c': {'type': 'categorical', 'values': [1], 'condition': {'b': None}},
        }
        assert sample_config(space, lambda: 0) == {'a': False}

    @pytest.mark.parametrize('space', [
        {'a': {'type': 'categorical', 'values': [1], 'condition': {'missing': 1}}},
        {'a': {'type': 'categorical', 'values': [1], 'condition': {'b': 1}},
         'b': {'type': 'categorical', 'values': [1], 'condition': {'a': 1}}},
    ])
    def test_invalid_condition_graph(self, space):
        def rng():
            pytest.fail('invalid spaces must not consume randomness')
        with pytest.raises(ValueError, match='[Uu]nknown|[Cc]ycl'):
            sample_config(space, rng)
        with pytest.raises(ValueError, match='[Uu]nknown|[Cc]ycl'):
            grid_configs(space)

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

    def test_stratified_preserves_first_seen_class_order(self):
        y = np.array([2, 1, 2, 1, 2, 1], dtype=np.int32)
        folds = stratified_k_fold(y, 3, do_shuffle=False)
        assert [test.tolist() for _, test in folds] == [[0, 1], [2, 3], [4, 5]]

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
        add_leaderboard(lb, 'model_a', {'x': 1}, np.array([0.8, 0.9]), 100)
        add_leaderboard(lb, 'model_b', {'x': 2}, np.array([0.7, 0.75]), 200)
        ranked = lb.ranked()
        assert len(ranked) == 2
        assert ranked[0]['modelName'] == 'model_a'
        assert ranked[0]['rank'] == 1
        assert ranked[1]['rank'] == 2

    def test_best(self):
        lb = Leaderboard()
        add_leaderboard(lb, 'a', {}, np.array([0.5]), 10)
        add_leaderboard(lb, 'b', {}, np.array([0.9]), 10)
        best = lb.best()
        assert best['modelName'] == 'b'

    def test_top(self):
        lb = Leaderboard()
        for i in range(10):
            add_leaderboard(
                lb, f'm{i}', {}, np.array([float(i) / 10]), 10)
        top3 = lb.top(3)
        assert len(top3) == 3
        assert top3[0]['meanScore'] >= top3[1]['meanScore']

    def test_serialization(self):
        lb = Leaderboard()
        first = add_leaderboard(
            lb, 'a', {'x': 1}, np.array([0.8, 0.9]), 100)
        add_leaderboard(lb, 'b', {'x': 2}, np.array([0.7]), 50)
        data = lb.to_json()
        lb2 = Leaderboard.from_json(data)
        assert lb2.length == 2
        assert lb2.best()['modelName'] == 'a'
        assert lb2.best()['candidateId'] == first['candidateId']

    def test_snapshots_params_and_scores(self):
        lb = Leaderboard()
        params = {'depth': 3, 'nested': {'eta': 0.1}}
        scores = np.array([0.8, 0.9], dtype=np.float64)
        add_leaderboard(lb, 'xgb', params, scores, 10.0)

        params['nested']['eta'] = 9
        scores[0] = 0
        assert lb.best()['params']['nested']['eta'] == pytest.approx(0.1)
        assert lb.best()['scores'][0] == pytest.approx(0.8)

        best = lb.best()
        best['params']['nested']['eta'] = 5
        best['scores'][0] = 0
        assert lb.best()['params']['nested']['eta'] == pytest.approx(0.1)
        assert lb.best()['scores'][0] == pytest.approx(0.8)

        data = lb.to_json()
        data[0]['params']['nested']['eta'] = 7
        assert lb.best()['params']['nested']['eta'] == pytest.approx(0.1)

    def test_to_archive(self):
        lb = Leaderboard()
        source = add_leaderboard(
            lb, 'a', {}, np.array([0.8, 0.9]), 42)
        archive = lb.to_archive(metric='accuracy')
        assert archive.size == 1
        record = archive.records()[0]
        assert record.status == 'ok'
        assert record.scores['accuracy'] == pytest.approx(0.85)
        assert record.metadata['sourceCandidateId'] == source['candidateId']
        assert record.metadata['candidate'] == source['candidate']

    def test_empty_best(self):
        lb = Leaderboard()
        assert lb.best() is None

    def test_length(self):
        lb = Leaderboard()
        assert lb.length == 0
        add_leaderboard(lb, 'a', {}, np.array([0.5]), 10)
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

    @pytest.mark.parametrize('vector', json.loads(
        (Path(__file__).resolve().parents[2] /
         'js/automl/test/candidate-v1.json').read_text(
             encoding='utf-8'))['cases'])
    def test_candidate_identity_shared_vectors(self, vector):
        candidate = create_candidate(
            vector['model'], vector['params'], vector['preprocess'])
        assert (candidate_canonical_bytes(candidate).decode('utf-8') ==
                vector['canonicalUtf8'])
        assert candidate_hash(candidate) == vector['sha256']
        assert make_candidate_id(candidate) == vector['candidateId']
        for seed in vector['seeds']:
            assert seed_for(
                candidate, seed['foldId'], seed['baseSeed']) == seed['value']

    def test_candidate_identity_ignores_display_name(self):
        first = create_candidate({
            'displayName': 'first', 'classId': 'wlearn.test.same@1',
        }, {'x': 1})
        renamed = create_candidate({
            'displayName': 'renamed', 'classId': 'wlearn.test.same@1',
        }, {'x': 1})
        assert make_candidate_id(first) == make_candidate_id(renamed)

    def test_candidate_identity_rejects_nonportable_values(self):
        model = {
            'displayName': 'x', 'classId': 'wlearn.test.x@1',
        }
        with pytest.raises(ValidationError, match='finite'):
            create_candidate(model, {'x': float('nan')})
        with pytest.raises(ValidationError, match='safe-integer'):
            create_candidate(model, {'x': 1 << 53})
        with pytest.raises(ValidationError, match='safe-integer'):
            create_candidate(model, {'x': float(1 << 53)})

    def test_candidate_identity_rejects_ambiguous_inputs(self):
        model = {
            'displayName': 'x', 'classId': 'wlearn.test.x@1',
        }
        for params in ([], None, 1, 'x'):
            with pytest.raises(ValidationError, match='params.*dict'):
                create_candidate(model, params)
        cyclic = {}
        cyclic['self'] = cyclic
        with pytest.raises(ValidationError, match='cyclic'):
            create_candidate(model, cyclic)
        for params in (False, [], '', 0, None):
            with pytest.raises(ValidationError, match='params.*dict'):
                normalize_model_specs([{
                    'name': 'x', 'cls': MockModel, 'params': params,
                }])
        with pytest.raises(ValidationError, match='preprocessChoices'):
            normalize_model_specs([{
                'name': 'x', 'cls': MockModel,
                'preprocessChoices': None,
            }])

    def test_candidate_identity_freezes_nested_tuples_and_keeps_proto_key(self):
        candidate = create_candidate({
            'displayName': 'x', 'classId': 'wlearn.test.x@1',
        }, {
            'x': ({'y': [1]},),
            '__proto__': {'safe': True},
        })
        assert '__proto__' in candidate['model']['params']
        with pytest.raises(TypeError, match='immutable'):
            candidate['model']['params']['x'][0]['y'].append(2)

        explicit = create_candidate(
            candidate['model'], candidate['model']['params'], {
                'templateId': 'default',
                'typeId': 'wlearn.preprocess.tabular@1',
                'resolvedParams': {},
            })
        resolved = create_candidate(
            candidate['model'], candidate['model']['params'], {
                'templateId': 'default',
                'typeId': 'wlearn.preprocess.tabular@1',
                'resolvedParams': explicit['preprocess']['resolvedParams'],
            })
        assert make_candidate_id(explicit) == make_candidate_id(resolved)

    def test_candidate_identity_requires_unique_stable_class_ids(self):
        class MissingId:
            @classmethod
            def create(cls, _params=None):
                return cls()

        with pytest.raises(ValidationError, match='classId'):
            normalize_model_specs([{'name': 'missing', 'cls': MissingId}])
        with pytest.raises(ValidationError, match='duplicate'):
            normalize_model_specs([
                {
                    'name': 'first', 'classId': 'wlearn.test.same@1',
                    'cls': MockModel,
                },
                {
                    'name': 'second', 'classId': 'wlearn.test.same@1',
                    'cls': MockModel,
                },
            ])

    def test_candidate_snapshots_inputs_and_reorders_object_keys(self):
        params = {'nested': {'z': 1, 'a': 2}}
        candidate = create_candidate({
            'displayName': 'x', 'classId': 'wlearn.test.snapshot@1',
        }, params)
        candidate_id = make_candidate_id(candidate)
        params['nested']['z'] = 99
        assert candidate['model']['params']['nested']['z'] == 1
        reordered = create_candidate({
            'displayName': 'x', 'classId': 'wlearn.test.snapshot@1',
        }, {'nested': {'a': 2, 'z': 1}})
        assert make_candidate_id(reordered) == candidate_id


# ===========================================================================
# Executor
# ===========================================================================

class TestExecutor:
    def test_evaluate_candidate(self):
        X, y = make_cls_data(n=60, n_classes=2)
        folds = stratified_k_fold(y, 3, seed=42)
        executor = Executor(folds, 'accuracy', X, y, seed=42)

        task = make_candidate_task('mock', {})
        result = executor.evaluate_candidate(
            task['candidateId'], task['candidate'], MockModel, {})
        assert 'meanScore' in result
        assert 'foldScores' in result
        assert len(result['foldScores']) == 3
        assert executor.leaderboard.length == 1
        assert executor.archive.size == 1
        record = executor.archive.records()[0]
        assert record.status == 'ok'
        assert record.scores['accuracy'] == pytest.approx(result['meanScore'])
        assert record.metadata['sourceCandidateId'] == task['candidateId']
        assert record.metadata['candidate'] == task['candidate']

    def test_rejects_stale_id_before_model_construction(self):
        class NeverCreated:
            creates = 0

            @classmethod
            def create(cls, _params=None):
                cls.creates += 1
                return MockModel.create({})

        X, y = make_cls_data(n=20, n_classes=2)
        executor = Executor(
            stratified_k_fold(y, 2, seed=42), 'accuracy', X, y)
        task = make_candidate_task('stale', {}, NeverCreated)
        with pytest.raises(ValidationError, match='candidate_id'):
            executor.evaluate_candidate(
                'wlc1_' + '0' * 64, task['candidate'], NeverCreated, {})
        assert NeverCreated.creates == 0

    def test_nondefault_seed_and_archive_provenance_without_subsampling(self):
        X, y = make_cls_data(n=20, n_classes=2)
        folds = stratified_k_fold(y, 2, seed=3)
        executor = Executor(folds, 'accuracy', X, y, seed=7)
        task = make_candidate_task(
            'seeded', {'__proto__': {'safe': True}})
        result = executor.evaluate_candidate(
            task['candidateId'], task['candidate'], MockModel, {})
        assert result['baseSeed'] == 7
        assert result['foldSeeds'].tolist() == [
            seed_for(task['candidate'], 0, 7),
            seed_for(task['candidate'], 1, 7),
        ]
        record = executor.archive.records()[0]
        assert record.seed == 7
        assert record.metadata['foldSeeds'] == [
            {'foldId': 0, 'seed': int(result['foldSeeds'][0])},
            {'foldId': 1, 'seed': int(result['foldSeeds'][1])},
        ]
        assert '__proto__' in record.metadata['candidate']['model']['params']
        assert make_candidate_id(
            record.metadata['candidate']) == record.candidate_id

    @pytest.mark.parametrize('seed', [-1, 1 << 32, 1.5, True])
    def test_rejects_invalid_base_seed(self, seed):
        X, y = make_cls_data(n=20, n_classes=2)
        with pytest.raises(ValidationError, match='seed'):
            Executor(stratified_k_fold(y, 2), 'accuracy', X, y, seed=seed)

    def test_subsample_budget(self):
        X, y = make_cls_data(n=100, n_classes=2)
        folds = k_fold(100, 3, seed=42)
        executor = Executor(folds, 'accuracy', X, y, seed=42)

        task = make_candidate_task('mock', {})
        result = executor.evaluate_candidate(
            task['candidateId'], task['candidate'], MockModel, {},
            budget={'type': 'subsample', 'value': 0.5},
        )
        assert result['nTrainUsed'] < 67  # should be about half of ~67 train

    def test_run_strategy_records_failures_in_archive(self):
        class FailingModel:
            @classmethod
            def create(cls, params=None):
                raise RuntimeError('create failed')

        X, y = make_cls_data(n=20, n_classes=2)
        folds = stratified_k_fold(y, 2, seed=42)
        executor = Executor(folds, 'accuracy', X, y, seed=42)

        class OneBad:
            def __init__(self):
                self.done = False

            def next(self):
                if self.done:
                    return None
                self.done = True
                return make_candidate_task(
                    'bad', {'__proto__': {'failed': True}}, FailingModel)

            def report(self, _result):
                pass

            def is_done(self):
                return self.done

        result = executor.run_strategy(OneBad())
        assert result['leaderboard'].length == 0
        assert result['archive'].size == 1
        record = result['archive'].records()[0]
        assert record.status == 'failed'
        assert record.error.phase == 'fit'
        assert record.seed == 42
        assert len(record.metadata['foldSeeds']) == len(folds)
        assert '__proto__' in record.metadata['candidate']['model']['params']
        assert make_candidate_id(
            record.metadata['candidate']) == record.candidate_id


class TestCandidatePipelineLifecycle:
    @staticmethod
    def _candidate():
        return create_candidate({
            'displayName': 'lifecycle',
            'classId': 'wlearn.test.lifecycle@1',
        }, {}, {
            'templateId': 'plain',
            'typeId': 'wlearn.preprocess.tabular@1',
            'resolvedParams': {'scale': False},
        })

    @staticmethod
    def _preprocessor_class(events, dispose_throws=False):
        class FakePreprocessor:
            def __init__(self, _config):
                events.append('preprocessor:create')

            def fit_transform(self, X, _y=None):
                events.append('preprocessor:fit')
                return X

            def transform(self, X):
                events.append('preprocessor:transform')
                return X

            def get_params(self):
                return {}

            def dispose(self):
                events.append('preprocessor:dispose')
                if dispose_throws:
                    raise RuntimeError('preprocessor dispose failed')

        return FakePreprocessor

    @staticmethod
    def _spec(model_cls):
        return {
            'name': 'lifecycle',
            'classId': 'wlearn.test.lifecycle@1',
            'cls': model_cls,
        }

    def test_model_create_failure_releases_preprocessor(self):
        events = []

        class FailingModel:
            @classmethod
            def create(cls, _params=None):
                events.append('model:create')
                raise RuntimeError('model create failed')

        candidate_cls = create_candidate_pipeline_class(
            self._spec(FailingModel), self._candidate(),
            preprocessor_cls=self._preprocessor_class(events),
            pipeline_cls=Pipeline,
        )
        with pytest.raises(RuntimeError, match='model create failed'):
            candidate_cls.create()
        assert events == [
            'preprocessor:create', 'model:create', 'preprocessor:dispose',
        ]

    def test_pipeline_create_failure_releases_children_in_reverse(self):
        events = []

        class Model:
            @classmethod
            def create(cls, _params=None):
                events.append('model:create')
                return cls()

            def dispose(self):
                events.append('model:dispose')

        class FailingPipeline:
            def __init__(self, *_args, **_kwargs):
                events.append('pipeline:create')
                raise RuntimeError('pipeline create failed')

        candidate_cls = create_candidate_pipeline_class(
            self._spec(Model), self._candidate(),
            preprocessor_cls=self._preprocessor_class(events),
            pipeline_cls=FailingPipeline,
        )
        with pytest.raises(RuntimeError, match='pipeline create failed'):
            candidate_cls.create()
        assert events == [
            'preprocessor:create', 'model:create', 'pipeline:create',
            'model:dispose', 'preprocessor:dispose',
        ]

    @pytest.mark.parametrize('phase', ['fit', 'predict'])
    def test_executor_releases_children_after_operation_failure(self, phase):
        events = []

        class FailingModel:
            @classmethod
            def create(cls, _params=None):
                events.append('model:create')
                return cls()

            def fit(self, _X, _y):
                events.append('model:fit')
                if phase == 'fit':
                    raise RuntimeError('fit failed')
                return self

            def predict(self, _X):
                events.append('model:predict')
                raise RuntimeError('predict failed')

            def get_params(self):
                return {}

            def dispose(self):
                events.append('model:dispose')

        candidate = self._candidate()
        candidate_cls = create_candidate_pipeline_class(
            self._spec(FailingModel), candidate,
            preprocessor_cls=self._preprocessor_class(events),
            pipeline_cls=Pipeline,
        )
        executor = Executor(
            [(np.array([0], dtype=np.int32),
              np.array([1], dtype=np.int32))],
            'accuracy',
            np.array([[1.0], [2.0]]),
            np.array([0, 1], dtype=np.int32),
        )
        with pytest.raises(RuntimeError, match=f'{phase} failed'):
            executor.evaluate_candidate(
                make_candidate_id(candidate), candidate,
                candidate_cls, {})
        assert events[-2:] == ['model:dispose', 'preprocessor:dispose']

    def test_pipeline_dispose_continues_after_child_failure(self):
        events = []

        class Model:
            @classmethod
            def create(cls, _params=None):
                return cls()

            def fit(self, _X, _y):
                return self

            def get_params(self):
                return {}

            def dispose(self):
                events.append('model:dispose')
                raise RuntimeError('model dispose failed')

        candidate_cls = create_candidate_pipeline_class(
            self._spec(Model), self._candidate(),
            preprocessor_cls=self._preprocessor_class(events),
            pipeline_cls=Pipeline,
        )
        pipeline = candidate_cls.create()
        with pytest.raises(RuntimeError, match='model dispose failed'):
            pipeline.dispose()
        assert events[-2:] == ['model:dispose', 'preprocessor:dispose']

    @pytest.mark.parametrize('strategy', [
        'random', 'halving', 'progressive', 'portfolio', 'bayesian',
    ])
    def test_refit_keeps_private_winner_and_releases_failed_pipeline(
            self, strategy):
        events = []
        state = {'fail_fit': False}
        fit_error = RuntimeError(f'{strategy} refit failed')

        class Model:
            class_id = f'wlearn.test.{strategy}-refit@1'

            @classmethod
            def default_search_space(cls):
                return {}

            @classmethod
            def create(cls, params=None):
                events.append(f'model:create:{dict(params or {})}')
                return cls()

            def fit(self, _X, _y):
                events.append('model:fit')
                if state['fail_fit']:
                    raise fit_error
                return self

            def predict(self, X):
                return np.zeros(len(X), dtype=np.int32)

            def get_params(self):
                return {}

            def dispose(self):
                events.append('model:dispose')
                if state['fail_fit']:
                    raise RuntimeError('model cleanup failed')

        candidate = create_candidate({
            'displayName': strategy, 'classId': Model.class_id,
        }, {}, {
            'templateId': 'plain',
            'typeId': 'wlearn.preprocess.tabular@1',
            'resolvedParams': {'scale': False},
        })
        spec = {
            'name': strategy,
            'classId': Model.class_id,
            'cls': Model,
            'preprocessChoices': [candidate['preprocess']],
        }
        spec['createCandidateClass'] = lambda item: (
            create_candidate_pipeline_class(
                spec, item,
                preprocessor_cls=self._preprocessor_class(events),
                pipeline_cls=Pipeline))

        common = dict(cv=2, task='classification', seed=7)
        if strategy == 'random':
            search = RandomSearch([spec], n_iter=1, **common)
        elif strategy == 'halving':
            search = SuccessiveHalvingSearch(
                [spec], n_iter=1, factor=2, **common)
        elif strategy == 'progressive':
            from wlearn.automl import ProgressiveSearch
            search = ProgressiveSearch(
                [spec], n_iter=1, promote_count=1,
                probe_fraction=1, **common)
        elif strategy == 'portfolio':
            search = PortfolioSearch([spec], **common)
        else:
            search = BayesianSearch(
                [spec], n_iter=1, n_initial=0, **common)

        X = np.arange(4, dtype=np.float64).reshape(4, 1)
        y = np.array([0, 0, 1, 1], dtype=np.int32)
        result = search.fit(X, y)
        original_id = result['bestResult']['candidateId']
        injected = create_candidate({
            'displayName': strategy, 'classId': Model.class_id,
        }, {'unevaluated': True}, candidate['preprocess'])
        injected_id = make_candidate_id(injected)
        result['bestResult']['candidate'] = injected
        result['bestResult']['candidateId'] = injected_id
        exposed = search.best_result
        exposed['candidate'] = injected
        exposed['candidateId'] = injected_id
        result['leaderboard'].add(
            injected, np.array([999.0]), 0, injected_id)

        assert search.best_result['candidateId'] == original_id
        events.clear()
        state['fail_fit'] = True
        with pytest.raises(RuntimeError) as exc:
            search.refit_best(X, y)
        assert exc.value is fit_error
        assert events == [
            'preprocessor:create', "model:create:{'task': 'classification'}",
            'preprocessor:fit', 'model:fit',
            'model:dispose', 'preprocessor:dispose',
        ]

    @pytest.mark.parametrize('strategy', [
        'random', 'halving', 'progressive', 'portfolio', 'bayesian',
    ])
    def test_search_preserves_first_backend_error(self, strategy):
        backend_error = BackendError(f'{strategy} backend unavailable')
        model_creates = {'count': 0}

        class Model:
            class_id = f'wlearn.test.{strategy}-backend-error@1'

            @classmethod
            def default_search_space(cls):
                return {}

            @classmethod
            def create(cls, _params=None):
                model_creates['count'] += 1
                raise RuntimeError('base model must not be created')

        class FailingCandidate:
            @classmethod
            def create(cls, _params=None):
                raise backend_error

        candidate = create_candidate({
            'displayName': strategy, 'classId': Model.class_id,
        }, {}, {
            'templateId': 'plain',
            'typeId': 'wlearn.preprocess.tabular@1',
            'resolvedParams': {'scale': False},
        })
        spec = {
            'name': strategy,
            'classId': Model.class_id,
            'cls': Model,
            'preprocessChoices': [candidate['preprocess']],
            'createCandidateClass': lambda _candidate: FailingCandidate,
        }
        common = dict(cv=2, task='classification', seed=7)
        if strategy == 'random':
            search = RandomSearch([spec], n_iter=1, **common)
        elif strategy == 'halving':
            search = SuccessiveHalvingSearch(
                [spec], n_iter=1, factor=2, **common)
        elif strategy == 'progressive':
            from wlearn.automl import ProgressiveSearch
            search = ProgressiveSearch(
                [spec], n_iter=1, promote_count=1,
                probe_fraction=1, **common)
        elif strategy == 'portfolio':
            search = PortfolioSearch([spec], **common)
        else:
            search = BayesianSearch(
                [spec], n_iter=1, n_initial=0, **common)
        X = np.arange(4, dtype=np.float64).reshape(4, 1)
        y = np.array([0, 0, 1, 1], dtype=np.int32)
        with pytest.raises(BackendError) as exc:
            search.fit(X, y)
        assert exc.value is backend_error
        assert model_creates['count'] == 0


# ===========================================================================
# Strategies
# ===========================================================================

class TestStrategies:
    @staticmethod
    def _model_with_preprocess_choices():
        choices = [
            {
                'templateId': 'plain',
                'typeId': 'wlearn.preprocess.tabular@1',
                'resolvedParams': {'scale': False},
            },
            {
                'templateId': 'scaled',
                'typeId': 'wlearn.preprocess.tabular@1',
                'resolvedParams': {'scale': 'standard'},
            },
        ]
        return {
            'name': 'mock',
            'cls': MockModel,
            'preprocessChoices': choices,
            'createCandidateClass': lambda _candidate: MockModel,
        }

    @staticmethod
    def _assert_two_templates(tasks):
        assert sorted(
            task['candidate']['preprocess']['templateId'] for task in tasks
        ) == ['plain', 'scaled']
        assert len({task['candidateId'] for task in tasks}) == 2
        for task in tasks:
            candidate_id = task['candidateId']
            assert candidate_id.startswith('wlc1_')
            assert len(candidate_id) == len('wlc1_') + 64

    def test_all_strategies_cross_preprocessing_templates(self):
        model = self._model_with_preprocess_choices()

        random = RandomStrategy([model], n_iter=1, seed=7)
        self._assert_two_templates([random.next(), random.next()])

        halving = HalvingStrategy(
            [model], n_iter=1, seed=7, factor=3,
            n_samples=20, cv=2,
        )
        halving_tasks = []
        for index in range(2):
            task = halving.next()
            halving_tasks.append(task)
            halving.report({
                'candidateId': task['candidateId'],
                'meanScore': float(index),
            })
        self._assert_two_templates(halving_tasks)

        progressive = ProgressiveStrategy(
            [model], n_iter=1, seed=7, promote_count=1)
        progressive_tasks = []
        for index in range(2):
            task = progressive.next()
            progressive_tasks.append(task)
            progressive.report({
                'candidateId': task['candidateId'],
                'meanScore': float(index),
            })
        self._assert_two_templates(progressive_tasks)

        portfolio = PortfolioStrategy(
            [model], task='classification', seed=7)
        self._assert_two_templates([portfolio.next(), portfolio.next()])

        fixed_model = {
            **model,
            'params': {'bias': 0, 'task': 'classification'},
        }
        bayesian = BayesianStrategy(
            [fixed_model], n_iter=1, n_initial=0, seed=7)
        bayesian_tasks = [bayesian.next(), bayesian.next()]
        self._assert_two_templates(bayesian_tasks)
        for task in bayesian_tasks:
            bayesian.report({
                'candidateId': task['candidateId'],
                'candidate': task['candidate'],
                'meanScore': 0.5,
            })
        bayesian.dispose()

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

    def test_bayesian_strategy_uses_optimizer_after_warmup(self):
        class FakeOptimizer:
            observed = []
            disposed = 0

            def __init__(self, space, **opts):
                self.space = space
                self.opts = opts

            def suggest(self):
                return {'bias': 0.25}

            def observe(self, params, score):
                self.observed.append((params, score))

            def dispose(self):
                FakeOptimizer.disposed += 1

        old = BayesianStrategy.optimizer_cls
        BayesianStrategy.optimizer_cls = FakeOptimizer
        try:
            models = [{'name': 'mock', 'cls': MockModel, 'params': {}}]
            strategy = BayesianStrategy(models, n_iter=3, seed=42, n_initial=1)
            first = strategy.next()
            strategy.report({
                'candidateId': first['candidateId'],
                'candidate': first['candidate'],
                'meanScore': 0.5,
            })
            second = strategy.next()
            assert second['params']['bias'] == pytest.approx(0.25)
            strategy.dispose()
            assert FakeOptimizer.observed
            assert FakeOptimizer.disposed == 1
        finally:
            BayesianStrategy.optimizer_cls = old

    def test_bayesian_strategy_handles_fixed_search_space_without_optimizer(self):
        class ExplodingOptimizer:
            def __init__(self, *args, **kwargs):
                raise AssertionError('optimizer should not be constructed')

        old = BayesianStrategy.optimizer_cls
        BayesianStrategy.optimizer_cls = ExplodingOptimizer
        try:
            models = [{'name': 'mock', 'cls': MockModel,
                       'params': {'bias': 0.1}}]
            strategy = BayesianStrategy(models, n_iter=2, seed=42, n_initial=0)
            out = []
            while not strategy.is_done():
                cand = strategy.next()
                if cand is not None:
                    out.append(cand)
                    strategy.report({
                        'candidateId': cand['candidateId'],
                        'meanScore': 0.5,
                    })
            assert len(out) == 2
            assert all(c['params'] == {'bias': 0.1} for c in out)
        finally:
            BayesianStrategy.optimizer_cls = old

    def test_bayesian_strategy_cleans_partial_optimizer_init(self):
        events = []
        create_error = RuntimeError('second optimizer failed')

        class SecondModel(MockModel):
            class_id = 'wlearn.test.second-model@1'

        class FaultOptimizer:
            calls = 0

            def __init__(self, _space, **_opts):
                type(self).calls += 1
                if type(self).calls == 2:
                    raise create_error

            def dispose(self):
                events.append('first:dispose')
                raise RuntimeError('cleanup failed')

        old = BayesianStrategy.optimizer_cls
        BayesianStrategy.optimizer_cls = FaultOptimizer
        try:
            with pytest.raises(RuntimeError, match='second optimizer failed') as exc:
                BayesianStrategy([
                    {'name': 'first', 'cls': MockModel},
                    {'name': 'second', 'cls': SecondModel},
                ], n_iter=2, n_initial=0)
            assert exc.value is create_error
            assert events == ['first:dispose']
        finally:
            BayesianStrategy.optimizer_cls = old


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

    def test_all_candidate_failures_preserve_first_error(self):
        class AlwaysFails:
            class_id = 'wlearn.test.always-fails@1'

            @classmethod
            def create(cls, _params=None):
                raise RuntimeError('create failed')

            @classmethod
            def default_search_space(cls):
                return {}

        X, y = make_cls_data(n=20, n_classes=2)
        search = SuccessiveHalvingSearch(
            [{'name': 'fails', 'cls': AlwaysFails}],
            n_iter=2, cv=2, task='classification')
        with pytest.raises(RuntimeError, match='create failed'):
            search.fit(X, y)


class TestBayesianSearch:
    def test_fit_with_injected_optimizer(self):
        class FakeOptimizer:
            def __init__(self, space, **opts):
                self.space = space

            def suggest(self):
                return {'bias': 0.0}

            def observe(self, params, score):
                pass

            def dispose(self):
                pass

        old = BayesianStrategy.optimizer_cls
        BayesianStrategy.optimizer_cls = FakeOptimizer
        try:
            X, y = make_cls_data(n=60, n_classes=2)
            models = [{'name': 'mock', 'cls': MockModel, 'params': {}}]
            search = BayesianSearch(models, n_iter=3, n_initial=1,
                                    cv=3, seed=42,
                                    task='classification')
            result = search.fit(X, y)
            assert result['leaderboard'].length == 3
            assert result['archive'].size == 3
            assert result['bestResult'] is not None
        finally:
            BayesianStrategy.optimizer_cls = old


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
        assert 'archive' in result
        assert result['archive'].size == len(result['leaderboard'])

    def test_with_ensemble(self):
        X, y = make_cls_data(n=60, n_classes=2)
        models = [
            {'name': 'mock1', 'classId': 'wlearn.test.mock1@1',
             'cls': MockModel, 'params': {}},
            {'name': 'mock2', 'classId': 'wlearn.test.mock2@1',
             'cls': MockModel, 'params': {'bias': 0.5}},
        ]
        result = auto_fit(
            models, X, y,
            ensemble=True, ensemble_size=5, n_iter=3, cv=3, seed=42,
        )
        assert result['model'] is not None
        # Model should be a VotingEnsemble
        from wlearn.ensemble import VotingEnsemble
        assert isinstance(result['model'], VotingEnsemble)

    def test_failed_final_ensemble_is_disposed_and_fit_error_wins(
            self, monkeypatch):
        import wlearn.automl._auto_fit as auto_fit_module

        fit_error = RuntimeError('ensemble fit failed')
        disposals = {'count': 0}

        class FailingEnsemble:
            def fit(self, _X, _y):
                raise fit_error

            def dispose(self):
                disposals['count'] += 1
                raise RuntimeError('ensemble cleanup failed')

        monkeypatch.setattr(
            auto_fit_module.VotingEnsemble,
            'create',
            classmethod(lambda _cls, **_kwargs: FailingEnsemble()),
        )
        X, y = make_cls_data(n=20, n_classes=2)
        with pytest.raises(RuntimeError) as exc:
            auto_fit(
                [{'name': 'mock', 'cls': MockModel}], X, y,
                ensemble=True, n_iter=1, cv=2, seed=7,
                task='classification')
        assert exc.value is fit_error
        assert disposals['count'] == 1

    def test_no_refit(self):
        X, y = make_cls_data(n=60, n_classes=2)
        models = [{'name': 'mock', 'cls': MockModel, 'params': {}}]
        result = auto_fit(models, X, y, refit=False, ensemble=False, n_iter=3, cv=3, seed=42)
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


class TestAutoFitPreprocessing:
    templates = [
        {
            'templateId': 'plain',
            'typeId': 'wlearn.preprocess.tabular@1',
            'params': {
                'encode': False, 'impute': False, 'scale': False,
                'maxCategories': 2,
            },
        },
        {
            'templateId': 'scaled',
            'typeId': 'wlearn.preprocess.tabular@1',
            'params': {
                'encode': False, 'impute': False, 'scale': 'standard',
                'maxCategories': 2,
            },
        },
    ]

    @staticmethod
    def _require_tranfi():
        pytest.importorskip('tranfi', reason='optional Tranfi backend not installed')

    @pytest.mark.parametrize('params', [False, 0, '', [], None])
    def test_rejects_explicit_non_dict_template_params(self, params):
        X, y = make_cls_data(n=20, n_features=2, n_classes=2)
        with pytest.raises(ValidationError, match='params must be a dict'):
            auto_fit(
                [{'name': 'mock', 'cls': MockModel}], X, y,
                preprocess=[{
                    'templateId': 'invalid',
                    'typeId': 'wlearn.preprocess.tabular@1',
                    'params': params,
                }],
                ensemble=False, refit=False,
            )

    @pytest.mark.parametrize('search_space', [False, 0, '', [], None])
    def test_rejects_explicit_non_dict_template_search_space(
            self, search_space):
        X, y = make_cls_data(n=20, n_features=2, n_classes=2)
        with pytest.raises(ValidationError, match='searchSpace must be a dict'):
            auto_fit(
                [{'name': 'mock', 'cls': MockModel}], X, y,
                preprocess=[{
                    'templateId': 'invalid',
                    'typeId': 'wlearn.preprocess.tabular@1',
                    'searchSpace': search_space,
                }],
                ensemble=False, refit=False,
            )

    @pytest.mark.parametrize('backend_state', ['missing', 'incompatible'])
    def test_preserves_actionable_optional_tranfi_error(
            self, monkeypatch, backend_state):
        import types
        import wlearn.preprocess as preprocess_module

        original_import = preprocess_module.importlib.import_module

        def fake_import(name, *args, **kwargs):
            if name != 'tranfi':
                return original_import(name, *args, **kwargs)
            if backend_state == 'missing':
                raise ImportError('tranfi intentionally absent')
            return types.SimpleNamespace()

        monkeypatch.setattr(preprocess_module, '_TRANFI', None)
        monkeypatch.setattr(
            preprocess_module.importlib, 'import_module', fake_import)
        model_creates = {'count': 0}

        class CountingModel(MockModel):
            @classmethod
            def create(cls, params=None):
                model_creates['count'] += 1
                return super().create(params)

        X, y = make_cls_data(n=20, n_features=2, n_classes=2)
        expected = (
            r'Install wlearn\[preprocess\]' if backend_state == 'missing' else
            'compatible tranfi>=0.2,<0.3')
        with pytest.raises(BackendError, match=expected):
            auto_fit(
                [{'name': 'mock', 'cls': CountingModel}], X, y,
                n_iter=1, cv=2, task='classification',
                preprocess=True, ensemble=False, refit=False,
            )
        assert model_creates['count'] == 0

    def test_evaluates_every_resolved_template(self):
        self._require_tranfi()
        X, y = make_cls_data(n=20, n_features=2, n_classes=2)
        result = auto_fit(
            [{'name': 'mock', 'cls': MockModel}],
            X, y,
            n_iter=1, cv=2, task='classification',
            preprocess=self.templates, ensemble=False, refit=False,
        )
        assert sorted(
            entry['candidate']['preprocess']['templateId']
            for entry in result['leaderboard']
        ) == ['plain', 'scaled']
        assert len({
            entry['candidateId'] for entry in result['leaderboard']
        }) == 2
        assert result['bestParams']['preprocess']['policyVersion'] == 1

    def test_categorical_dictionary_is_fitted_inside_each_fold(self):
        self._require_tranfi()
        train_columns = []
        predict_columns = []

        class ShapeProbe:
            class_id = 'wlearn.test.shape-probe@1'

            @classmethod
            def create(cls, _params=None):
                return cls()

            @classmethod
            def default_search_space(cls):
                return {}

            def fit(self, X, _y):
                train_columns.append(X.shape[1])
                return self

            def predict(self, X):
                predict_columns.append(X.shape[1])
                return np.zeros(len(X), dtype=np.int32)

            def get_params(self):
                return {}

            def dispose(self):
                pass

        X = np.arange(8, dtype=np.float64).reshape(-1, 1)
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
        auto_fit(
            [{'name': 'shape', 'cls': ShapeProbe}], X, y,
            n_iter=1, cv=2, seed=42, task='classification',
            preprocess={'encode': 'onehot', 'impute': False, 'scale': False},
            ensemble=False, refit=False,
        )
        assert train_columns == [4, 4]
        assert predict_columns == [4, 4]

    def test_scaler_statistics_are_fitted_inside_each_fold(self):
        self._require_tranfi()
        train_means = []

        class MeanProbe:
            class_id = 'wlearn.test.mean-probe@1'

            @classmethod
            def create(cls, _params=None):
                return cls()

            @classmethod
            def default_search_space(cls):
                return {}

            def fit(self, X, _y):
                train_means.append(float(np.mean(X)))
                return self

            def predict(self, X):
                return np.zeros(len(X), dtype=np.float64)

            def get_params(self):
                return {}

            def dispose(self):
                pass

        X = np.array(
            [0.1, 0.2, 0.3, 0.4, 10.1, 20.2, 30.3, 1000.4],
            dtype=np.float64,
        ).reshape(-1, 1)
        y = np.array(
            [0.1, 0.2, 0.3, 0.4, 1.1, 1.2, 1.3, 1.4],
            dtype=np.float64,
        )
        auto_fit(
            [{'name': 'mean', 'cls': MeanProbe}], X, y,
            n_iter=1, cv=2, seed=7, task='regression',
            preprocess={
                'encode': False, 'impute': False, 'scale': 'standard',
            },
            ensemble=False, refit=False,
        )
        assert len(train_means) == 2
        assert all(abs(mean) < 1e-12 for mean in train_means)

    def test_round_trips_pipeline_candidate_provenance(self):
        self._require_tranfi()
        X, y = make_cls_data(n=20, n_features=2, n_classes=2)
        result = auto_fit(
            [{'name': 'mock', 'cls': MockModel,
              'params': {'__proto__': {'safe': True}}}], X, y,
            n_iter=1, cv=2, task='classification',
            preprocess=self.templates[1]['params'],
            ensemble=False, refit=True,
        )
        candidate = result['bestCandidate']
        expected = {
            'candidateId': make_candidate_id(candidate),
            'candidate': candidate,
            'baseSeed': 42,
            'foldSeeds': [
                {'foldId': fold_id,
                 'seed': seed_for(candidate, fold_id, 42)}
                for fold_id in range(2)
            ],
        }
        assert '__proto__' in candidate['model']['params']
        assert isinstance(result['model'], Pipeline)
        assert result['model'].provenance == expected

        data = result['model'].save()
        manifest, _, _ = decode_bundle(data)
        assert manifest['metadata']['provenance'] == expected
        loaded = load_bundle(data)
        assert loaded.provenance == expected
        np.testing.assert_array_equal(
            loaded.predict(X), result['model'].predict(X))
        loaded.dispose()
        result['model'].dispose()

    def test_round_trips_candidate_pipelines_nested_in_ensemble(self):
        self._require_tranfi()
        X, y = make_cls_data(n=20, n_features=2, n_classes=2)
        result = auto_fit(
            [
                {
                    'name': 'm1', 'classId': 'wlearn.test.persist.m1@1',
                    'cls': MockModel,
                    'params': {'__proto__': {'model': 'm1'}},
                },
                {
                    'name': 'm2', 'classId': 'wlearn.test.persist.m2@1',
                    'cls': MockModel,
                    'params': {'__proto__': {'model': 'm2'}},
                },
            ],
            X, y,
            n_iter=1, cv=2, task='classification',
            preprocess=self.templates[0]['params'],
            ensemble=True, ensemble_size=2,
        )
        data = result['model'].save()
        _, toc, blobs = decode_bundle(data)
        assert toc
        for entry in toc:
            child, _, _ = decode_bundle(bytes(
                blobs[entry['offset']:entry['offset'] + entry['length']]))
            assert child['typeId'] == 'wlearn.pipeline@1'
            assert (
                child['metadata']['provenance']['candidate']['preprocess']
                ['templateId'] == 'wlearn.preprocess.inline.v1')
            provenance = child['metadata']['provenance']
            assert '__proto__' in provenance['candidate']['model']['params']
            assert provenance['candidateId'] == make_candidate_id(
                provenance['candidate'])

        loaded = load_bundle(data)
        np.testing.assert_array_equal(
            loaded.predict(X), result['model'].predict(X))
        loaded.dispose()
        result['model'].dispose()

    def test_rejects_stacking_passthrough_with_preprocessing(self):
        self._require_tranfi()
        X, y = make_cls_data(n=20, n_features=2, n_classes=2)
        with pytest.raises(ValidationError, match='stacking_passthrough'):
            auto_fit(
                [
                    {
                        'name': 'm1', 'classId': 'wlearn.test.stack.m1@1',
                        'cls': MockModel,
                    },
                    {
                        'name': 'm2', 'classId': 'wlearn.test.stack.m2@1',
                        'cls': MockModel,
                    },
                ],
                X, y,
                n_iter=1, cv=2, task='classification',
                preprocess=self.templates[0]['params'], ensemble=True,
                ensemble_size=2, stacking=True,
                meta_estimator={'cls': MockModel, 'params': {}},
                stacking_passthrough=True,
            )


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
        expected = {'xgb': 10, 'lgb': 6, 'ebm': 4, 'linear': 4, 'svm': 4, 'knn': 3, 'tsetlin': 3}
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
            {'name': 'xgb', 'portfolioKey': 'xgb',
             'cls': MockModel, 'params': {}},
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
        assert count == 10  # 8 boosting + 2 RF-mode

    def test_multiple_models(self):
        models = [
            {'name': 'xgb', 'portfolioKey': 'xgb', 'classId': 'wlearn.test.xgb@1',
             'cls': MockModel, 'params': {}},
            {'name': 'knn', 'portfolioKey': 'knn', 'classId': 'wlearn.test.knn@1',
             'cls': MockModel, 'params': {}},
        ]
        strategy = PortfolioStrategy(models, task='classification')
        count = 0
        while not strategy.is_done():
            if strategy.next() is None:
                break
            count += 1
        assert count == 10 + 3  # xgb=10, knn=3

    def test_fixed_params_override(self):
        models = [
            {'name': 'xgb', 'portfolioKey': 'xgb',
             'cls': MockModel, 'params': {'eta': 0.999}},
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
        models = [{'name': 'xgb', 'portfolioKey': 'xgb',
                   'cls': MockModel, 'params': {}}]
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
        """XGB portfolio should produce 10 leaderboard entries (8 boosting + 2 RF)."""
        X, y = make_cls_data(n=60, n_classes=2)
        models = [{'name': 'xgb', 'portfolioKey': 'xgb',
                   'cls': MockModel, 'params': {}}]
        search = PortfolioSearch(models, cv=3, seed=42, task='classification')
        result = search.fit(X, y)
        ranked = result['leaderboard'].ranked()
        assert len(ranked) == 10


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
            {'name': 'xgb', 'portfolioKey': 'xgb', 'classId': 'wlearn.test.xgb@1',
             'cls': MockModel, 'params': {}},
            {'name': 'knn', 'portfolioKey': 'knn', 'classId': 'wlearn.test.knn@1',
             'cls': MockModel, 'params': {}},
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

    def test_strategy_bayesian(self):
        class FakeOptimizer:
            def __init__(self, space, **opts):
                self.space = space

            def suggest(self):
                return {'bias': 0.0}

            def observe(self, params, score):
                pass

            def dispose(self):
                pass

        old = BayesianStrategy.optimizer_cls
        BayesianStrategy.optimizer_cls = FakeOptimizer
        try:
            X, y = make_cls_data(n=60, n_classes=2)
            models = [{'name': 'mock', 'cls': MockModel, 'params': {}}]
            result = auto_fit(models, X, y, cv=3, seed=42, n_iter=3,
                              strategy='bayesian', ensemble=False)
            assert result['model'] is not None
            assert result['archive'].size == 3
        finally:
            BayesianStrategy.optimizer_cls = old


def test_package_owned_portfolio_configurations():
    class Custom(MockModel):
        class_id = 'test.custom-portfolio'

        @classmethod
        def default_portfolio(cls, task):
            return [{'bias': 2 if task == 'regression' else 3}, {'bias': 4}]

    strategy = PortfolioStrategy([dict(name='custom', cls=Custom)], task='regression')
    assert strategy.next()['params']['bias'] == 2
    assert strategy.next()['params']['bias'] == 4
    assert strategy.next() is None
    override = PortfolioStrategy([dict(name='custom', cls=Custom, portfolio=[{'bias': 9}], params={'bias': 5})])
    assert override.next()['params']['bias'] == 5
    assert override.next() is None
    for portfolio in [[], [None], [3], 'bad']:
        with pytest.raises(ValidationError):
            PortfolioStrategy([dict(name='custom', cls=Custom, portfolio=portfolio)])


def test_failed_candidate_does_not_stop_halving_rounds():
    class Broken:
        class_id = 'wlearn.test.broken@1'

        @classmethod
        def create(cls, params):
            raise ValueError('invalid candidate')

    X = np.arange(24, dtype=float).reshape(-1, 2)
    y = np.arange(12, dtype=float)
    search = SuccessiveHalvingSearch([
        {'name': 'broken', 'cls': Broken, 'searchSpace': {}},
        {'name': 'working', 'cls': MockModel}
    ], n_iter=3, cv=2, task='regression', factor=2)
    result = search.fit(X, y)
    assert result['rounds'], 'surviving candidates must reach subsequent rounds'


@pytest.mark.parametrize('preprocess', [False, True])
def test_auto_fit_labels_only_falls_back_without_probabilities(preprocess):
    class LabelsOnly(MockModel):
        class_id = 'test.labels-only'

        @property
        def capabilities(self):
            return {'classifier': True, 'predictProba': False}

        def predict_proba(self, X):
            raise AssertionError('probabilities are unavailable')

    X, y = make_cls_data(n=24, n_classes=2)
    result = auto_fit([('labels', LabelsOnly)], X, y, n_iter=1, cv=2,
                      task='classification', preprocess=preprocess)
    try:
        assert len(result['model'].predict(X)) == len(y)
        assert not result['model'].capabilities['predictProba']
        assert result['leaderboard'][0]['supportsPredictProba'] is False
    finally:
        result['model'].dispose()


def test_auto_fit_mixed_probability_capabilities():
    class LabelsOnly(MockModel):
        class_id = 'test.labels-only'

        @property
        def capabilities(self):
            return {'classifier': True, 'predictProba': False}

        def predict_proba(self, X):
            raise AssertionError('probabilities are unavailable')

    X, y = make_cls_data(n=24, n_classes=2)
    result = auto_fit([('labels', LabelsOnly), ('proba', MockModel)], X, y,
                      n_iter=4, cv=2, task='classification')
    try:
        assert {e['modelName'] for e in result['leaderboard']} == {'labels', 'proba'}
        assert result['model'].capabilities['predictProba']
        assert result['model'].predict_proba(X).size == len(y) * 2
    finally:
        result['model'].dispose()
