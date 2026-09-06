"""Tests for Python Pipeline fit/predict/predict_proba/score."""

import numpy as np
import pytest

from wlearn.pipeline import Pipeline
from wlearn.errors import NotFittedError, DisposedError, ValidationError
from wlearn.bundle import decode_bundle

def make_binary_data(seed=42, n=100, n_features=3):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def make_regression_data(seed=42, n=100, n_features=2):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    y = 2 * X[:, 0] + 3 * X[:, 1] + rng.randn(n) * 0.5
    return X, y


class MockTransformer:
    """A simple z-score transformer for testing pipeline fit/predict."""

    def __init__(self):
        self._mean = None
        self._std = None
        self._fitted = False

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std == 0] = 1.0
        self._fitted = True
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self._mean) / self._std

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def save(self):
        # Minimal bundle for testing -- not a real wlearn bundle
        from wlearn.bundle import encode_bundle
        import json
        data = json.dumps({
            'mean': self._mean.tolist(),
            'std': self._std.tolist(),
        }).encode()
        return encode_bundle(
            {'typeId': 'test.mock_transformer@1'},
            [{'id': 'params', 'data': data}],
        )

    def get_params(self):
        return {}

    def set_params(self, params):
        if params.get('fail'):
            raise RuntimeError('mutation failed')
        return self

    def dispose(self):
        pass

    @property
    def is_fitted(self):
        return self._fitted


class TestPipelineFit:
    @pytest.mark.parametrize('fail_at', ['transform', 'model'])
    def test_failed_refit_invalidates_pipeline_until_recovery(self, fail_at):
        class Step:
            fail = False
            offset = 0

            def fit(self, X, y=None):
                self.offset += 1
                if self.fail:
                    raise ValueError('refit failed')
                return self

            def transform(self, X):
                return np.asarray(X) + self.offset

            def predict(self, X):
                return np.asarray(X).ravel()

            def score(self, X, y):
                return 1.0

            def save(self):
                from wlearn.bundle import encode_bundle
                return encode_bundle({'typeId': 'test.refit@1'}, [])

        transform, model = Step(), Step()
        pipe = Pipeline([('transform', transform), ('model', model)])
        pipe.fit([[0]], [0])
        np.testing.assert_array_equal(pipe.predict([[0]]), [1])
        failing = transform if fail_at == 'transform' else model
        failing.fail = True
        with pytest.raises(ValueError, match='refit failed'):
            pipe.fit([[0]], [0])
        for operation in [lambda: pipe.predict([[0]]), lambda: pipe.score([[0]], [0]), pipe.save]:
            with pytest.raises(NotFittedError):
                operation()
        failing.fail = False
        pipe.fit([[0]], [0])
        np.testing.assert_array_equal(pipe.predict([[0]]), [3])

    """Test Pipeline fit/predict with a single estimator (no transformer)."""

    def test_fit_predict_single_step(self):
        pytest.importorskip('liblinear', reason='liblinear-official not installed')
        from wlearn.liblinear import LinearModel
        X, y = make_binary_data()
        model = LinearModel.create({'solver': 0, 'C': 1.0})
        pipe = Pipeline([('model', model)])
        pipe.fit(X, y)
        assert pipe.is_fitted
        preds = pipe.predict(X)
        assert len(preds) == len(y)

    def test_score_single_step(self):
        pytest.importorskip('liblinear', reason='liblinear-official not installed')
        from wlearn.liblinear import LinearModel
        X, y = make_binary_data()
        model = LinearModel.create({'solver': 0, 'C': 1.0})
        pipe = Pipeline([('model', model)])
        pipe.fit(X, y)
        acc = pipe.score(X, y)
        assert acc > 0.7

    def test_predict_proba_single_step(self):
        pytest.importorskip('liblinear', reason='liblinear-official not installed')
        from wlearn.liblinear import LinearModel
        X, y = make_binary_data()
        model = LinearModel.create({'solver': 0, 'C': 1.0})
        pipe = Pipeline([('model', model)])
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)
        assert len(proba) > 0
        assert np.all(proba >= 0)

    def test_capabilities_forwarded_as_defensive_copy(self):
        pytest.importorskip('liblinear', reason='liblinear-official not installed')
        from wlearn.liblinear import LinearModel
        model = LinearModel.create({'solver': 0})
        pipe = Pipeline([('model', model)])
        capabilities = pipe.capabilities
        assert capabilities['classifier'] is True
        assert capabilities['predictProba'] is True
        capabilities['predictProba'] = False
        assert pipe.capabilities['predictProba'] is True

    def test_not_fitted_errors(self):
        pytest.importorskip('liblinear', reason='liblinear-official not installed')
        from wlearn.liblinear import LinearModel
        model = LinearModel.create({'solver': 0})
        pipe = Pipeline([('model', model)])
        with pytest.raises(NotFittedError):
            pipe.predict(np.zeros((1, 3)))
        with pytest.raises(NotFittedError):
            pipe.score(np.zeros((1, 3)), np.zeros(1))
        with pytest.raises(NotFittedError):
            pipe.predict_proba(np.zeros((1, 3)))


class TestPipelineWithTransformer:
    """Test Pipeline with transformer + estimator chain."""

    def test_fit_predict_with_transformer(self):
        pytest.importorskip('liblinear', reason='liblinear-official not installed')
        from wlearn.liblinear import LinearModel
        X, y = make_binary_data()
        scaler = MockTransformer()
        model = LinearModel.create({'solver': 0, 'C': 1.0})
        pipe = Pipeline([('scaler', scaler), ('model', model)])
        pipe.fit(X, y)
        assert pipe.is_fitted
        preds = pipe.predict(X)
        assert len(preds) == len(y)
        acc = pipe.score(X, y)
        assert acc > 0.7

    def test_transformer_is_fitted(self):
        pytest.importorskip('liblinear', reason='liblinear-official not installed')
        from wlearn.liblinear import LinearModel
        X, y = make_binary_data()
        scaler = MockTransformer()
        model = LinearModel.create({'solver': 0, 'C': 1.0})
        pipe = Pipeline([('scaler', scaler), ('model', model)])
        pipe.fit(X, y)
        assert scaler.is_fitted

    def test_set_params_invalidates_before_child_mutation(self):
        pytest.importorskip('liblinear', reason='liblinear-official not installed')
        from wlearn.liblinear import LinearModel
        X, y = make_binary_data()
        scaler = MockTransformer()
        model = LinearModel.create({'solver': 0, 'C': 1.0})
        pipe = Pipeline([('scaler', scaler), ('model', model)])
        pipe.fit(X, y)
        pipe.set_params({'scaler': {}})
        assert not pipe.is_fitted
        with pytest.raises(NotFittedError):
            pipe.predict(X)

        pipe.fit(X, y)
        with pytest.raises(RuntimeError, match='mutation failed'):
            pipe.set_params({'scaler': {'fail': True}})
        assert not pipe.is_fitted

    def test_set_params_rejects_unknown_and_preflights_all_setters(self):
        class MissingSetterEstimator:
            capabilities = {}

            def fit(self, X, y):
                return self

            def predict(self, X):
                return np.zeros(len(X), dtype=np.int32)

            def dispose(self):
                pass

        X, y = make_binary_data()
        scaler = MockTransformer()
        mutations = {'count': 0}

        def mutate(params):
            mutations['count'] += 1
            return scaler

        scaler.set_params = mutate
        pipe = Pipeline([
            ('scaler', scaler), ('model', MissingSetterEstimator())])
        pipe.fit(X, y)

        with pytest.raises(ValidationError, match='Unknown.*modle'):
            pipe.set_params({'modle': {}})
        assert pipe.is_fitted
        with pytest.raises(ValidationError, match='model.*set_params'):
            pipe.set_params({'scaler': {}, 'model': {}})
        assert mutations['count'] == 0
        assert pipe.is_fitted
        pipe.dispose()

    def test_fit_transform_used(self):
        """Verify fit_transform is preferred over separate fit + transform."""
        pytest.importorskip('liblinear', reason='liblinear-official not installed')
        from wlearn.liblinear import LinearModel
        X, y = make_binary_data()

        calls = []

        class TrackingTransformer(MockTransformer):
            def fit(self, X, y=None):
                calls.append('fit')
                return super().fit(X, y)

            def transform(self, X):
                calls.append('transform')
                return super().transform(X)

            def fit_transform(self, X, y=None):
                calls.append('fit_transform')
                # Implement directly to avoid calling fit/transform
                X = np.asarray(X, dtype=np.float64)
                self._mean = X.mean(axis=0)
                self._std = X.std(axis=0)
                self._std[self._std == 0] = 1.0
                self._fitted = True
                return (X - self._mean) / self._std

        scaler = TrackingTransformer()
        model = LinearModel.create({'solver': 0, 'C': 1.0})
        pipe = Pipeline([('scaler', scaler), ('model', model)])
        pipe.fit(X, y)
        # Pipeline should call fit_transform, not fit + transform
        assert calls == ['fit_transform']

    def test_regression_pipeline(self):
        pytest.importorskip('liblinear', reason='liblinear-official not installed')
        from wlearn.liblinear import LinearModel
        X, y = make_regression_data()
        scaler = MockTransformer()
        model = LinearModel.create({'solver': 11, 'C': 1.0})
        pipe = Pipeline([('scaler', scaler), ('model', model)])
        pipe.fit(X, y)
        r2 = pipe.score(X, y)
        assert r2 > 0.5

    def test_predict_proba_unsupported(self):
        """Last step without predict_proba raises ValidationError."""
        X, y = make_binary_data()
        scaler = MockTransformer()

        class NoProbaModel:
            def fit(self, X, y): return self
            def predict(self, X): return np.zeros(len(X))
            def score(self, X, y): return 0.0

        model = NoProbaModel()
        pipe = Pipeline([('scaler', scaler), ('model', model)])
        pipe.fit(X, y)
        with pytest.raises(ValidationError, match='predict_proba'):
            pipe.predict_proba(X)


class TestPipelineDispose:
    def test_dispose_propagates(self):
        pytest.importorskip('liblinear', reason='liblinear-official not installed')
        from wlearn.liblinear import LinearModel
        X, y = make_binary_data()
        model = LinearModel.create({'solver': 0})
        pipe = Pipeline([('model', model)])
        pipe.fit(X, y)
        pipe.dispose()
        assert not pipe.is_fitted
        with pytest.raises(DisposedError):
            pipe.predict(X)

    def test_double_dispose_safe(self):
        pytest.importorskip('liblinear', reason='liblinear-official not installed')
        from wlearn.liblinear import LinearModel
        model = LinearModel.create({'solver': 0})
        pipe = Pipeline([('model', model)])
        pipe.dispose()
        pipe.dispose()  # should not raise


class TestPipelineSaveLoad:
    def test_save_load_roundtrip(self):
        pytest.importorskip('liblinear', reason='liblinear-official not installed')
        from wlearn.liblinear import LinearModel
        X, y = make_binary_data()
        model = LinearModel.create({'solver': 0, 'C': 1.0})
        pipe = Pipeline([('model', model)])
        pipe.fit(X, y)
        preds_orig = pipe.predict(X)

        bundle_bytes = pipe.save()
        loaded = Pipeline.load(bundle_bytes)
        preds_loaded = loaded.predict(X)
        assert np.array_equal(preds_orig, preds_loaded)


def test_save_reports_missing_step_serializer_before_serializing_children():
    class First:
        saves = 0

        def fit_transform(self, X, y):
            return X

        def save(self):
            self.saves += 1
            return b''

    class Last:
        def fit(self, X, y):
            return self

    first = First()
    pipe = Pipeline([('first', first), ('broken', Last())])
    pipe.fit([[0], [1]], [0, 1])
    with pytest.raises(ValidationError, match='broken.*save'):
        pipe.save()
    assert first.saves == 0


def test_pipeline_routes_weights_and_validates_before_mutation():
    calls = []
    weights = np.array([1.0, 3.0])

    class Map:
        capabilities = dict(transformer=True, sampleWeight=True)

        def fit_transform(self, X, y, sample_weight=None):
            calls.append(('map', sample_weight))
            return X

    class Model:
        capabilities = dict(sampleWeight=True)

        def fit(self, X, y, sample_weight=None):
            calls.append(('model', sample_weight))
            return self

    pipe = Pipeline([('map', Map()), ('model', Model())])
    pipe.fit([[0], [1]], [0, 1], sample_weight=weights)
    assert calls[0][0] == 'map' and calls[1][0] == 'model'
    np.testing.assert_array_equal(calls[0][1], weights)
    np.testing.assert_array_equal(calls[1][1], weights)
    for bad in [[1], [0, 0], [-1, 2], [float('nan'), 1]]:
        with pytest.raises(ValidationError):
            pipe.fit([[0], [1]], [0, 1], sample_weight=bad)
    Model.capabilities['sampleWeight'] = False
    with pytest.raises(ValidationError, match='sample_weight'):
        pipe.fit([[0], [1]], [0, 1], sample_weight=weights)
    assert len(calls) == 2
