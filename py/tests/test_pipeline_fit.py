"""Tests for Python Pipeline fit/predict/predict_proba/score."""

import numpy as np
import pytest

from wlearn.pipeline import Pipeline
from wlearn.errors import NotFittedError, DisposedError, ValidationError
from wlearn.bundle import decode_bundle

liblinear = pytest.importorskip('liblinear', reason='liblinear-official not installed')


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

    def dispose(self):
        pass

    @property
    def is_fitted(self):
        return self._fitted


class TestPipelineFit:
    """Test Pipeline fit/predict with a single estimator (no transformer)."""

    def test_fit_predict_single_step(self):
        from wlearn.liblinear import LinearModel
        X, y = make_binary_data()
        model = LinearModel.create({'solver': 0, 'C': 1.0})
        pipe = Pipeline([('model', model)])
        pipe.fit(X, y)
        assert pipe.is_fitted
        preds = pipe.predict(X)
        assert len(preds) == len(y)

    def test_score_single_step(self):
        from wlearn.liblinear import LinearModel
        X, y = make_binary_data()
        model = LinearModel.create({'solver': 0, 'C': 1.0})
        pipe = Pipeline([('model', model)])
        pipe.fit(X, y)
        acc = pipe.score(X, y)
        assert acc > 0.7

    def test_predict_proba_single_step(self):
        from wlearn.liblinear import LinearModel
        X, y = make_binary_data()
        model = LinearModel.create({'solver': 0, 'C': 1.0})
        pipe = Pipeline([('model', model)])
        pipe.fit(X, y)
        proba = pipe.predict_proba(X)
        assert len(proba) > 0
        assert np.all(proba >= 0)

    def test_not_fitted_errors(self):
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
        from wlearn.liblinear import LinearModel
        X, y = make_binary_data()
        scaler = MockTransformer()
        model = LinearModel.create({'solver': 0, 'C': 1.0})
        pipe = Pipeline([('scaler', scaler), ('model', model)])
        pipe.fit(X, y)
        assert scaler.is_fitted

    def test_fit_transform_used(self):
        """Verify fit_transform is preferred over separate fit + transform."""
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
        from wlearn.liblinear import LinearModel
        model = LinearModel.create({'solver': 0})
        pipe = Pipeline([('model', model)])
        pipe.dispose()
        pipe.dispose()  # should not raise


class TestPipelineSaveLoad:
    def test_save_load_roundtrip(self):
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
