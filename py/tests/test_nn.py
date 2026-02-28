"""Tests for wlearn.nn (MLPClassifier, MLPRegressor)."""

import numpy as np
import pytest
from wlearn.nn import MLPClassifier, MLPRegressor
from wlearn.errors import NotFittedError, DisposedError
from wlearn.bundle import decode_bundle


def make_binary_data(seed=42, n=50, n_features=2):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)
    return X, y


def make_regression_data(seed=42, n=50, n_features=2):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features).astype(np.float32)
    y = (2.0 * X[:, 0] + 3.0 * X[:, 1] + 0.5).astype(np.float32)
    return X, y


class TestMLPClassifier:
    def test_create_unfitted(self):
        model = MLPClassifier.create()
        assert not model.is_fitted
        with pytest.raises(NotFittedError):
            model.predict(np.zeros((1, 2)))

    def test_binary_classification(self):
        X, y = make_binary_data()
        model = MLPClassifier.create({
            'hidden_sizes': [8], 'epochs': 50, 'lr': 0.01,
            'optimizer': 'sgd', 'seed': 42
        })
        model.fit(X, y)
        assert model.is_fitted
        assert model.score(X, y) > 0.6

    def test_predict_shape(self):
        X, y = make_binary_data(n=20)
        model = MLPClassifier.create({
            'hidden_sizes': [4], 'epochs': 10, 'lr': 0.01,
            'optimizer': 'sgd', 'seed': 42
        })
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 20
        assert preds.dtype == np.float64

    def test_predict_proba(self):
        X, y = make_binary_data(n=20)
        model = MLPClassifier.create({
            'hidden_sizes': [4], 'epochs': 10, 'lr': 0.01,
            'optimizer': 'sgd', 'seed': 42
        })
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert len(proba) == 20 * 2  # binary: 2 classes
        proba_2d = proba.reshape(-1, 2)
        np.testing.assert_allclose(proba_2d.sum(axis=1), 1.0, atol=1e-5)
        assert np.all(proba >= 0)

    def test_classes(self):
        X, y = make_binary_data()
        model = MLPClassifier.create({
            'hidden_sizes': [4], 'epochs': 5, 'lr': 0.01,
            'optimizer': 'sgd', 'seed': 42
        })
        model.fit(X, y)
        assert len(model.classes) == 2
        assert 0 in model.classes
        assert 1 in model.classes

    def test_save_load_roundtrip(self):
        X, y = make_binary_data(n=20)
        model = MLPClassifier.create({
            'hidden_sizes': [4], 'epochs': 20, 'lr': 0.01,
            'optimizer': 'sgd', 'seed': 42
        })
        model.fit(X, y)

        bundle_bytes = model.save()
        assert len(bundle_bytes) > 0

        # Decode and reload
        manifest, toc, blobs = decode_bundle(bundle_bytes)
        assert manifest['typeId'] == 'wlearn.nn.mlp.classifier@1'

        loaded = MLPClassifier._from_bundle(manifest, toc, blobs)
        assert loaded.is_fitted

        # Predictions must match
        np.testing.assert_array_equal(
            model.predict(X), loaded.predict(X))

        model.dispose()
        loaded.dispose()

    def test_dispose(self):
        model = MLPClassifier.create({
            'hidden_sizes': [4], 'epochs': 5, 'lr': 0.01,
            'optimizer': 'sgd', 'seed': 42
        })
        X, y = make_binary_data(n=10)
        model.fit(X, y)
        model.dispose()
        assert not model.is_fitted
        with pytest.raises(DisposedError):
            model.predict(X)
        # Dispose is idempotent
        model.dispose()

    def test_capabilities(self):
        model = MLPClassifier.create()
        caps = model.capabilities
        assert caps['classifier'] is True
        assert caps['regressor'] is False
        assert caps['predictProba'] is True

    def test_get_set_params(self):
        model = MLPClassifier.create({'lr': 0.01})
        assert model.get_params()['lr'] == 0.01
        model.set_params({'lr': 0.001})
        assert model.get_params()['lr'] == 0.001


class TestMLPRegressor:
    def test_create_unfitted(self):
        model = MLPRegressor.create()
        assert not model.is_fitted
        with pytest.raises(NotFittedError):
            model.predict(np.zeros((1, 2)))

    def test_regression(self):
        X, y = make_regression_data()
        model = MLPRegressor.create({
            'hidden_sizes': [16], 'epochs': 100, 'lr': 0.01,
            'optimizer': 'sgd', 'seed': 42
        })
        model.fit(X, y)
        assert model.is_fitted
        # R2 should be positive (better than mean prediction)
        assert model.score(X, y) > 0.0

    def test_predict_shape(self):
        X, y = make_regression_data(n=20)
        model = MLPRegressor.create({
            'hidden_sizes': [4], 'epochs': 10, 'lr': 0.01,
            'optimizer': 'sgd', 'seed': 42
        })
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 20
        assert preds.dtype == np.float64

    def test_save_load_roundtrip(self):
        X, y = make_regression_data(n=20)
        model = MLPRegressor.create({
            'hidden_sizes': [4], 'epochs': 20, 'lr': 0.01,
            'optimizer': 'sgd', 'seed': 42
        })
        model.fit(X, y)

        bundle_bytes = model.save()
        manifest, toc, blobs = decode_bundle(bundle_bytes)
        assert manifest['typeId'] == 'wlearn.nn.mlp.regressor@1'

        loaded = MLPRegressor._from_bundle(manifest, toc, blobs)
        np.testing.assert_allclose(
            model.predict(X), loaded.predict(X), atol=1e-5)

        model.dispose()
        loaded.dispose()

    def test_dispose(self):
        model = MLPRegressor.create({
            'hidden_sizes': [4], 'epochs': 5, 'lr': 0.01,
            'optimizer': 'sgd', 'seed': 42
        })
        X, y = make_regression_data(n=10)
        model.fit(X, y)
        model.dispose()
        assert not model.is_fitted
        with pytest.raises(DisposedError):
            model.predict(X)

    def test_capabilities(self):
        model = MLPRegressor.create()
        caps = model.capabilities
        assert caps['classifier'] is False
        assert caps['regressor'] is True
        assert caps['predictProba'] is False

    def test_get_set_params(self):
        model = MLPRegressor.create({'lr': 0.01})
        assert model.get_params()['lr'] == 0.01
        model.set_params({'epochs': 50})
        assert model.get_params()['epochs'] == 50
