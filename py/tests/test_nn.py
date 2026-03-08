"""Tests for wlearn.nn (MLP, TabM, NAM classifiers and regressors)."""

import numpy as np
import pytest
from wlearn.nn import (MLPClassifier, MLPRegressor,
                        TabMClassifier, TabMRegressor,
                        NAMClassifier, NAMRegressor)
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
        assert model.score(X, y) > 0.9

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
        assert model.score(X, y) > 0.9

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


class TestTabMClassifier:
    def test_create_unfitted(self):
        model = TabMClassifier.create()
        assert not model.is_fitted
        with pytest.raises(NotFittedError):
            model.predict(np.zeros((1, 2)))

    def test_binary_classification(self):
        X, y = make_binary_data()
        model = TabMClassifier.create({
            'hidden_sizes': [8], 'epochs': 50, 'lr': 0.01,
            'optimizer': 'sgd', 'seed': 42, 'n_ensemble': 4
        })
        model.fit(X, y)
        assert model.is_fitted
        assert model.score(X, y) > 0.9

    def test_predict_shape(self):
        X, y = make_binary_data(n=20)
        model = TabMClassifier.create({
            'hidden_sizes': [4], 'epochs': 10, 'lr': 0.01,
            'optimizer': 'sgd', 'seed': 42, 'n_ensemble': 4
        })
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 20
        assert preds.dtype == np.float64

    def test_predict_proba(self):
        X, y = make_binary_data(n=20)
        model = TabMClassifier.create({
            'hidden_sizes': [4], 'epochs': 10, 'lr': 0.01,
            'seed': 42, 'n_ensemble': 4
        })
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert len(proba) == 20 * 2
        proba_2d = proba.reshape(-1, 2)
        np.testing.assert_allclose(proba_2d.sum(axis=1), 1.0, atol=1e-5)
        assert np.all(proba >= 0)

    def test_save_load_roundtrip(self):
        X, y = make_binary_data(n=20)
        model = TabMClassifier.create({
            'hidden_sizes': [4], 'epochs': 20, 'lr': 0.01,
            'seed': 42, 'n_ensemble': 4
        })
        model.fit(X, y)

        bundle_bytes = model.save()
        assert len(bundle_bytes) > 0

        manifest, toc, blobs = decode_bundle(bundle_bytes)
        assert manifest['typeId'] == 'wlearn.nn.tabm.classifier@1'

        loaded = TabMClassifier._from_bundle(manifest, toc, blobs)
        assert loaded.is_fitted

        np.testing.assert_array_equal(
            model.predict(X), loaded.predict(X))

        model.dispose()
        loaded.dispose()

    def test_dispose(self):
        model = TabMClassifier.create({
            'hidden_sizes': [4], 'epochs': 5, 'seed': 42, 'n_ensemble': 4
        })
        X, y = make_binary_data(n=10)
        model.fit(X, y)
        model.dispose()
        assert not model.is_fitted
        with pytest.raises(DisposedError):
            model.predict(X)
        model.dispose()

    def test_capabilities(self):
        model = TabMClassifier.create()
        caps = model.capabilities
        assert caps['classifier'] is True
        assert caps['regressor'] is False
        assert caps['predictProba'] is True
        assert caps['earlyStopping'] is True

    def test_get_set_params(self):
        model = TabMClassifier.create({'n_ensemble': 8})
        assert model.get_params()['n_ensemble'] == 8
        model.set_params({'n_ensemble': 16})
        assert model.get_params()['n_ensemble'] == 16


class TestTabMRegressor:
    def test_create_unfitted(self):
        model = TabMRegressor.create()
        assert not model.is_fitted
        with pytest.raises(NotFittedError):
            model.predict(np.zeros((1, 2)))

    def test_regression(self):
        X, y = make_regression_data()
        model = TabMRegressor.create({
            'hidden_sizes': [16], 'epochs': 100, 'lr': 0.01,
            'optimizer': 'adam', 'seed': 42, 'n_ensemble': 4
        })
        model.fit(X, y)
        assert model.is_fitted
        assert model.score(X, y) > 0.9

    def test_predict_shape(self):
        X, y = make_regression_data(n=20)
        model = TabMRegressor.create({
            'hidden_sizes': [4], 'epochs': 10, 'lr': 0.01,
            'seed': 42, 'n_ensemble': 4
        })
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 20
        assert preds.dtype == np.float64

    def test_save_load_roundtrip(self):
        X, y = make_regression_data(n=20)
        model = TabMRegressor.create({
            'hidden_sizes': [4], 'epochs': 20, 'lr': 0.01,
            'seed': 42, 'n_ensemble': 4
        })
        model.fit(X, y)

        bundle_bytes = model.save()
        manifest, toc, blobs = decode_bundle(bundle_bytes)
        assert manifest['typeId'] == 'wlearn.nn.tabm.regressor@1'

        loaded = TabMRegressor._from_bundle(manifest, toc, blobs)
        np.testing.assert_allclose(
            model.predict(X), loaded.predict(X), atol=1e-5)

        model.dispose()
        loaded.dispose()

    def test_dispose(self):
        model = TabMRegressor.create({
            'hidden_sizes': [4], 'epochs': 5, 'seed': 42, 'n_ensemble': 4
        })
        X, y = make_regression_data(n=10)
        model.fit(X, y)
        model.dispose()
        assert not model.is_fitted
        with pytest.raises(DisposedError):
            model.predict(X)

    def test_capabilities(self):
        model = TabMRegressor.create()
        caps = model.capabilities
        assert caps['classifier'] is False
        assert caps['regressor'] is True
        assert caps['predictProba'] is False
        assert caps['earlyStopping'] is True

    def test_get_set_params(self):
        model = TabMRegressor.create({'n_ensemble': 8})
        assert model.get_params()['n_ensemble'] == 8


class TestNAMClassifier:
    def test_create_unfitted(self):
        model = NAMClassifier.create()
        assert not model.is_fitted
        with pytest.raises(NotFittedError):
            model.predict(np.zeros((1, 2)))

    def test_binary_classification_relu(self):
        X, y = make_binary_data()
        model = NAMClassifier.create({
            'hidden_sizes': [8], 'epochs': 80, 'lr': 0.01,
            'optimizer': 'adam', 'seed': 42, 'activation': 'relu'
        })
        model.fit(X, y)
        assert model.is_fitted
        assert model.score(X, y) > 0.9

    def test_binary_classification_exu(self):
        X, y = make_binary_data()
        model = NAMClassifier.create({
            'hidden_sizes': [8], 'epochs': 80, 'lr': 0.01,
            'optimizer': 'adam', 'seed': 42, 'activation': 'exu'
        })
        model.fit(X, y)
        assert model.is_fitted
        assert model.score(X, y) > 0.9

    def test_predict_shape(self):
        X, y = make_binary_data(n=20)
        model = NAMClassifier.create({
            'hidden_sizes': [4], 'epochs': 10, 'lr': 0.01,
            'seed': 42, 'activation': 'relu'
        })
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 20
        assert preds.dtype == np.float64

    def test_predict_proba(self):
        X, y = make_binary_data(n=20)
        model = NAMClassifier.create({
            'hidden_sizes': [4], 'epochs': 10, 'lr': 0.01,
            'seed': 42, 'activation': 'relu'
        })
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert len(proba) == 20 * 2
        proba_2d = proba.reshape(-1, 2)
        np.testing.assert_allclose(proba_2d.sum(axis=1), 1.0, atol=1e-5)
        assert np.all(proba >= 0)

    def test_save_load_roundtrip(self):
        X, y = make_binary_data(n=20)
        model = NAMClassifier.create({
            'hidden_sizes': [4], 'epochs': 20, 'lr': 0.01,
            'seed': 42, 'activation': 'relu'
        })
        model.fit(X, y)

        bundle_bytes = model.save()
        assert len(bundle_bytes) > 0

        manifest, toc, blobs = decode_bundle(bundle_bytes)
        assert manifest['typeId'] == 'wlearn.nn.nam.classifier@1'

        loaded = NAMClassifier._from_bundle(manifest, toc, blobs)
        assert loaded.is_fitted

        np.testing.assert_array_equal(
            model.predict(X), loaded.predict(X))

        model.dispose()
        loaded.dispose()

    def test_dispose(self):
        model = NAMClassifier.create({
            'hidden_sizes': [4], 'epochs': 5, 'seed': 42, 'activation': 'relu'
        })
        X, y = make_binary_data(n=10)
        model.fit(X, y)
        model.dispose()
        assert not model.is_fitted
        with pytest.raises(DisposedError):
            model.predict(X)
        model.dispose()

    def test_capabilities(self):
        model = NAMClassifier.create()
        caps = model.capabilities
        assert caps['classifier'] is True
        assert caps['regressor'] is False
        assert caps['predictProba'] is True
        assert caps['earlyStopping'] is True

    def test_get_set_params(self):
        model = NAMClassifier.create({'activation': 'exu'})
        assert model.get_params()['activation'] == 'exu'
        model.set_params({'activation': 'relu'})
        assert model.get_params()['activation'] == 'relu'


class TestNAMRegressor:
    def test_create_unfitted(self):
        model = NAMRegressor.create()
        assert not model.is_fitted
        with pytest.raises(NotFittedError):
            model.predict(np.zeros((1, 2)))

    def test_regression_relu(self):
        X, y = make_regression_data()
        model = NAMRegressor.create({
            'hidden_sizes': [16], 'epochs': 150, 'lr': 0.01,
            'optimizer': 'adam', 'seed': 42, 'activation': 'relu'
        })
        model.fit(X, y)
        assert model.is_fitted
        assert model.score(X, y) > 0.9

    def test_regression_exu(self):
        X, y = make_regression_data()
        model = NAMRegressor.create({
            'hidden_sizes': [16], 'epochs': 150, 'lr': 0.01,
            'optimizer': 'adam', 'seed': 42, 'activation': 'exu'
        })
        model.fit(X, y)
        assert model.is_fitted
        assert model.score(X, y) > 0.9

    def test_predict_shape(self):
        X, y = make_regression_data(n=20)
        model = NAMRegressor.create({
            'hidden_sizes': [4], 'epochs': 10, 'lr': 0.01,
            'seed': 42, 'activation': 'relu'
        })
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 20
        assert preds.dtype == np.float64

    def test_save_load_roundtrip(self):
        X, y = make_regression_data(n=20)
        model = NAMRegressor.create({
            'hidden_sizes': [4], 'epochs': 20, 'lr': 0.01,
            'seed': 42, 'activation': 'relu'
        })
        model.fit(X, y)

        bundle_bytes = model.save()
        manifest, toc, blobs = decode_bundle(bundle_bytes)
        assert manifest['typeId'] == 'wlearn.nn.nam.regressor@1'

        loaded = NAMRegressor._from_bundle(manifest, toc, blobs)
        np.testing.assert_allclose(
            model.predict(X), loaded.predict(X), atol=1e-5)

        model.dispose()
        loaded.dispose()

    def test_dispose(self):
        model = NAMRegressor.create({
            'hidden_sizes': [4], 'epochs': 5, 'seed': 42, 'activation': 'relu'
        })
        X, y = make_regression_data(n=10)
        model.fit(X, y)
        model.dispose()
        assert not model.is_fitted
        with pytest.raises(DisposedError):
            model.predict(X)

    def test_capabilities(self):
        model = NAMRegressor.create()
        caps = model.capabilities
        assert caps['classifier'] is False
        assert caps['regressor'] is True
        assert caps['predictProba'] is False
        assert caps['earlyStopping'] is True

    def test_get_set_params(self):
        model = NAMRegressor.create({'activation': 'exu'})
        assert model.get_params()['activation'] == 'exu'
