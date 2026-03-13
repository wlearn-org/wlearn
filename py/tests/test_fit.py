"""Tests for Python fit() on all model wrappers."""

import numpy as np
import pytest

from wlearn.bundle import decode_bundle
from wlearn.errors import NotFittedError, DisposedError


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


def make_multiclass_data(seed=42, n=150, n_classes=3):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)
    scores = X[:, 0] + X[:, 1]
    boundaries = np.quantile(scores, [1/n_classes * i for i in range(1, n_classes)])
    y = np.digitize(scores, boundaries)
    return X, y


# ===========================================================================
# XGBoost
# ===========================================================================

class TestXGBoost:
    def test_binary_classification(self):
        from wlearn.xgboost import XGBModel
        X, y = make_binary_data()
        model = XGBModel.create({'objective': 'binary:logistic', 'numRound': 50})
        model.fit(X, y)
        assert model.is_fitted
        accuracy = model.score(X, y)
        assert accuracy > 0.7

    def test_regression(self):
        from wlearn.xgboost import XGBModel
        X, y = make_regression_data()
        model = XGBModel.create({'objective': 'reg:squarederror', 'numRound': 50})
        model.fit(X, y)
        r2 = model.score(X, y)
        assert r2 > 0.5

    def test_multiclass(self):
        from wlearn.xgboost import XGBModel
        X, y = make_multiclass_data()
        model = XGBModel.create({'objective': 'multi:softprob', 'numRound': 50})
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1, 2})

    def test_predict_proba(self):
        from wlearn.xgboost import XGBModel
        X, y = make_binary_data()
        model = XGBModel.create({'objective': 'binary:logistic', 'numRound': 50})
        model.fit(X, y)
        proba = model.predict_proba(X)
        proba_2d = proba.reshape(-1, 2)
        assert np.allclose(proba_2d.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(proba >= 0)

    def test_save_load_roundtrip(self):
        from wlearn.xgboost import XGBModel
        X, y = make_binary_data()
        model = XGBModel.create({'objective': 'binary:logistic', 'numRound': 50})
        model.fit(X, y)
        preds_orig = model.predict(X)

        bundle = model.save()
        manifest, toc, blobs = decode_bundle(bundle)
        loaded = XGBModel._from_bundle(manifest, toc, blobs)
        preds_loaded = loaded.predict(X)
        assert np.array_equal(preds_orig, preds_loaded)

    def test_create_unfitted(self):
        from wlearn.xgboost import XGBModel
        model = XGBModel.create()
        assert not model.is_fitted
        with pytest.raises(NotFittedError):
            model.predict(np.zeros((1, 2)))

    def test_dispose(self):
        from wlearn.xgboost import XGBModel
        X, y = make_binary_data()
        model = XGBModel.create({'objective': 'binary:logistic', 'numRound': 20})
        model.fit(X, y)
        model.dispose()
        with pytest.raises(DisposedError):
            model.predict(X)


# ===========================================================================
# Liblinear
# ===========================================================================

class TestLiblinear:
    def test_classification(self):
        from wlearn.liblinear import LinearModel
        X, y = make_binary_data()
        model = LinearModel.create({'solver': 0, 'C': 1.0})
        model.fit(X, y)
        assert model.is_fitted
        accuracy = model.score(X, y)
        assert accuracy > 0.7

    def test_regression(self):
        from wlearn.liblinear import LinearModel
        X, y = make_regression_data()
        model = LinearModel.create({'solver': 11, 'C': 1.0})
        model.fit(X, y)
        r2 = model.score(X, y)
        assert r2 > 0.5

    def test_predict_proba(self):
        from wlearn.liblinear import LinearModel
        X, y = make_binary_data()
        model = LinearModel.create({'solver': 0, 'C': 1.0})
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert len(proba) > 0
        assert np.all(proba >= 0)

    def test_save_load_roundtrip(self):
        from wlearn.liblinear import LinearModel
        X, y = make_binary_data()
        model = LinearModel.create({'solver': 0, 'C': 1.0})
        model.fit(X, y)
        preds_orig = model.predict(X)

        bundle = model.save()
        manifest, toc, blobs = decode_bundle(bundle)
        loaded = LinearModel._from_bundle(manifest, toc, blobs)
        preds_loaded = loaded.predict(X)
        assert np.array_equal(preds_orig, preds_loaded)

    def test_create_unfitted(self):
        from wlearn.liblinear import LinearModel
        model = LinearModel.create()
        assert not model.is_fitted
        with pytest.raises(NotFittedError):
            model.predict(np.zeros((1, 2)))

    def test_dispose(self):
        from wlearn.liblinear import LinearModel
        X, y = make_binary_data()
        model = LinearModel.create({'solver': 0})
        model.fit(X, y)
        model.dispose()
        with pytest.raises(DisposedError):
            model.predict(X)


# ===========================================================================
# Libsvm
# ===========================================================================

class TestLibsvm:
    def test_classification(self):
        from wlearn.libsvm import SVMModel
        X, y = make_binary_data()
        model = SVMModel.create({'svmType': 0, 'kernel': 2, 'C': 1.0})
        model.fit(X, y)
        assert model.is_fitted
        accuracy = model.score(X, y)
        assert accuracy > 0.7

    def test_regression(self):
        from wlearn.libsvm import SVMModel
        X, y = make_regression_data()
        model = SVMModel.create({'svmType': 3, 'kernel': 2, 'C': 1.0})
        model.fit(X, y)
        r2 = model.score(X, y)
        assert r2 > 0.3

    def test_predict_proba(self):
        from wlearn.libsvm import SVMModel
        X, y = make_binary_data()
        model = SVMModel.create({'svmType': 0, 'kernel': 2, 'probability': 1})
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert len(proba) > 0
        assert np.all(proba >= 0)

    def test_save_load_roundtrip(self):
        from wlearn.libsvm import SVMModel
        X, y = make_binary_data()
        model = SVMModel.create({'svmType': 0, 'kernel': 2})
        model.fit(X, y)
        preds_orig = model.predict(X)

        bundle = model.save()
        manifest, toc, blobs = decode_bundle(bundle)
        loaded = SVMModel._from_bundle(manifest, toc, blobs)
        preds_loaded = loaded.predict(X)
        assert np.array_equal(preds_orig, preds_loaded)

    def test_create_unfitted(self):
        from wlearn.libsvm import SVMModel
        model = SVMModel.create()
        assert not model.is_fitted
        with pytest.raises(NotFittedError):
            model.predict(np.zeros((1, 2)))

    def test_dispose(self):
        from wlearn.libsvm import SVMModel
        X, y = make_binary_data()
        model = SVMModel.create({'svmType': 0, 'kernel': 2})
        model.fit(X, y)
        model.dispose()
        with pytest.raises(DisposedError):
            model.predict(X)


# ===========================================================================
# Nanoflann (KNN)
# ===========================================================================

class TestNanoflann:
    def test_classification(self):
        from wlearn.nanoflann import KNNModel
        X, y = make_binary_data()
        model = KNNModel.create({'k': 5, 'task': 'classification'})
        model.fit(X, y)
        assert model.is_fitted
        accuracy = model.score(X, y)
        assert accuracy > 0.7

    def test_regression(self):
        from wlearn.nanoflann import KNNModel
        X, y = make_regression_data()
        model = KNNModel.create({'k': 5, 'task': 'regression'})
        model.fit(X, y)
        r2 = model.score(X, y)
        assert r2 > 0.3

    def test_predict_proba(self):
        from wlearn.nanoflann import KNNModel
        X, y = make_binary_data()
        model = KNNModel.create({'k': 5, 'task': 'classification'})
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert len(proba) > 0
        assert np.all(proba >= 0)

    def test_save_load_roundtrip(self):
        from wlearn.nanoflann import KNNModel
        X, y = make_binary_data()
        model = KNNModel.create({'k': 5, 'task': 'classification'})
        model.fit(X, y)
        preds_orig = model.predict(X)

        bundle = model.save()
        manifest, toc, blobs = decode_bundle(bundle)
        loaded = KNNModel._from_bundle(manifest, toc, blobs)
        preds_loaded = loaded.predict(X)
        assert np.array_equal(preds_orig, preds_loaded)

    def test_create_unfitted(self):
        from wlearn.nanoflann import KNNModel
        model = KNNModel.create()
        assert not model.is_fitted
        with pytest.raises(NotFittedError):
            model.predict(np.zeros((1, 2)))

    def test_dispose(self):
        from wlearn.nanoflann import KNNModel
        X, y = make_binary_data()
        model = KNNModel.create({'k': 5, 'task': 'classification'})
        model.fit(X, y)
        model.dispose()
        with pytest.raises(DisposedError):
            model.predict(X)


# ===========================================================================
# LightGBM
# ===========================================================================

class TestLightGBM:
    def test_binary_classification(self):
        from wlearn.lightgbm import LGBModel
        X, y = make_binary_data()
        model = LGBModel.create({'objective': 'binary', 'numRound': 50, 'verbosity': -1})
        model.fit(X, y)
        assert model.is_fitted
        accuracy = model.score(X, y)
        assert accuracy > 0.7

    def test_regression(self):
        from wlearn.lightgbm import LGBModel
        X, y = make_regression_data()
        model = LGBModel.create({'objective': 'regression', 'numRound': 50, 'verbosity': -1})
        model.fit(X, y)
        r2 = model.score(X, y)
        assert r2 > 0.5

    def test_multiclass(self):
        from wlearn.lightgbm import LGBModel
        X, y = make_multiclass_data()
        model = LGBModel.create({'objective': 'multiclass', 'numRound': 50, 'verbosity': -1})
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1, 2})

    def test_predict_proba(self):
        from wlearn.lightgbm import LGBModel
        X, y = make_binary_data()
        model = LGBModel.create({'objective': 'binary', 'numRound': 50, 'verbosity': -1})
        model.fit(X, y)
        proba = model.predict_proba(X)
        proba_2d = proba.reshape(-1, 2)
        assert np.allclose(proba_2d.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(proba >= 0)

    def test_save_load_roundtrip(self):
        from wlearn.lightgbm import LGBModel
        X, y = make_binary_data()
        model = LGBModel.create({'objective': 'binary', 'numRound': 50, 'verbosity': -1})
        model.fit(X, y)
        preds_orig = model.predict(X)

        bundle = model.save()
        manifest, toc, blobs = decode_bundle(bundle)
        loaded = LGBModel._from_bundle(manifest, toc, blobs)
        preds_loaded = loaded.predict(X)
        assert np.array_equal(preds_orig, preds_loaded)

    def test_create_unfitted(self):
        from wlearn.lightgbm import LGBModel
        model = LGBModel.create()
        assert not model.is_fitted
        with pytest.raises(NotFittedError):
            model.predict(np.zeros((1, 2)))

    def test_dispose(self):
        from wlearn.lightgbm import LGBModel
        X, y = make_binary_data()
        model = LGBModel.create({'objective': 'binary', 'numRound': 20, 'verbosity': -1})
        model.fit(X, y)
        model.dispose()
        with pytest.raises(DisposedError):
            model.predict(X)


# ===========================================================================
# RF (C11 core via ctypes)
# ===========================================================================

rf_available = True
try:
    from wlearn.rf import RFClassifier, RFRegressor
except RuntimeError:
    rf_available = False


@pytest.mark.skipif(not rf_available, reason='librf not found (set RF_LIB_PATH)')
class TestRF:
    def test_classification(self):
        from wlearn.rf import RFClassifier
        X, y = make_binary_data()
        model = RFClassifier.create({'n_estimators': 50, 'seed': 42})
        model.fit(X, y)
        assert model.is_fitted
        accuracy = model.score(X, y)
        assert accuracy > 0.8
        model.dispose()

    def test_regression(self):
        from wlearn.rf import RFRegressor
        X, y = make_regression_data()
        model = RFRegressor.create({'n_estimators': 50, 'seed': 42})
        model.fit(X, y)
        r2 = model.score(X, y)
        assert r2 > 0.7
        model.dispose()

    def test_multiclass(self):
        from wlearn.rf import RFClassifier
        X, y = make_multiclass_data()
        model = RFClassifier.create({'n_estimators': 50, 'seed': 42})
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)
        model.dispose()

    def test_predict_proba(self):
        from wlearn.rf import RFClassifier
        X, y = make_binary_data()
        model = RFClassifier.create({'n_estimators': 50, 'seed': 42})
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(proba >= 0)
        model.dispose()

    def test_histogram_binning(self):
        from wlearn.rf import RFClassifier
        X, y = make_binary_data(n=200)
        model = RFClassifier.create({
            'n_estimators': 50, 'histogram_binning': 1, 'seed': 42})
        model.fit(X, y)
        accuracy = model.score(X, y)
        assert accuracy > 0.8
        model.dispose()

    def test_jarf(self):
        from wlearn.rf import RFClassifier
        X, y = make_binary_data(n=200)
        model = RFClassifier.create({
            'n_estimators': 50, 'jarf': 1, 'seed': 42})
        model.fit(X, y)
        accuracy = model.score(X, y)
        assert accuracy > 0.8
        model.dispose()

    def test_sample_weight(self):
        from wlearn.rf import RFClassifier
        X, y = make_binary_data()
        sw = np.ones(len(y))
        model = RFClassifier.create({
            'n_estimators': 50, 'sample_weight': sw, 'seed': 42})
        model.fit(X, y)
        assert model.is_fitted
        model.dispose()

    def test_class_weight_balanced(self):
        from wlearn.rf import RFClassifier
        X, y = make_binary_data()
        model = RFClassifier.create({
            'n_estimators': 50, 'class_weight': 'balanced', 'seed': 42})
        model.fit(X, y)
        assert model.is_fitted
        model.dispose()

    def test_permutation_importance(self):
        from wlearn.rf import RFClassifier
        X, y = make_binary_data()
        model = RFClassifier.create({'n_estimators': 50, 'seed': 42})
        model.fit(X, y)
        imp = model.permutation_importance(X, y, n_repeats=3, seed=42)
        assert imp.shape == (X.shape[1],)
        model.dispose()

    def test_proximity(self):
        from wlearn.rf import RFClassifier
        X, y = make_binary_data(n=30)
        model = RFClassifier.create({'n_estimators': 50, 'seed': 42})
        model.fit(X, y)
        prox = model.proximity(X)
        assert prox.shape == (30, 30)
        assert np.allclose(np.diag(prox), 1.0)
        assert np.allclose(prox, prox.T)
        model.dispose()

    def test_feature_importances(self):
        from wlearn.rf import RFClassifier
        X, y = make_binary_data()
        model = RFClassifier.create({'n_estimators': 50, 'seed': 42})
        model.fit(X, y)
        imp = model.feature_importances()
        assert imp.shape == (X.shape[1],)
        assert np.all(imp >= 0)
        model.dispose()

    def test_save_load_roundtrip(self):
        from wlearn.rf import RFClassifier
        X, y = make_binary_data()
        model = RFClassifier.create({'n_estimators': 50, 'seed': 42})
        model.fit(X, y)
        preds_orig = model.predict(X)

        bundle = model.save()
        manifest, toc, blobs = decode_bundle(bundle)
        loaded = RFClassifier._from_bundle(manifest, toc, blobs)
        preds_loaded = loaded.predict(X)
        assert np.array_equal(preds_orig, preds_loaded)
        model.dispose()
        loaded.dispose()

    def test_create_unfitted(self):
        from wlearn.rf import RFClassifier
        model = RFClassifier.create()
        assert not model.is_fitted
        with pytest.raises(NotFittedError):
            model.predict(np.zeros((1, 2)))

    def test_dispose(self):
        from wlearn.rf import RFClassifier
        X, y = make_binary_data()
        model = RFClassifier.create({'n_estimators': 20, 'seed': 42})
        model.fit(X, y)
        model.dispose()
        with pytest.raises(DisposedError):
            model.predict(X)
