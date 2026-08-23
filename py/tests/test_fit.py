"""Tests for Python fit() on all model wrappers."""

import importlib.util

import numpy as np
import pytest

from wlearn.bundle import decode_bundle, encode_bundle
from wlearn.errors import NotFittedError, DisposedError


HAS_XGBOOST = importlib.util.find_spec('xgboost') is not None
HAS_LIBLINEAR = importlib.util.find_spec('liblinear') is not None
HAS_LIBSVM = importlib.util.find_spec('libsvm') is not None
HAS_NANOFLANN = importlib.util.find_spec('pynanoflann') is not None
HAS_LIGHTGBM = importlib.util.find_spec('lightgbm') is not None


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

@pytest.mark.skipif(not HAS_XGBOOST, reason='xgboost not installed')
class TestXGBoost:
    def test_task_resolution_matches_unified_api(self):
        from wlearn.xgboost import XGBModel
        X, y = make_binary_data(n=40)

        inferred = XGBModel.create({'numRound': 2})
        inferred.fit(X, y)
        assert inferred.get_params()['task'] == 'classification'
        assert inferred.get_params()['objective'] == 'binary:logistic'
        assert inferred.capabilities['classifier'] is True

        explicit_regression = XGBModel.create({
            'task': 'regression', 'numRound': 2})
        explicit_regression.fit(X, y.astype(np.int32))
        assert explicit_regression.get_params()['objective'] == \
            'reg:squarederror'
        assert explicit_regression.capabilities['regressor'] is True

        explicit_classification = XGBModel.create({
            'task': 'classification', 'numRound': 2})
        explicit_classification.fit(X, np.where(y == 0, 3, 7))
        assert explicit_classification.get_params()['objective'] == \
            'binary:logistic'
        np.testing.assert_array_equal(
            explicit_classification.classes, [3, 7])

        invalid = XGBModel.create({'task': 'clustering', 'numRound': 1})
        with pytest.raises(ValueError, match='Unknown task'):
            invalid.fit(X, y)

    @pytest.mark.parametrize('num_round', [0, -1, 1.5, True, 2 ** 53])
    def test_invalid_num_round(self, num_round):
        from wlearn.xgboost import XGBModel
        X, y = make_binary_data(n=40)
        model = XGBModel.create({
            'task': 'classification', 'numRound': num_round})
        with pytest.raises(ValueError, match='positive safe integer'):
            model.fit(X, y)
        assert not model.is_fitted

    def test_binary_classification(self):
        from wlearn.xgboost import XGBModel
        X, y = make_binary_data()
        model = XGBModel.create({'objective': 'binary:logistic', 'numRound': 50})
        model.fit(X, y)
        assert model.is_fitted
        assert model.predict(X).dtype == np.int32
        accuracy = model.score(X, y)
        assert accuracy > 0.7

    def test_noncontiguous_labels_and_invalid_refit(self):
        from wlearn.xgboost import XGBModel
        X, y = make_binary_data()
        y = np.where(y == 0, 3, 7).astype(np.int32)
        model = XGBModel.create({
            'objective': 'binary:logistic', 'numRound': 20})
        model.fit(X, y)
        before = model.predict(X)
        assert before.dtype == np.int32
        np.testing.assert_array_equal(model.classes, [3, 7])
        with pytest.raises(ValueError, match='y length'):
            model.fit(X, y[:-1])
        np.testing.assert_array_equal(model.predict(X), before)
        with pytest.raises(ValueError, match='y length'):
            model.score(X, y[:-1])

    @pytest.mark.parametrize('labels', [
        np.zeros(120),
        np.r_[0.5, np.zeros(119)],
        np.r_[2147483648, np.zeros(119)],
    ])
    def test_invalid_classifier_labels(self, labels):
        from wlearn.xgboost import XGBModel
        X, _ = make_binary_data()
        model = XGBModel.create({
            'objective': 'binary:logistic', 'numRound': 1})
        with pytest.raises(ValueError):
            model.fit(X, labels)

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
        classes = np.array([3, 7, 11], dtype=np.int32)
        y = classes[y]
        model = XGBModel.create({'objective': 'multi:softprob', 'numRound': 50})
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset(set(classes))

        proba = model.predict_proba(X)
        assert proba.dtype == np.float64
        assert proba.shape == (len(X) * len(classes),)
        proba_2d = proba.reshape(len(X), len(classes))
        np.testing.assert_allclose(proba_2d.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_array_equal(classes[proba_2d.argmax(axis=1)], preds)

        loaded = XGBModel._from_bundle(*decode_bundle(model.save()))
        np.testing.assert_array_equal(loaded.classes, classes)
        np.testing.assert_allclose(loaded.predict_proba(X), proba, atol=0)

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

    def test_bundle_rejects_swapped_model_identity_and_artifact_drift(self):
        from wlearn.xgboost import XGBModel
        Xc, yc = make_binary_data(n=40)
        Xr, yr = make_regression_data(n=40, n_features=3)
        classifier = XGBModel.create({
            'objective': 'binary:logistic', 'numRound': 2})
        regressor = XGBModel.create({
            'objective': 'reg:squarederror', 'numRound': 2})
        classifier.fit(Xc, yc)
        regressor.fit(Xr, yr)
        classifier_parts = decode_bundle(classifier.save())
        regressor_parts = decode_bundle(regressor.save())

        def artifact(parts):
            _, toc, blobs = parts
            entry = toc[0]
            return bytes(
                blobs[entry['offset']:entry['offset'] + entry['length']])

        manifest = classifier_parts[0]
        swapped = encode_bundle(
            manifest, [{'id': 'model', 'data': artifact(regressor_parts)}])
        with pytest.raises(ValueError, match='model objective'):
            decode = decode_bundle(swapped)
            XGBModel._from_bundle(*decode)

        extra = encode_bundle(manifest, [
            {'id': 'model', 'data': artifact(classifier_parts)},
            {'id': 'extra', 'data': b'x'},
        ])
        with pytest.raises(ValueError, match='exactly one'):
            XGBModel._from_bundle(*decode_bundle(extra))

        wrong_features = {
            **manifest,
            'metadata': {
                **manifest['metadata'],
                'nFeatures': manifest['metadata']['nFeatures'] + 1,
            },
        }
        mismatch = encode_bundle(
            wrong_features,
            [{'id': 'model', 'data': artifact(classifier_parts)}],
        )
        with pytest.raises(ValueError, match='feature count'):
            XGBModel._from_bundle(*decode_bundle(mismatch))

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

@pytest.mark.skipif(not HAS_LIBLINEAR, reason='liblinear-official not installed')
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

@pytest.mark.skipif(not HAS_LIBSVM, reason='libsvm-official not installed')
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

@pytest.mark.skipif(not HAS_NANOFLANN, reason='pynanoflann not installed')
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

@pytest.mark.skipif(not HAS_LIGHTGBM, reason='lightgbm not installed')
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
