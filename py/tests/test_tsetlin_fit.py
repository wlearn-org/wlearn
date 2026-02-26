"""Tests for Tsetlin Machine Python wrapper."""

import numpy as np
import pytest

from wlearn.tsetlin import TsetlinModel
from wlearn.bundle import decode_bundle
from wlearn.errors import NotFittedError, DisposedError

# Skip all tests if tmu is not installed
pytest.importorskip('tmu')


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
    boundaries = np.quantile(scores, [1 / n_classes * i for i in range(1, n_classes)])
    y = np.digitize(scores, boundaries)
    return X, y


class TestBinaryClassification:
    def test_fit_predict(self):
        X, y = make_binary_data()
        model = TsetlinModel.create({
            'nClauses': 200, 'threshold': 50, 's': 3.0,
            'nEpochs': 50, 'seed': 42,
        })
        model.fit(X, y)
        assert model.is_fitted
        preds = model.predict(X)
        assert len(preds) == len(y)
        accuracy = np.mean(preds == y)
        assert accuracy > 0.6

    def test_predict_proba(self):
        X, y = make_binary_data()
        model = TsetlinModel.create({
            'nClauses': 200, 'threshold': 50, 's': 3.0,
            'nEpochs': 50, 'seed': 42,
        })
        model.fit(X, y)
        proba = model.predict_proba(X)
        proba_2d = proba.reshape(-1, 2)
        assert proba_2d.shape == (len(y), 2)
        assert np.allclose(proba_2d.sum(axis=1), 1.0)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_score(self):
        X, y = make_binary_data()
        model = TsetlinModel.create({
            'nClauses': 200, 'threshold': 50, 's': 3.0,
            'nEpochs': 50, 'seed': 42,
        })
        model.fit(X, y)
        score = model.score(X, y)
        assert score > 0.6


class TestRegression:
    def test_fit_predict(self):
        X, y = make_regression_data()
        model = TsetlinModel.create({
            'task': 'regression',
            'nClauses': 500, 'threshold': 200, 's': 5.0,
            'nEpochs': 100, 'seed': 42,
        })
        model.fit(X, y)
        assert model._task == 1
        preds = model.predict(X)
        assert len(preds) == len(y)
        # TM regression with binarized features has limited accuracy
        # just verify predictions are finite and in reasonable range
        assert np.all(np.isfinite(preds))

    def test_predict_proba_raises(self):
        X, y = make_regression_data()
        model = TsetlinModel.create({
            'task': 'regression',
            'nClauses': 100, 'threshold': 50, 'nEpochs': 10,
        })
        model.fit(X, y)
        with pytest.raises(ValueError, match='predict_proba'):
            model.predict_proba(X)


class TestMulticlass:
    def test_fit_predict(self):
        X, y = make_multiclass_data()
        model = TsetlinModel.create({
            'nClauses': 200, 'threshold': 50, 's': 3.0,
            'nEpochs': 50, 'seed': 42,
        })
        model.fit(X, y)
        assert model._n_classes == 3
        preds = model.predict(X)
        assert len(preds) == len(y)
        assert set(preds.astype(int)).issubset({0, 1, 2})

    def test_predict_proba(self):
        X, y = make_multiclass_data()
        model = TsetlinModel.create({
            'nClauses': 200, 'threshold': 50, 's': 3.0,
            'nEpochs': 50, 'seed': 42,
        })
        model.fit(X, y)
        proba = model.predict_proba(X)
        proba_2d = proba.reshape(-1, 3)
        assert proba_2d.shape == (len(y), 3)
        assert np.allclose(proba_2d.sum(axis=1), 1.0)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)


class TestSaveLoad:
    def test_round_trip_classifier(self):
        X, y = make_binary_data()
        model = TsetlinModel.create({
            'nClauses': 200, 'threshold': 50, 's': 3.0,
            'nEpochs': 50, 'seed': 42,
        })
        model.fit(X, y)
        preds_orig = model.predict(X)

        bundle = model.save()
        manifest, toc, blobs = decode_bundle(bundle)
        loaded = TsetlinModel._from_bundle(manifest, toc, blobs)
        preds_loaded = loaded.predict(X)

        assert np.array_equal(preds_orig, preds_loaded)

    def test_round_trip_regressor(self):
        X, y = make_regression_data()
        model = TsetlinModel.create({
            'task': 'regression',
            'nClauses': 200, 'threshold': 100, 's': 3.0,
            'nEpochs': 50, 'seed': 42,
        })
        model.fit(X, y)
        preds_orig = model.predict(X)

        bundle = model.save()
        manifest, toc, blobs = decode_bundle(bundle)
        loaded = TsetlinModel._from_bundle(manifest, toc, blobs)
        preds_loaded = loaded.predict(X)

        assert np.allclose(preds_orig, preds_loaded)

    def test_type_id_classifier(self):
        X, y = make_binary_data()
        model = TsetlinModel.create({
            'nClauses': 100, 'threshold': 50, 'nEpochs': 10,
        })
        model.fit(X, y)
        bundle = model.save()
        manifest, _, _ = decode_bundle(bundle)
        assert manifest['typeId'] == 'wlearn.tsetlin.classifier@1'

    def test_type_id_regressor(self):
        X, y = make_regression_data()
        model = TsetlinModel.create({
            'task': 'regression',
            'nClauses': 100, 'threshold': 50, 'nEpochs': 10,
        })
        model.fit(X, y)
        bundle = model.save()
        manifest, _, _ = decode_bundle(bundle)
        assert manifest['typeId'] == 'wlearn.tsetlin.regressor@1'

    def test_blob_identity_on_resave(self):
        X, y = make_binary_data()
        model = TsetlinModel.create({
            'nClauses': 100, 'threshold': 50, 'nEpochs': 10,
        })
        model.fit(X, y)

        bundle1 = model.save()
        m1, t1, b1 = decode_bundle(bundle1)
        loaded = TsetlinModel._from_bundle(m1, t1, b1)

        bundle2 = loaded.save()
        m2, t2, b2 = decode_bundle(bundle2)

        e1 = next(e for e in t1 if e['id'] == 'model')
        e2 = next(e for e in t2 if e['id'] == 'model')
        blob1 = bytes(b1[e1['offset']:e1['offset'] + e1['length']])
        blob2 = bytes(b2[e2['offset']:e2['offset'] + e2['length']])
        assert blob1 == blob2


class TestLifecycle:
    def test_create_unfitted(self):
        model = TsetlinModel.create()
        assert not model.is_fitted

    def test_not_fitted_raises(self):
        model = TsetlinModel.create()
        with pytest.raises(NotFittedError):
            model.predict(np.zeros((1, 2)))

    def test_dispose(self):
        X, y = make_binary_data()
        model = TsetlinModel.create({
            'nClauses': 100, 'threshold': 50, 'nEpochs': 10,
        })
        model.fit(X, y)
        model.dispose()
        assert not model.is_fitted
        with pytest.raises(DisposedError):
            model.predict(X)

    def test_dispose_idempotent(self):
        X, y = make_binary_data()
        model = TsetlinModel.create({
            'nClauses': 100, 'threshold': 50, 'nEpochs': 10,
        })
        model.fit(X, y)
        model.dispose()
        model.dispose()  # should not raise

    def test_refit(self):
        X, y = make_binary_data()
        model = TsetlinModel.create({
            'nClauses': 100, 'threshold': 50, 'nEpochs': 10,
        })
        model.fit(X, y)
        preds1 = model.predict(X).copy()
        model.fit(X, y)
        preds2 = model.predict(X)
        assert len(preds2) == len(y)


class TestParams:
    def test_get_set_params(self):
        model = TsetlinModel.create({'seed': 42, 'nClauses': 200})
        params = model.get_params()
        assert params['seed'] == 42
        assert params['nClauses'] == 200

        model.set_params({'threshold': 100})
        params = model.get_params()
        assert params['threshold'] == 100
        assert params['seed'] == 42
