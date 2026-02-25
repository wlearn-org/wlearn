"""Tests for EBM Python fit() wrapping interpret package."""

import numpy as np
import pytest

from wlearn.ebm import EBMModel
from wlearn.bundle import decode_bundle
from wlearn.errors import NotFittedError, DisposedError

# Skip all tests if interpret is not installed
pytest.importorskip('interpret')


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
    # Create separable classes using feature sums
    scores = X[:, 0] + X[:, 1]
    boundaries = np.quantile(scores, [1/n_classes * i for i in range(1, n_classes)])
    y = np.digitize(scores, boundaries)
    return X, y


class TestBinaryClassification:
    def test_fit_predict(self):
        X, y = make_binary_data()
        model = EBMModel.create({'seed': 42, 'maxRounds': 100, 'maxInteractions': 0})
        model.fit(X, y)
        assert model.is_fitted
        preds = model.predict(X)
        assert len(preds) == len(y)
        accuracy = np.mean(preds == y)
        assert accuracy > 0.7

    def test_predict_proba(self):
        X, y = make_binary_data()
        model = EBMModel.create({'seed': 42, 'maxRounds': 100, 'maxInteractions': 0})
        model.fit(X, y)
        proba = model.predict_proba(X)
        proba_2d = proba.reshape(-1, 2)
        assert proba_2d.shape == (len(y), 2)
        assert np.allclose(proba_2d.sum(axis=1), 1.0)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_score(self):
        X, y = make_binary_data()
        model = EBMModel.create({'seed': 42, 'maxRounds': 100, 'maxInteractions': 0})
        model.fit(X, y)
        score = model.score(X, y)
        assert score > 0.7

    def test_explain(self):
        X, y = make_binary_data()
        model = EBMModel.create({'seed': 42, 'maxRounds': 100, 'maxInteractions': 0})
        model.fit(X, y)
        expl = model.explain(X[:5])
        assert expl['nSamples'] == 5
        assert expl['nTerms'] == model._n_terms
        assert expl['nScores'] == 1
        # Contributions sum + intercept should approximate raw scores
        contrib = np.array(expl['contributions']).reshape(5, model._n_terms, 1)
        scores_from_contrib = contrib.sum(axis=1)[:, 0] + expl['intercept'][0]
        raw_scores = model._predict_scores(X[:5])[:, 0]
        assert np.allclose(scores_from_contrib, raw_scores, atol=1e-10)

    def test_feature_importances(self):
        X, y = make_binary_data()
        model = EBMModel.create({'seed': 42, 'maxRounds': 100, 'maxInteractions': 0})
        model.fit(X, y)
        imp = model.feature_importances()
        assert len(imp) == model._n_terms
        assert np.all(imp >= 0)


class TestRegression:
    def test_fit_predict(self):
        X, y = make_regression_data()
        model = EBMModel.create({'seed': 42, 'maxRounds': 200, 'maxInteractions': 0,
                                  'objective': 'regression'})
        model.fit(X, y)
        assert model._task == 'regression'
        r2 = model.score(X, y)
        assert r2 > 0.5

    def test_predict_proba_raises(self):
        X, y = make_regression_data()
        model = EBMModel.create({'seed': 42, 'maxRounds': 100, 'objective': 'regression'})
        model.fit(X, y)
        with pytest.raises(ValueError, match='predict_proba'):
            model.predict_proba(X)


class TestMulticlass:
    def test_fit_predict(self):
        X, y = make_multiclass_data()
        model = EBMModel.create({'seed': 42, 'maxRounds': 200, 'maxInteractions': 0})
        model.fit(X, y)
        assert model._n_classes == 3
        assert model._n_scores == 3
        preds = model.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1, 2})

    def test_predict_proba(self):
        X, y = make_multiclass_data()
        model = EBMModel.create({'seed': 42, 'maxRounds': 200, 'maxInteractions': 0})
        model.fit(X, y)
        proba = model.predict_proba(X)
        proba_2d = proba.reshape(-1, 3)
        assert proba_2d.shape == (len(y), 3)
        assert np.allclose(proba_2d.sum(axis=1), 1.0)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)


class TestInteractions:
    def test_interactions_detected(self):
        X, y = make_binary_data(n=200)
        model = EBMModel.create({'seed': 42, 'maxRounds': 100, 'maxInteractions': 3})
        model.fit(X, y)
        # With 3 features + up to 3 interactions, should have > 3 terms
        assert model._n_terms > 3


class TestSaveLoad:
    def test_round_trip_classifier(self):
        X, y = make_binary_data()
        model = EBMModel.create({'seed': 42, 'maxRounds': 100, 'maxInteractions': 0})
        model.fit(X, y)
        preds_orig = model.predict(X)

        bundle = model.save()
        manifest, toc, blobs = decode_bundle(bundle)
        loaded = EBMModel._from_bundle(manifest, toc, blobs)
        preds_loaded = loaded.predict(X)

        assert np.array_equal(preds_orig, preds_loaded)

    def test_round_trip_regressor(self):
        X, y = make_regression_data()
        model = EBMModel.create({'seed': 42, 'maxRounds': 100, 'objective': 'regression'})
        model.fit(X, y)
        preds_orig = model.predict(X)

        bundle = model.save()
        manifest, toc, blobs = decode_bundle(bundle)
        loaded = EBMModel._from_bundle(manifest, toc, blobs)
        preds_loaded = loaded.predict(X)

        assert np.allclose(preds_orig, preds_loaded)

    def test_type_id_classifier(self):
        X, y = make_binary_data()
        model = EBMModel.create({'seed': 42, 'maxRounds': 50, 'maxInteractions': 0})
        model.fit(X, y)
        bundle = model.save()
        manifest, _, _ = decode_bundle(bundle)
        assert manifest['typeId'] == 'wlearn.ebm.classifier@1'

    def test_type_id_regressor(self):
        X, y = make_regression_data()
        model = EBMModel.create({'seed': 42, 'maxRounds': 50, 'objective': 'regression'})
        model.fit(X, y)
        bundle = model.save()
        manifest, _, _ = decode_bundle(bundle)
        assert manifest['typeId'] == 'wlearn.ebm.regressor@1'

    def test_blob_identity_on_resave(self):
        X, y = make_binary_data()
        model = EBMModel.create({'seed': 42, 'maxRounds': 50, 'maxInteractions': 0})
        model.fit(X, y)

        bundle1 = model.save()
        m1, t1, b1 = decode_bundle(bundle1)
        loaded = EBMModel._from_bundle(m1, t1, b1)

        bundle2 = loaded.save()
        m2, t2, b2 = decode_bundle(bundle2)

        e1 = next(e for e in t1 if e['id'] == 'model')
        e2 = next(e for e in t2 if e['id'] == 'model')
        blob1 = bytes(b1[e1['offset']:e1['offset'] + e1['length']])
        blob2 = bytes(b2[e2['offset']:e2['offset'] + e2['length']])
        assert blob1 == blob2


class TestLifecycle:
    def test_create_unfitted(self):
        model = EBMModel.create()
        assert not model.is_fitted

    def test_not_fitted_raises(self):
        model = EBMModel.create()
        with pytest.raises(NotFittedError):
            model.predict(np.zeros((1, 2)))

    def test_dispose(self):
        X, y = make_binary_data()
        model = EBMModel.create({'seed': 42, 'maxRounds': 50, 'maxInteractions': 0})
        model.fit(X, y)
        model.dispose()
        assert not model.is_fitted
        with pytest.raises(DisposedError):
            model.predict(X)

    def test_dispose_idempotent(self):
        X, y = make_binary_data()
        model = EBMModel.create({'seed': 42, 'maxRounds': 50, 'maxInteractions': 0})
        model.fit(X, y)
        model.dispose()
        model.dispose()  # should not raise

    def test_refit(self):
        X, y = make_binary_data()
        model = EBMModel.create({'seed': 42, 'maxRounds': 50, 'maxInteractions': 0})
        model.fit(X, y)
        preds1 = model.predict(X).copy()
        model.fit(X, y)
        preds2 = model.predict(X)
        assert len(preds2) == len(y)


class TestParams:
    def test_get_set_params(self):
        model = EBMModel.create({'seed': 42, 'learningRate': 0.05})
        params = model.get_params()
        assert params['seed'] == 42
        assert params['learningRate'] == 0.05

        model.set_params({'maxRounds': 500})
        params = model.get_params()
        assert params['maxRounds'] == 500
        assert params['seed'] == 42

    def test_map_params(self):
        mapped = EBMModel._map_params({
            'learningRate': 0.05,
            'maxRounds': 1000,
            'maxInteractions': 5,
            'maxBins': 128,
            'seed': 123,
            'objective': 'classification',
        })
        assert mapped['learning_rate'] == 0.05
        assert mapped['max_rounds'] == 1000
        assert mapped['interactions'] == 5
        assert mapped['max_bins'] == 128
        assert mapped['max_interaction_bins'] == 128  # Matches max_bins
        assert mapped['random_state'] == 123
        assert 'objective' not in mapped
