"""Tests for stochtree Python fit() wrapping upstream stochtree package."""

import numpy as np
import pytest

from wlearn.stochtree import BARTModel
from wlearn.bundle import decode_bundle
from wlearn.errors import NotFittedError, DisposedError

# Skip all tests if stochtree is not installed
pytest.importorskip('stochtree')


def make_regression_data(seed=42, n=100, n_features=2):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    y = 2 * X[:, 0] + 3 * X[:, 1] + rng.randn(n) * 0.5
    return X, y


def make_binary_data(seed=42, n=100, n_features=3):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    return X, y


class TestRegression:
    def test_fit_predict(self):
        X, y = make_regression_data()
        model = BARTModel.create({
            'numTrees': 50, 'numGfr': 5, 'numSamples': 20,
            'seed': 42, 'objective': 'regression',
        })
        model.fit(X, y)
        assert model.is_fitted
        preds = model.predict(X)
        assert len(preds) == len(y)

    def test_score(self):
        X, y = make_regression_data()
        model = BARTModel.create({
            'numTrees': 50, 'numGfr': 5, 'numSamples': 20,
            'seed': 42, 'objective': 'regression',
        })
        model.fit(X, y)
        r2 = model.score(X, y)
        assert r2 > 0.5

    def test_predict_proba_raises(self):
        X, y = make_regression_data()
        model = BARTModel.create({
            'numTrees': 30, 'numGfr': 5, 'numSamples': 10,
            'seed': 42, 'objective': 'regression',
        })
        model.fit(X, y)
        with pytest.raises(ValueError, match='predict_proba'):
            model.predict_proba(X)


class TestClassification:
    def test_fit_predict(self):
        X, y = make_binary_data()
        model = BARTModel.create({
            'numTrees': 50, 'numGfr': 5, 'numSamples': 20,
            'seed': 42, 'objective': 'classification',
        })
        model.fit(X, y)
        assert model.is_fitted
        preds = model.predict(X)
        assert len(preds) == len(y)
        accuracy = np.mean(preds == y)
        assert accuracy > 0.7

    def test_predict_proba(self):
        X, y = make_binary_data()
        model = BARTModel.create({
            'numTrees': 50, 'numGfr': 5, 'numSamples': 20,
            'seed': 42, 'objective': 'classification',
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
        model = BARTModel.create({
            'numTrees': 50, 'numGfr': 5, 'numSamples': 20,
            'seed': 42, 'objective': 'classification',
        })
        model.fit(X, y)
        score = model.score(X, y)
        assert score > 0.7

    def test_auto_detect_task(self):
        X, y = make_binary_data()
        model = BARTModel.create({
            'numTrees': 50, 'numGfr': 5, 'numSamples': 20,
            'seed': 42,
        })
        model.fit(X, y)
        assert model._task == 1  # auto-detected classification
        preds = model.predict(X)
        accuracy = np.mean(preds == y)
        assert accuracy > 0.7


class TestSaveLoad:
    def test_round_trip_regressor(self):
        X, y = make_regression_data()
        model = BARTModel.create({
            'numTrees': 30, 'numGfr': 5, 'numSamples': 10,
            'seed': 42, 'objective': 'regression',
        })
        model.fit(X, y)
        preds_orig = model.predict(X)

        bundle = model.save()
        manifest, toc, blobs = decode_bundle(bundle)
        loaded = BARTModel._from_bundle(manifest, toc, blobs)
        preds_loaded = loaded.predict(X)

        assert np.allclose(preds_orig, preds_loaded)

    def test_round_trip_classifier(self):
        X, y = make_binary_data()
        model = BARTModel.create({
            'numTrees': 30, 'numGfr': 5, 'numSamples': 10,
            'seed': 42, 'objective': 'classification',
        })
        model.fit(X, y)
        preds_orig = model.predict(X)

        bundle = model.save()
        manifest, toc, blobs = decode_bundle(bundle)
        loaded = BARTModel._from_bundle(manifest, toc, blobs)
        preds_loaded = loaded.predict(X)

        assert np.array_equal(preds_orig, preds_loaded)

    def test_type_id_regressor(self):
        X, y = make_regression_data()
        model = BARTModel.create({
            'numTrees': 30, 'numGfr': 5, 'numSamples': 10,
            'seed': 42, 'objective': 'regression',
        })
        model.fit(X, y)
        bundle = model.save()
        manifest, _, _ = decode_bundle(bundle)
        assert manifest['typeId'] == 'wlearn.stochtree.regressor@1'

    def test_type_id_classifier(self):
        X, y = make_binary_data()
        model = BARTModel.create({
            'numTrees': 30, 'numGfr': 5, 'numSamples': 10,
            'seed': 42, 'objective': 'classification',
        })
        model.fit(X, y)
        bundle = model.save()
        manifest, _, _ = decode_bundle(bundle)
        assert manifest['typeId'] == 'wlearn.stochtree.classifier@1'

    def test_blob_identity_on_resave(self):
        X, y = make_regression_data()
        model = BARTModel.create({
            'numTrees': 30, 'numGfr': 5, 'numSamples': 10,
            'seed': 42, 'objective': 'regression',
        })
        model.fit(X, y)

        bundle1 = model.save()
        m1, t1, b1 = decode_bundle(bundle1)
        loaded = BARTModel._from_bundle(m1, t1, b1)

        bundle2 = loaded.save()
        m2, t2, b2 = decode_bundle(bundle2)

        e1 = next(e for e in t1 if e['id'] == 'model')
        e2 = next(e for e in t2 if e['id'] == 'model')
        blob1 = bytes(b1[e1['offset']:e1['offset'] + e1['length']])
        blob2 = bytes(b2[e2['offset']:e2['offset'] + e2['length']])
        assert blob1 == blob2


class TestLifecycle:
    def test_create_unfitted(self):
        model = BARTModel.create()
        assert not model.is_fitted

    def test_not_fitted_raises(self):
        model = BARTModel.create()
        with pytest.raises(NotFittedError):
            model.predict(np.zeros((1, 2)))

    def test_dispose(self):
        X, y = make_regression_data()
        model = BARTModel.create({
            'numTrees': 30, 'numGfr': 5, 'numSamples': 10,
            'seed': 42, 'objective': 'regression',
        })
        model.fit(X, y)
        model.dispose()
        assert not model.is_fitted
        with pytest.raises(DisposedError):
            model.predict(X)

    def test_dispose_idempotent(self):
        X, y = make_regression_data()
        model = BARTModel.create({
            'numTrees': 30, 'numGfr': 5, 'numSamples': 10,
            'seed': 42, 'objective': 'regression',
        })
        model.fit(X, y)
        model.dispose()
        model.dispose()  # should not raise

    def test_refit(self):
        X, y = make_regression_data()
        model = BARTModel.create({
            'numTrees': 30, 'numGfr': 5, 'numSamples': 10,
            'seed': 42, 'objective': 'regression',
        })
        model.fit(X, y)
        preds1 = model.predict(X).copy()
        model.fit(X, y)
        preds2 = model.predict(X)
        assert len(preds2) == len(y)


class TestParams:
    def test_get_set_params(self):
        model = BARTModel.create({'seed': 42, 'numTrees': 100})
        params = model.get_params()
        assert params['seed'] == 42
        assert params['numTrees'] == 100

        model.set_params({'numSamples': 50})
        params = model.get_params()
        assert params['numSamples'] == 50
        assert params['seed'] == 42

    def test_default_search_space(self):
        space = BARTModel.default_search_space()
        assert 'numTrees' in space
        assert 'alpha' in space
        assert 'minSamplesLeaf' in space
        assert space['numTrees']['type'] == 'int_uniform'
        assert space['alpha']['type'] == 'uniform'


class TestCrossLanguageParity:
    """Bundles from Python fit() should be loadable by JS and vice versa."""

    def test_python_fit_produces_valid_bundle(self):
        X, y = make_regression_data()
        model = BARTModel.create({
            'numTrees': 30, 'numGfr': 5, 'numSamples': 10,
            'seed': 42, 'objective': 'regression',
        })
        model.fit(X, y)
        bundle = model.save()

        # Verify bundle is valid
        manifest, toc, blobs = decode_bundle(bundle)
        assert manifest['typeId'] == 'wlearn.stochtree.regressor@1'
        assert len(toc) == 1
        assert toc[0]['id'] == 'model'

        # Verify blob is valid JSON with expected structure
        entry = toc[0]
        blob = bytes(blobs[entry['offset']:entry['offset'] + entry['length']])
        import json
        model_json = json.loads(blob.decode('utf-8'))
        assert 'forest_container' in model_json
        assert 'y_bar' in model_json
        assert 'y_std' in model_json
        assert model_json['task'] == 0

    def test_python_classifier_bundle_structure(self):
        X, y = make_binary_data()
        model = BARTModel.create({
            'numTrees': 30, 'numGfr': 5, 'numSamples': 10,
            'seed': 42, 'objective': 'classification',
        })
        model.fit(X, y)
        bundle = model.save()

        manifest, toc, blobs = decode_bundle(bundle)
        assert manifest['typeId'] == 'wlearn.stochtree.classifier@1'
        metadata = manifest.get('metadata', {})
        assert metadata['nrClass'] == 2
        assert len(metadata['classes']) == 2

        import json
        entry = toc[0]
        blob = bytes(blobs[entry['offset']:entry['offset'] + entry['length']])
        model_json = json.loads(blob.decode('utf-8'))
        assert model_json['task'] == 1
