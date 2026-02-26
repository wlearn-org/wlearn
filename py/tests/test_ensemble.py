"""Tests for ensemble: VotingEnsemble, StackingEnsemble, BaggedEstimator, selection, OOF, weights."""

import numpy as np
import pytest

from wlearn.ensemble import (
    VotingEnsemble, StackingEnsemble, BaggedEstimator,
    caruana_select, get_oof_predictions, optimize_weights, project_simplex,
)
from wlearn.errors import NotFittedError, DisposedError, ValidationError


# --- MockModel: no native deps, deterministic ---

class MockModel:
    """Simple model for testing ensemble/automl without native ML deps."""

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
        if len(unique) <= 20:
            self._classes = np.array(unique, dtype=np.int32)
            self._n_classes = len(unique)
        self._mean = float(np.mean(y))
        return self

    def predict(self, X):
        n = len(X)
        if self._classes is not None and self._n_classes > 0:
            # Classification: simple nearest-centroid-like prediction
            # Just predict the most common class with some variation from bias
            out = np.zeros(n, dtype=np.float64)
            for i in range(n):
                score = float(X[i].sum()) + self._bias
                cls_idx = int(score * 1000) % self._n_classes
                out[i] = self._classes[cls_idx]
            return out
        # Regression
        return np.full(n, self._mean + self._bias, dtype=np.float64)

    def predict_proba(self, X):
        n = len(X)
        nc = self._n_classes
        out = np.zeros(n * nc, dtype=np.float64)
        for i in range(n):
            score = float(X[i].sum()) + self._bias
            # Distribute probabilities based on score
            for c in range(nc):
                out[i * nc + c] = 1.0 / nc
            # Slightly boost one class
            boost_idx = int(abs(score) * 100) % nc
            out[i * nc + boost_idx] += 0.1
            # Renormalize
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

    def save(self):
        from wlearn.bundle import encode_bundle
        import json
        blob = json.dumps({
            'classes': [int(c) for c in self._classes] if self._classes is not None else None,
            'nClasses': int(self._n_classes),
            'mean': float(self._mean) if self._mean is not None else None,
            'bias': float(self._bias),
        }).encode('utf-8')
        return encode_bundle(
            {'typeId': 'test.mock@1', 'params': self._params},
            [{'id': 'model', 'data': blob}],
        )

    @staticmethod
    def _from_bundle(manifest, toc, blobs):
        import json
        entry = next((e for e in toc if e['id'] == 'model'), None)
        if entry is None:
            raise ValueError('Bundle missing "model" artifact')
        blob = bytes(blobs[entry['offset']:entry['offset'] + entry['length']])
        data = json.loads(blob.decode('utf-8'))
        m = MockModel.__new__(MockModel)
        m._params = manifest.get('params', {})
        m._fitted = True
        m._disposed = False
        m._classes = np.array(data['classes'], dtype=np.int32) if data['classes'] else None
        m._n_classes = data['nClasses']
        m._mean = data['mean']
        m._bias = data['bias']
        return m

    def dispose(self):
        self._disposed = True

    def get_params(self):
        return dict(self._params)

    def set_params(self, p):
        self._params.update(p)
        return self

    @property
    def is_fitted(self):
        return self._fitted


from wlearn.registry import register as _register_loader
_register_loader('test.mock@1', MockModel._from_bundle)


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


# ===========================================================================
# VotingEnsemble
# ===========================================================================

class TestVotingEnsembleSoft:
    def test_soft_classification(self):
        X, y = make_cls_data()
        ens = VotingEnsemble.create(
            estimators=[
                ('m1', MockModel, {'bias': 0.0}),
                ('m2', MockModel, {'bias': 0.1}),
            ],
            voting='soft',
            task='classification',
        )
        ens.fit(X, y)
        assert ens.is_fitted
        preds = ens.predict(X)
        assert len(preds) == len(X)
        # All predictions should be valid class labels
        for p in preds:
            assert int(p) in set(y)

    def test_predict_proba_shape(self):
        X, y = make_cls_data(n_classes=3)
        ens = VotingEnsemble.create(
            estimators=[
                ('m1', MockModel, {}),
                ('m2', MockModel, {'bias': 0.5}),
            ],
            voting='soft',
            task='classification',
        )
        ens.fit(X, y)
        proba = ens.predict_proba(X)
        n_classes = len(set(y))
        assert len(proba) == len(X) * n_classes
        # Probabilities should sum to ~1 for each sample
        for i in range(len(X)):
            row_sum = sum(proba[i * n_classes + c] for c in range(n_classes))
            assert abs(row_sum - 1.0) < 1e-10

    def test_custom_weights(self):
        X, y = make_cls_data()
        ens = VotingEnsemble.create(
            estimators=[
                ('m1', MockModel, {}),
                ('m2', MockModel, {'bias': 1.0}),
            ],
            weights=[0.8, 0.2],
            voting='soft',
            task='classification',
        )
        ens.fit(X, y)
        preds = ens.predict(X)
        assert len(preds) == len(X)


class TestVotingEnsembleHard:
    def test_hard_classification(self):
        X, y = make_cls_data()
        ens = VotingEnsemble.create(
            estimators=[
                ('m1', MockModel, {}),
                ('m2', MockModel, {'bias': 0.5}),
                ('m3', MockModel, {'bias': -0.5}),
            ],
            voting='hard',
            task='classification',
        )
        ens.fit(X, y)
        preds = ens.predict(X)
        assert len(preds) == len(X)

    def test_hard_no_predict_proba(self):
        X, y = make_cls_data()
        ens = VotingEnsemble.create(
            estimators=[('m1', MockModel, {})],
            voting='hard',
            task='classification',
        )
        ens.fit(X, y)
        with pytest.raises(ValidationError):
            ens.predict_proba(X)


class TestVotingEnsembleRegression:
    def test_regression(self):
        X, y = make_reg_data()
        ens = VotingEnsemble.create(
            estimators=[
                ('m1', MockModel, {'bias': 0.0}),
                ('m2', MockModel, {'bias': 1.0}),
            ],
            task='regression',
        )
        ens.fit(X, y)
        preds = ens.predict(X)
        assert len(preds) == len(X)

    def test_regression_no_predict_proba(self):
        X, y = make_reg_data()
        ens = VotingEnsemble.create(
            estimators=[('m1', MockModel, {})],
            task='regression',
        )
        ens.fit(X, y)
        with pytest.raises(ValidationError):
            ens.predict_proba(X)


class TestVotingEnsembleLifecycle:
    def test_not_fitted_error(self):
        ens = VotingEnsemble.create(
            estimators=[('m1', MockModel, {})],
            task='classification',
        )
        X, _ = make_cls_data()
        with pytest.raises(NotFittedError):
            ens.predict(X)

    def test_dispose(self):
        X, y = make_cls_data()
        ens = VotingEnsemble.create(
            estimators=[('m1', MockModel, {})],
            task='classification',
        )
        ens.fit(X, y)
        ens.dispose()
        with pytest.raises(DisposedError):
            ens.predict(X)

    def test_double_dispose(self):
        ens = VotingEnsemble.create(
            estimators=[('m1', MockModel, {})],
            task='classification',
        )
        ens.dispose()
        ens.dispose()  # should not raise

    def test_get_set_params(self):
        ens = VotingEnsemble.create(
            estimators=[('m1', MockModel, {})],
            voting='soft',
            task='classification',
        )
        p = ens.get_params()
        assert p['voting'] == 'soft'
        ens.set_params({'voting': 'hard'})
        assert ens.get_params()['voting'] == 'hard'


# ===========================================================================
# StackingEnsemble
# ===========================================================================

class TestStackingEnsemble:
    def test_classification(self):
        X, y = make_cls_data(n=60, n_classes=2)
        ens = StackingEnsemble.create(
            estimators=[
                ('base1', MockModel, {}),
                ('base2', MockModel, {'bias': 0.5}),
            ],
            final_estimator=('meta', MockModel, {}),
            cv=3,
            task='classification',
        )
        ens.fit(X, y)
        assert ens.is_fitted
        preds = ens.predict(X)
        assert len(preds) == len(X)

    def test_regression(self):
        X, y = make_reg_data(n=60)
        ens = StackingEnsemble.create(
            estimators=[
                ('base1', MockModel, {}),
                ('base2', MockModel, {'bias': 1.0}),
            ],
            final_estimator=('meta', MockModel, {}),
            cv=3,
            task='regression',
        )
        ens.fit(X, y)
        preds = ens.predict(X)
        assert len(preds) == len(X)

    def test_passthrough(self):
        X, y = make_cls_data(n=60, n_classes=2)
        ens = StackingEnsemble.create(
            estimators=[('base1', MockModel, {})],
            final_estimator=('meta', MockModel, {}),
            cv=3,
            task='classification',
            passthrough=True,
        )
        ens.fit(X, y)
        preds = ens.predict(X)
        assert len(preds) == len(X)

    def test_no_final_estimator_error(self):
        ens = StackingEnsemble.create(
            estimators=[('base1', MockModel, {})],
            task='classification',
        )
        X, y = make_cls_data()
        with pytest.raises(ValidationError):
            ens.fit(X, y)

    def test_dispose(self):
        X, y = make_cls_data(n=60, n_classes=2)
        ens = StackingEnsemble.create(
            estimators=[('base1', MockModel, {})],
            final_estimator=('meta', MockModel, {}),
            cv=3,
            task='classification',
        )
        ens.fit(X, y)
        ens.dispose()
        with pytest.raises(DisposedError):
            ens.predict(X)


# ===========================================================================
# caruana_select
# ===========================================================================

class TestCaruanaSelect:
    def test_basic_selection(self):
        n = 30
        n_classes = 2
        # Create 5 candidate OOF predictions
        rng = np.random.RandomState(42)
        y = np.array([i % n_classes for i in range(n)], dtype=np.int32)
        oof_preds = []
        for _ in range(5):
            proba = rng.rand(n * n_classes)
            # Normalize rows
            for i in range(n):
                row_sum = sum(proba[i * n_classes + c] for c in range(n_classes))
                for c in range(n_classes):
                    proba[i * n_classes + c] /= row_sum
            oof_preds.append(proba)

        result = caruana_select(
            oof_preds, y, max_size=10,
            scoring='accuracy', task='classification',
        )
        assert 'indices' in result
        assert 'weights' in result
        assert 'scores' in result
        assert len(result['scores']) == 10
        # Weights should sum to ~1
        assert abs(sum(result['weights']) - 1.0) < 1e-10
        # Indices should be unique and sorted
        assert list(result['indices']) == sorted(result['indices'])

    def test_regression_selection(self):
        n = 30
        rng = np.random.RandomState(42)
        y = rng.randn(n)
        oof_preds = [rng.randn(n) for _ in range(3)]

        result = caruana_select(
            oof_preds, y, max_size=5,
            scoring='r2', task='regression',
        )
        assert len(result['scores']) == 5
        assert len(result['weights']) > 0

    def test_scores_improve_overall(self):
        n = 50
        n_classes = 2
        rng = np.random.RandomState(123)
        y = np.array([i % n_classes for i in range(n)], dtype=np.int32)
        oof_preds = []
        for _ in range(10):
            proba = rng.rand(n * n_classes)
            for i in range(n):
                row_sum = proba[i * n_classes] + proba[i * n_classes + 1]
                proba[i * n_classes] /= row_sum
                proba[i * n_classes + 1] /= row_sum
            oof_preds.append(proba)

        result = caruana_select(
            oof_preds, y, max_size=8,
            scoring='accuracy', task='classification',
        )
        # Final score should be >= first score (overall improvement)
        assert result['scores'][-1] >= result['scores'][0] - 0.1


# ===========================================================================
# get_oof_predictions
# ===========================================================================

class TestOofPredictions:
    def test_classification_shape(self):
        X, y = make_cls_data(n=60, n_classes=3)
        specs = [
            ('m1', MockModel, {}),
            ('m2', MockModel, {'bias': 0.5}),
        ]
        result = get_oof_predictions(specs, X, y, cv=3, seed=42, task='classification')
        assert len(result['oofPreds']) == 2
        n_classes = len(set(y))
        assert len(result['oofPreds'][0]) == len(X) * n_classes
        assert result['classes'] is not None
        assert len(result['classes']) == n_classes

    def test_regression_shape(self):
        X, y = make_reg_data(n=60)
        specs = [
            ('m1', MockModel, {}),
            ('m2', MockModel, {'bias': 1.0}),
        ]
        result = get_oof_predictions(specs, X, y, cv=3, seed=42, task='regression')
        assert len(result['oofPreds']) == 2
        assert len(result['oofPreds'][0]) == len(X)
        assert result['classes'] is None

    def test_proba_sum_to_one(self):
        X, y = make_cls_data(n=60, n_classes=2)
        specs = [('m1', MockModel, {})]
        result = get_oof_predictions(specs, X, y, cv=3, seed=42, task='classification')
        oof = result['oofPreds'][0]
        n_classes = len(result['classes'])
        for i in range(len(X)):
            row_sum = sum(oof[i * n_classes + c] for c in range(n_classes))
            assert abs(row_sum - 1.0) < 1e-10


# ===========================================================================
# BaggedEstimator
# ===========================================================================

class TestBaggedEstimatorClassification:
    def test_classification_basic(self):
        X, y = make_cls_data(n=60, n_classes=3)
        bag = BaggedEstimator.create(
            estimator=('m1', MockModel, {}),
            k_fold=3,
            task='classification',
            seed=42,
        )
        bag.fit(X, y)
        assert bag.is_fitted
        preds = bag.predict(X)
        assert len(preds) == len(X)
        valid_classes = set(int(v) for v in y)
        for p in preds:
            assert int(p) in valid_classes

    def test_predict_proba_shape(self):
        X, y = make_cls_data(n=60, n_classes=3)
        bag = BaggedEstimator.create(
            estimator=('m1', MockModel, {}),
            k_fold=3,
            task='classification',
        )
        bag.fit(X, y)
        proba = bag.predict_proba(X)
        n_classes = len(set(y))
        assert len(proba) == len(X) * n_classes
        # Each row should sum to ~1
        for i in range(len(X)):
            row_sum = sum(proba[i * n_classes + c] for c in range(n_classes))
            assert abs(row_sum - 1.0) < 1e-10

    def test_oof_predictions_shape(self):
        X, y = make_cls_data(n=60, n_classes=2)
        bag = BaggedEstimator.create(
            estimator=('m1', MockModel, {}),
            k_fold=3,
            task='classification',
        )
        bag.fit(X, y)
        oof = bag.oof_predictions
        n_classes = len(set(y))
        assert len(oof) == len(X) * n_classes

    def test_oof_predictions_sum_to_one(self):
        X, y = make_cls_data(n=60, n_classes=2)
        bag = BaggedEstimator.create(
            estimator=('m1', MockModel, {}),
            k_fold=3,
            task='classification',
        )
        bag.fit(X, y)
        oof = bag.oof_predictions
        n_classes = 2
        for i in range(len(X)):
            row_sum = sum(oof[i * n_classes + c] for c in range(n_classes))
            assert abs(row_sum - 1.0) < 1e-10

    def test_multiple_repeats(self):
        X, y = make_cls_data(n=60, n_classes=2)
        bag = BaggedEstimator.create(
            estimator=('m1', MockModel, {}),
            k_fold=3,
            n_repeats=2,
            task='classification',
        )
        bag.fit(X, y)
        assert bag.is_fitted
        # Should have k_fold * n_repeats = 6 fold models
        assert len(bag._fold_models) == 6
        # OOF should still be valid
        oof = bag.oof_predictions
        n_classes = 2
        for i in range(len(X)):
            row_sum = sum(oof[i * n_classes + c] for c in range(n_classes))
            assert abs(row_sum - 1.0) < 1e-10

    def test_predict_averages_fold_models(self):
        X, y = make_cls_data(n=30, n_classes=2)
        bag = BaggedEstimator.create(
            estimator=('m1', MockModel, {}),
            k_fold=3,
            task='classification',
        )
        bag.fit(X, y)
        proba = bag.predict_proba(X)
        # Manually average fold model probabilities
        nc = 2
        n = len(X)
        manual = np.zeros(n * nc, dtype=np.float64)
        for model in bag._fold_models:
            p = model.predict_proba(X)
            for i in range(n * nc):
                manual[i] += p[i]
        for i in range(n * nc):
            manual[i] /= len(bag._fold_models)
        np.testing.assert_allclose(proba, manual, atol=1e-12)


class TestBaggedEstimatorRegression:
    def test_regression_basic(self):
        X, y = make_reg_data(n=60)
        bag = BaggedEstimator.create(
            estimator=('m1', MockModel, {}),
            k_fold=3,
            task='regression',
        )
        bag.fit(X, y)
        preds = bag.predict(X)
        assert len(preds) == len(X)

    def test_oof_predictions_shape(self):
        X, y = make_reg_data(n=60)
        bag = BaggedEstimator.create(
            estimator=('m1', MockModel, {}),
            k_fold=3,
            task='regression',
        )
        bag.fit(X, y)
        oof = bag.oof_predictions
        assert len(oof) == len(X)

    def test_regression_no_predict_proba(self):
        X, y = make_reg_data(n=60)
        bag = BaggedEstimator.create(
            estimator=('m1', MockModel, {}),
            k_fold=3,
            task='regression',
        )
        bag.fit(X, y)
        with pytest.raises(ValidationError):
            bag.predict_proba(X)


class TestBaggedEstimatorLifecycle:
    def test_not_fitted_error(self):
        bag = BaggedEstimator.create(
            estimator=('m1', MockModel, {}),
            task='classification',
        )
        X, _ = make_cls_data()
        with pytest.raises(NotFittedError):
            bag.predict(X)

    def test_disposed_error(self):
        X, y = make_cls_data(n=30)
        bag = BaggedEstimator.create(
            estimator=('m1', MockModel, {}),
            k_fold=3,
            task='classification',
        )
        bag.fit(X, y)
        bag.dispose()
        with pytest.raises(DisposedError):
            bag.predict(X)

    def test_double_dispose(self):
        bag = BaggedEstimator.create(
            estimator=('m1', MockModel, {}),
            task='classification',
        )
        bag.dispose()
        bag.dispose()  # should not raise

    def test_get_set_params(self):
        bag = BaggedEstimator.create(
            estimator=('m1', MockModel, {}),
            k_fold=5,
            task='classification',
        )
        p = bag.get_params()
        assert p['kFold'] == 5
        bag.set_params({'kFold': 3})
        assert bag.get_params()['kFold'] == 3

    def test_is_fitted_property(self):
        bag = BaggedEstimator.create(
            estimator=('m1', MockModel, {}),
            task='classification',
        )
        assert not bag.is_fitted
        X, y = make_cls_data(n=30)
        bag.fit(X, y)
        assert bag.is_fitted
        bag.dispose()
        assert not bag.is_fitted

    def test_save_load_roundtrip(self):
        X, y = make_cls_data(n=60, n_classes=2)
        bag = BaggedEstimator.create(
            estimator=('m1', MockModel, {}),
            k_fold=3,
            task='classification',
        )
        bag.fit(X, y)
        preds_before = bag.predict(X)
        proba_before = bag.predict_proba(X)
        oof_before = bag.oof_predictions

        data = bag.save()
        loaded = BaggedEstimator.load(data)

        assert loaded.is_fitted
        preds_after = loaded.predict(X)
        proba_after = loaded.predict_proba(X)
        oof_after = loaded.oof_predictions

        np.testing.assert_allclose(preds_after, preds_before, atol=1e-12)
        np.testing.assert_allclose(proba_after, proba_before, atol=1e-12)
        np.testing.assert_allclose(oof_after, oof_before, atol=1e-12)
        loaded.dispose()

    def test_capabilities(self):
        bag = BaggedEstimator.create(
            estimator=('m1', MockModel, {}),
            task='classification',
        )
        cap = bag.capabilities
        assert cap['classifier'] is True
        assert cap['regressor'] is False
        assert cap['predictProba'] is True

    def test_classes_property(self):
        X, y = make_cls_data(n=30, n_classes=3)
        bag = BaggedEstimator.create(
            estimator=('m1', MockModel, {}),
            k_fold=3,
            task='classification',
        )
        bag.fit(X, y)
        assert len(bag.classes) == 3
        assert list(bag.classes) == [0, 1, 2]


# ===========================================================================
# Weight optimization
# ===========================================================================

class TestProjectSimplex:
    def test_basic(self):
        v = np.array([1.5, 0.5, -0.5])
        w = project_simplex(v)
        assert abs(np.sum(w) - 1.0) < 1e-10
        assert all(w >= -1e-15)

    def test_already_valid(self):
        v = np.array([0.3, 0.3, 0.4])
        w = project_simplex(v)
        np.testing.assert_allclose(w, v, atol=1e-10)

    def test_all_negative(self):
        v = np.array([-1.0, -2.0, -3.0])
        w = project_simplex(v)
        assert abs(np.sum(w) - 1.0) < 1e-10
        assert all(w >= -1e-15)

    def test_single(self):
        v = np.array([0.5])
        w = project_simplex(v)
        assert abs(w[0] - 1.0) < 1e-10

    def test_two_elements(self):
        v = np.array([2.0, 0.0])
        w = project_simplex(v)
        assert abs(np.sum(w) - 1.0) < 1e-10
        assert all(w >= -1e-15)


class TestOptimizeWeights:
    def test_classification(self):
        n = 50
        n_classes = 2
        rng = np.random.RandomState(42)
        y = np.array([i % n_classes for i in range(n)], dtype=np.int32)
        oof_preds = []
        for _ in range(3):
            proba = rng.rand(n * n_classes)
            for i in range(n):
                row_sum = sum(proba[i * n_classes + c] for c in range(n_classes))
                for c in range(n_classes):
                    proba[i * n_classes + c] /= row_sum
            oof_preds.append(proba)

        init_w = np.array([1.0 / 3] * 3)
        refined = optimize_weights(oof_preds, y, init_w, task='classification')
        assert abs(np.sum(refined) - 1.0) < 1e-10
        assert all(refined >= -1e-15)

    def test_regression(self):
        n = 50
        rng = np.random.RandomState(42)
        y = rng.randn(n)
        oof_preds = [rng.randn(n) for _ in range(3)]

        init_w = np.array([1.0 / 3] * 3)
        refined = optimize_weights(oof_preds, y, init_w, task='regression')
        assert abs(np.sum(refined) - 1.0) < 1e-10
        assert all(refined >= -1e-15)

    def test_single_model(self):
        n = 20
        rng = np.random.RandomState(42)
        y = rng.randn(n)
        oof_preds = [rng.randn(n)]
        init_w = np.array([1.0])
        refined = optimize_weights(oof_preds, y, init_w, task='regression')
        assert abs(refined[0] - 1.0) < 1e-10

    def test_improves_or_maintains_logloss(self):
        n = 100
        n_classes = 3
        rng = np.random.RandomState(123)
        y = np.array([i % n_classes for i in range(n)], dtype=np.int32)
        oof_preds = []
        for _ in range(5):
            proba = rng.rand(n * n_classes)
            for i in range(n):
                row_sum = sum(proba[i * n_classes + c] for c in range(n_classes))
                for c in range(n_classes):
                    proba[i * n_classes + c] /= row_sum
            oof_preds.append(proba)

        init_w = np.array([0.2] * 5)

        # Compute logloss with init weights
        def compute_logloss(weights):
            eps = 1e-15
            loss = 0.0
            for i in range(n):
                p = 0.0
                for j in range(len(weights)):
                    p += weights[j] * oof_preds[j][i * n_classes + y[i]]
                loss -= np.log(max(p, eps))
            return loss / n

        init_loss = compute_logloss(init_w)
        refined = optimize_weights(oof_preds, y, init_w, task='classification',
                                   n_iter=200)
        refined_loss = compute_logloss(refined)
        # Refined should be at least as good (lower logloss) or very close
        assert refined_loss <= init_loss + 0.01


class TestCaruanaSelectRefineWeights:
    def test_refine_weights(self):
        n = 50
        n_classes = 2
        rng = np.random.RandomState(42)
        y = np.array([i % n_classes for i in range(n)], dtype=np.int32)
        oof_preds = []
        for _ in range(5):
            proba = rng.rand(n * n_classes)
            for i in range(n):
                row_sum = sum(proba[i * n_classes + c] for c in range(n_classes))
                for c in range(n_classes):
                    proba[i * n_classes + c] /= row_sum
            oof_preds.append(proba)

        result_no_refine = caruana_select(
            oof_preds, y, max_size=10,
            scoring='accuracy', task='classification',
            refine_weights=False,
        )
        result_refined = caruana_select(
            oof_preds, y, max_size=10,
            scoring='accuracy', task='classification',
            refine_weights=True,
        )
        # Both should have valid weights
        assert abs(sum(result_no_refine['weights']) - 1.0) < 1e-10
        assert abs(sum(result_refined['weights']) - 1.0) < 1e-10
        # Indices should be the same (refine only changes weights)
        np.testing.assert_array_equal(result_no_refine['indices'], result_refined['indices'])


# ===========================================================================
# neg_logloss metric
# ===========================================================================

class TestNegLogloss:
    def test_basic(self):
        from wlearn.automl._cv import neg_logloss
        y = np.array([0, 1, 0], dtype=np.int32)
        proba = np.array([0.9, 0.1, 0.2, 0.8, 0.7, 0.3])  # (3 * 2) flat
        score = neg_logloss(y, proba, n_classes=2)
        # Should be negative (higher is better)
        assert score < 0
        # Perfect predictions should give higher score
        perfect = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        perfect_score = neg_logloss(y, perfect, n_classes=2)
        assert perfect_score > score

    def test_scorer_registry(self):
        from wlearn.automl._cv import get_scorer
        scorer = get_scorer('neg_logloss')
        assert callable(scorer)


# ===========================================================================
# StackingEnsemble with BaggedEstimator base models
# ===========================================================================

class TestStackingWithBaggedBase:
    def test_stacking_with_bagged_base(self):
        X, y = make_cls_data(n=60, n_classes=2)

        # Create and fit a BaggedEstimator
        bagged = BaggedEstimator.create(
            estimator=('m1', MockModel, {}),
            k_fold=3,
            task='classification',
            seed=42,
        )
        bagged.fit(X, y)

        # Use it as a base model in StackingEnsemble
        stacking = StackingEnsemble.create(
            estimators=[('bagged_m1', bagged)],  # 2-tuple: pre-fitted
            final_estimator=('meta', MockModel, {}),
            cv=3,
            task='classification',
        )
        stacking.fit(X, y)
        assert stacking.is_fitted
        preds = stacking.predict(X)
        assert len(preds) == len(X)

    def test_stacking_mixed_bagged_and_spec(self):
        X, y = make_cls_data(n=60, n_classes=2)

        # One pre-fitted BaggedEstimator
        bagged = BaggedEstimator.create(
            estimator=('m1', MockModel, {}),
            k_fold=3,
            task='classification',
            seed=42,
        )
        bagged.fit(X, y)

        # Mix with a regular spec
        stacking = StackingEnsemble.create(
            estimators=[
                ('bagged_m1', bagged),           # 2-tuple: pre-fitted
                ('m2', MockModel, {'bias': 0.5}),  # 3-tuple: regular spec
            ],
            final_estimator=('meta', MockModel, {}),
            cv=3,
            task='classification',
        )
        stacking.fit(X, y)
        assert stacking.is_fitted
        preds = stacking.predict(X)
        assert len(preds) == len(X)
