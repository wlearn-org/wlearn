"""Tests for ensemble: VotingEnsemble, StackingEnsemble, BaggedEstimator, selection, OOF, weights."""

import numpy as np
import pytest

from wlearn.pipeline import Pipeline
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
        unique = list(dict.fromkeys(int(v) for v in y))
        if self._params.get('classOrder') == 'descending':
            unique.sort(reverse=True)
        elif self._params.get('classOrder') != 'firstSeen':
            unique.sort()
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

    @property
    def classes(self):
        return self._classes

    @property
    def capabilities(self):
        classifier = self._classes is not None
        return {
            'classifier': classifier,
            'regressor': not classifier,
            'predictProba': classifier,
            'decisionFunction': False,
            'sampleWeight': False,
            'csr': False,
            'earlyStopping': False,
        }


class HardOnlyMock(MockModel):
    predict_proba = None

    @property
    def classes(self):
        return None


class NoProbabilityCapabilityMock(MockModel):
    @property
    def capabilities(self):
        return {**super().capabilities, 'predictProba': False}


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
        assert preds.dtype == np.int32
        assert len(preds) == len(X)

    def test_accepts_pipeline_ending_in_probability_classifier(self):
        class PipelineFactory:
            @classmethod
            def create(cls, params=None):
                return Pipeline([
                    ('model', MockModel.create(params or {})),
                ])

        X, y = make_cls_data()
        ens = VotingEnsemble.create(
            estimators=[('pipeline', PipelineFactory, {})],
            voting='soft',
            task='classification',
        )
        ens.fit(X, y)
        assert ens.capabilities['predictProba'] is True
        assert ens.predict(X).dtype == np.int32
        ens.dispose()

    def test_aligns_child_probability_class_order(self):
        X, _ = make_cls_data(n=30, n_classes=2)
        y = np.array([2] * 20 + [1] * 10, dtype=np.int32)
        ens = VotingEnsemble.create(
            estimators=[(
                'reversed', MockModel, {'classOrder': 'descending'})],
            task='classification',
        )
        ens.fit(X, y)
        np.testing.assert_array_equal(ens.classes, [1, 2])
        proba = ens.predict_proba(X)
        direct = ens._models[0].predict_proba(X)
        np.testing.assert_allclose(proba.reshape(-1, 2),
                                   direct.reshape(-1, 2)[:, ::-1])


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
        assert preds.dtype == np.int32
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

    def test_hard_does_not_require_probability_class_metadata(self):
        X, y = make_cls_data()
        ens = VotingEnsemble.create(
            estimators=[('hard-only', HardOnlyMock, {})],
            voting='hard',
            task='classification',
        )
        ens.fit(X, y)
        assert len(ens.predict(X)) == len(X)
        with pytest.raises(ValidationError, match='predict_proba'):
            ens.set_params({'voting': 'soft'})
        assert ens.get_params()['voting'] == 'hard'
        assert ens.is_fitted
        assert len(ens.predict(X)) == len(X)

    def test_hard_predicts_once_per_child_and_validates_labels(self):
        class CountingMock(MockModel):
            def __init__(self, params=None):
                super().__init__(params)
                self.predict_calls = 0

            def predict(self, X):
                self.predict_calls += 1
                return super().predict(X)

        X, y = make_cls_data()
        ens = VotingEnsemble.create(
            estimators=[('m1', CountingMock, {}), ('m2', CountingMock, {})],
            voting='hard',
            task='classification',
        )
        ens.fit(X, y)
        ens.predict(X)
        assert [model.predict_calls for model in ens._models] == [1, 1]
        ens.dispose()

        invalid_outputs = [
            np.empty(0),
            np.full(len(X), np.nan),
            np.full(len(X), 0.5),
            np.full(len(X), 99),
        ]
        for output in invalid_outputs:
            class InvalidOutputMock(MockModel):
                def predict(self, _X):
                    return output

            invalid = VotingEnsemble.create(
                estimators=[('bad', InvalidOutputMock, {})],
                voting='hard',
                task='classification',
            )
            invalid.fit(X, y)
            with pytest.raises(ValidationError):
                invalid.predict(X)
            invalid.dispose()


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

    def test_normalizes_relative_weights(self):
        X, y = make_reg_data()
        ens = VotingEnsemble.create(
            estimators=[
                ('m1', MockModel, {'bias': 0.0}),
                ('m2', MockModel, {'bias': 2.0}),
            ],
            weights=[1, 1],
            task='regression',
        )
        ens.fit(X, y)
        assert ens.get_params()['weights'] == [0.5, 0.5]
        expected = (
            ens._models[0].predict(X) + ens._models[1].predict(X)) / 2
        np.testing.assert_allclose(ens.predict(X), expected)


class TestVotingEnsembleLifecycle:
    def test_rejects_invalid_fit_configuration_before_training(self):
        X, y = make_cls_data()
        wrong_weights = VotingEnsemble.create(
            estimators=[('m1', MockModel, {}), ('m2', MockModel, {})],
            weights=[1],
            task='classification',
        )
        with pytest.raises(ValidationError):
            wrong_weights.fit(X, y)
        wrong_weights.dispose()

        nonfinite = VotingEnsemble.create(
            estimators=[('m1', MockModel, {})],
            weights=[float('nan')],
            task='classification',
        )
        with pytest.raises(ValidationError):
            nonfinite.fit(X, y)
        nonfinite.dispose()

        nonnumeric = VotingEnsemble.create(
            estimators=[('m1', MockModel, {})],
            weights=['1'],
            task='classification',
        )
        with pytest.raises(ValidationError):
            nonnumeric.fit(X, y)
        nonnumeric.dispose()

        boolean = VotingEnsemble.create(
            estimators=[('m1', MockModel, {})],
            weights=[True],
            task='classification',
        )
        with pytest.raises(ValidationError):
            boolean.fit(X, y)
        boolean.dispose()

        for weights in ([0], [-1]):
            invalid_weights = VotingEnsemble.create(
                estimators=[('m1', MockModel, {})],
                weights=weights,
                task='classification',
            )
            with pytest.raises(ValidationError):
                invalid_weights.fit(X, y)
            invalid_weights.dispose()

        duplicate_names = VotingEnsemble.create(
            estimators=[('m1', MockModel, {}), ('m1', MockModel, {})],
            task='classification',
        )
        with pytest.raises(ValidationError):
            duplicate_names.fit(X, y)
        duplicate_names.dispose()

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
        assert not ens.is_fitted
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

    def test_invalid_set_params_preserves_fitted_inference_config(self):
        X, y = make_cls_data()
        ens = VotingEnsemble.create(
            estimators=[('m1', MockModel, {}), ('m2', MockModel, {})],
            weights=[0.5, 0.5],
            task='classification',
        )
        ens.fit(X, y)
        before = ens.get_params()
        with pytest.raises(ValidationError):
            ens.set_params({'weights': [1]})
        with pytest.raises(ValidationError):
            ens.set_params({'voting': 'invalid'})
        with pytest.raises(ValidationError, match='Unknown.*task'):
            ens.set_params({'task': 'regression'})
        assert ens.is_fitted
        assert ens.get_params() == before
        assert len(ens.predict(X)) == len(X)
        ens.dispose()

    def test_later_child_failure_releases_reverse_and_preserves_error(self):
        events = []
        live = {'count': 0}
        fit_error = RuntimeError('second fit failed')

        def model_class(label, fail=False, cleanup_throws=False):
            class Model:
                def __init__(self):
                    self.disposed = False

                @classmethod
                def create(cls, _params=None):
                    live['count'] += 1
                    events.append(f'{label}:create')
                    return cls()

                def fit(self, _X, _y):
                    events.append(f'{label}:fit')
                    if fail:
                        raise fit_error
                    self.classes = np.array(
                        sorted(set(int(value) for value in _y)),
                        dtype=np.int32)
                    return self

                def dispose(self):
                    if self.disposed:
                        return
                    self.disposed = True
                    live['count'] -= 1
                    events.append(f'{label}:dispose')
                    if cleanup_throws:
                        raise RuntimeError(f'{label} cleanup failed')

            return Model

        first = model_class('first', cleanup_throws=True)
        second = model_class('second', fail=True)
        X, y = make_cls_data(n=20)
        ensemble = VotingEnsemble.create(
            estimators=[('first', first, {}), ('second', second, {})],
            voting='hard',
            task='classification')
        with pytest.raises(RuntimeError) as exc:
            ensemble.fit(X, y)
        assert exc.value is fit_error
        assert live['count'] == 0
        assert events == [
            'first:create', 'first:fit', 'second:create', 'second:fit',
            'second:dispose', 'first:dispose',
        ]
        ensemble.dispose()
        assert live['count'] == 0


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
        assert preds.dtype == np.int32
        assert len(preds) == len(X)

    def test_derives_probability_capability_from_meta_model(self):
        X, y = make_cls_data(n=60, n_classes=2)
        ens = StackingEnsemble.create(
            estimators=[('base', MockModel, {})],
            final_estimator=('meta', NoProbabilityCapabilityMock, {}),
            cv=3,
            task='classification',
        )
        ens.fit(X, y)
        assert ens.capabilities['predictProba'] is False
        assert ens.predict(X).dtype == np.int32
        with pytest.raises(ValidationError, match='does not support'):
            ens.predict_proba(X)
        ens.dispose()

    def test_rejects_base_without_probability_capability(self):
        X, y = make_cls_data(n=30, n_classes=2)
        ens = StackingEnsemble.create(
            estimators=[('base', NoProbabilityCapabilityMock, {})],
            final_estimator=('meta', MockModel, {}),
            cv=2,
            task='classification',
        )
        with pytest.raises(ValidationError, match='capability'):
            ens.fit(X, y)
        assert not ens.is_fitted
        ens.dispose()

    def test_aligns_base_and_meta_probability_class_order(self):
        X, _ = make_cls_data(n=30, n_classes=2)
        y = np.array([2] * 20 + [1] * 10, dtype=np.int32)
        params = {'classOrder': 'descending'}
        ens = StackingEnsemble.create(
            estimators=[('base', MockModel, params)],
            final_estimator=('meta', MockModel, params),
            cv=2,
            task='classification',
        )
        ens.fit(X, y)
        np.testing.assert_array_equal(ens.classes, [1, 2])
        proba = ens.predict_proba(X)
        raw = ens._meta_model.predict_proba(ens._build_meta_features(X))
        np.testing.assert_allclose(proba.reshape(-1, 2),
                                   raw.reshape(-1, 2)[:, ::-1])

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
        assert not ens.is_fitted
        with pytest.raises(DisposedError):
            ens.predict(X)

    def test_training_param_change_invalidates_fitted_state(self):
        X, y = make_cls_data(n=60, n_classes=2)
        ens = StackingEnsemble.create(
            estimators=[('base1', MockModel, {})],
            final_estimator=('meta', MockModel, {}),
            cv=3,
            task='classification',
        )
        ens.fit(X, y)
        assert ens.is_fitted
        ens.set_params({'seed': 123})
        assert not ens.is_fitted
        with pytest.raises(NotFittedError):
            ens.save()
        ens.dispose()

    def test_rejects_invalid_training_config_transactionally(self):
        X, y = make_cls_data(n=30, n_classes=2)
        ens = StackingEnsemble.create(
            estimators=[('base', MockModel, {})],
            final_estimator=('meta', MockModel, {}),
            cv=2,
            task='classification',
        )
        ens.fit(X, y)
        with pytest.raises(ValidationError):
            ens.set_params({'cv': 1})
        with pytest.raises(ValidationError, match='Unknown.*task'):
            ens.set_params({'task': 'regression'})
        assert ens.get_params()['cv'] == 2
        assert ens.is_fitted

        invalid = StackingEnsemble.create(
            estimators=[('base', MockModel, {})],
            final_estimator=('meta', MockModel, {}),
            cv=2,
            task='unknown',
        )
        with pytest.raises(ValidationError):
            invalid.fit(X, y)
        ens.dispose()
        invalid.dispose()

    @pytest.mark.parametrize('failure', ['later-base', 'meta'])
    def test_transactional_full_data_failure_cleanup(self, failure):
        events = []
        live = {'count': 0}
        fit_error = RuntimeError(f'{failure} fit failed')

        def model_class(label, fail_full=False, cleanup_throws=False):
            class Model:
                def __init__(self):
                    self.disposed = False
                    self.full = False

                @classmethod
                def create(cls, _params=None):
                    live['count'] += 1
                    return cls()

                def fit(self, X, _y):
                    self.full = len(X) == 20
                    if self.full and fail_full:
                        raise fit_error
                    self.classes = np.array(
                        sorted(set(int(value) for value in _y)),
                        dtype=np.int32)
                    return self

                def predict(self, X):
                    return np.zeros(len(X), dtype=np.int32)

                @property
                def capabilities(self):
                    return {'predictProba': True}

                def predict_proba(self, X):
                    return np.full(len(X) * 2, 0.5, dtype=np.float64)

                def dispose(self):
                    if self.disposed:
                        return
                    self.disposed = True
                    live['count'] -= 1
                    phase = 'full' if self.full else 'oof'
                    events.append(f'{label}:dispose:{phase}')
                    if self.full and cleanup_throws:
                        raise RuntimeError(f'{label} cleanup failed')

            return Model

        base1 = model_class('base1', cleanup_throws=True)
        base2 = model_class(
            'base2', fail_full=failure == 'later-base')
        meta = model_class('meta', fail_full=failure == 'meta')
        X, y = make_cls_data(n=20, n_classes=2)
        ensemble = StackingEnsemble.create(
            estimators=[('base1', base1, {}), ('base2', base2, {})],
            final_estimator=('meta', meta, {}),
            cv=2,
            task='classification')
        with pytest.raises(RuntimeError) as exc:
            ensemble.fit(X, y)
        assert exc.value is fit_error
        assert live['count'] == 0
        full_disposals = [
            event for event in events if event.endswith(':full')]
        assert full_disposals == (
            ['base2:dispose:full', 'base1:dispose:full']
            if failure == 'later-base' else
            ['meta:dispose:full', 'base2:dispose:full',
             'base1:dispose:full'])
        ensemble.dispose()
        assert live['count'] == 0


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

    def test_noncontiguous_class_labels_follow_probability_columns(self):
        y = np.array([2, 2, 5, 5], dtype=np.int32)
        bad = np.array([
            0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9, 0.1])
        good = np.array([
            0.9, 0.1, 0.9, 0.1, 0.1, 0.9, 0.1, 0.9])
        result = caruana_select(
            [bad, good], y, max_size=1, task='classification',
            classes=[2, 5], refine_weights=False)
        np.testing.assert_array_equal(result['indices'], [1])
        np.testing.assert_array_equal(result['scores'], [1])
        reversed_result = caruana_select(
            [bad.reshape(-1, 2)[:, ::-1].reshape(-1),
             good.reshape(-1, 2)[:, ::-1].reshape(-1)],
            y, max_size=1, task='classification',
            classes=[5, 2], refine_weights=False)
        np.testing.assert_array_equal(reversed_result['indices'], [1])
        np.testing.assert_array_equal(reversed_result['scores'], [1])

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

    def test_rejects_class_metadata_inconsistent_with_probability_width(self):
        with pytest.raises(ValidationError, match='n_classes must match'):
            caruana_select(
                [np.array([0.5, 0.5, 0.5, 0.5])],
                np.array([2, 5], dtype=np.int32),
                task='classification', n_classes=3, classes=[2, 5, 9])

    @pytest.mark.parametrize(
        'classes',
        [[2.5, 5], [np.nan, 5], [2 ** 31, 5], ['2', '5'], [True, 5], [2, 2]],
    )
    def test_rejects_class_metadata_outside_int32_contract(self, classes):
        with pytest.raises(ValidationError, match='unique int32'):
            caruana_select(
                [np.array([0.8, 0.2, 0.2, 0.8])],
                np.array([2, 5], dtype=np.int32),
                max_size=1, task='classification', classes=classes,
                refine_weights=False)

    @pytest.mark.parametrize(
        'bad',
        [np.array([0.8, 0.2]),
         np.array([0.8, 0.2, np.nan, 0.8])],
    )
    def test_rejects_mismatched_or_nonfinite_candidates(self, bad):
        good = np.array([0.8, 0.2, 0.2, 0.8])
        with pytest.raises(ValidationError):
            caruana_select(
                [good, bad], np.array([2, 5], dtype=np.int32),
                max_size=1, task='classification', classes=[2, 5],
                refine_weights=False)


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

    def test_aligns_classes_and_requires_probability_capability(self):
        class ReversedProbabilityMock(MockModel):
            def predict_proba(self, X):
                output = np.empty(len(X) * 2, dtype=np.float64)
                output[0::2] = 0.9
                output[1::2] = 0.1
                return output

        X, _ = make_cls_data(n=30, n_classes=2)
        y = np.array([2] * 20 + [1] * 10, dtype=np.int32)
        result = get_oof_predictions([
            ('reversed', ReversedProbabilityMock,
             {'classOrder': 'descending'}),
        ], X, y, cv=2, seed=42, task='classification')
        np.testing.assert_array_equal(result['classes'], [1, 2])
        np.testing.assert_allclose(
            result['oofPreds'][0].reshape(-1, 2),
            np.tile([0.1, 0.9], (len(X), 1)),
        )

        with pytest.raises(ValidationError, match='capability'):
            get_oof_predictions([
                ('no-probability', NoProbabilityCapabilityMock, {}),
            ], X, y, cv=2, seed=42, task='classification')


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
        assert preds.dtype == np.int32
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

    def test_aligns_fold_model_probability_class_order(self):
        X, _ = make_cls_data(n=30, n_classes=2)
        y = np.array([2] * 20 + [1] * 10, dtype=np.int32)
        bag = BaggedEstimator.create(
            estimator=(
                'reversed', MockModel, {'classOrder': 'descending'}),
            k_fold=2,
            task='classification',
        )
        bag.fit(X, y)
        np.testing.assert_array_equal(bag.classes, [1, 2])
        proba = bag.predict_proba(X)
        manual = np.zeros_like(proba)
        for model in bag._fold_models:
            manual += model.predict_proba(X).reshape(-1, 2)[:, ::-1].reshape(-1)
        manual /= len(bag._fold_models)
        np.testing.assert_allclose(proba, manual)

    def test_rejects_child_without_probability_capability(self):
        X, y = make_cls_data(n=30, n_classes=2)
        bag = BaggedEstimator.create(
            estimator=('no-probability', NoProbabilityCapabilityMock, {}),
            k_fold=2,
            task='classification',
        )
        with pytest.raises(ValidationError, match='capability'):
            bag.fit(X, y)
        assert not bag.is_fitted
        bag.dispose()


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
        assert preds.dtype == np.float64
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

    def test_training_param_change_invalidates_fitted_state(self):
        X, y = make_cls_data(n=30, n_classes=2)
        bag = BaggedEstimator.create(
            estimator=('m1', MockModel, {}),
            k_fold=2,
            task='classification',
        )
        bag.fit(X, y)
        assert bag.is_fitted
        bag.set_params({'nRepeats': 2})
        assert not bag.is_fitted
        with pytest.raises(NotFittedError):
            bag.save()
        bag.dispose()

    def test_rejects_invalid_training_config_transactionally(self):
        X, y = make_cls_data(n=30, n_classes=2)
        bag = BaggedEstimator.create(
            estimator=('m1', MockModel, {}),
            k_fold=2,
            task='classification',
        )
        bag.fit(X, y)
        with pytest.raises(ValidationError):
            bag.set_params({'nRepeats': 0})
        with pytest.raises(ValidationError, match='Unknown.*estimator'):
            bag.set_params({'estimator': None})
        assert bag.get_params()['nRepeats'] == 1
        assert bag.is_fitted

        invalid = BaggedEstimator.create(
            estimator=('m1', MockModel, {}),
            k_fold=2,
            n_repeats=-1,
            task='classification',
        )
        with pytest.raises(ValidationError):
            invalid.fit(X, y)
        bag.dispose()
        invalid.dispose()

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

    def test_failed_refit_preserves_previous_fitted_state(self):
        class TrackingModel:
            fail = False
            live = 0

            def __init__(self):
                self.disposed = False

            @classmethod
            def create(cls, _params=None):
                cls.live += 1
                return cls()

            def fit(self, _X, _y):
                if type(self).fail:
                    raise RuntimeError('refit failed')
                self.classes = np.array(
                    sorted(set(int(value) for value in _y)),
                    dtype=np.int32)
                return self

            def predict_proba(self, X):
                result = np.empty(len(X) * 2, dtype=np.float64)
                result[0::2] = 0.75
                result[1::2] = 0.25
                return result

            @property
            def capabilities(self):
                return {'predictProba': True}

            def dispose(self):
                if self.disposed:
                    return
                self.disposed = True
                type(self).live -= 1

        X, y = make_cls_data(n=20, n_classes=2)
        bag = BaggedEstimator.create(
            estimator=('tracking', TrackingModel, {}),
            k_fold=2,
            task='classification',
        )
        bag.fit(X, y)
        assert TrackingModel.live == 2
        before = bag.predict(X).copy()
        TrackingModel.fail = True
        with pytest.raises(RuntimeError, match='refit failed'):
            bag.fit(X, y)
        assert bag.is_fitted
        np.testing.assert_array_equal(bag.predict(X), before)
        assert TrackingModel.live == 2
        bag.dispose()
        assert TrackingModel.live == 0

    def test_committed_refit_ignores_old_model_cleanup_failure(self):
        class CleanupModel:
            generation = 1
            live = 0

            def __init__(self):
                self.generation = type(self).generation
                self.disposed = False

            @classmethod
            def create(cls, _params=None):
                cls.live += 1
                return cls()

            def fit(self, _X, _y):
                self.classes = np.array(
                    sorted(set(int(value) for value in _y)),
                    dtype=np.int32)
                return self

            def predict_proba(self, X):
                return np.full(len(X) * 2, 0.5, dtype=np.float64)

            @property
            def capabilities(self):
                return {'predictProba': True}

            def dispose(self):
                if self.disposed:
                    return
                self.disposed = True
                type(self).live -= 1
                if self.generation == 1:
                    raise RuntimeError('old cleanup failed')

        X, y = make_cls_data(n=20, n_classes=2)
        bag = BaggedEstimator.create(
            estimator=('cleanup', CleanupModel, {}),
            k_fold=2,
            task='classification',
        )
        bag.fit(X, y)
        CleanupModel.generation = 2
        bag.fit(X, y)
        assert bag.is_fitted
        assert CleanupModel.live == 2
        bag.dispose()
        assert CleanupModel.live == 0

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
    def test_classification_backtracks_loss_increase(self):
        predictions = [np.array([.9, .1, .9, .1]), np.array([.1, .9, .1, .9])]
        refined = optimize_weights(predictions, np.array([0, 1]), np.array([.9, .1]), lr=100)
        np.testing.assert_allclose(refined, [.5, .5], atol=1e-6)

    @pytest.mark.parametrize('scale', [1., 100.])
    def test_regression_interior_optimum_across_target_scales(self, scale):
        predictions = [np.array([scale, 0.]), np.array([-3*scale, 0.])]
        refined = optimize_weights(predictions, np.zeros(2), np.array([.5, .5]), task='regression')
        # The optimum cancels both predictors, independently of target units.
        np.testing.assert_allclose(refined, [.75, .25], atol=1e-6)

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

    def test_noncontiguous_class_labels_follow_probability_columns(self):
        y_mapped = np.array([0, 0, 1, 1], dtype=np.int32)
        y_labels = np.array([2, 2, 5, 5], dtype=np.int32)
        oof_preds = [
            np.array([0.9, 0.1, 0.8, 0.2, 0.4, 0.6, 0.3, 0.7]),
            np.array([0.6, 0.4, 0.7, 0.3, 0.2, 0.8, 0.1, 0.9]),
        ]
        initial = np.array([0.5, 0.5])
        expected = optimize_weights(
            oof_preds, y_mapped, initial, task='classification')
        actual = optimize_weights(
            oof_preds, y_labels, initial, task='classification',
            classes=[2, 5])
        np.testing.assert_array_equal(actual, expected)
        reversed_oof = [
            values.reshape(-1, 2)[:, ::-1].reshape(-1)
            for values in oof_preds]
        reversed_actual = optimize_weights(
            reversed_oof, y_labels, initial, task='classification',
            classes=[5, 2])
        np.testing.assert_array_equal(reversed_actual, expected)

    def test_rejects_trailing_probability_values(self):
        y = np.array([2, 2, 5, 5], dtype=np.int32)
        predictions = [np.full(9, 0.5), np.full(9, 0.5)]
        with pytest.raises(ValidationError, match='divisible by n'):
            optimize_weights(
                predictions, y, np.array([0.5, 0.5]),
                task='classification', classes=[2, 5])

    @pytest.mark.parametrize(
        'classes',
        [[2.5, 5], [np.nan, 5], [2 ** 31, 5], ['2', '5'], [True, 5], [2, 2]],
    )
    def test_rejects_class_metadata_outside_int32_contract(self, classes):
        y = np.array([2, 5], dtype=np.int32)
        predictions = [
            np.array([0.8, 0.2, 0.2, 0.8]),
            np.array([0.7, 0.3, 0.3, 0.7]),
        ]
        with pytest.raises(ValidationError, match='unique int32'):
            optimize_weights(
                predictions, y, np.array([0.5, 0.5]),
                task='classification', classes=classes)

    @pytest.mark.parametrize(
        'bad',
        [np.array([0.8, 0.2]),
         np.array([0.8, 0.2, np.inf, 0.8])],
    )
    def test_rejects_mismatched_or_nonfinite_candidates(self, bad):
        good = np.array([0.8, 0.2, 0.2, 0.8])
        with pytest.raises(ValidationError):
            optimize_weights(
                [good, bad], np.array([2, 5], dtype=np.int32),
                np.array([0.5, 0.5]), task='classification',
                classes=[2, 5])

    @pytest.mark.parametrize(
        'bad',
        [np.array([0.8, 0.2]),
         np.array([0.8, 0.2, np.nan, 0.8])],
    )
    def test_validates_single_candidate_before_unit_weight(self, bad):
        with pytest.raises(ValidationError):
            optimize_weights(
                [bad], np.array([2, 5], dtype=np.int32),
                np.array([1.0]), task='classification', classes=[2, 5])

    def test_single_candidate_rejects_unknown_label(self):
        with pytest.raises(ValidationError, match='missing from classes'):
            optimize_weights(
                [np.array([0.8, 0.2, 0.2, 0.8])],
                np.array([2, 9], dtype=np.int32), np.array([1.0]),
                task='classification', classes=[2, 5])

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

    def test_nonzero_labels_follow_declared_probability_columns(self):
        from wlearn.automl._cv import neg_logloss
        y = np.array([2, 5], dtype=np.int32)
        proba = np.array([0.8, 0.2, 0.1, 0.9])
        expected = (np.log(0.8) + np.log(0.9)) / 2
        assert neg_logloss(
            y, proba, n_classes=2, classes=[2, 5]) == pytest.approx(expected)
        assert neg_logloss(y, proba, n_classes=2) == pytest.approx(expected)
        reversed_proba = proba.reshape(-1, 2)[:, ::-1].reshape(-1)
        assert neg_logloss(
            y, reversed_proba, n_classes=2,
            classes=[5, 2]) == pytest.approx(expected)

    def test_explicit_classes_cover_absent_labels(self):
        from wlearn.automl._cv import neg_logloss
        y = np.array([2, 2], dtype=np.int32)
        proba = np.array([0.8, 0.2, 0.7, 0.3])
        assert neg_logloss(
            y, proba, n_classes=2, classes=[2, 5]) == pytest.approx(
                (np.log(0.8) + np.log(0.7)) / 2)
        with pytest.raises(ValidationError, match='classes are required'):
            neg_logloss(y, proba, n_classes=2)

    def test_zero_based_subset_does_not_guess_probability_columns(self):
        from wlearn.automl._cv import neg_logloss
        y = np.array([0, 2], dtype=np.int32)
        proba = np.array([0.8, 0.1, 0.1, 0.1, 0.8, 0.1])
        with pytest.raises(ValidationError, match='classes are required'):
            neg_logloss(y, proba, n_classes=3)
        assert neg_logloss(
            y, proba, n_classes=3, classes=[0, 2, 5]) == pytest.approx(
                (np.log(0.8) + np.log(0.8)) / 2)

    @pytest.mark.parametrize('proba, match', [
        (np.array([0.8, 0.2, 0.7]), 'length'),
        (np.array([0.8, 0.2, np.nan, 0.3]), 'finite'),
        (np.array([0.8, 0.2, 1.1, -0.1]), r'\[0, 1\]'),
    ])
    def test_rejects_malformed_probabilities(self, proba, match):
        from wlearn.automl._cv import neg_logloss
        with pytest.raises(ValidationError, match=match):
            neg_logloss(
                np.array([2, 5], dtype=np.int32), proba,
                n_classes=2, classes=[2, 5])

    def test_rejects_unknown_label(self):
        from wlearn.automl._cv import neg_logloss
        with pytest.raises(ValidationError, match='missing from classes'):
            neg_logloss(
                np.array([2, 9], dtype=np.int32),
                np.array([0.8, 0.2, 0.1, 0.9]),
                n_classes=2, classes=[2, 5])

    def test_scorer_registry(self):
        from wlearn.automl._cv import get_scorer
        # CV scorers consume predict() labels. Probability-aware scoring needs
        # a separate scorer/executor contract and is not silently approximated.
        with pytest.raises(ValidationError, match='Unknown scoring'):
            get_scorer('neg_logloss')


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

    def test_rejects_incompatible_bagged_oof_rows_without_ownership(self):
        X, y = make_cls_data(n=60, n_classes=2)
        bagged = BaggedEstimator.create(
            estimator=('m1', MockModel, {}),
            k_fold=3,
            task='classification',
            seed=42,
        )
        bagged.fit(X, y)
        stacking = StackingEnsemble.create(
            estimators=[('bagged_m1', bagged)],
            final_estimator=('meta', MockModel, {}),
            cv=3,
            task='classification',
        )
        with pytest.raises(ValidationError, match='OOF shape'):
            stacking.fit(X[:50], y[:50])
        assert bagged.is_fitted
        stacking.dispose()
        bagged.dispose()

    def test_rejects_nonfinite_bagged_oof_without_ownership(self):
        X, y = make_cls_data(n=60, n_classes=2)

        class Prefitted:
            is_fitted = True
            disposed = False
            classes = np.array([0, 1], dtype=np.int32)
            oof_predictions = np.full(len(X) * 2, 0.5, dtype=np.float64)
            oof_predictions[0] = np.nan

            @staticmethod
            def get_params():
                return {'task': 'classification'}

            def dispose(self):
                self.disposed = True

        prefitted = Prefitted()
        stacking = StackingEnsemble.create(
            estimators=[('prefitted', prefitted)],
            final_estimator=('meta', MockModel, {}),
            cv=3,
            task='classification',
        )
        with pytest.raises(ValidationError, match='OOF predictions must be finite'):
            stacking.fit(X, y)
        assert not prefitted.disposed
        assert not stacking.is_fitted
        stacking.dispose()

    def test_rejects_bagged_task_and_classes_without_ownership(self):
        X, y = make_cls_data(n=60, n_classes=2)
        classification_bag = BaggedEstimator.create(
            estimator=('m1', MockModel, {}),
            k_fold=3,
            task='classification',
            seed=42,
        )
        classification_bag.fit(X, y)
        class_mismatch = StackingEnsemble.create(
            estimators=[('bagged_m1', classification_bag)],
            final_estimator=('meta', MockModel, {}),
            cv=3,
            task='classification',
        )
        with pytest.raises(ValidationError, match='classes'):
            class_mismatch.fit(X, y + 1)
        assert classification_bag.is_fitted
        class_mismatch.dispose()
        classification_bag.dispose()

        _, y_reg = make_reg_data(n=60)
        regression_bag = BaggedEstimator.create(
            estimator=('m1', MockModel, {}),
            k_fold=3,
            task='regression',
            seed=42,
        )
        regression_bag.fit(X, y_reg)
        task_mismatch = StackingEnsemble.create(
            estimators=[('bagged_m1', regression_bag)],
            final_estimator=('meta', MockModel, {}),
            cv=3,
            task='classification',
        )
        with pytest.raises(ValidationError, match='task'):
            task_mismatch.fit(X, y)
        assert regression_bag.is_fitted
        task_mismatch.dispose()
        regression_bag.dispose()


def test_voting_rejects_malformed_regression_predictions():
    X, y = make_reg_data(n=20)

    class MalformedRegression(MockModel):
        mode = 'short'

        def predict(self, values):
            if type(self).mode == 'short':
                return np.zeros(max(0, len(values) - 1), dtype=np.float64)
            return np.full(len(values), np.nan, dtype=np.float64)

    ensemble = VotingEnsemble.create(
        estimators=[('malformed', MalformedRegression, {})],
        task='regression')
    ensemble.fit(X, y)
    with pytest.raises(ValidationError, match='wrong shape'):
        ensemble.predict(X)
    MalformedRegression.mode = 'nonfinite'
    with pytest.raises(ValidationError, match='finite numbers'):
        ensemble.predict(X)
    ensemble.dispose()


def test_bagging_and_oof_reject_malformed_regression_predictions():
    X, y = make_reg_data(n=20)

    class NonfiniteRegression(MockModel):
        def predict(self, values):
            return np.full(len(values), np.inf, dtype=np.float64)

    with pytest.raises(ValidationError, match='finite numbers'):
        get_oof_predictions(
            [('nonfinite', NonfiniteRegression, {})], X, y,
            cv=2, task='regression')

    bag = BaggedEstimator.create(
        estimator=('nonfinite', NonfiniteRegression, {}),
        k_fold=2, task='regression')
    with pytest.raises(ValidationError, match='finite numbers'):
        bag.fit(X, y)
    assert not bag.is_fitted
    bag.dispose()


def test_stacking_rejects_malformed_regression_oof_predictions():
    X, y = make_reg_data(n=20)

    class ShortRegression(MockModel):
        def predict(self, values):
            return np.zeros(max(0, len(values) - 1), dtype=np.float64)

    stacking = StackingEnsemble.create(
        estimators=[('short', ShortRegression, {})],
        final_estimator=('meta', MockModel, {}),
        cv=2, task='regression')
    with pytest.raises(ValidationError, match='wrong shape'):
        stacking.fit(X, y)
    stacking.dispose()


def test_stacking_validates_regression_base_and_meta_inference():
    X, y = make_reg_data(n=20)

    class ToggleRegression(MockModel):
        instances = []

        def __init__(self, params=None):
            super().__init__(params)
            self.malformed = False
            type(self).instances.append(self)

        def predict(self, values):
            if self.malformed:
                return np.full(len(values), np.nan, dtype=np.float64)
            return super().predict(values)

    base_malformed = StackingEnsemble.create(
        estimators=[('base', ToggleRegression, {})],
        final_estimator=('meta', MockModel, {}),
        cv=2, task='regression')
    base_malformed.fit(X, y)
    for model in ToggleRegression.instances:
        model.malformed = True
    with pytest.raises(ValidationError, match='finite numbers'):
        base_malformed.predict(X)
    base_malformed.dispose()

    ToggleRegression.instances = []
    meta_malformed = StackingEnsemble.create(
        estimators=[('base', MockModel, {})],
        final_estimator=('meta', ToggleRegression, {}),
        cv=2, task='regression')
    meta_malformed.fit(X, y)
    for model in ToggleRegression.instances:
        model.malformed = True
    with pytest.raises(ValidationError, match='finite numbers'):
        meta_malformed.predict(X)
    meta_malformed.dispose()
