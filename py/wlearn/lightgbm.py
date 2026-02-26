"""Python wrapper for LightGBM in wlearn.

Wraps native Python lightgbm package with the wlearn estimator interface.
Supports classification and regression with save/load via WLRN bundles.
"""

import tempfile
import os

import numpy as np

from .errors import NotFittedError, DisposedError
from .bundle import encode_bundle
from .registry import register

try:
    import lightgbm as lgb
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False

CLASSIFIER_OBJECTIVES = frozenset([
    'binary', 'multiclass', 'multiclassova', 'cross_entropy',
])

PROBA_OBJECTIVES = frozenset(['binary', 'multiclass', 'multiclassova'])

WLEARN_PARAMS = frozenset(['numRound', 'coerce'])


def _check_lightgbm():
    if not _HAS_LIGHTGBM:
        raise ImportError(
            'lightgbm is required for LGBModel. '
            'Install it with: pip install lightgbm')


class LGBModel:
    def __init__(self, booster, params, nr_class=0, classes=None):
        _check_lightgbm()
        self._booster = booster
        self._params = dict(params)
        self._nr_class = nr_class
        self._classes = (np.array(classes, dtype=np.int32)
                         if classes is not None and len(classes) > 0
                         else np.array([], dtype=np.int32))
        self._fitted = True
        self._disposed = False

    @classmethod
    def create(cls, params=None):
        """Create an unfitted LightGBM model."""
        _check_lightgbm()
        obj = cls.__new__(cls)
        obj._booster = None
        obj._params = dict(params) if params else {}
        obj._nr_class = 0
        obj._classes = np.array([], dtype=np.int32)
        obj._fitted = False
        obj._disposed = False
        return obj

    def fit(self, X, y):
        """Train a LightGBM model."""
        if self._disposed:
            raise DisposedError('LGBModel has been disposed.')
        _check_lightgbm()

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        obj = self._params.get('objective', 'regression')
        num_round = self._params.get('numRound', 100)

        # Build lgb params (exclude wlearn-only params)
        lgb_params = {k: v for k, v in self._params.items()
                      if k not in WLEARN_PARAMS}
        lgb_params.setdefault('objective', obj)
        lgb_params.setdefault('verbosity', -1)

        # Detect classes for classification
        if obj in CLASSIFIER_OBJECTIVES:
            unique = np.unique(y)
            classes = np.sort(unique).astype(np.int32)
            self._classes = classes
            self._nr_class = len(classes)

            if obj in ('multiclass', 'multiclassova'):
                lgb_params['num_class'] = len(classes)

            # Remap to 0-based contiguous
            class_map = {int(c): i for i, c in enumerate(classes)}
            y_train = np.array([class_map[int(v)] for v in y],
                               dtype=np.float32)
        else:
            self._classes = np.array([], dtype=np.int32)
            self._nr_class = 0
            y_train = y.astype(np.float64)

        dtrain = lgb.Dataset(X, label=y_train, free_raw_data=False)
        self._booster = lgb.train(lgb_params, dtrain,
                                  num_boost_round=num_round)
        self._fitted = True
        return self

    @staticmethod
    def _from_bundle(manifest, toc, blobs):
        _check_lightgbm()
        entry = next((e for e in toc if e['id'] == 'model'), None)
        if entry is None:
            raise ValueError('Bundle missing "model" artifact')

        model_bytes = bytes(
            blobs[entry['offset']:entry['offset'] + entry['length']])

        fd, path = tempfile.mkstemp(suffix='.lgb')
        try:
            os.write(fd, model_bytes)
            os.close(fd)
            booster = lgb.Booster(model_file=path)
        finally:
            os.unlink(path)

        params = manifest.get('params', {})
        meta = manifest.get('metadata', {})
        return LGBModel(
            booster, params,
            nr_class=meta.get('nrClass', 0),
            classes=meta.get('classes'),
        )

    def predict(self, X):
        self._ensure_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        raw = self._booster.predict(X)
        obj = self._params.get('objective', 'regression')

        if obj not in CLASSIFIER_OBJECTIVES:
            return np.asarray(raw, dtype=np.float64)

        rows = X.shape[0]
        result = np.empty(rows, dtype=np.float64)

        if obj == 'binary':
            idx = (np.asarray(raw) > 0.5).astype(int)
            for i in range(rows):
                result[i] = self._classes[idx[i]]
        elif obj in ('multiclass', 'multiclassova'):
            proba = np.asarray(raw)
            if proba.ndim == 1:
                proba = proba.reshape(rows, -1)
            best = proba.argmax(axis=1)
            for i in range(rows):
                result[i] = self._classes[best[i]]
        else:
            idx = (np.asarray(raw) > 0.5).astype(int)
            for i in range(rows):
                result[i] = self._classes[idx[i]]

        return result

    def predict_proba(self, X):
        self._ensure_fitted()
        obj = self._params.get('objective', 'regression')
        if obj not in PROBA_OBJECTIVES:
            raise ValueError(
                f'predict_proba requires classification objective, got "{obj}"')

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        raw = self._booster.predict(X)
        rows = X.shape[0]

        if obj == 'binary':
            proba = np.asarray(raw, dtype=np.float64)
            result = np.empty(rows * 2, dtype=np.float64)
            for i in range(rows):
                result[i * 2] = 1.0 - proba[i]
                result[i * 2 + 1] = proba[i]
            return result

        # multiclass / multiclassova
        proba = np.asarray(raw, dtype=np.float64)
        if proba.ndim == 2:
            return proba.ravel()
        return proba

    def score(self, X, y):
        preds = self.predict(X)
        y = np.asarray(y, dtype=np.float64)
        obj = self._params.get('objective', 'regression')

        if obj in CLASSIFIER_OBJECTIVES:
            return float(np.mean(preds == y))

        y_mean = y.mean()
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        return 0.0 if ss_tot == 0 else float(1 - ss_res / ss_tot)

    def save(self):
        self._ensure_fitted()
        fd, path = tempfile.mkstemp(suffix='.lgb')
        try:
            os.close(fd)
            self._booster.save_model(path)
            with open(path, 'rb') as f:
                model_bytes = f.read()
        finally:
            os.unlink(path)

        obj = self._params.get('objective', 'regression')
        type_id = ('wlearn.lightgbm.classifier@1'
                   if obj in CLASSIFIER_OBJECTIVES
                   else 'wlearn.lightgbm.regressor@1')

        return encode_bundle(
            {
                'typeId': type_id,
                'params': self.get_params(),
                'metadata': {
                    'nrClass': self._nr_class,
                    'classes': self._classes.tolist(),
                    'objective': obj,
                },
            },
            [{'id': 'model', 'data': model_bytes}],
        )

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        self._booster = None
        self._fitted = False

    def get_params(self):
        return dict(self._params)

    def set_params(self, p):
        self._params.update(p)
        return self

    @property
    def is_fitted(self):
        return self._fitted and not self._disposed

    def _ensure_fitted(self):
        if self._disposed:
            raise DisposedError('LGBModel has been disposed.')
        if not self._fitted:
            raise NotFittedError('LGBModel is not fitted.')

    @classmethod
    def default_search_space(cls):
        return {
            'objective': {
                'type': 'categorical',
                'values': ['binary', 'regression'],
            },
            'max_depth': {'type': 'int_uniform', 'low': 3, 'high': 12},
            'learning_rate': {'type': 'log_uniform', 'low': 0.01, 'high': 0.3},
            'numRound': {'type': 'int_uniform', 'low': 50, 'high': 500},
            'subsample': {'type': 'uniform', 'low': 0.5, 'high': 1.0},
            'colsample_bytree': {'type': 'uniform', 'low': 0.5, 'high': 1.0},
            'min_child_weight': {'type': 'log_uniform', 'low': 1, 'high': 10},
            'reg_lambda': {'type': 'log_uniform', 'low': 1e-3, 'high': 10},
            'reg_alpha': {'type': 'log_uniform', 'low': 1e-3, 'high': 10},
            'num_leaves': {'type': 'int_uniform', 'low': 15, 'high': 127},
        }


if _HAS_LIGHTGBM:
    register('wlearn.lightgbm.classifier@1', LGBModel._from_bundle)
    register('wlearn.lightgbm.regressor@1', LGBModel._from_bundle)
