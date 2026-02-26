"""Python wrapper for @wlearn/xgboost bundles.

Loads WLRN bundles produced by JS @wlearn/xgboost, predicts using
native Python xgboost, and saves back to WLRN bundles that JS can load.
"""

import tempfile
import os

import numpy as np
import xgboost as xgb

from .errors import NotFittedError, DisposedError
from .bundle import encode_bundle
from .registry import register

CLASSIFIER_OBJECTIVES = frozenset([
    'binary:logistic', 'binary:logitraw', 'binary:hinge',
    'multi:softmax', 'multi:softprob',
])

PROBA_OBJECTIVES = frozenset(['binary:logistic', 'multi:softprob'])

WLEARN_PARAMS = frozenset(['numRound', 'coerce'])


class XGBModel:
    def __init__(self, booster, params, nr_class=0, classes=None):
        self._booster = booster
        self._params = dict(params)
        self._nr_class = nr_class
        self._classes = np.array(classes, dtype=np.int32) if classes else np.array([], dtype=np.int32)
        self._fitted = True
        self._disposed = False

    @classmethod
    def create(cls, params=None):
        """Create an unfitted XGBoost model."""
        obj = cls.__new__(cls)
        obj._booster = None
        obj._params = dict(params) if params else {}
        obj._nr_class = 0
        obj._classes = np.array([], dtype=np.int32)
        obj._fitted = False
        obj._disposed = False
        return obj

    def fit(self, X, y):
        """Train an XGBoost model.

        Params are passed directly to xgboost.train(). The wlearn-only
        param ``numRound`` controls the number of boosting rounds (default 100).
        """
        if self._disposed:
            raise DisposedError('XGBModel has been disposed.')

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        obj = self._params.get('objective', 'reg:squarederror')
        num_round = self._params.get('numRound', 100)

        # Build xgboost params (exclude wlearn-only params)
        xgb_params = {k: v for k, v in self._params.items()
                      if k not in WLEARN_PARAMS}
        xgb_params.setdefault('objective', obj)
        xgb_params.setdefault('verbosity', 0)

        # Detect classes for classification
        if obj in CLASSIFIER_OBJECTIVES:
            unique = np.unique(y)
            classes = np.sort(unique).astype(np.int32)
            self._classes = classes
            self._nr_class = len(classes)

            # Remap to 0-based contiguous for multi:softmax/softprob
            if obj in ('multi:softmax', 'multi:softprob'):
                xgb_params['num_class'] = len(classes)
                class_map = {int(c): i for i, c in enumerate(classes)}
                y_train = np.array([class_map[int(v)] for v in y], dtype=np.float32)
            else:
                # Binary: remap to 0/1
                class_map = {int(c): i for i, c in enumerate(classes)}
                y_train = np.array([class_map[int(v)] for v in y], dtype=np.float32)
        else:
            self._classes = np.array([], dtype=np.int32)
            self._nr_class = 0
            y_train = y.astype(np.float32)

        dtrain = xgb.DMatrix(X, label=y_train)
        self._booster = xgb.train(xgb_params, dtrain, num_boost_round=num_round)
        self._fitted = True
        return self

    @staticmethod
    def _from_bundle(manifest, toc, blobs):
        entry = next((e for e in toc if e['id'] == 'model'), None)
        if entry is None:
            raise ValueError('Bundle missing "model" artifact')

        model_bytes = bytes(blobs[entry['offset']:entry['offset'] + entry['length']])

        fd, path = tempfile.mkstemp(suffix='.ubj')
        try:
            os.write(fd, model_bytes)
            os.close(fd)
            booster = xgb.Booster()
            booster.load_model(path)
        finally:
            os.unlink(path)

        params = manifest.get('params', {})
        meta = manifest.get('metadata', {})
        return XGBModel(
            booster, params,
            nr_class=meta.get('nrClass', 0),
            classes=meta.get('classes'),
        )

    def predict(self, X):
        self._ensure_fitted()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        dm = xgb.DMatrix(X)
        raw = self._booster.predict(dm)
        obj = self._params.get('objective', 'reg:squarederror')

        if obj not in CLASSIFIER_OBJECTIVES:
            return raw.astype(np.float64)

        rows = X.shape[0]
        result = np.empty(rows, dtype=np.float64)

        if obj == 'binary:logistic':
            idx = (raw > 0.5).astype(int)
            for i in range(rows):
                result[i] = self._classes[idx[i]]
        elif obj == 'multi:softprob':
            nc = self._nr_class
            reshaped = raw.reshape(rows, nc)
            best = reshaped.argmax(axis=1)
            for i in range(rows):
                result[i] = self._classes[best[i]]
        elif obj == 'multi:softmax':
            for i in range(rows):
                idx = int(round(raw[i]))
                result[i] = self._classes[idx] if idx < len(self._classes) else raw[i]
        else:
            # binary:logitraw, binary:hinge
            idx = (raw > 0).astype(int)
            for i in range(rows):
                result[i] = self._classes[idx[i]]

        return result

    def predict_proba(self, X):
        self._ensure_fitted()
        obj = self._params.get('objective', 'reg:squarederror')
        if obj not in PROBA_OBJECTIVES:
            raise ValueError(
                f'predict_proba requires binary:logistic or multi:softprob, got "{obj}"')

        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        dm = xgb.DMatrix(X)
        raw = self._booster.predict(dm)
        rows = X.shape[0]

        if obj == 'binary:logistic':
            result = np.empty(rows * 2, dtype=np.float64)
            for i in range(rows):
                result[i * 2] = 1 - raw[i]
                result[i * 2 + 1] = raw[i]
            return result

        # multi:softprob
        return raw.astype(np.float64)

    def score(self, X, y):
        preds = self.predict(X)
        y = np.asarray(y, dtype=np.float64)
        obj = self._params.get('objective', 'reg:squarederror')

        if obj in CLASSIFIER_OBJECTIVES:
            return float(np.mean(preds == y))

        # R-squared
        y_mean = y.mean()
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        return 0.0 if ss_tot == 0 else float(1 - ss_res / ss_tot)

    def save(self):
        self._ensure_fitted()
        fd, path = tempfile.mkstemp(suffix='.ubj')
        try:
            os.close(fd)
            self._booster.save_model(path)
            with open(path, 'rb') as f:
                model_bytes = f.read()
        finally:
            os.unlink(path)

        obj = self._params.get('objective', 'reg:squarederror')
        type_id = ('wlearn.xgboost.classifier@1'
                   if obj in CLASSIFIER_OBJECTIVES
                   else 'wlearn.xgboost.regressor@1')

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
            raise DisposedError('XGBModel has been disposed.')
        if not self._fitted:
            raise NotFittedError('XGBModel is not fitted.')


    @classmethod
    def default_search_space(cls):
        return {
            'objective': {'type': 'categorical', 'values': ['binary:logistic', 'reg:squarederror']},
            'max_depth': {'type': 'int_uniform', 'low': 3, 'high': 10},
            'eta': {'type': 'log_uniform', 'low': 0.01, 'high': 0.3},
            'numRound': {'type': 'int_uniform', 'low': 50, 'high': 500},
            'subsample': {'type': 'uniform', 'low': 0.5, 'high': 1.0},
            'colsample_bytree': {'type': 'uniform', 'low': 0.5, 'high': 1.0},
            'min_child_weight': {'type': 'log_uniform', 'low': 1, 'high': 10},
            'lambda': {'type': 'log_uniform', 'low': 1e-3, 'high': 10},
            'alpha': {'type': 'log_uniform', 'low': 1e-3, 'high': 10},
        }


register('wlearn.xgboost.classifier@1', XGBModel._from_bundle)
register('wlearn.xgboost.regressor@1', XGBModel._from_bundle)
