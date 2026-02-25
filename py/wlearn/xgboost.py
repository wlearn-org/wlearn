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


register('wlearn.xgboost.classifier@1', XGBModel._from_bundle)
register('wlearn.xgboost.regressor@1', XGBModel._from_bundle)
