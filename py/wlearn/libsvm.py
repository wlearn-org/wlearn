"""Python wrapper for @wlearn/libsvm bundles.

Loads WLRN bundles produced by JS @wlearn/libsvm, predicts using
native Python libsvm, and saves back to WLRN bundles that JS can load.
"""

import tempfile
import os

import numpy as np
from libsvm.svmutil import svm_load_model, svm_save_model, svm_predict

from .errors import NotFittedError, DisposedError
from .bundle import encode_bundle
from .registry import register

SVR_TYPES = frozenset([3, 4])  # EPSILON_SVR, NU_SVR


class SVMModel:
    def __init__(self, model, params, raw_bytes=None):
        self._model = model
        self._params = dict(params)
        self._raw_bytes = raw_bytes
        self._fitted = True
        self._disposed = False

    @staticmethod
    def _from_bundle(manifest, toc, blobs):
        entry = next((e for e in toc if e['id'] == 'model'), None)
        if entry is None:
            raise ValueError('Bundle missing "model" artifact')

        model_bytes = bytes(blobs[entry['offset']:entry['offset'] + entry['length']])

        fd, path = tempfile.mkstemp(suffix='.model')
        try:
            os.write(fd, model_bytes)
            os.close(fd)
            model = svm_load_model(path)
        finally:
            os.unlink(path)

        params = manifest.get('params', {})
        return SVMModel(model, params, raw_bytes=model_bytes)

    def predict(self, X):
        self._ensure_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_list = X.tolist()
        y_dummy = [0] * len(X_list)
        p_labels, _, _ = svm_predict(y_dummy, X_list, self._model, '-q')

        svm_type = self._params.get('svmType', 0)
        if svm_type in SVR_TYPES:
            return np.array(p_labels, dtype=np.float64)
        return np.array(p_labels, dtype=np.float64)

    def predict_proba(self, X):
        self._ensure_fitted()
        probability = self._params.get('probability', 0)
        if not probability:
            raise ValueError(
                'predict_proba requires probability=1 in model params')

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_list = X.tolist()
        y_dummy = [0] * len(X_list)
        _, _, p_vals = svm_predict(y_dummy, X_list, self._model, '-b 1 -q')
        return np.array(p_vals, dtype=np.float64).ravel()

    def score(self, X, y):
        preds = self.predict(X)
        y = np.asarray(y, dtype=np.float64)
        svm_type = self._params.get('svmType', 0)

        if svm_type in SVR_TYPES:
            y_mean = y.mean()
            ss_res = np.sum((y - preds) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)
            return 0.0 if ss_tot == 0 else float(1 - ss_res / ss_tot)

        return float(np.mean(preds == y))

    def save(self):
        self._ensure_fitted()

        fd, path = tempfile.mkstemp(suffix='.model')
        try:
            os.close(fd)
            svm_save_model(path, self._model)
            with open(path, 'rb') as f:
                model_bytes = f.read()
        finally:
            os.unlink(path)

        svm_type = self._params.get('svmType', 0)
        type_id = ('wlearn.libsvm.regressor@1'
                   if svm_type in SVR_TYPES
                   else 'wlearn.libsvm.classifier@1')

        return encode_bundle(
            {'typeId': type_id, 'params': self.get_params()},
            [{'id': 'model', 'data': model_bytes}],
        )

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        self._model = None
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
            raise DisposedError('SVMModel has been disposed.')
        if not self._fitted:
            raise NotFittedError('SVMModel is not fitted.')


register('wlearn.libsvm.classifier@1', SVMModel._from_bundle)
register('wlearn.libsvm.regressor@1', SVMModel._from_bundle)
