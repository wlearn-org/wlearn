"""Numeric preprocessing transformers (v1: numpy arrays only).

StandardScaler and MinMaxScaler mirror JS @wlearn/core implementations.
Cross-language bundle round-trips are supported: JS-saved bundles can be
loaded in Python and vice versa.
"""

import json

import numpy as np

from .errors import NotFittedError, DisposedError, ValidationError
from .bundle import encode_bundle
from .registry import register

STANDARD_SCALER_TYPE_ID = 'wlearn.preprocess.standard_scaler@1'
MINMAX_SCALER_TYPE_ID = 'wlearn.preprocess.minmax_scaler@1'


class StandardScaler:
    """Standardizes features by removing the mean and scaling to unit variance.

    Uses Welford's algorithm for numerical stability (matching JS implementation).
    Stores sample std (ddof=1) to match JS behavior.
    """

    def __init__(self, params=None):
        self._params = dict(params) if params else {}
        self._means = None
        self._stds = None
        self._fitted = False
        self._disposed = False

    @classmethod
    def create(cls, params=None):
        return cls(params)

    def fit(self, X, y=None):
        self._ensure_alive()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[0] == 0:
            raise ValidationError('Cannot fit on empty data')

        self._means = X.mean(axis=0)
        # Sample std (ddof=1) matching JS Welford's (rows-1 denominator)
        self._stds = X.std(axis=0, ddof=1) if X.shape[0] > 1 else np.zeros(X.shape[1])
        self._fitted = True
        return self

    def transform(self, X):
        self._ensure_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != len(self._means):
            raise ValidationError(
                f'Expected {len(self._means)} columns, got {X.shape[1]}')
        stds = self._stds.copy()
        stds[stds == 0] = 1.0  # avoid division by zero, output 0
        result = (X - self._means) / stds
        # Where std was 0, output should be 0 (matching JS behavior)
        result[:, self._stds == 0] = 0.0
        return result

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def save(self):
        self._ensure_fitted()
        artifact = json.dumps(
            {'means': self._means.tolist(), 'stds': self._stds.tolist()},
            sort_keys=True, separators=(',', ':'),
        ).encode()
        return encode_bundle(
            {'typeId': STANDARD_SCALER_TYPE_ID, 'params': self.get_params()},
            [{'id': 'params', 'data': artifact, 'mediaType': 'application/json'}],
        )

    @staticmethod
    def _from_bundle(manifest, toc, blobs):
        entry = next((e for e in toc if e['id'] == 'params'), None)
        if entry is None:
            raise ValidationError('Bundle missing "params" artifact')
        data = json.loads(bytes(blobs[entry['offset']:entry['offset'] + entry['length']]))
        scaler = StandardScaler(manifest.get('params'))
        scaler._means = np.array(data['means'], dtype=np.float64)
        scaler._stds = np.array(data['stds'], dtype=np.float64)
        scaler._fitted = True
        return scaler

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        self._means = None
        self._stds = None
        self._fitted = False

    def get_params(self):
        return dict(self._params)

    def set_params(self, p):
        self._params.update(p)
        return self

    @property
    def is_fitted(self):
        return self._fitted and not self._disposed

    def _ensure_alive(self):
        if self._disposed:
            raise DisposedError('StandardScaler has been disposed.')

    def _ensure_fitted(self):
        self._ensure_alive()
        if not self._fitted:
            raise NotFittedError('StandardScaler is not fitted.')


class MinMaxScaler:
    """Scales features to [0, 1] range based on per-column min and max."""

    def __init__(self, params=None):
        self._params = dict(params) if params else {}
        self._mins = None
        self._maxs = None
        self._fitted = False
        self._disposed = False

    @classmethod
    def create(cls, params=None):
        return cls(params)

    def fit(self, X, y=None):
        self._ensure_alive()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[0] == 0:
            raise ValidationError('Cannot fit on empty data')

        self._mins = X.min(axis=0)
        self._maxs = X.max(axis=0)
        self._fitted = True
        return self

    def transform(self, X):
        self._ensure_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != len(self._mins):
            raise ValidationError(
                f'Expected {len(self._mins)} columns, got {X.shape[1]}')
        ranges = self._maxs - self._mins
        result = np.zeros_like(X)
        mask = ranges > 0
        result[:, mask] = (X[:, mask] - self._mins[mask]) / ranges[mask]
        return result

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def save(self):
        self._ensure_fitted()
        artifact = json.dumps(
            {'mins': self._mins.tolist(), 'maxs': self._maxs.tolist()},
            sort_keys=True, separators=(',', ':'),
        ).encode()
        return encode_bundle(
            {'typeId': MINMAX_SCALER_TYPE_ID, 'params': self.get_params()},
            [{'id': 'params', 'data': artifact, 'mediaType': 'application/json'}],
        )

    @staticmethod
    def _from_bundle(manifest, toc, blobs):
        entry = next((e for e in toc if e['id'] == 'params'), None)
        if entry is None:
            raise ValidationError('Bundle missing "params" artifact')
        data = json.loads(bytes(blobs[entry['offset']:entry['offset'] + entry['length']]))
        scaler = MinMaxScaler(manifest.get('params'))
        scaler._mins = np.array(data['mins'], dtype=np.float64)
        scaler._maxs = np.array(data['maxs'], dtype=np.float64)
        scaler._fitted = True
        return scaler

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        self._mins = None
        self._maxs = None
        self._fitted = False

    def get_params(self):
        return dict(self._params)

    def set_params(self, p):
        self._params.update(p)
        return self

    @property
    def is_fitted(self):
        return self._fitted and not self._disposed

    def _ensure_alive(self):
        if self._disposed:
            raise DisposedError('MinMaxScaler has been disposed.')

    def _ensure_fitted(self):
        self._ensure_alive()
        if not self._fitted:
            raise NotFittedError('MinMaxScaler is not fitted.')


register(STANDARD_SCALER_TYPE_ID, StandardScaler._from_bundle)
register(MINMAX_SCALER_TYPE_ID, MinMaxScaler._from_bundle)
