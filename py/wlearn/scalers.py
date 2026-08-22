"""Numeric preprocessing transformers (v1: numpy arrays only).

StandardScaler and MinMaxScaler mirror JS @wlearn/core implementations.
Cross-language bundle round-trips are supported: JS-saved bundles can be
loaded in Python and vice versa.
"""

import json

import numpy as np

from .errors import NotFittedError, DisposedError, ValidationError
from .bundle import encode_bundle, write_bundle_output
from .registry import register

STANDARD_SCALER_TYPE_ID_V1 = 'wlearn.preprocess.standard_scaler@1'
STANDARD_SCALER_TYPE_ID = 'wlearn.preprocess.standard_scaler@2'
MINMAX_SCALER_TYPE_ID_V1 = 'wlearn.preprocess.minmax_scaler@1'
MINMAX_SCALER_TYPE_ID = 'wlearn.preprocess.minmax_scaler@2'


def _validated_artifact_vectors(data, first_key, second_key, label):
    if not isinstance(data, dict):
        raise ValidationError(f'{label} artifact must be an object')
    first = data.get(first_key)
    second = data.get(second_key)
    if (not isinstance(first, list) or not isinstance(second, list) or
            not first or len(first) != len(second)):
        raise ValidationError(
            f'{label} artifact must contain non-empty, equal-length '
            f'{first_key} and {second_key}')
    for value in first + second:
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, float)) or
                not np.isfinite(value)):
            raise ValidationError(
                f'{label} artifact statistics must be finite numbers')
    return (np.asarray(first, dtype=np.float64),
            np.asarray(second, dtype=np.float64))


class StandardScaler:
    """Standardizes features by removing the mean and scaling to unit variance.

    Uses Welford's algorithm for numerical stability (matching JS implementation).
    Stores population std (ddof=0), matching sklearn and Tranfi preprocessing.
    """

    def __init__(self, params=None):
        self._params = dict(params) if params else {}
        self._means = None
        self._stds = None
        self._fitted = False
        self._disposed = False
        self._legacy_constant_scale = False

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
        if X.shape[1] == 0:
            raise ValidationError('Cannot fit data with zero columns')
        if not np.all(np.isfinite(X)):
            raise ValidationError(
                'StandardScaler fit data must contain only finite numbers')

        self._means = X.mean(axis=0)
        self._stds = X.std(axis=0, ddof=0)
        self._legacy_constant_scale = False
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
        if self._legacy_constant_scale:
            result = np.zeros_like(X)
            mask = self._stds > 0
            result[:, mask] = (
                X[:, mask] - self._means[mask]) / self._stds[mask]
            return result
        stds = np.where(self._stds == 0, 1.0, self._stds)
        return (X - self._means) / stds

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def save(self, path=None):
        self._ensure_fitted()
        artifact = json.dumps(
            {'means': self._means.tolist(), 'stds': self._stds.tolist()},
            sort_keys=True, separators=(',', ':'),
        ).encode()
        bundle = encode_bundle(
            {
                'typeId': (STANDARD_SCALER_TYPE_ID_V1
                           if self._legacy_constant_scale
                           else STANDARD_SCALER_TYPE_ID),
                'params': self.get_params(),
            },
            [{'id': 'params', 'data': artifact, 'mediaType': 'application/json'}],
        )
        return write_bundle_output(bundle, path)

    @staticmethod
    def _from_bundle(manifest, toc, blobs):
        entry = next((e for e in toc if e['id'] == 'params'), None)
        if entry is None:
            raise ValidationError('Bundle missing "params" artifact')
        data = json.loads(bytes(blobs[entry['offset']:entry['offset'] + entry['length']]))
        means, stds = _validated_artifact_vectors(
            data, 'means', 'stds', 'StandardScaler')
        if np.any(stds < 0):
            raise ValidationError(
                'StandardScaler artifact standard deviations must be '
                'non-negative')
        scaler = StandardScaler(manifest.get('params'))
        scaler._means = means
        scaler._stds = stds
        scaler._legacy_constant_scale = (
            manifest.get('typeId') == STANDARD_SCALER_TYPE_ID_V1)
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
        self._legacy_constant_scale = False

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
        if X.shape[1] == 0:
            raise ValidationError('Cannot fit data with zero columns')
        if not np.all(np.isfinite(X)):
            raise ValidationError(
                'MinMaxScaler fit data must contain only finite numbers')

        self._mins = X.min(axis=0)
        self._maxs = X.max(axis=0)
        self._legacy_constant_scale = False
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
        if self._legacy_constant_scale:
            result = np.zeros_like(X)
            mask = ranges > 0
            result[:, mask] = (
                X[:, mask] - self._mins[mask]) / ranges[mask]
            return result
        scales = np.where(ranges == 0, 1.0, ranges)
        return (X - self._mins) / scales

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def save(self, path=None):
        self._ensure_fitted()
        artifact = json.dumps(
            {'mins': self._mins.tolist(), 'maxs': self._maxs.tolist()},
            sort_keys=True, separators=(',', ':'),
        ).encode()
        bundle = encode_bundle(
            {
                'typeId': (MINMAX_SCALER_TYPE_ID_V1
                           if self._legacy_constant_scale
                           else MINMAX_SCALER_TYPE_ID),
                'params': self.get_params(),
            },
            [{'id': 'params', 'data': artifact, 'mediaType': 'application/json'}],
        )
        return write_bundle_output(bundle, path)

    @staticmethod
    def _from_bundle(manifest, toc, blobs):
        entry = next((e for e in toc if e['id'] == 'params'), None)
        if entry is None:
            raise ValidationError('Bundle missing "params" artifact')
        data = json.loads(bytes(blobs[entry['offset']:entry['offset'] + entry['length']]))
        mins, maxs = _validated_artifact_vectors(
            data, 'mins', 'maxs', 'MinMaxScaler')
        if np.any(maxs < mins):
            raise ValidationError(
                'MinMaxScaler artifact maxima must not be below minima')
        scaler = MinMaxScaler(manifest.get('params'))
        scaler._mins = mins
        scaler._maxs = maxs
        scaler._legacy_constant_scale = (
            manifest.get('typeId') == MINMAX_SCALER_TYPE_ID_V1)
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


register(STANDARD_SCALER_TYPE_ID_V1, StandardScaler._from_bundle)
register(STANDARD_SCALER_TYPE_ID, StandardScaler._from_bundle)
register(MINMAX_SCALER_TYPE_ID_V1, MinMaxScaler._from_bundle)
register(MINMAX_SCALER_TYPE_ID, MinMaxScaler._from_bundle)
