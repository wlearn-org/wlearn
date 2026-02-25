"""Python wrapper for @wlearn/nanoflann bundles.

Loads WLRN bundles produced by JS @wlearn/nanoflann, predicts using
native pynanoflann, and saves back to WLRN bundles that JS can load.

Blob format (NF01):
  [0..3]   magic "NF01" (4 bytes ASCII)
  [4..7]   nSamples  (uint32 LE)
  [8..11]  nFeatures (uint32 LE)
  [12]     task: 0 = classification, 1 = regression
  [13..15] reserved (zeros)
  [16..)   X data (float64 LE, row-major)
  [..)     y data: int32 LE (classification) or float64 LE (regression)
"""

import struct

import numpy as np
import pynanoflann

from .errors import NotFittedError, DisposedError
from .bundle import encode_bundle
from .registry import register

NF01_MAGIC = b'NF01'
NF01_HEADER = 16
# struct: 4s = magic, I = uint32 nSamples, I = uint32 nFeatures, B = task, 3x = padding
NF01_FMT = '<4sIIB3x'


class KNNModel:
    def __init__(self, X, y, params, tree=None):
        self._X = np.ascontiguousarray(X, dtype=np.float64)
        self._params = dict(params)
        self._disposed = False
        self._fitted = True

        task = self._params.get('task', 'classification')
        if task == 'classification':
            self._y = np.asarray(y, dtype=np.int32)
            self._classes = np.unique(self._y)
            self._n_classes = len(self._classes)
        else:
            self._y = np.asarray(y, dtype=np.float64)
            self._classes = None
            self._n_classes = 0

        if tree is not None:
            self._tree = tree
        else:
            self._build_tree()

    @classmethod
    def create(cls, params=None):
        """Create an unfitted KNNModel."""
        obj = cls.__new__(cls)
        obj._X = None
        obj._y = None
        obj._params = dict(params) if params else {}
        obj._tree = None
        obj._classes = None
        obj._n_classes = 0
        obj._disposed = False
        obj._fitted = False
        return obj

    def fit(self, X, y):
        """Train a KNN model (stores data and builds KD-tree).

        Params: ``k`` (int, default 5), ``metric`` (str, default 'l2'),
        ``leafMaxSize`` (int, default 10), ``task`` ('classification' or 'regression').
        """
        if self._disposed:
            raise DisposedError('KNNModel has been disposed.')

        self._X = np.ascontiguousarray(X, dtype=np.float64)
        if self._X.ndim == 1:
            self._X = self._X.reshape(1, -1)

        task = self._params.get('task', 'classification')
        if task == 'classification':
            self._y = np.asarray(y, dtype=np.int32)
            self._classes = np.unique(self._y)
            self._n_classes = len(self._classes)
        else:
            self._y = np.asarray(y, dtype=np.float64)
            self._classes = None
            self._n_classes = 0

        self._build_tree()
        self._fitted = True
        return self

    def _build_tree(self):
        k = self._params.get('k', 5)
        metric = self._params.get('metric', 'l2').upper()
        leaf_max_size = self._params.get('leafMaxSize', 10)
        self._tree = pynanoflann.KDTree(
            n_neighbors=k, metric=metric, leaf_size=leaf_max_size)
        self._tree.fit(self._X)

    @staticmethod
    def _from_bundle(manifest, toc, blobs):
        entry = next((e for e in toc if e['id'] == 'model'), None)
        if entry is None:
            raise ValueError('Bundle missing "model" artifact')

        blob = bytes(blobs[entry['offset']:entry['offset'] + entry['length']])

        if len(blob) < NF01_HEADER:
            raise ValueError('Blob too short for NF01 header')

        magic, ns, nf, task = struct.unpack_from(NF01_FMT, blob, 0)
        if magic != NF01_MAGIC:
            raise ValueError(f'Invalid blob magic: {magic!r} (expected NF01)')

        x_bytes = ns * nf * 8
        x_end = NF01_HEADER + x_bytes

        X = np.frombuffer(blob, dtype='<f8', count=ns * nf,
                          offset=NF01_HEADER).reshape(ns, nf).copy()

        if task == 0:
            y = np.frombuffer(blob, dtype='<i4', count=ns,
                              offset=x_end).copy()
        else:
            y = np.frombuffer(blob, dtype='<f8', count=ns,
                              offset=x_end).copy()

        params = manifest.get('params', {})
        return KNNModel(X, y, params)

    def predict(self, X):
        self._ensure_fitted()
        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        k = min(self._params.get('k', 5), len(self._X))
        self._tree.n_neighbors = k
        distances, indices = self._tree.kneighbors(X)
        neighbor_labels = self._y[indices]

        task = self._params.get('task', 'classification')
        if task == 'classification':
            preds = np.empty(len(X), dtype=np.float64)
            for i, row in enumerate(neighbor_labels):
                counts = np.bincount(row.astype(np.intp),
                                     minlength=self._n_classes)
                # Tie-break: smallest class label wins (argmax picks first max)
                preds[i] = float(self._classes[counts.argmax()])
            return preds
        else:
            return neighbor_labels.mean(axis=1)

    def predict_proba(self, X):
        self._ensure_fitted()
        task = self._params.get('task', 'classification')
        if task != 'classification':
            raise ValueError('predict_proba only for classification')

        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        k = min(self._params.get('k', 5), len(self._X))
        self._tree.n_neighbors = k
        distances, indices = self._tree.kneighbors(X)
        neighbor_labels = self._y[indices]

        n_queries = len(X)
        proba = np.zeros((n_queries, self._n_classes), dtype=np.float64)
        for i, row in enumerate(neighbor_labels):
            counts = np.bincount(row.astype(np.intp),
                                 minlength=self._n_classes)
            proba[i] = counts / counts.sum()
        return proba.ravel()

    def score(self, X, y):
        preds = self.predict(X)
        y = np.asarray(y, dtype=np.float64)
        task = self._params.get('task', 'classification')
        if task != 'classification':
            y_mean = y.mean()
            ss_res = np.sum((y - preds) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)
            return 0.0 if ss_tot == 0 else float(1 - ss_res / ss_tot)
        return float(np.mean(preds == y))

    def save(self):
        self._ensure_fitted()

        task = self._params.get('task', 'classification')
        task_byte = 0 if task != 'regression' else 1
        ns, nf = self._X.shape

        # NF01 header
        header = struct.pack(NF01_FMT, NF01_MAGIC, ns, nf, task_byte)

        # X as float64 LE
        x_bytes = np.ascontiguousarray(self._X, dtype='<f8').tobytes()

        # y as int32 LE (classification) or float64 LE (regression)
        if task_byte == 0:
            y_bytes = np.ascontiguousarray(self._y, dtype='<i4').tobytes()
        else:
            y_bytes = np.ascontiguousarray(self._y, dtype='<f8').tobytes()

        model_blob = header + x_bytes + y_bytes

        type_id = ('wlearn.nanoflann.regressor@1'
                   if task == 'regression'
                   else 'wlearn.nanoflann.classifier@1')

        metadata = {
            'nSamples': int(ns),
            'nFeatures': int(nf),
        }
        if self._classes is not None:
            metadata['nClasses'] = self._n_classes
            metadata['classes'] = self._classes.tolist()

        return encode_bundle(
            {'typeId': type_id, 'params': self.get_params(),
             'metadata': metadata},
            [{'id': 'model', 'data': model_blob}],
        )

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        self._tree = None
        self._X = None
        self._y = None
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
            raise DisposedError('KNNModel has been disposed.')
        if not self._fitted:
            raise NotFittedError('KNNModel is not fitted.')


register('wlearn.nanoflann.classifier@1', KNNModel._from_bundle)
register('wlearn.nanoflann.regressor@1', KNNModel._from_bundle)
