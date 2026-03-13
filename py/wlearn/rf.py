"""Python wrapper for @wlearn/rf bundles.

Loads WLRN bundles produced by JS @wlearn/rf, predicts using
native librf.so (C11 FFI via ctypes), and saves back to WLRN bundles
that JS can load.

Provides RFClassifier and RFRegressor following the wlearn estimator contract.
Requires RF_LIB_PATH to point to the compiled librf.so.
"""

import ctypes
import ctypes.util
import os
import platform

import numpy as np

from .errors import NotFittedError, DisposedError
from .bundle import encode_bundle
from .registry import register

# ---------------------------------------------------------------------------
# FFI loading (shared across all RF models in this module)
# ---------------------------------------------------------------------------

_lib = None


def _find_lib():
    env_path = os.environ.get('RF_LIB_PATH')
    if env_path and os.path.isfile(env_path):
        return env_path

    system = platform.system()
    lib_name = {'Darwin': 'librf.dylib', 'Windows': 'rf.dll'}.get(
        system, 'librf.so')

    this_dir = os.path.dirname(os.path.abspath(__file__))
    for rel in [lib_name,
                os.path.join('..', '..', '..', '..', 'rf', 'build', lib_name)]:
        path = os.path.abspath(os.path.join(this_dir, rel))
        if os.path.isfile(path):
            return path

    found = ctypes.util.find_library('rf')
    return found


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib

    lib_path = _find_lib()
    if not lib_path:
        raise RuntimeError(
            'Could not find librf. Set RF_LIB_PATH or build the C core: '
            'cd rf && mkdir build && cd build && cmake .. && make')

    _lib = ctypes.CDLL(lib_path)

    # rf_params_init
    _lib.rf_params_init.argtypes = [ctypes.c_void_p]
    _lib.rf_params_init.restype = None

    # rf_fit
    _lib.rf_fit.argtypes = [
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_void_p, ctypes.c_void_p,
    ]
    _lib.rf_fit.restype = ctypes.c_void_p

    # rf_predict
    _lib.rf_predict.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p,
    ]
    _lib.rf_predict.restype = ctypes.c_int

    # rf_predict_proba
    _lib.rf_predict_proba.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p,
    ]
    _lib.rf_predict_proba.restype = ctypes.c_int

    # rf_save / rf_load
    _lib.rf_save.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int32),
    ]
    _lib.rf_save.restype = ctypes.c_int
    _lib.rf_load.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    _lib.rf_load.restype = ctypes.c_void_p

    # rf_free / rf_free_buffer / rf_get_error
    _lib.rf_free.argtypes = [ctypes.c_void_p]
    _lib.rf_free.restype = None
    _lib.rf_free_buffer.argtypes = [ctypes.c_void_p]
    _lib.rf_free_buffer.restype = None
    _lib.rf_get_error.argtypes = []
    _lib.rf_get_error.restype = ctypes.c_char_p

    # Getters
    for name in ['n_trees', 'n_features', 'n_classes', 'task',
                 'criterion', 'heterogeneous', 'oob_weighting', 'leaf_model']:
        fn = getattr(_lib, f'wl_rf_get_{name}')
        fn.argtypes = [ctypes.c_void_p]
        fn.restype = ctypes.c_int

    _lib.wl_rf_get_oob_score.argtypes = [ctypes.c_void_p]
    _lib.wl_rf_get_oob_score.restype = ctypes.c_double
    _lib.wl_rf_get_sample_rate.argtypes = [ctypes.c_void_p]
    _lib.wl_rf_get_sample_rate.restype = ctypes.c_double
    _lib.wl_rf_get_alpha_trim.argtypes = [ctypes.c_void_p]
    _lib.wl_rf_get_alpha_trim.restype = ctypes.c_double
    _lib.wl_rf_get_feature_importance.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _lib.wl_rf_get_feature_importance.restype = ctypes.c_double

    # rf_permutation_importance
    _lib.rf_permutation_importance.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_uint32, ctypes.c_void_p,
    ]
    _lib.rf_permutation_importance.restype = ctypes.c_int

    # rf_proximity
    _lib.rf_proximity.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p,
    ]
    _lib.rf_proximity.restype = ctypes.c_int

    return _lib


# ---------------------------------------------------------------------------
# RFParams struct (must match C layout exactly)
# ---------------------------------------------------------------------------

class _RFParams(ctypes.Structure):
    _fields_ = [
        ('n_estimators', ctypes.c_int32),
        ('max_depth', ctypes.c_int32),
        ('min_samples_split', ctypes.c_int32),
        ('min_samples_leaf', ctypes.c_int32),
        ('max_features', ctypes.c_int32),
        ('max_leaf_nodes', ctypes.c_int32),
        ('bootstrap', ctypes.c_int32),
        ('compute_oob', ctypes.c_int32),
        ('extra_trees', ctypes.c_int32),
        ('seed', ctypes.c_uint32),
        ('task', ctypes.c_int32),
        ('criterion', ctypes.c_int32),
        ('heterogeneous', ctypes.c_int32),
        ('oob_weighting', ctypes.c_int32),
        ('leaf_model', ctypes.c_int32),
        ('store_leaf_samples', ctypes.c_int32),
        ('sample_rate', ctypes.c_double),
        ('alpha_trim', ctypes.c_double),
        ('monotonic_cst', ctypes.POINTER(ctypes.c_int32)),
        ('n_monotonic_cst', ctypes.c_int32),
        ('sample_weight', ctypes.POINTER(ctypes.c_double)),
        ('n_sample_weight', ctypes.c_int32),
        ('histogram', ctypes.c_int32),
        ('max_bins', ctypes.c_int32),
        ('jarf', ctypes.c_int32),
        ('jarf_n_estimators', ctypes.c_int32),
        ('jarf_max_depth', ctypes.c_int32),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CLS_CRITERIA = {'gini': 0, 'entropy': 1, 'hellinger': 2}
_REG_CRITERIA = {'mse': 0, 'squared_error': 0, 'mae': 1, 'absolute_error': 1}


def _resolve_criterion(value, task):
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        table = _CLS_CRITERIA if task == 0 else _REG_CRITERIA
        return table.get(value.lower(), 0)
    return 0


def _resolve_max_features(value, ncol, task):
    if isinstance(value, str):
        s = value.lower()
        if s == 'sqrt':
            return max(1, round(ncol ** 0.5))
        if s == 'log2':
            return max(1, round(np.log2(ncol)))
        if s == 'third':
            return max(1, round(ncol / 3))
    if isinstance(value, (int, float)) and value > 0:
        return int(value)
    # auto
    if task == 1:
        return max(1, round(ncol / 3))
    return max(1, round(ncol ** 0.5))


def _get_error():
    err = _load_lib().rf_get_error()
    return err.decode('utf-8') if err else 'unknown error'


# ---------------------------------------------------------------------------
# Base class (shared by RFClassifier and RFRegressor)
# ---------------------------------------------------------------------------

class _RFBase:
    _task_int = 0  # overridden in subclass

    def __init__(self, params):
        self._handle = None
        self._fitted = False
        self._disposed = False
        self._params = dict(params) if params else {}
        self._n_features = 0
        self._n_classes = 0
        self._classes = None
        # prevent GC of ctypes buffers
        self._sw_buf = None
        self._mono_buf = None

    @classmethod
    def create(cls, params=None):
        return cls(params)

    def fit(self, X, y):
        if self._disposed:
            raise DisposedError('RFModel has been disposed.')

        lib = _load_lib()

        if self._handle:
            lib.rf_free(self._handle)
            self._handle = None

        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f'X must be 2-dimensional, got {X.ndim}')
        nrow, ncol = X.shape
        if y.shape[0] != nrow:
            raise ValueError(
                f'y length ({y.shape[0]}) does not match X rows ({nrow})')

        task = self._task_int

        # For classifiers, record classes for label mapping
        if task == 0:
            self._classes = np.sort(np.unique(y)).astype(np.int32)
            self._n_classes = len(self._classes)

        params = _RFParams()
        lib.rf_params_init(ctypes.byref(params))
        params.n_estimators = self._params.get('n_estimators', 100)
        params.max_depth = self._params.get('max_depth', 0)
        params.min_samples_split = self._params.get('min_samples_split', 2)
        params.min_samples_leaf = self._params.get('min_samples_leaf', 1)
        params.max_features = _resolve_max_features(
            self._params.get('max_features', 0), ncol, task)
        params.max_leaf_nodes = self._params.get('max_leaf_nodes', 0)
        params.bootstrap = self._params.get('bootstrap', 1)
        params.compute_oob = self._params.get('compute_oob', 1)
        params.extra_trees = self._params.get('extra_trees', 0)
        params.seed = self._params.get('seed', 42)
        params.task = task
        params.criterion = _resolve_criterion(
            self._params.get('criterion', 0), task)
        params.heterogeneous = self._params.get('heterogeneous', 0)
        params.oob_weighting = self._params.get('oob_weighting', 0)
        params.leaf_model = self._params.get('leaf_model', 0)
        params.store_leaf_samples = self._params.get(
            'store_leaf_samples', 1 if task == 1 else 0)
        params.sample_rate = self._params.get('sample_rate', 1.0)
        params.alpha_trim = self._params.get('alpha_trim', 0.0)

        # Sample weights
        sw = self._params.get('sample_weight', None)
        class_weight = self._params.get('class_weight', None)
        if sw is None and class_weight == 'balanced' and task == 0:
            classes, counts = np.unique(y, return_counts=True)
            weight_map = {int(c): nrow / (len(classes) * cnt)
                          for c, cnt in zip(classes, counts)}
            sw = np.array([weight_map[int(yi)] for yi in y], dtype=np.float64)

        if sw is not None:
            sw_arr = np.ascontiguousarray(sw, dtype=np.float64)
            if sw_arr.shape[0] != nrow:
                raise ValueError(
                    f'sample_weight length ({sw_arr.shape[0]}) != X rows ({nrow})')
            self._sw_buf = sw_arr
            params.sample_weight = sw_arr.ctypes.data_as(
                ctypes.POINTER(ctypes.c_double))
            params.n_sample_weight = nrow
        else:
            params.sample_weight = None
            params.n_sample_weight = 0

        # Histogram binning
        params.histogram = self._params.get('histogram_binning', 0)
        params.max_bins = self._params.get('max_bins', 256)

        # JARF
        params.jarf = self._params.get('jarf', 0)
        params.jarf_n_estimators = self._params.get('jarf_n_estimators', 50)
        params.jarf_max_depth = self._params.get('jarf_max_depth', 6)

        # Monotonic constraints
        mono = self._params.get('monotonic_cst', None)
        if mono is not None:
            mono_arr = np.asarray(mono, dtype=np.int32)
            if mono_arr.shape[0] == ncol:
                self._mono_buf = mono_arr
                params.monotonic_cst = mono_arr.ctypes.data_as(
                    ctypes.POINTER(ctypes.c_int32))
                params.n_monotonic_cst = ncol
            else:
                params.monotonic_cst = None
                params.n_monotonic_cst = 0
        else:
            params.monotonic_cst = None
            params.n_monotonic_cst = 0

        handle = lib.rf_fit(X.ctypes.data, nrow, ncol,
                            y.ctypes.data, ctypes.byref(params))
        if not handle:
            raise RuntimeError(f'rf_fit failed: {_get_error()}')

        self._handle = handle
        self._fitted = True
        self._n_features = lib.wl_rf_get_n_features(handle)
        if task == 0:
            self._n_classes = lib.wl_rf_get_n_classes(handle)
        return self

    def predict(self, X):
        self._ensure_fitted()
        lib = _load_lib()
        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f'X must be 2-dimensional, got {X.ndim}')
        nrow, ncol = X.shape
        out = np.zeros(nrow, dtype=np.float64)
        ret = lib.rf_predict(self._handle, X.ctypes.data, nrow, ncol,
                             out.ctypes.data)
        if ret != 0:
            raise RuntimeError(f'rf_predict failed: {_get_error()}')
        return out

    def score(self, X, y):
        preds = self.predict(X)
        y = np.asarray(y, dtype=np.float64)
        if self._task_int == 0:
            return float(np.mean(preds.astype(int) == y.astype(int)))
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def feature_importances(self):
        self._ensure_fitted()
        lib = _load_lib()
        result = np.zeros(self._n_features, dtype=np.float64)
        for i in range(self._n_features):
            result[i] = lib.wl_rf_get_feature_importance(self._handle, i)
        return result

    def permutation_importance(self, X, y, n_repeats=5, seed=42):
        self._ensure_fitted()
        lib = _load_lib()
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        nrow, ncol = X.shape
        out = np.zeros(ncol, dtype=np.float64)
        ret = lib.rf_permutation_importance(
            self._handle, X.ctypes.data, nrow, ncol,
            y.ctypes.data, n_repeats, seed, out.ctypes.data)
        if ret != 0:
            raise RuntimeError(
                f'rf_permutation_importance failed: {_get_error()}')
        return out

    def proximity(self, X):
        self._ensure_fitted()
        lib = _load_lib()
        X = np.ascontiguousarray(X, dtype=np.float64)
        nrow, ncol = X.shape
        out = np.zeros(nrow * nrow, dtype=np.float64)
        ret = lib.rf_proximity(self._handle, X.ctypes.data, nrow, ncol,
                               out.ctypes.data)
        if ret != 0:
            raise RuntimeError(f'rf_proximity failed: {_get_error()}')
        return out.reshape(nrow, nrow)

    def oob_score(self):
        self._ensure_fitted()
        return _load_lib().wl_rf_get_oob_score(self._handle)

    def save(self):
        self._ensure_fitted()
        lib = _load_lib()

        out_buf = ctypes.c_void_p()
        out_len = ctypes.c_int32()
        ret = lib.rf_save(self._handle, ctypes.byref(out_buf),
                          ctypes.byref(out_len))
        if ret != 0:
            raise RuntimeError(f'rf_save failed: {_get_error()}')

        buf_len = out_len.value
        model_bytes = bytes(
            (ctypes.c_char * buf_len).from_address(out_buf.value))
        lib.rf_free_buffer(out_buf)

        type_id = ('wlearn.rf.classifier@1' if self._task_int == 0
                   else 'wlearn.rf.regressor@1')

        return encode_bundle(
            {
                'typeId': type_id,
                'params': self.get_params(),
                'metadata': {
                    'nrClass': self._n_classes,
                    'classes': self._classes.tolist() if self._classes is not None else [],
                },
            },
            [{'id': 'model', 'data': model_bytes}],
        )

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        if self._handle:
            _load_lib().rf_free(self._handle)
            self._handle = None
        self._fitted = False

    def get_params(self):
        return dict(self._params)

    def set_params(self, p):
        self._params.update(p)
        return self

    @property
    def is_fitted(self):
        return self._fitted and not self._disposed

    @property
    def n_features(self):
        return self._n_features

    @property
    def n_classes(self):
        return self._n_classes

    @property
    def n_trees(self):
        if not self._handle or self._disposed:
            return 0
        return _load_lib().wl_rf_get_n_trees(self._handle)

    def _ensure_fitted(self):
        if self._disposed:
            raise DisposedError('RFModel has been disposed.')
        if not self._fitted:
            raise NotFittedError('RFModel is not fitted. Call fit() first.')

    def __del__(self):
        if hasattr(self, '_handle') and self._handle and not self._disposed:
            try:
                self.dispose()
            except Exception:
                pass

    @classmethod
    def _from_bundle(cls, manifest, toc, blobs):
        entry = next((e for e in toc if e['id'] == 'model'), None)
        if entry is None:
            raise ValueError('Bundle missing "model" artifact')

        model_bytes = bytes(
            blobs[entry['offset']:entry['offset'] + entry['length']])

        lib = _load_lib()
        buf = ctypes.create_string_buffer(model_bytes)
        handle = lib.rf_load(buf, len(model_bytes))
        if not handle:
            raise RuntimeError(f'rf_load failed: {_get_error()}')

        params = manifest.get('params', {})
        obj = cls(params)
        obj._handle = handle
        obj._fitted = True
        obj._n_features = lib.wl_rf_get_n_features(handle)
        obj._n_classes = lib.wl_rf_get_n_classes(handle)

        meta = manifest.get('metadata', {})
        classes = meta.get('classes')
        if classes:
            obj._classes = np.array(classes, dtype=np.int32)

        return obj

    @staticmethod
    def default_search_space():
        return {
            'n_estimators': {'type': 'int_uniform', 'low': 50, 'high': 500},
            'max_depth': {'type': 'int_uniform', 'low': 3, 'high': 20},
            'max_features': {'type': 'categorical',
                             'values': ['sqrt', 'log2', 'third']},
            'min_samples_split': {'type': 'int_uniform', 'low': 2, 'high': 20},
            'min_samples_leaf': {'type': 'int_uniform', 'low': 1, 'high': 10},
            'histogram_binning': {'type': 'categorical', 'values': [0, 1]},
            'jarf': {'type': 'categorical', 'values': [0, 1]},
        }


# ---------------------------------------------------------------------------
# Public classes
# ---------------------------------------------------------------------------

class RFClassifier(_RFBase):
    """Random Forest classifier backed by C11 core."""
    _task_int = 0

    def predict_proba(self, X):
        self._ensure_fitted()
        lib = _load_lib()
        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f'X must be 2-dimensional, got {X.ndim}')
        nrow, ncol = X.shape
        nc = self._n_classes
        out = np.zeros(nrow * nc, dtype=np.float64)
        ret = lib.rf_predict_proba(self._handle, X.ctypes.data, nrow, ncol,
                                   out.ctypes.data)
        if ret != 0:
            raise RuntimeError(f'rf_predict_proba failed: {_get_error()}')
        return out.reshape(nrow, nc)


class RFRegressor(_RFBase):
    """Random Forest regressor backed by C11 core."""
    _task_int = 1


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

register('wlearn.rf.classifier@1', RFClassifier._from_bundle)
register('wlearn.rf.regressor@1', RFRegressor._from_bundle)
