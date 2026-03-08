"""Python wrapper for @wlearn/nn bundles.

Loads WLRN bundles produced by JS @wlearn/nn, predicts using
native polygrad (C FFI), and saves back to WLRN bundles that JS can load.

Provides MLPClassifier, MLPRegressor, TabMClassifier, TabMRegressor,
NAMClassifier, and NAMRegressor following the wlearn estimator contract.
"""

import numpy as np

from .errors import NotFittedError, DisposedError
from .bundle import encode_bundle
from .registry import register

# Lazy import polygrad to avoid hard dependency at module level
_Instance = None

def _get_instance_class():
    global _Instance
    if _Instance is None:
        from polygrad.instance import Instance
        _Instance = Instance
    return _Instance


def _make_mlp_spec(n_features, hidden_sizes, n_outputs, activation,
                   loss, seed):
    """Build the poly_mlp_instance JSON spec."""
    layers = [n_features] + list(hidden_sizes) + [n_outputs]
    return {
        'layers': layers,
        'activation': activation,
        'bias': True,
        'loss': loss,
        'batch_size': 1,
        'seed': seed,
    }


class MLPClassifier:
    """MLP classifier using polygrad Instance backend.

    Parameters (passed via create() or set_params()):
        hidden_sizes: list of hidden layer sizes (default [64])
        activation: activation function (default 'relu')
        lr: learning rate (default 0.01)
        epochs: number of training epochs (default 100)
        optimizer: 'sgd' or 'adam' (default 'adam')
        seed: random seed (default 42)
    """

    def __init__(self, instance, params, nr_class=0, classes=None,
                 n_features=0):
        self._instance = instance
        self._params = dict(params)
        self._nr_class = nr_class
        self._classes = (np.array(classes, dtype=np.int32)
                         if classes is not None and len(classes) > 0
                         else np.array([], dtype=np.int32))
        self._n_features = n_features
        self._fitted = True
        self._disposed = False

    @classmethod
    def create(cls, params=None):
        """Create an unfitted MLP classifier."""
        obj = cls.__new__(cls)
        obj._instance = None
        obj._params = dict(params) if params else {}
        obj._nr_class = 0
        obj._classes = np.array([], dtype=np.int32)
        obj._n_features = 0
        obj._fitted = False
        obj._disposed = False
        return obj

    def fit(self, X, y):
        if self._disposed:
            raise DisposedError('MLPClassifier has been disposed.')

        Instance = _get_instance_class()
        from polygrad.instance import OPTIM_SGD, OPTIM_ADAM

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples, n_features = X.shape
        self._n_features = n_features

        # Detect classes
        unique = np.sort(np.unique(y))
        self._classes = unique.astype(np.int32)
        self._nr_class = len(unique)
        class_map = {int(c): i for i, c in enumerate(unique)}

        # One-hot encode targets
        y_onehot = np.zeros((n_samples, self._nr_class), dtype=np.float32)
        for i in range(n_samples):
            y_onehot[i, class_map[int(y[i])]] = 1.0

        # Build MLP spec
        hidden = self._params.get('hidden_sizes', [64])
        activation = self._params.get('activation', 'relu')
        seed = self._params.get('seed', 42)
        lr = self._params.get('lr', 0.01)
        epochs = self._params.get('epochs', 100)
        optimizer = self._params.get('optimizer', 'adam')

        spec = _make_mlp_spec(n_features, hidden, self._nr_class,
                              activation, 'cross_entropy', seed)

        # Create instance
        if self._instance:
            self._instance.free()
        self._instance = Instance.mlp(spec)

        # Set optimizer
        optim_kind = OPTIM_ADAM if optimizer == 'adam' else OPTIM_SGD
        self._instance.set_optimizer(optim_kind, lr=lr)

        # Train
        for epoch in range(epochs):
            for i in range(n_samples):
                xi = X[i:i+1].flatten()
                yi = y_onehot[i]
                self._instance.train_step(x=xi, y=yi)

        self._fitted = True
        return self

    @staticmethod
    def _from_bundle(manifest, toc, blobs):
        Instance = _get_instance_class()

        ir_entry = next((e for e in toc if e['id'] == 'ir'), None)
        w_entry = next((e for e in toc if e['id'] == 'weights'), None)
        if ir_entry is None or w_entry is None:
            raise ValueError('Bundle missing "ir" or "weights" artifact')

        ir_bytes = bytes(blobs[ir_entry['offset']:
                               ir_entry['offset'] + ir_entry['length']])
        w_bytes = bytes(blobs[w_entry['offset']:
                              w_entry['offset'] + w_entry['length']])

        instance = Instance.from_ir(ir_bytes, w_bytes)
        params = manifest.get('params', {})
        meta = manifest.get('metadata', {})
        return MLPClassifier(
            instance, params,
            nr_class=meta.get('nrClass', 0),
            classes=meta.get('classes'),
            n_features=meta.get('nFeatures', 0),
        )

    def predict(self, X):
        self._ensure_fitted()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]
        result = np.empty(n_samples, dtype=np.float64)

        for i in range(n_samples):
            out = self._instance.forward(x=X[i])
            logits = out['output']
            pred_idx = int(np.argmax(logits))
            result[i] = self._classes[pred_idx] if pred_idx < len(self._classes) else pred_idx

        return result

    def predict_proba(self, X):
        self._ensure_fitted()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]
        nc = self._nr_class
        result = np.empty(n_samples * nc, dtype=np.float64)

        for i in range(n_samples):
            out = self._instance.forward(x=X[i])
            logits = out['output']
            # softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            result[i * nc:(i + 1) * nc] = probs

        return result

    def score(self, X, y):
        preds = self.predict(X)
        y = np.asarray(y, dtype=np.float64)
        return float(np.mean(preds == y))

    def save(self):
        self._ensure_fitted()
        ir_bytes = self._instance.export_ir()
        w_bytes = self._instance.export_weights()

        return encode_bundle(
            {
                'typeId': 'wlearn.nn.mlp.classifier@1',
                'params': self.get_params(),
                'metadata': {
                    'nrClass': self._nr_class,
                    'classes': self._classes.tolist(),
                    'nFeatures': self._n_features,
                },
            },
            [
                {'id': 'ir', 'data': ir_bytes},
                {'id': 'weights', 'data': w_bytes},
            ],
        )

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        if self._instance:
            self._instance.free()
            self._instance = None
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
    def classes(self):
        return np.array(self._classes, dtype=np.int32)

    @property
    def capabilities(self):
        return {
            'classifier': True,
            'regressor': False,
            'predictProba': True,
            'decisionFunction': False,
            'sampleWeight': False,
            'csr': False,
            'earlyStopping': False,
        }

    @classmethod
    def default_search_space(cls):
        return {
            'hidden_sizes': {'type': 'categorical',
                             'values': [[64], [128], [64, 64], [128, 64]]},
            'activation': {'type': 'categorical',
                           'values': ['relu', 'gelu', 'silu']},
            'lr': {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-1},
            'epochs': {'type': 'int_uniform', 'low': 10, 'high': 200},
            'optimizer': {'type': 'categorical', 'values': ['adam', 'sgd']},
        }

    def _ensure_fitted(self):
        if self._disposed:
            raise DisposedError('MLPClassifier has been disposed.')
        if not self._fitted:
            raise NotFittedError('MLPClassifier is not fitted.')


class MLPRegressor:
    """MLP regressor using polygrad Instance backend.

    Parameters (passed via create() or set_params()):
        hidden_sizes: list of hidden layer sizes (default [64])
        activation: activation function (default 'relu')
        lr: learning rate (default 0.01)
        epochs: number of training epochs (default 100)
        optimizer: 'sgd' or 'adam' (default 'adam')
        seed: random seed (default 42)
    """

    def __init__(self, instance, params, n_features=0):
        self._instance = instance
        self._params = dict(params)
        self._n_features = n_features
        self._fitted = True
        self._disposed = False

    @classmethod
    def create(cls, params=None):
        """Create an unfitted MLP regressor."""
        obj = cls.__new__(cls)
        obj._instance = None
        obj._params = dict(params) if params else {}
        obj._n_features = 0
        obj._fitted = False
        obj._disposed = False
        return obj

    def fit(self, X, y):
        if self._disposed:
            raise DisposedError('MLPRegressor has been disposed.')

        Instance = _get_instance_class()
        from polygrad.instance import OPTIM_SGD, OPTIM_ADAM

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples, n_features = X.shape
        n_outputs = y.shape[1]
        self._n_features = n_features

        # Build MLP spec
        hidden = self._params.get('hidden_sizes', [64])
        activation = self._params.get('activation', 'relu')
        seed = self._params.get('seed', 42)
        lr = self._params.get('lr', 0.01)
        epochs = self._params.get('epochs', 100)
        optimizer = self._params.get('optimizer', 'adam')

        spec = _make_mlp_spec(n_features, hidden, n_outputs,
                              activation, 'mse', seed)

        if self._instance:
            self._instance.free()
        self._instance = Instance.mlp(spec)

        optim_kind = OPTIM_ADAM if optimizer == 'adam' else OPTIM_SGD
        self._instance.set_optimizer(optim_kind, lr=lr)

        for epoch in range(epochs):
            for i in range(n_samples):
                xi = X[i:i+1].flatten()
                yi = y[i].flatten()
                self._instance.train_step(x=xi, y=yi)

        self._fitted = True
        return self

    @staticmethod
    def _from_bundle(manifest, toc, blobs):
        Instance = _get_instance_class()

        ir_entry = next((e for e in toc if e['id'] == 'ir'), None)
        w_entry = next((e for e in toc if e['id'] == 'weights'), None)
        if ir_entry is None or w_entry is None:
            raise ValueError('Bundle missing "ir" or "weights" artifact')

        ir_bytes = bytes(blobs[ir_entry['offset']:
                               ir_entry['offset'] + ir_entry['length']])
        w_bytes = bytes(blobs[w_entry['offset']:
                              w_entry['offset'] + w_entry['length']])

        instance = Instance.from_ir(ir_bytes, w_bytes)
        params = manifest.get('params', {})
        meta = manifest.get('metadata', {})
        return MLPRegressor(
            instance, params,
            n_features=meta.get('nFeatures', 0),
        )

    def predict(self, X):
        self._ensure_fitted()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]
        results = []
        for i in range(n_samples):
            out = self._instance.forward(x=X[i])
            results.append(out['output'].copy())

        result = np.array(results, dtype=np.float64)
        if result.shape[1] == 1:
            return result.flatten()
        return result

    def score(self, X, y):
        preds = self.predict(X)
        y = np.asarray(y, dtype=np.float64)
        y_mean = y.mean()
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        return 0.0 if ss_tot == 0 else float(1 - ss_res / ss_tot)

    def save(self):
        self._ensure_fitted()
        ir_bytes = self._instance.export_ir()
        w_bytes = self._instance.export_weights()

        return encode_bundle(
            {
                'typeId': 'wlearn.nn.mlp.regressor@1',
                'params': self.get_params(),
                'metadata': {
                    'nFeatures': self._n_features,
                },
            },
            [
                {'id': 'ir', 'data': ir_bytes},
                {'id': 'weights', 'data': w_bytes},
            ],
        )

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        if self._instance:
            self._instance.free()
            self._instance = None
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
    def capabilities(self):
        return {
            'classifier': False,
            'regressor': True,
            'predictProba': False,
            'decisionFunction': False,
            'sampleWeight': False,
            'csr': False,
            'earlyStopping': False,
        }

    @classmethod
    def default_search_space(cls):
        return {
            'hidden_sizes': {'type': 'categorical',
                             'values': [[64], [128], [64, 64], [128, 64]]},
            'activation': {'type': 'categorical',
                           'values': ['relu', 'gelu', 'silu']},
            'lr': {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-1},
            'epochs': {'type': 'int_uniform', 'low': 10, 'high': 200},
            'optimizer': {'type': 'categorical', 'values': ['adam', 'sgd']},
        }

    def _ensure_fitted(self):
        if self._disposed:
            raise DisposedError('MLPRegressor has been disposed.')
        if not self._fitted:
            raise NotFittedError('MLPRegressor is not fitted.')


def _make_tabm_spec(n_features, hidden_sizes, n_outputs, n_ensemble,
                    activation, loss, seed):
    """Build the poly_tabm_instance JSON spec."""
    layers = [n_features] + list(hidden_sizes) + [n_outputs]
    return {
        'layers': layers,
        'n_ensemble': n_ensemble,
        'activation': activation,
        'bias': True,
        'loss': loss,
        'batch_size': 1,
        'seed': seed,
    }


def _make_nam_spec(n_features, hidden_sizes, n_outputs, activation,
                   loss, seed):
    """Build the poly_nam_instance JSON spec."""
    return {
        'n_features': n_features,
        'hidden_sizes': list(hidden_sizes),
        'activation': activation,
        'n_outputs': n_outputs,
        'loss': loss,
        'batch_size': 1,
        'seed': seed,
    }


class TabMClassifier:
    """TabM classifier (BatchEnsemble MLP) using polygrad Instance backend.

    Parameters (passed via create() or set_params()):
        hidden_sizes: list of hidden layer sizes (default [64])
        n_ensemble: number of ensemble members (default 32)
        activation: activation function (default 'relu')
        lr: learning rate (default 0.01)
        epochs: number of training epochs (default 100)
        optimizer: 'sgd' or 'adam' (default 'adam')
        seed: random seed (default 42)
    """

    def __init__(self, instance, params, nr_class=0, classes=None,
                 n_features=0):
        self._instance = instance
        self._params = dict(params)
        self._nr_class = nr_class
        self._classes = (np.array(classes, dtype=np.int32)
                         if classes is not None and len(classes) > 0
                         else np.array([], dtype=np.int32))
        self._n_features = n_features
        self._fitted = True
        self._disposed = False

    @classmethod
    def create(cls, params=None):
        obj = cls.__new__(cls)
        obj._instance = None
        obj._params = dict(params) if params else {}
        obj._nr_class = 0
        obj._classes = np.array([], dtype=np.int32)
        obj._n_features = 0
        obj._fitted = False
        obj._disposed = False
        return obj

    def fit(self, X, y):
        if self._disposed:
            raise DisposedError('TabMClassifier has been disposed.')

        Instance = _get_instance_class()
        from polygrad.instance import OPTIM_SGD, OPTIM_ADAM

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples, n_features = X.shape
        self._n_features = n_features

        unique = np.sort(np.unique(y))
        self._classes = unique.astype(np.int32)
        self._nr_class = len(unique)
        class_map = {int(c): i for i, c in enumerate(unique)}

        y_onehot = np.zeros((n_samples, self._nr_class), dtype=np.float32)
        for i in range(n_samples):
            y_onehot[i, class_map[int(y[i])]] = 1.0

        hidden = self._params.get('hidden_sizes', [64])
        n_ensemble = self._params.get('n_ensemble', 32)
        activation = self._params.get('activation', 'relu')
        seed = self._params.get('seed', 42)
        lr = self._params.get('lr', 0.01)
        epochs = self._params.get('epochs', 100)
        optimizer = self._params.get('optimizer', 'adam')

        spec = _make_tabm_spec(n_features, hidden, self._nr_class,
                               n_ensemble, activation, 'cross_entropy', seed)

        if self._instance:
            self._instance.free()
        self._instance = Instance.tabm(spec)

        optim_kind = OPTIM_ADAM if optimizer == 'adam' else OPTIM_SGD
        self._instance.set_optimizer(optim_kind, lr=lr)

        for epoch in range(epochs):
            for i in range(n_samples):
                xi = X[i:i+1].flatten()
                yi = y_onehot[i]
                self._instance.train_step(x=xi, y=yi)

        self._fitted = True
        return self

    @staticmethod
    def _from_bundle(manifest, toc, blobs):
        Instance = _get_instance_class()

        ir_entry = next((e for e in toc if e['id'] == 'ir'), None)
        w_entry = next((e for e in toc if e['id'] == 'weights'), None)
        if ir_entry is None or w_entry is None:
            raise ValueError('Bundle missing "ir" or "weights" artifact')

        ir_bytes = bytes(blobs[ir_entry['offset']:
                               ir_entry['offset'] + ir_entry['length']])
        w_bytes = bytes(blobs[w_entry['offset']:
                              w_entry['offset'] + w_entry['length']])

        instance = Instance.from_ir(ir_bytes, w_bytes)
        params = manifest.get('params', {})
        meta = manifest.get('metadata', {})
        return TabMClassifier(
            instance, params,
            nr_class=meta.get('nrClass', 0),
            classes=meta.get('classes'),
            n_features=meta.get('nFeatures', 0),
        )

    def predict(self, X):
        self._ensure_fitted()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]
        result = np.empty(n_samples, dtype=np.float64)

        for i in range(n_samples):
            out = self._instance.forward(x=X[i])
            logits = out['output']
            pred_idx = int(np.argmax(logits))
            result[i] = self._classes[pred_idx] if pred_idx < len(self._classes) else pred_idx

        return result

    def predict_proba(self, X):
        self._ensure_fitted()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]
        nc = self._nr_class
        result = np.empty(n_samples * nc, dtype=np.float64)

        for i in range(n_samples):
            out = self._instance.forward(x=X[i])
            logits = out['output']
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            result[i * nc:(i + 1) * nc] = probs

        return result

    def score(self, X, y):
        preds = self.predict(X)
        y = np.asarray(y, dtype=np.float64)
        return float(np.mean(preds == y))

    def save(self):
        self._ensure_fitted()
        ir_bytes = self._instance.export_ir()
        w_bytes = self._instance.export_weights()

        return encode_bundle(
            {
                'typeId': 'wlearn.nn.tabm.classifier@1',
                'params': self.get_params(),
                'metadata': {
                    'nrClass': self._nr_class,
                    'classes': self._classes.tolist(),
                    'nFeatures': self._n_features,
                },
            },
            [
                {'id': 'ir', 'data': ir_bytes},
                {'id': 'weights', 'data': w_bytes},
            ],
        )

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        if self._instance:
            self._instance.free()
            self._instance = None
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
    def classes(self):
        return np.array(self._classes, dtype=np.int32)

    @property
    def capabilities(self):
        return {
            'classifier': True, 'regressor': False,
            'predictProba': True, 'decisionFunction': False,
            'sampleWeight': False, 'csr': False, 'earlyStopping': True,
        }

    @classmethod
    def default_search_space(cls):
        return {
            'hidden_sizes': {'type': 'categorical',
                             'values': [[64], [128], [64, 64], [128, 64]]},
            'n_ensemble': {'type': 'categorical', 'values': [4, 8, 16, 32]},
            'activation': {'type': 'categorical',
                           'values': ['relu', 'gelu', 'silu']},
            'lr': {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-1},
            'epochs': {'type': 'int_uniform', 'low': 10, 'high': 200},
            'optimizer': {'type': 'categorical', 'values': ['adam', 'sgd']},
        }

    def _ensure_fitted(self):
        if self._disposed:
            raise DisposedError('TabMClassifier has been disposed.')
        if not self._fitted:
            raise NotFittedError('TabMClassifier is not fitted.')


class TabMRegressor:
    """TabM regressor (BatchEnsemble MLP) using polygrad Instance backend."""

    def __init__(self, instance, params, n_features=0):
        self._instance = instance
        self._params = dict(params)
        self._n_features = n_features
        self._fitted = True
        self._disposed = False

    @classmethod
    def create(cls, params=None):
        obj = cls.__new__(cls)
        obj._instance = None
        obj._params = dict(params) if params else {}
        obj._n_features = 0
        obj._fitted = False
        obj._disposed = False
        return obj

    def fit(self, X, y):
        if self._disposed:
            raise DisposedError('TabMRegressor has been disposed.')

        Instance = _get_instance_class()
        from polygrad.instance import OPTIM_SGD, OPTIM_ADAM

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples, n_features = X.shape
        n_outputs = y.shape[1]
        self._n_features = n_features

        hidden = self._params.get('hidden_sizes', [64])
        n_ensemble = self._params.get('n_ensemble', 32)
        activation = self._params.get('activation', 'relu')
        seed = self._params.get('seed', 42)
        lr = self._params.get('lr', 0.01)
        epochs = self._params.get('epochs', 100)
        optimizer = self._params.get('optimizer', 'adam')

        spec = _make_tabm_spec(n_features, hidden, n_outputs,
                               n_ensemble, activation, 'mse', seed)

        if self._instance:
            self._instance.free()
        self._instance = Instance.tabm(spec)

        optim_kind = OPTIM_ADAM if optimizer == 'adam' else OPTIM_SGD
        self._instance.set_optimizer(optim_kind, lr=lr)

        for epoch in range(epochs):
            for i in range(n_samples):
                xi = X[i:i+1].flatten()
                yi = y[i].flatten()
                self._instance.train_step(x=xi, y=yi)

        self._fitted = True
        return self

    @staticmethod
    def _from_bundle(manifest, toc, blobs):
        Instance = _get_instance_class()

        ir_entry = next((e for e in toc if e['id'] == 'ir'), None)
        w_entry = next((e for e in toc if e['id'] == 'weights'), None)
        if ir_entry is None or w_entry is None:
            raise ValueError('Bundle missing "ir" or "weights" artifact')

        ir_bytes = bytes(blobs[ir_entry['offset']:
                               ir_entry['offset'] + ir_entry['length']])
        w_bytes = bytes(blobs[w_entry['offset']:
                              w_entry['offset'] + w_entry['length']])

        instance = Instance.from_ir(ir_bytes, w_bytes)
        params = manifest.get('params', {})
        meta = manifest.get('metadata', {})
        return TabMRegressor(
            instance, params,
            n_features=meta.get('nFeatures', 0),
        )

    def predict(self, X):
        self._ensure_fitted()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]
        results = []
        for i in range(n_samples):
            out = self._instance.forward(x=X[i])
            results.append(out['output'].copy())

        result = np.array(results, dtype=np.float64)
        if result.shape[1] == 1:
            return result.flatten()
        return result

    def score(self, X, y):
        preds = self.predict(X)
        y = np.asarray(y, dtype=np.float64)
        y_mean = y.mean()
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        return 0.0 if ss_tot == 0 else float(1 - ss_res / ss_tot)

    def save(self):
        self._ensure_fitted()
        ir_bytes = self._instance.export_ir()
        w_bytes = self._instance.export_weights()

        return encode_bundle(
            {
                'typeId': 'wlearn.nn.tabm.regressor@1',
                'params': self.get_params(),
                'metadata': {
                    'nFeatures': self._n_features,
                },
            },
            [
                {'id': 'ir', 'data': ir_bytes},
                {'id': 'weights', 'data': w_bytes},
            ],
        )

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        if self._instance:
            self._instance.free()
            self._instance = None
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
    def capabilities(self):
        return {
            'classifier': False, 'regressor': True,
            'predictProba': False, 'decisionFunction': False,
            'sampleWeight': False, 'csr': False, 'earlyStopping': True,
        }

    @classmethod
    def default_search_space(cls):
        return {
            'hidden_sizes': {'type': 'categorical',
                             'values': [[64], [128], [64, 64], [128, 64]]},
            'n_ensemble': {'type': 'categorical', 'values': [4, 8, 16, 32]},
            'activation': {'type': 'categorical',
                           'values': ['relu', 'gelu', 'silu']},
            'lr': {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-1},
            'epochs': {'type': 'int_uniform', 'low': 10, 'high': 200},
            'optimizer': {'type': 'categorical', 'values': ['adam', 'sgd']},
        }

    def _ensure_fitted(self):
        if self._disposed:
            raise DisposedError('TabMRegressor has been disposed.')
        if not self._fitted:
            raise NotFittedError('TabMRegressor is not fitted.')


class NAMClassifier:
    """NAM classifier (Neural Additive Model) using polygrad Instance backend.

    Parameters (passed via create() or set_params()):
        hidden_sizes: list of hidden layer sizes (default [64])
        activation: 'exu', 'relu', 'gelu', or 'silu' (default 'exu')
        lr: learning rate (default 0.01)
        epochs: number of training epochs (default 100)
        optimizer: 'sgd' or 'adam' (default 'adam')
        seed: random seed (default 42)
    """

    def __init__(self, instance, params, nr_class=0, classes=None,
                 n_features=0):
        self._instance = instance
        self._params = dict(params)
        self._nr_class = nr_class
        self._classes = (np.array(classes, dtype=np.int32)
                         if classes is not None and len(classes) > 0
                         else np.array([], dtype=np.int32))
        self._n_features = n_features
        self._fitted = True
        self._disposed = False

    @classmethod
    def create(cls, params=None):
        obj = cls.__new__(cls)
        obj._instance = None
        obj._params = dict(params) if params else {}
        obj._nr_class = 0
        obj._classes = np.array([], dtype=np.int32)
        obj._n_features = 0
        obj._fitted = False
        obj._disposed = False
        return obj

    def fit(self, X, y):
        if self._disposed:
            raise DisposedError('NAMClassifier has been disposed.')

        Instance = _get_instance_class()
        from polygrad.instance import OPTIM_SGD, OPTIM_ADAM

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples, n_features = X.shape
        self._n_features = n_features

        unique = np.sort(np.unique(y))
        self._classes = unique.astype(np.int32)
        self._nr_class = len(unique)
        class_map = {int(c): i for i, c in enumerate(unique)}

        y_onehot = np.zeros((n_samples, self._nr_class), dtype=np.float32)
        for i in range(n_samples):
            y_onehot[i, class_map[int(y[i])]] = 1.0

        hidden = self._params.get('hidden_sizes', [64])
        activation = self._params.get('activation', 'exu')
        seed = self._params.get('seed', 42)
        lr = self._params.get('lr', 0.01)
        epochs = self._params.get('epochs', 100)
        optimizer = self._params.get('optimizer', 'adam')

        spec = _make_nam_spec(n_features, hidden, self._nr_class,
                              activation, 'cross_entropy', seed)

        if self._instance:
            self._instance.free()
        self._instance = Instance.nam(spec)

        optim_kind = OPTIM_ADAM if optimizer == 'adam' else OPTIM_SGD
        self._instance.set_optimizer(optim_kind, lr=lr)

        for epoch in range(epochs):
            for i in range(n_samples):
                xi = X[i:i+1].flatten()
                yi = y_onehot[i]
                self._instance.train_step(x=xi, y=yi)

        self._fitted = True
        return self

    @staticmethod
    def _from_bundle(manifest, toc, blobs):
        Instance = _get_instance_class()

        ir_entry = next((e for e in toc if e['id'] == 'ir'), None)
        w_entry = next((e for e in toc if e['id'] == 'weights'), None)
        if ir_entry is None or w_entry is None:
            raise ValueError('Bundle missing "ir" or "weights" artifact')

        ir_bytes = bytes(blobs[ir_entry['offset']:
                               ir_entry['offset'] + ir_entry['length']])
        w_bytes = bytes(blobs[w_entry['offset']:
                              w_entry['offset'] + w_entry['length']])

        instance = Instance.from_ir(ir_bytes, w_bytes)
        params = manifest.get('params', {})
        meta = manifest.get('metadata', {})
        return NAMClassifier(
            instance, params,
            nr_class=meta.get('nrClass', 0),
            classes=meta.get('classes'),
            n_features=meta.get('nFeatures', 0),
        )

    def predict(self, X):
        self._ensure_fitted()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]
        result = np.empty(n_samples, dtype=np.float64)

        for i in range(n_samples):
            out = self._instance.forward(x=X[i])
            logits = out['output']
            pred_idx = int(np.argmax(logits))
            result[i] = self._classes[pred_idx] if pred_idx < len(self._classes) else pred_idx

        return result

    def predict_proba(self, X):
        self._ensure_fitted()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]
        nc = self._nr_class
        result = np.empty(n_samples * nc, dtype=np.float64)

        for i in range(n_samples):
            out = self._instance.forward(x=X[i])
            logits = out['output']
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            result[i * nc:(i + 1) * nc] = probs

        return result

    def score(self, X, y):
        preds = self.predict(X)
        y = np.asarray(y, dtype=np.float64)
        return float(np.mean(preds == y))

    def save(self):
        self._ensure_fitted()
        ir_bytes = self._instance.export_ir()
        w_bytes = self._instance.export_weights()

        return encode_bundle(
            {
                'typeId': 'wlearn.nn.nam.classifier@1',
                'params': self.get_params(),
                'metadata': {
                    'nrClass': self._nr_class,
                    'classes': self._classes.tolist(),
                    'nFeatures': self._n_features,
                },
            },
            [
                {'id': 'ir', 'data': ir_bytes},
                {'id': 'weights', 'data': w_bytes},
            ],
        )

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        if self._instance:
            self._instance.free()
            self._instance = None
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
    def classes(self):
        return np.array(self._classes, dtype=np.int32)

    @property
    def capabilities(self):
        return {
            'classifier': True, 'regressor': False,
            'predictProba': True, 'decisionFunction': False,
            'sampleWeight': False, 'csr': False, 'earlyStopping': True,
        }

    @classmethod
    def default_search_space(cls):
        return {
            'hidden_sizes': {'type': 'categorical',
                             'values': [[32], [64], [64, 32], [128]]},
            'activation': {'type': 'categorical',
                           'values': ['exu', 'relu', 'gelu']},
            'lr': {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-1},
            'epochs': {'type': 'int_uniform', 'low': 10, 'high': 200},
            'optimizer': {'type': 'categorical', 'values': ['adam', 'sgd']},
        }

    def _ensure_fitted(self):
        if self._disposed:
            raise DisposedError('NAMClassifier has been disposed.')
        if not self._fitted:
            raise NotFittedError('NAMClassifier is not fitted.')


class NAMRegressor:
    """NAM regressor (Neural Additive Model) using polygrad Instance backend."""

    def __init__(self, instance, params, n_features=0):
        self._instance = instance
        self._params = dict(params)
        self._n_features = n_features
        self._fitted = True
        self._disposed = False

    @classmethod
    def create(cls, params=None):
        obj = cls.__new__(cls)
        obj._instance = None
        obj._params = dict(params) if params else {}
        obj._n_features = 0
        obj._fitted = False
        obj._disposed = False
        return obj

    def fit(self, X, y):
        if self._disposed:
            raise DisposedError('NAMRegressor has been disposed.')

        Instance = _get_instance_class()
        from polygrad.instance import OPTIM_SGD, OPTIM_ADAM

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples, n_features = X.shape
        n_outputs = y.shape[1]
        self._n_features = n_features

        hidden = self._params.get('hidden_sizes', [64])
        activation = self._params.get('activation', 'exu')
        seed = self._params.get('seed', 42)
        lr = self._params.get('lr', 0.01)
        epochs = self._params.get('epochs', 100)
        optimizer = self._params.get('optimizer', 'adam')

        spec = _make_nam_spec(n_features, hidden, n_outputs,
                              activation, 'mse', seed)

        if self._instance:
            self._instance.free()
        self._instance = Instance.nam(spec)

        optim_kind = OPTIM_ADAM if optimizer == 'adam' else OPTIM_SGD
        self._instance.set_optimizer(optim_kind, lr=lr)

        for epoch in range(epochs):
            for i in range(n_samples):
                xi = X[i:i+1].flatten()
                yi = y[i].flatten()
                self._instance.train_step(x=xi, y=yi)

        self._fitted = True
        return self

    @staticmethod
    def _from_bundle(manifest, toc, blobs):
        Instance = _get_instance_class()

        ir_entry = next((e for e in toc if e['id'] == 'ir'), None)
        w_entry = next((e for e in toc if e['id'] == 'weights'), None)
        if ir_entry is None or w_entry is None:
            raise ValueError('Bundle missing "ir" or "weights" artifact')

        ir_bytes = bytes(blobs[ir_entry['offset']:
                               ir_entry['offset'] + ir_entry['length']])
        w_bytes = bytes(blobs[w_entry['offset']:
                              w_entry['offset'] + w_entry['length']])

        instance = Instance.from_ir(ir_bytes, w_bytes)
        params = manifest.get('params', {})
        meta = manifest.get('metadata', {})
        return NAMRegressor(
            instance, params,
            n_features=meta.get('nFeatures', 0),
        )

    def predict(self, X):
        self._ensure_fitted()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]
        results = []
        for i in range(n_samples):
            out = self._instance.forward(x=X[i])
            results.append(out['output'].copy())

        result = np.array(results, dtype=np.float64)
        if result.shape[1] == 1:
            return result.flatten()
        return result

    def score(self, X, y):
        preds = self.predict(X)
        y = np.asarray(y, dtype=np.float64)
        y_mean = y.mean()
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        return 0.0 if ss_tot == 0 else float(1 - ss_res / ss_tot)

    def save(self):
        self._ensure_fitted()
        ir_bytes = self._instance.export_ir()
        w_bytes = self._instance.export_weights()

        return encode_bundle(
            {
                'typeId': 'wlearn.nn.nam.regressor@1',
                'params': self.get_params(),
                'metadata': {
                    'nFeatures': self._n_features,
                },
            },
            [
                {'id': 'ir', 'data': ir_bytes},
                {'id': 'weights', 'data': w_bytes},
            ],
        )

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        if self._instance:
            self._instance.free()
            self._instance = None
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
    def capabilities(self):
        return {
            'classifier': False, 'regressor': True,
            'predictProba': False, 'decisionFunction': False,
            'sampleWeight': False, 'csr': False, 'earlyStopping': True,
        }

    @classmethod
    def default_search_space(cls):
        return {
            'hidden_sizes': {'type': 'categorical',
                             'values': [[32], [64], [64, 32], [128]]},
            'activation': {'type': 'categorical',
                           'values': ['exu', 'relu', 'gelu']},
            'lr': {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-1},
            'epochs': {'type': 'int_uniform', 'low': 10, 'high': 200},
            'optimizer': {'type': 'categorical', 'values': ['adam', 'sgd']},
        }

    def _ensure_fitted(self):
        if self._disposed:
            raise DisposedError('NAMRegressor has been disposed.')
        if not self._fitted:
            raise NotFittedError('NAMRegressor is not fitted.')


register('wlearn.nn.mlp.classifier@1', MLPClassifier._from_bundle)
register('wlearn.nn.mlp.regressor@1', MLPRegressor._from_bundle)
register('wlearn.nn.tabm.classifier@1', TabMClassifier._from_bundle)
register('wlearn.nn.tabm.regressor@1', TabMRegressor._from_bundle)
register('wlearn.nn.nam.classifier@1', NAMClassifier._from_bundle)
register('wlearn.nn.nam.regressor@1', NAMRegressor._from_bundle)
