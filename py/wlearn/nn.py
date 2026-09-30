"""Tabular estimators over Polygrad Model, with WLRN @2 persistence."""

from copy import deepcopy
import numpy as np

from .errors import (ValidationError, BundleError, BackendError,
                     NotFittedError, DisposedError)
from .bundle import encode_bundle, decode_bundle, write_bundle_output
from .registry import register


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValidationError(f'{name} must be a positive integer')
    return int(value)


def _matrix(X):
    X = np.ascontiguousarray(X, dtype=np.float32)
    if X.ndim != 2 or not all(X.shape) or not np.isfinite(X).all():
        raise ValidationError('X must be a nonempty finite float32 matrix')
    return X


def _probabilities(values):
    result = np.exp(values - values.max(axis=1, keepdims=True))
    return result / result.sum(axis=1, keepdims=True)


def _forward(model, X, batch, outputs):
    result = np.empty((len(X), outputs), dtype=np.float64)
    # Inference padding is safe for these independent-row tabular families.
    data = np.zeros((batch, X.shape[1]), dtype=np.float32)
    for start in range(0, len(X), batch):
        count = min(batch, len(X) - start)
        data.fill(0)
        data[:count] = X[start:start + count]
        values = model.forward(x=data)['output']
        if values.shape != (batch, outputs) or not np.isfinite(values).all():
            raise BackendError('Polygrad returned invalid predictions')
        result[start:start + count] = values[:count]
    return result


class _NeuralEstimator:
    def __init__(self, params=None):
        params = dict(params or {})
        self._polygrad = params.pop('polygrad', None)
        self._params = deepcopy(params)
        self._runtime = None
        self._owns_runtime = False
        self._model = None
        self._disposed = False
        self._classes = []

    @classmethod
    def create(cls, params=None):
        return cls(params)

    def _acquire(self):
        if self._runtime is None:
            import polygrad
            if self._polygrad is not None and hasattr(self._polygrad, 'Model'):
                self._runtime = self._polygrad
            else:
                self._runtime = polygrad.create(**(self._polygrad or {}))
                self._owns_runtime = True
        return self._runtime

    def _check(self, fitted=True):
        if self._disposed:
            raise DisposedError()
        if fitted and self._model is None:
            raise NotFittedError()

    def fit(self, X, y):
        self._check(False)
        X = _matrix(X)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim != 1 or len(y) != len(X) or not np.isfinite(y).all():
            raise ValidationError('y must have one finite value per row')
        p = self._params
        epochs = _positive(p.get('epochs', 100), 'epochs')
        patience = _positive(p.get('patience', 10), 'patience')
        requested_batch = _positive(p.get('batch_size', 1), 'batch_size')
        fraction, lr = p.get('validation_fraction', 0), p.get('lr', .01)
        if not np.isfinite(lr) or lr <= 0:
            raise ValidationError('lr must be positive and finite')
        if not np.isfinite(fraction) or not 0 <= fraction < 1:
            raise ValidationError('validation_fraction must be in [0, 1)')
        n_val = max(1, int(len(X) * fraction)) if fraction else 0
        n_train = len(X) - n_val
        if n_train == 0:
            raise ValidationError('Validation split leaves no training rows')
        batch = min(requested_batch, n_train)
        if n_train % batch:
            raise ValidationError('Polygrad tabular training requires complete fixed-size batches; choose batch_size dividing the training rows (or 1)')
        if self._family == 'TabM' and batch != 1:
            raise ValidationError('Polygrad 0.6 TabM currently requires batch_size=1')
        optimizer = p.get('optimizer', 'adam')
        if optimizer not in ('adam', 'sgd'):
            raise ValidationError('optimizer must be adam or sgd')
        classes = np.unique(y).tolist() if self._classifier else []
        if self._classifier and len(classes) < 2:
            raise ValidationError('Classification requires at least two classes')
        outputs = len(classes) or 1
        target = (np.eye(outputs, dtype=np.float32)[np.searchsorted(classes, y)]
                  if classes else y.astype(np.float32).reshape(-1, 1))
        if not np.isfinite(target).all():
            raise ValidationError('y exceeds float32 range')
        hidden = p.get('hidden_sizes', p.get('hiddenSizes', [64]))
        spec = dict(activation=p.get('activation', 'exu' if self._family == 'NAM' else 'relu'),
                    loss='cross_entropy' if classes else 'mse', batch_size=batch,
                    seed=p.get('seed', 42))
        if self._family == 'NAM':
            spec.update(n_features=X.shape[1], hidden_sizes=hidden, n_outputs=outputs)
        else:
            spec.update(layers=[X.shape[1], *hidden, outputs], n_ensemble=p.get('n_ensemble', 32))
        model = None
        try:
            model = getattr(self._acquire().models, self._family)(spec)
            model.set_optimizer(optimizer, lr=lr)
            rng = np.random.RandomState(p.get('seed', 42))
            best_loss, best_weights, stale = float('inf'), None, 0
            for _ in range(epochs):
                order = rng.permutation(n_train)
                for start in range(0, n_train, batch):
                    indices = order[start:start + batch]
                    loss = model.train_step(x=X[indices], y=target[indices])
                    if not np.isfinite(loss):
                        raise BackendError('Non-finite training loss')
                if n_val:
                    values = _forward(model, X[n_train:], batch, outputs)
                    if classes:
                        probs = _probabilities(values)
                        loss = -np.log(np.maximum(1e-15, probs[np.arange(n_val), np.searchsorted(classes, y[n_train:])])).mean()
                    else:
                        loss = np.square(values[:, 0] - y[n_train:]).mean()
                    if loss < best_loss:
                        best_loss, stale = loss, 0
                        best_weights = model.export_weights(include_optimizer=False)
                    else:
                        stale += 1
                        if stale >= patience:
                            break
            # Restore even when the epoch budget expires before patience.
            if best_weights is not None:
                model.import_weights(best_weights)
        except BaseException:
            if model is not None:
                model.dispose()
            raise
        if self._model is not None:
            self._model.dispose()
        self._model, self._classes = model, classes
        self._n_features, self._batch_size = X.shape[1], batch
        return self

    def _predict(self, X):
        self._check()
        X = _matrix(X)
        if X.shape[1] != self._n_features:
            raise ValidationError(f'Expected {self._n_features} features, got {X.shape[1]}')
        return _forward(self._model, X, self._batch_size, len(self._classes) or 1)

    def predict(self, X):
        values = self._predict(X)
        if self._classifier:
            return np.asarray(self._classes, dtype=np.float64)[values.argmax(axis=1)]
        return values[:, 0]

    def score(self, X, y):
        pred = self.predict(X)
        y = np.asarray(y, dtype=np.float64)
        if y.shape != pred.shape:
            raise ValidationError('y shape does not match predictions')
        if self._classifier:
            return float(np.mean(pred == y))
        total = np.square(y - y.mean()).sum()
        return float(1 - np.square(y - pred).sum() / total) if total else 0.0

    @classmethod
    def _type_id(cls, version=2):
        task = 'classifier' if cls._classifier else 'regressor'
        return f'wlearn.nn.{cls._family.lower()}.{task}@{version}'

    def save(self, path=None):
        self._check()
        bundle = encode_bundle(dict(
            typeId=self._type_id(), params=self.get_params(), metadata=dict(
                nFeatures=self._n_features, batchSize=self._batch_size,
                nrClass=len(self._classes), classes=self._classes)),
            [dict(id='model', data=self._model.save_bundle(include_optimizer=False))])
        return write_bundle_output(bundle, path)

    @classmethod
    def load(cls, source, *, polygrad=None):
        return cls._from_bundle(*decode_bundle(source), polygrad=polygrad)

    @classmethod
    def _from_bundle(cls, manifest, toc, blobs, *, polygrad=None):
        if manifest['typeId'] != cls._type_id():
            raise BundleError('NN requires a @2 Model bundle; legacy @1 Instance bundles must be retrained')
        entry = next((e for e in toc if e['id'] == 'model'), None)
        meta = manifest.get('metadata', {})
        if (entry is None or any(type(meta.get(k)) is not int or meta[k] < 1
                                 for k in ('nFeatures', 'batchSize'))
                or not isinstance(meta.get('classes'), list)
                or not all(type(v) in (float, int) and np.isfinite(v) for v in meta['classes'])
                or len(set(meta['classes'])) != len(meta['classes'])
                or (not cls._classifier and meta['classes'])
                or meta.get('nrClass') != len(meta['classes'])
                or (cls._classifier and meta['nrClass'] < 2)):
            raise BundleError('Invalid NN Model bundle metadata')
        result = cls.create({**manifest.get('params', {}), 'polygrad': polygrad})
        try:
            data = bytes(blobs[entry['offset']:entry['offset'] + entry['length']])
            result._model = result._acquire().Model.from_bundle(data)
            bindings = {b['name']: b for b in result._model.bindings()}
            for name, shape in [('x', [meta['batchSize'], meta['nFeatures']]),
                                ('output', [meta['batchSize'], meta['nrClass'] or 1])]:
                if (name not in bindings or list(bindings[name]['shape']) != shape
                        or bindings[name]['dtype'] != 'float32'):
                    raise BundleError('NN metadata does not match the Polygrad signature')
            result._n_features, result._batch_size = meta['nFeatures'], meta['batchSize']
            result._classes = list(meta['classes'])
            return result
        except BaseException:
            result.dispose()
            raise

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        if self._model is not None:
            self._model.dispose()
            self._model = None
        if self._owns_runtime:
            self._runtime.dispose()

    def get_params(self):
        return deepcopy(self._params)

    def set_params(self, params):
        self._check(False)
        self._params.update({k: v for k, v in params.items() if k != 'polygrad'})
        return self

    @property
    def is_fitted(self):
        return self._model is not None and not self._disposed

    @property
    def capabilities(self):
        return dict(classifier=self._classifier, regressor=not self._classifier,
                    predictProba=self._classifier, decisionFunction=False,
                    sampleWeight=False, csr=False, earlyStopping=True)

    @classmethod
    def default_search_space(cls):
        return deepcopy(_SEARCH_SPACES[cls._family])


class _Classifier(_NeuralEstimator):
    _classifier = True

    def predict_proba(self, X):
        return _probabilities(self._predict(X)).reshape(-1)

    @property
    def classes(self):
        return list(self._classes)

    @property
    def nr_class(self):
        return len(self._classes)


class _Regressor(_NeuralEstimator):
    _classifier = False


class MLPClassifier(_Classifier):
    _family = 'MLP'


class MLPRegressor(_Regressor):
    _family = 'MLP'


class TabMClassifier(_Classifier):
    _family = 'TabM'


class TabMRegressor(_Regressor):
    _family = 'TabM'


class NAMClassifier(_Classifier):
    _family = 'NAM'


class NAMRegressor(_Regressor):
    _family = 'NAM'


def _legacy_bundle(*args):
    raise BundleError('Legacy NN @1 Instance bundles require retraining for Polygrad 0.6')


for _cls in (MLPClassifier, MLPRegressor, TabMClassifier, TabMRegressor,
             NAMClassifier, NAMRegressor):
    register(_cls._type_id(), _cls._from_bundle)
    register(_cls._type_id(1), _legacy_bundle)

_SEARCH_SPACES = {'MLP': {'activation': {'type': 'categorical', 'values': ['relu', 'gelu', 'silu']},
         'epochs': {'high': 200, 'low': 10, 'type': 'int_uniform'},
         'hidden_sizes': {'type': 'categorical',
                          'values': [[64], [128], [64, 64], [128, 64]]},
         'lr': {'high': 0.1, 'low': 0.0001, 'type': 'log_uniform'},
         'optimizer': {'type': 'categorical', 'values': ['adam', 'sgd']}},
 'NAM': {'activation': {'type': 'categorical', 'values': ['exu', 'relu', 'gelu']},
         'epochs': {'high': 200, 'low': 10, 'type': 'int_uniform'},
         'hidden_sizes': {'type': 'categorical',
                          'values': [[32], [64], [64, 32], [128]]},
         'lr': {'high': 0.1, 'low': 0.0001, 'type': 'log_uniform'},
         'optimizer': {'type': 'categorical', 'values': ['adam', 'sgd']}},
 'TabM': {'activation': {'type': 'categorical', 'values': ['relu', 'gelu', 'silu']},
          'epochs': {'high': 200, 'low': 10, 'type': 'int_uniform'},
          'hidden_sizes': {'type': 'categorical',
                           'values': [[64], [128], [64, 64], [128, 64]]},
          'lr': {'high': 0.1, 'low': 0.0001, 'type': 'log_uniform'},
          'n_ensemble': {'type': 'categorical', 'values': [4, 8, 16, 32]},
          'optimizer': {'type': 'categorical', 'values': ['adam', 'sgd']}}}
