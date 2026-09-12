"""Independent target heads; constant binary heads preserve the label axes."""
import numpy as np

from ..targets import normalize_targets, validate_sample_weight
from ..task import task_params, validate_estimator_task
from ..errors import ValidationError, NotFittedError, DisposedError
from ..bundle import encode_bundle, validate_bundle, write_bundle_output
from ..registry import register, load, _load_with_context, assert_required_loaders
from ..prediction import create_prediction, validate_prediction
from ..cv import r2_score
from ._class_order import class_column_map, require_probability_model, validate_probability_output, validate_regression_output

_TYPES = {'multioutput': 'wlearn.ensemble.multioutput.regressor@1',
          'multilabel': 'wlearn.ensemble.multilabel.classifier@1'}


def _dispose(models, suppress=False):
    first = None
    for model in models:
        if model is not None:
            try:
                model.dispose()
            except Exception as error:
                first = first or error
    if first is not None and not suppress:
        raise first


def _names(names, count):
    names = [f'target_{i}' for i in range(count)] if names is None else names
    if not isinstance(names, (list, tuple)) or len(names) != count or any(not isinstance(v, str) or not v for v in names) or len(set(names)) != count:
        raise ValidationError('target_names must uniquely name every target')
    return list(names)


def _spec(spec):
    if not isinstance(spec, (list, tuple)) or not 2 <= len(spec) <= 3 or not isinstance(spec[0], str) or not spec[0] or not callable(getattr(spec[1], 'create', None)):
        raise ValidationError('estimator must be (name, EstimatorClass, params); loaded composites need a new specification before refitting')
    params = spec[2] if len(spec) == 3 else {}
    if params is not None and not isinstance(params, dict):
        raise ValidationError('estimator params must be a dict')
    return (spec[0], spec[1], dict(params or {}))


class _MultiTarget:
    _kind = None

    def __init__(self, estimator=None, target_names=None, task=None):
        if task is not None and task != self._kind:
            raise ValidationError(f'task must be {self._kind}')
        self._spec = _spec(estimator) if estimator is not None else None
        self._names = list(target_names) if target_names is not None else None
        self._models, self._constants, self._columns = [], [], []
        self._features = 0
        self._fitted = self._disposed = False

    @classmethod
    def create(cls, params=None, **kwargs):
        return cls(**{**(params or {}), **kwargs})

    def _alive(self):
        if self._disposed:
            raise DisposedError('Multi-target estimator has been disposed')

    def _ready(self):
        self._alive()
        if not self._fitted:
            raise NotFittedError('Multi-target estimator is not fitted')

    def fit(self, X, y, sample_weight=None):
        self._alive()
        name, cls, params = _spec(self._spec)
        xn = np.asarray(X)
        yn = normalize_targets(y, self._kind)
        if xn.ndim != 2 or xn.shape[1] < 1 or len(xn) != len(yn):
            raise ValidationError('X and target rows must match')
        names = _names(self._names, yn.shape[1])
        weights = None if sample_weight is None else validate_sample_weight(sample_weight, len(yn))
        task = 'classification' if self._kind == 'multilabel' else 'regression'
        task_params(params, task)
        models, constants, columns = [], [], []
        try:
            for t in range(yn.shape[1]):
                column = np.ascontiguousarray(yn[:, t])
                constant = int(column[0]) if task == 'classification' and np.all(column == column[0]) else None
                constants.append(constant)
                if constant is not None:
                    models.append(None)
                    columns.append(None)
                    continue
                model = cls.create(task_params(params, task))
                models.append(model)
                validate_estimator_task(model, task)
                if task == 'classification':
                    require_probability_model(model, name)
                if weights is not None and not model.capabilities.get('sampleWeight'):
                    raise ValidationError(f'{name} does not support sample_weight')
                model.fit(xn, column, **({'sample_weight': weights} if weights is not None else {}))
                validate_estimator_task(model, task)
                columns.append(class_column_map(model, [0, 1], name) if task == 'classification' else None)
        except Exception:
            _dispose(models, True)
            raise
        previous = self._models
        self._models, self._constants, self._columns = models, constants, columns
        self._names, self._features, self._fitted = names, xn.shape[1], True
        _dispose(previous, True)
        return self

    def _matrix(self, X):
        self._ready()
        a = np.asarray(X)
        if a.ndim != 2 or a.shape[1] != self._features or not len(a):
            raise ValidationError('Prediction feature count differs from training')
        return a

    def predict(self, X):
        if self._kind == 'multilabel':
            return (self.predict_proba(X) >= 0.5).astype(np.int32)
        xn = self._matrix(X)
        return np.column_stack([validate_regression_output(m.predict(xn), len(xn), self._names[t]) for t, m in enumerate(self._models)])

    def predict_proba(self, X):
        if self._kind != 'multilabel':
            raise ValidationError('Multioutput regression does not support probabilities')
        xn = self._matrix(X)
        result = np.empty((len(xn), len(self._models)), dtype=np.float64)
        for t, model in enumerate(self._models):
            if model is None:
                result[:, t] = self._constants[t]
            else:
                p = validate_probability_output(model.predict_proba(xn), len(xn), 2, self._names[t])
                result[:, t] = np.asarray(p).reshape(len(xn), 2)[:, self._columns[t][1]]
        return result

    def predict_quantiles(self, X, levels):
        xn = self._matrix(X)
        if not self.capabilities['predictQuantiles']:
            raise ValidationError('Every target estimator must support quantiles')
        levels = create_prediction(rows=1, quantile_levels=levels,
                                   quantiles=np.zeros(len(levels))).quantile_levels.copy()
        outputs = [m.predict_quantiles(xn, levels) for m in self._models]
        for p in outputs:
            validate_prediction(p)
            if p.rows != len(xn) or p.target_count != 1 or p.quantiles is None or not np.array_equal(p.quantile_levels, levels):
                raise ValidationError('Target quantile dimensions and levels must match the request')
        result = np.stack([p.quantiles.reshape(len(xn), len(levels)) for p in outputs], axis=1)
        return create_prediction(rows=len(xn), task_kind='multioutput', target_count=len(outputs),
                                 target_names=self.target_names, quantiles=result.ravel(), quantile_levels=levels)

    def score(self, X, y):
        yn = normalize_targets(y, self._kind)
        pred = self.predict(X)
        if pred.shape != yn.shape:
            raise ValidationError('Score target shape mismatch')
        if self._kind == 'multilabel':
            return float(np.mean(np.all(pred == yn, axis=1)))
        return float(np.mean([r2_score(yn[:, t], pred[:, t]) for t in range(yn.shape[1])]))

    @property
    def target_names(self):
        return None if self._names is None else list(self._names)

    @property
    def target_count(self):
        return len(self._models) or len(self._names or []) or None

    @property
    def is_fitted(self):
        return self._fitted and not self._disposed

    @property
    def capabilities(self):
        return dict(classifier=self._kind == 'multilabel', regressor=self._kind == 'multioutput',
                    multioutput=self._kind == 'multioutput', multilabel=self._kind == 'multilabel',
                    predictProba=self._kind == 'multilabel', decisionFunction=False, sampleWeight=True,
                    csr=False, earlyStopping=False,
                    predictQuantiles=self._kind == 'multioutput' and bool(self._models) and all(m.capabilities.get('predictQuantiles') for m in self._models))

    def get_params(self):
        return dict(task=self._kind, target_names=self.target_names,
                    estimator_name=self._spec[0] if self._spec else None,
                    estimator_params=dict(self._spec[2]) if self._spec else {})

    def set_params(self, params):
        self._alive()
        if not isinstance(params, dict) or set(params) - {'estimator', 'target_names'}:
            raise ValidationError('Supported params are estimator and target_names')
        spec = _spec(params['estimator']) if params.get('estimator') is not None else self._spec
        names = params.get('target_names', self._names)
        if names is not None:
            names = _names(names, len(names))
        previous = self._models
        self._fitted, self._models, self._spec, self._names = False, [], spec, names
        _dispose(previous)
        return self

    def dispose(self):
        if self._disposed:
            return
        self._disposed, self._fitted = True, False
        previous, self._models = self._models, []
        _dispose(previous)

    def save(self, path=None):
        self._ready()
        artifacts, requires = [], set()
        for t, model in enumerate(self._models):
            if model is None:
                continue
            data = model.save()
            manifest, _, _ = validate_bundle(data)
            requires.add(manifest['typeId'])
            requires.update(manifest.get('requires', []))
            artifacts.append(dict(id=f'target_{t}', data=data, mediaType='application/x-wlearn-bundle'))
        p = self.get_params()
        params = dict(task=self._kind, targetNames=self.target_names, estimatorName=p['estimator_name'],
                      estimatorParams=p['estimator_params'], nFeatures=self._features, constants=self._constants)
        data = encode_bundle(dict(typeId=_TYPES[self._kind], requires=sorted(requires), params=params), artifacts)
        return write_bundle_output(data, path)

    @classmethod
    def load(cls, data, **kwargs):
        manifest, _, _ = validate_bundle(data)
        if manifest['typeId'] != _TYPES[cls._kind]:
            raise ValidationError('Multi-target bundle type mismatch')
        return load(data, **kwargs)

    @staticmethod
    def _restore(manifest, toc, blobs, context):
        kind = 'multilabel' if manifest['typeId'] == _TYPES['multilabel'] else 'multioutput'
        p = manifest.get('params')
        if not isinstance(p, dict) or p.get('task') != kind or not isinstance(p.get('constants'), list) or not p['constants'] or type(p.get('nFeatures')) is not int or p['nFeatures'] < 1:
            raise ValidationError('Invalid multi-target manifest')
        if not isinstance(p.get('targetNames'), list) or p['nFeatures'] > 2**53 - 1:
            raise ValidationError('Invalid target names or feature count')
        names = _names(p['targetNames'], len(p['constants']))
        if any(v is not None and not (kind == 'multilabel' and type(v) is int and v in (0, 1)) for v in p['constants']):
            raise ValidationError('Invalid constant target')
        ids = [f'target_{i}' for i, v in enumerate(p['constants']) if v is None]
        if len(toc) != len(ids) or any(e['id'] not in ids or e.get('mediaType') != 'application/x-wlearn-bundle' for e in toc):
            raise ValidationError('Invalid multi-target artifacts')
        if p.get('estimatorName') is not None and (not isinstance(p['estimatorName'], str) or not isinstance(p.get('estimatorParams'), dict)):
            raise ValidationError('Invalid estimator description')
        assert_required_loaders(manifest)
        result = MultiLabelClassifier() if kind == 'multilabel' else MultiOutputRegressor()
        result._names, result._constants, result._features = names, p['constants'], p['nFeatures']
        result._spec = (p['estimatorName'], None, p['estimatorParams']) if p.get('estimatorName') else None
        try:
            for t in range(len(names)):
                if p['constants'][t] is not None:
                    result._models.append(None)
                    result._columns.append(None)
                    continue
                entry = next(e for e in toc if e['id'] == f'target_{t}')
                model = _load_with_context(bytes(blobs[entry['offset']:entry['offset']+entry['length']]), context)
                result._models.append(model)
                validate_estimator_task(model, 'classification' if kind == 'multilabel' else 'regression')
                if kind == 'multilabel':
                    require_probability_model(model, names[t])
                result._columns.append(class_column_map(model, [0, 1], names[t]) if kind == 'multilabel' else None)
            result._fitted = True
            return result
        except Exception:
            _dispose(result._models, True)
            raise


class MultiOutputRegressor(_MultiTarget):
    _kind = 'multioutput'


class MultiLabelClassifier(_MultiTarget):
    _kind = 'multilabel'


for _type in _TYPES.values():
    register(_type, _MultiTarget._restore, accepts_context=True)
