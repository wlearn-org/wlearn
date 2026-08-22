"""VotingEnsemble matching JS @wlearn/ensemble/voting.js."""

import math
import numpy as np

from ..errors import ValidationError, NotFittedError, DisposedError
from ..bundle import encode_bundle, validate_bundle, write_bundle_output
from ..registry import (
    register, load as registry_load, _load_with_context,
    assert_required_loaders,
)
from ..automl._cv import accuracy, r2_score
from ._manifest import validate_voting_manifest
from ._class_order import (
    class_column_map, require_probability_model, validate_label_output,
    validate_probability_output, validate_regression_output,
)

TYPE_ID_CLS = 'wlearn.ensemble.voting.classifier@1'
TYPE_ID_REG = 'wlearn.ensemble.voting.regressor@1'
_registered = False


class VotingEnsemble:
    def __init__(self, estimators=None, weights=None, voting='soft',
                 task='classification'):
        """
        Args:
            estimators: list of (name, cls, params) tuples
            weights: list/array of weights or None (equal weights)
            voting: 'soft' or 'hard'
            task: 'classification' or 'regression'
        """
        self._specs = estimators or []
        self._weights = weights
        self._voting = voting
        self._task = task
        self._models = None
        self._classes = None
        self._fitted = False
        self._disposed = False
        VotingEnsemble._register()

    @classmethod
    def create(cls, estimators=None, weights=None, voting='soft',
               task='classification'):
        return cls(estimators=estimators, weights=weights, voting=voting, task=task)

    def _ensure_alive(self):
        if self._disposed:
            raise DisposedError('VotingEnsemble has been disposed.')

    def _ensure_fitted(self):
        self._ensure_alive()
        if not self._fitted:
            raise NotFittedError('VotingEnsemble is not fitted. Call fit() first.')

    def fit(self, X, y):
        self._ensure_alive()
        weights = _validate_voting_config(
            self._specs, self._weights, self._voting, self._task)

        classes = self._classes
        if self._task == 'classification':
            labels = sorted(set(int(v) for v in y))
            classes = np.array(labels, dtype=np.int32)

        models = []
        try:
            for name, est_cls, params in self._specs:
                model = est_cls.create(params or {})
                models.append(model)
                model.fit(X, y)
                if (self._task == 'classification' and
                        self._voting == 'soft'):
                    _validate_soft_voting_model(
                        model, classes,
                        f'VotingEnsemble estimator "{name}"')
        except Exception as exc:
            _dispose_owned(models, exc)
            raise

        previous = self._models or []
        self._models = models
        self._classes = classes
        self._weights = weights
        self._fitted = True
        _dispose_replaced(previous)
        return self

    def predict(self, X):
        self._ensure_fitted()
        n = len(X)

        if self._task == 'regression':
            return self._weighted_average(X, n)

        if self._voting == 'soft':
            proba = self.predict_proba(X)
            nc = len(self._classes)
            out = np.zeros(n, dtype=np.int32)
            for i in range(n):
                best_c = 0
                best_v = -float('inf')
                for c in range(nc):
                    if proba[i * nc + c] > best_v:
                        best_v = proba[i * nc + c]
                        best_c = c
                out[i] = self._classes[best_c]
            return out

        return self._majority_vote(X, n)

    def predict_proba(self, X):
        self._ensure_fitted()
        if self._task != 'classification':
            raise ValidationError('predict_proba is only available for classification')
        if self._voting == 'hard':
            raise ValidationError('predict_proba requires voting="soft"')

        n = len(X)
        nc = len(self._classes)
        out = np.zeros(n * nc, dtype=np.float64)

        for m in range(len(self._models)):
            label = f'VotingEnsemble estimator "{self._specs[m][0]}"'
            proba = validate_probability_output(
                self._models[m].predict_proba(X), n, nc, label)
            columns = class_column_map(
                self._models[m], self._classes, label)
            w = self._weights[m]
            for row in range(n):
                for column in range(nc):
                    out[row * nc + column] += (
                        w * proba[row * nc + columns[column]])

        return out

    def score(self, X, y):
        self._ensure_fitted()
        preds = self.predict(X)
        if self._task == 'classification':
            return accuracy(y, preds)
        return r2_score(y, preds)

    def save(self, path=None):
        self._ensure_fitted()
        type_id = TYPE_ID_CLS if self._task == 'classification' else TYPE_ID_REG
        manifest = {
            'typeId': type_id,
            'params': {
                'task': self._task,
                'voting': self._voting,
                'weights': list(self._weights),
                'estimatorNames': [s[0] for s in self._specs],
                'classes': (
                    [int(value) for value in self._classes]
                    if self._classes is not None else None),
            },
        }
        artifacts = [
            {
                'id': self._specs[i][0],
                'data': self._models[i].save(),
                'mediaType': 'application/x-wlearn-bundle',
            }
            for i in range(len(self._models))
        ]
        return write_bundle_output(encode_bundle(manifest, artifacts), path)

    @classmethod
    def load(cls, data, *, loader_options=None):
        manifest, _, _ = validate_bundle(data)
        if manifest.get('typeId') not in (TYPE_ID_CLS, TYPE_ID_REG):
            raise ValidationError(
                f'VotingEnsemble.load expected typeId "{TYPE_ID_CLS}" or '
                f'"{TYPE_ID_REG}", '
                f'got "{manifest.get("typeId")}"')
        cls._register()
        return registry_load(data, loader_options=loader_options)

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        try:
            _dispose_owned(self._models or [])
        finally:
            self._models = None
            self._fitted = False

    def get_params(self):
        return {
            'task': self._task,
            'voting': self._voting,
            'weights': list(self._weights) if self._weights is not None else None,
            'estimatorNames': [s[0] for s in self._specs],
        }

    def set_params(self, p):
        self._ensure_alive()
        if not isinstance(p, dict):
            raise ValidationError('VotingEnsemble params must be a dict')
        unknown = set(p).difference(('voting', 'weights'))
        if unknown:
            raise ValidationError(
                f'Unknown VotingEnsemble parameter "{next(iter(unknown))}"')
        voting = p.get('voting', self._voting)
        requested_weights = p.get('weights', self._weights)
        weights = _validate_voting_config(
            self._specs, requested_weights, voting, self._task,
            require_constructors=False)
        if (self._fitted and self._task == 'classification' and
                voting == 'soft'):
            for index, model in enumerate(self._models):
                _validate_soft_voting_model(
                    model, self._classes,
                    f'VotingEnsemble estimator "{self._specs[index][0]}"')
        self._voting = voting
        self._weights = weights
        return self

    @property
    def capabilities(self):
        return {
            'classifier': self._task == 'classification',
            'regressor': self._task == 'regression',
            'predictProba': self._task == 'classification' and self._voting == 'soft',
            'decisionFunction': False,
            'sampleWeight': False,
            'csr': False,
            'earlyStopping': False,
        }

    @property
    def is_fitted(self):
        return self._fitted and not self._disposed

    @property
    def classes(self):
        return self._classes

    def _weighted_average(self, X, n):
        out = np.zeros(n, dtype=np.float64)
        for m in range(len(self._models)):
            preds = validate_regression_output(
                self._models[m].predict(X), n,
                f'VotingEnsemble estimator "{self._specs[m][0]}"')
            w = self._weights[m]
            for i in range(n):
                out[i] += w * float(preds[i])
        return out

    def _majority_vote(self, X, n):
        out = np.zeros(n, dtype=np.int32)
        nc = len(self._classes)
        predictions = [
            validate_label_output(
                model.predict(X), n, self._classes,
                f'VotingEnsemble estimator "{self._specs[index][0]}"')
            for index, model in enumerate(self._models)
        ]
        for i in range(n):
            votes = np.zeros(nc, dtype=np.float64)
            for m in range(len(predictions)):
                pred = int(predictions[m][i])
                for c in range(nc):
                    if self._classes[c] == pred:
                        votes[c] += self._weights[m]
                        break
            best_c = int(np.argmax(votes))
            out[i] = self._classes[best_c]
        return out

    @staticmethod
    def _register():
        global _registered
        if _registered:
            return
        _registered = True

        def loader(manifest, toc, blobs, context):
            return VotingEnsemble._load_from_parts(
                manifest, toc, blobs, context)

        register(TYPE_ID_CLS, loader, accepts_context=True)
        register(TYPE_ID_REG, loader, accepts_context=True)

    @staticmethod
    def _load_from_parts(manifest, toc, blobs, context):
        p = validate_voting_manifest(
            manifest, toc, TYPE_ID_CLS, TYPE_ID_REG)
        assert_required_loaders(manifest)
        specs = [(name, None, None) for name in p['estimatorNames']]
        weights = _validate_voting_config(
            specs, p['weights'], p['voting'], p['task'],
            require_constructors=False)
        ens = VotingEnsemble(
            task=p['task'],
            voting=p['voting'],
            weights=weights,
        )
        ens._classes = np.array(p['classes'], dtype=np.int32) if p.get('classes') else None
        ens._specs = specs
        ens._models = []
        try:
            for name in p['estimatorNames']:
                entry = next((t for t in toc if t['id'] == name), None)
                if entry is None:
                    raise ValidationError(
                        f'No artifact for estimator "{name}"')
                blob = bytes(
                    blobs[entry['offset']:entry['offset'] + entry['length']])
                model = _load_with_context(blob, context)
                ens._models.append(model)
                if (ens._task == 'classification' and
                        ens._voting == 'soft'):
                    _validate_soft_voting_model(
                        model, ens._classes,
                        f'VotingEnsemble estimator "{name}"')
            ens._fitted = True
            return ens
        except Exception:
            _dispose_loaded(ens._models)
            raise


def _validate_soft_voting_model(model, classes, label):
    require_probability_model(model, label)
    return class_column_map(model, classes, label)


def _validate_voting_config(
        specs, weights, voting, task, *, require_constructors=True):
    if task not in ('classification', 'regression'):
        raise ValidationError(
            'VotingEnsemble task must be "classification" or "regression"')
    if voting not in ('soft', 'hard'):
        raise ValidationError(
            'VotingEnsemble voting must be "soft" or "hard"')
    if not isinstance(specs, (list, tuple)) or not specs:
        raise ValidationError(
            'VotingEnsemble estimators must be a nonempty sequence')
    names = set()
    for index, spec in enumerate(specs):
        if (not isinstance(spec, (list, tuple)) or len(spec) < 2 or
                not isinstance(spec[0], str) or not spec[0] or
                (require_constructors and
                 not callable(getattr(spec[1], 'create', None)))):
            raise ValidationError(
                f'VotingEnsemble estimator {index} has an invalid '
                'specification')
        if spec[0] in names:
            raise ValidationError(
                'VotingEnsemble estimator names must be unique')
        names.add(spec[0])
    if weights is None:
        return np.full(len(specs), 1.0 / len(specs), dtype=np.float64)
    try:
        raw = np.asarray(weights, dtype=object)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(
            'VotingEnsemble weights must be finite numbers') from exc
    if raw.ndim != 1 or len(raw) != len(specs):
        raise ValidationError(
            'VotingEnsemble weights must match the estimator count')
    if any(
            isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, float, np.integer, np.floating)) or
            not math.isfinite(float(value)) or float(value) < 0
            for value in raw):
        raise ValidationError(
            'VotingEnsemble weights must contain only nonnegative finite '
            'numbers')
    resolved = np.asarray(raw, dtype=np.float64)
    total = float(np.sum(resolved))
    if not math.isfinite(total) or total <= 0:
        raise ValidationError(
            'VotingEnsemble weights must have a positive finite sum')
    return resolved / total


def _dispose_loaded(models):
    _dispose_owned(models, RuntimeError('preserve load error'))


def _dispose_replaced(models):
    _dispose_owned(models, RuntimeError('replacement already committed'))


def _dispose_owned(models, operation_error=None):
    first_error = None
    for model in reversed(models):
        try:
            if hasattr(model, 'dispose'):
                model.dispose()
        except Exception as exc:
            if first_error is None:
                first_error = exc
    if operation_error is None and first_error is not None:
        raise first_error
