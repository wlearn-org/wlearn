"""VotingEnsemble matching JS @wlearn/ensemble/voting.js."""

import numpy as np

from ..errors import ValidationError, NotFittedError, DisposedError
from ..bundle import encode_bundle, validate_bundle, write_bundle_output
from ..registry import (
    register, load as registry_load, _load_with_context,
    assert_required_loaders,
)
from ..automl._cv import accuracy, r2_score

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
        self._weights = np.array(weights, dtype=np.float64) if weights is not None else None
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

        classes = self._classes
        if self._task == 'classification':
            labels = sorted(set(int(v) for v in y))
            classes = np.array(labels, dtype=np.int32)

        weights = self._weights
        if weights is None:
            n = len(self._specs)
            weights = np.full(n, 1.0 / n, dtype=np.float64)

        models = []
        try:
            for _name, est_cls, params in self._specs:
                model = est_cls.create(params or {})
                models.append(model)
                model.fit(X, y)
        except Exception as exc:
            _dispose_owned(models, exc)
            raise

        previous = self._models or []
        self._models = models
        self._classes = classes
        self._weights = weights
        self._fitted = True
        _dispose_owned(previous)
        return self

    def predict(self, X):
        self._ensure_fitted()
        n = len(X)

        if self._task == 'regression':
            return self._weighted_average(X, n)

        if self._voting == 'soft':
            proba = self.predict_proba(X)
            nc = len(self._classes)
            out = np.zeros(n, dtype=np.float64)
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
            proba = self._models[m].predict_proba(X)
            w = self._weights[m]
            for i in range(n * nc):
                out[i] += w * proba[i]

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
        _dispose_owned(self._models or [])

    def get_params(self):
        return {
            'task': self._task,
            'voting': self._voting,
            'weights': list(self._weights) if self._weights is not None else None,
            'estimatorNames': [s[0] for s in self._specs],
        }

    def set_params(self, p):
        self._ensure_alive()
        if 'voting' in p:
            self._voting = p['voting']
        if 'weights' in p:
            self._weights = np.array(p['weights'], dtype=np.float64)
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
        return self._fitted

    @property
    def classes(self):
        return self._classes

    def _weighted_average(self, X, n):
        out = np.zeros(n, dtype=np.float64)
        for m in range(len(self._models)):
            preds = self._models[m].predict(X)
            w = self._weights[m]
            for i in range(n):
                out[i] += w * float(preds[i])
        return out

    def _majority_vote(self, X, n):
        out = np.zeros(n, dtype=np.float64)
        nc = len(self._classes)
        for i in range(n):
            votes = np.zeros(nc, dtype=np.float64)
            for m in range(len(self._models)):
                pred = float(self._models[m].predict(X)[i])
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
        p = manifest['params']
        expected_type_id = TYPE_ID_REG if p.get('task') == 'regression' \
            else TYPE_ID_CLS
        if manifest.get('typeId') != expected_type_id:
            raise ValidationError(
                f'VotingEnsemble.load expected typeId "{expected_type_id}", '
                f'got "{manifest.get("typeId")}"')
        if not isinstance(p.get('estimatorNames'), list):
            raise ValidationError(
                'VotingEnsemble manifest must declare estimatorNames')
        assert_required_loaders(manifest)
        ens = VotingEnsemble(
            task=p['task'],
            voting=p['voting'],
            weights=p['weights'],
        )
        ens._classes = np.array(p['classes'], dtype=np.int32) if p.get('classes') else None
        ens._specs = [(name, None, None) for name in p['estimatorNames']]
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
            ens._fitted = True
            return ens
        except Exception:
            _dispose_loaded(ens._models)
            raise


def _dispose_loaded(models):
    _dispose_owned(models, RuntimeError('preserve load error'))


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
