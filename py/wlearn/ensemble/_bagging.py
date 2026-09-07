"""BaggedEstimator: K-fold bagging with OOF storage.

Trains K copies of a base model on K folds, stores out-of-fold predictions,
and averages predictions at inference time. Supports multiple repeats with
different fold assignments.

typeIds:
  wlearn.ensemble.bagged.classifier@1
  wlearn.ensemble.bagged.regressor@1
"""


import numpy as np

from ..task import task_params, validate_estimator_task
from ..resampling import resolve_cv, serialize_cv, ResamplingPlan
from ..errors import ValidationError, NotFittedError, DisposedError
from ..bundle import encode_bundle, validate_bundle, write_bundle_output
from ..registry import (
    register, load as registry_load, _load_with_context,
    assert_required_loaders,
)
from ..cv import accuracy, r2_score, k_fold
from ._manifest import validate_bagging_manifest
from ._class_order import (
    class_column_map, require_probability_model,
    validate_probability_output, validate_regression_output,
)

TYPE_ID_CLS = 'wlearn.ensemble.bagged.classifier@1'
TYPE_ID_REG = 'wlearn.ensemble.bagged.regressor@1'
_registered = False


class BaggedEstimator:
    """K-fold bagged estimator with out-of-fold prediction storage.

    Trains K * n_repeats copies of a base model. Each repeat uses a different
    seed for fold assignment. OOF predictions are accumulated (sum + count)
    and averaged, matching AutoGluon's BaggedEnsembleModel pattern.
    """

    def __init__(self, estimator=None, k_fold=5, n_repeats=1,
                 task='classification', seed=42):
        """
        Args:
            estimator: (name, cls, params) tuple for the base model
            k_fold: number of CV folds per repeat
            n_repeats: number of bagging rounds (different fold assignments)
            task: 'classification' or 'regression'
            seed: random seed (each repeat uses seed + repeat_idx)
        """
        self._spec = estimator  # (name, cls, params)
        self._k_fold = k_fold
        self._n_repeats = n_repeats
        self._task = task
        self._seed = seed
        self._fold_models = None  # list of fitted models, length K * n_repeats
        self._classes = None
        self._n_classes = 0
        self._n_samples = 0
        self._oof_accum = None   # accumulated OOF predictions (sum)
        self._oof_counts = None  # per-sample prediction count (uint32)
        self._has_oof = False
        self._fitted = False
        self._disposed = False
        BaggedEstimator._register()

    @classmethod
    def create(cls, estimator=None, k_fold=5, n_repeats=1,
               task='classification', seed=42):
        return cls(estimator=estimator, k_fold=k_fold, n_repeats=n_repeats,
                   task=task, seed=seed)

    def _ensure_alive(self):
        if self._disposed:
            raise DisposedError('BaggedEstimator has been disposed.')

    def _ensure_fitted(self):
        self._ensure_alive()
        if not self._fitted:
            raise NotFittedError('BaggedEstimator is not fitted. Call fit() first.')

    def fit(self, X, y):
        self._ensure_alive()
        _validate_bagging_config(
            self._spec, self._k_fold, self._n_repeats,
            self._seed, self._task)
        n = len(X)

        classes = None
        n_classes = 0
        if self._task == 'classification':
            labels = sorted(set(int(v) for v in y))
            classes = np.array(labels, dtype=np.int32)
            n_classes = len(classes)

        oof_accum = np.zeros(
            n * n_classes if self._task == 'classification' else n,
            dtype=np.float64)
        oof_counts = np.zeros(n, dtype=np.uint32)

        name, est_cls = self._spec[:2]
        params = self._spec[2] if len(self._spec) > 2 else None
        fold_models = []
        try:
            for repeat in range(self._n_repeats):
                repeat_seed = self._seed + repeat

                folds = resolve_cv(self._k_fold, y, task=self._task,
                                   seed=repeat_seed, require_complete=True)

                for train_idx, val_idx in folds:
                    X_train, y_train = X[train_idx], y[train_idx]
                    X_val = X[val_idx]

                    model = est_cls.create(task_params(params, self._task))
                    fold_models.append(model)
                    model.fit(X_train, y_train)
                    validate_estimator_task(model, self._task)

                    if self._task == 'classification':
                        label = f'BaggedEstimator child "{name}"'
                        require_probability_model(model, label)
                        columns = class_column_map(model, classes, label)
                        proba = validate_probability_output(
                            model.predict_proba(X_val), len(val_idx),
                            n_classes, label)
                        for i in range(len(val_idx)):
                            row = val_idx[i]
                            for c in range(n_classes):
                                oof_accum[row * n_classes + c] += \
                                    proba[i * n_classes + columns[c]]
                    else:
                        preds = validate_regression_output(
                            model.predict(X_val), len(val_idx),
                            f'BaggedEstimator child "{name}"')
                        for i in range(len(val_idx)):
                            oof_accum[val_idx[i]] += float(preds[i])

                    for i in range(len(val_idx)):
                        oof_counts[val_idx[i]] += 1
        except Exception as exc:
            _dispose_owned(fold_models, exc)
            raise

        previous = self._fold_models or []
        self._n_samples = n
        self._classes = classes
        self._n_classes = n_classes
        self._oof_accum = oof_accum
        self._oof_counts = oof_counts
        self._has_oof = True
        self._fold_models = fold_models
        self._fitted = True
        _dispose_replaced(previous)
        return self

    def predict(self, X):
        self._ensure_fitted()
        n = len(X)

        if self._task == 'regression':
            out = np.zeros(n, dtype=np.float64)
            for index, model in enumerate(self._fold_models):
                preds = validate_regression_output(
                    model.predict(X), n,
                    f'BaggedEstimator child {index}')
                for i in range(n):
                    out[i] += float(preds[i])
            n_models = len(self._fold_models)
            for i in range(n):
                out[i] /= n_models
            return out

        # Classification: average probabilities, then argmax
        proba = self.predict_proba(X)
        nc = self._n_classes
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

    def predict_proba(self, X):
        self._ensure_fitted()
        if self._task != 'classification':
            raise ValidationError('predict_proba is only available for classification')

        n = len(X)
        nc = self._n_classes
        out = np.zeros(n * nc, dtype=np.float64)
        n_models = len(self._fold_models)

        for model_index, model in enumerate(self._fold_models):
            label = f'BaggedEstimator child {model_index}'
            proba = validate_probability_output(
                model.predict_proba(X), n, nc, label)
            columns = class_column_map(model, self._classes, label)
            for row in range(n):
                for column in range(nc):
                    out[row * nc + column] += \
                        proba[row * nc + columns[column]]

        for i in range(n * nc):
            out[i] /= n_models

        return out

    def score(self, X, y):
        self._ensure_fitted()
        preds = self.predict(X)
        if self._task == 'classification':
            return accuracy(y, preds)
        return r2_score(y, preds)

    @property
    def oof_predictions(self):
        """Return averaged OOF predictions.

        Classification: flat (n * n_classes,) row-major probabilities.
        Regression: flat (n,) predictions.
        """
        self._ensure_fitted()
        if not self._has_oof:
            raise ValidationError(
                'BaggedEstimator artifact does not include stored OOF '
                'predictions')
        counts = self._oof_counts.copy()
        counts[counts == 0] = 1  # avoid div-by-zero

        if self._task == 'classification':
            nc = self._n_classes
            oof = self._oof_accum.copy()
            for i in range(self._n_samples):
                c = counts[i]
                for j in range(nc):
                    oof[i * nc + j] /= c
            return oof

        return self._oof_accum / counts

    def save(self, path=None):
        self._ensure_fitted()
        type_id = TYPE_ID_CLS if self._task == 'classification' else TYPE_ID_REG

        manifest = {
            'typeId': type_id,
            'params': {
                'task': self._task,
                'kFold': serialize_cv(self._k_fold),
                'nRepeats': self._n_repeats,
                'seed': self._seed,
                'estimatorName': self._spec[0],
                'classes': [int(c) for c in self._classes] if self._classes is not None else None,
                'nClasses': int(self._n_classes),
                'nSamples': int(self._n_samples),
            },
        }

        artifacts = []
        for i, model in enumerate(self._fold_models):
            artifacts.append({
                'id': f'fold_{i}',
                'data': model.save(),
                'mediaType': 'application/x-wlearn-bundle',
            })

        # Store OOF data as raw float64 LE bytes
        oof = self.oof_predictions
        oof_bytes = np.ascontiguousarray(oof, dtype='<f8').tobytes()
        artifacts.append({
            'id': 'oof',
            'data': oof_bytes,
            'mediaType': 'application/octet-stream',
        })

        return write_bundle_output(encode_bundle(manifest, artifacts), path)

    @classmethod
    def load(cls, data, *, loader_options=None):
        manifest, _, _ = validate_bundle(data)
        if manifest.get('typeId') not in (TYPE_ID_CLS, TYPE_ID_REG):
            raise ValidationError(
                f'BaggedEstimator.load expected typeId "{TYPE_ID_CLS}" or '
                f'"{TYPE_ID_REG}", '
                f'got "{manifest.get("typeId")}"')
        cls._register()
        return registry_load(data, loader_options=loader_options)

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        try:
            _dispose_owned(self._fold_models or [])
        finally:
            self._fold_models = None
            self._oof_accum = None
            self._oof_counts = None
            self._has_oof = False
            self._fitted = False

    def get_params(self):
        return {
            'task': self._task,
            'kFold': serialize_cv(self._k_fold),
            'nRepeats': self._n_repeats,
            'seed': self._seed,
            'estimatorName': self._spec[0] if self._spec else None,
        }

    def set_params(self, p):
        self._ensure_alive()
        if not isinstance(p, dict):
            raise ValidationError('BaggedEstimator params must be a dict')
        unknown = set(p).difference(('kFold', 'nRepeats', 'seed'))
        if unknown:
            raise ValidationError(
                f'Unknown BaggedEstimator parameter "{next(iter(unknown))}"')
        k_fold = p.get('kFold', self._k_fold)
        n_repeats = p.get('nRepeats', self._n_repeats)
        seed = p.get('seed', self._seed)
        _validate_bagging_config(
            self._spec, k_fold, n_repeats, seed, self._task,
            require_constructor=False)
        if any(name in p for name in ('kFold', 'nRepeats', 'seed')):
            self._fitted = False
        self._k_fold = k_fold
        self._n_repeats = n_repeats
        self._seed = seed
        return self

    @property
    def capabilities(self):
        return {
            'classifier': self._task == 'classification',
            'regressor': self._task == 'regression',
            'predictProba': self._task == 'classification',
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

    @staticmethod
    def _register():
        global _registered
        if _registered:
            return
        _registered = True

        def loader(manifest, toc, blobs, context):
            return BaggedEstimator._load_from_parts(
                manifest, toc, blobs, context)

        register(TYPE_ID_CLS, loader, accepts_context=True)
        register(TYPE_ID_REG, loader, accepts_context=True)

    @staticmethod
    def _load_from_parts(manifest, toc, blobs, context):
        p = validate_bagging_manifest(
            manifest, toc, TYPE_ID_CLS, TYPE_ID_REG)
        assert_required_loaders(manifest)
        bag = BaggedEstimator(
            task=p['task'],
            k_fold=p.get('kFold', 5),
            n_repeats=p.get('nRepeats', 1),
            seed=p.get('seed', 42),
        )
        bag._classes = np.array(p['classes'], dtype=np.int32) if p.get('classes') else None
        bag._n_classes = p.get('nClasses', 0)
        bag._n_samples = p.get('nSamples', 0)
        bag._spec = (p.get('estimatorName', 'base'), None, None)

        # Load fold models
        n_fold_models = (bag._k_fold if isinstance(bag._k_fold, int) else len(bag._k_fold)) * bag._n_repeats
        bag._fold_models = []
        try:
            for i in range(n_fold_models):
                fold_id = f'fold_{i}'
                entry = next((t for t in toc if t['id'] == fold_id), None)
                if entry is None:
                    raise ValidationError(f'No artifact for "{fold_id}"')
                blob = bytes(
                    blobs[entry['offset']:entry['offset'] + entry['length']])
                model = _load_with_context(blob, context)
                bag._fold_models.append(model)
                if bag._task == 'classification':
                    label = f'BaggedEstimator child {i}'
                    require_probability_model(model, label)
                    class_column_map(model, bag._classes, label)

            # Load OOF data
            oof_entry = next((t for t in toc if t['id'] == 'oof'), None)
            if oof_entry is not None:
                oof_blob = bytes(
                    blobs[oof_entry['offset']:
                          oof_entry['offset'] + oof_entry['length']])
                oof = np.frombuffer(oof_blob, dtype='<f8').copy()
                # Store as accum with counts=1 so oof_predictions works.
                bag._oof_accum = oof
                bag._oof_counts = np.ones(
                    bag._n_samples, dtype=np.uint32)
                bag._has_oof = True
            else:
                # No OOF stored (loaded from older format)
                if bag._task == 'classification':
                    bag._oof_accum = np.zeros(
                        bag._n_samples * bag._n_classes, dtype=np.float64)
                else:
                    bag._oof_accum = np.zeros(
                        bag._n_samples, dtype=np.float64)
                bag._oof_counts = np.zeros(
                    bag._n_samples, dtype=np.uint32)
                bag._has_oof = False

            bag._fitted = True
            return bag
        except Exception:
            _dispose_loaded(bag._fold_models)
            raise


def _validate_bagging_config(
        spec, k_fold, n_repeats, seed, task, *,
        require_constructor=True):
    if task not in ('classification', 'regression'):
        raise ValidationError(
            'BaggedEstimator task must be "classification" or "regression"')
    if not isinstance(k_fold, (list, tuple, ResamplingPlan)) and (
            isinstance(k_fold, bool) or not isinstance(k_fold, int) or k_fold < 2 or k_fold > (1 << 53) - 1):
        raise ValidationError(
            'BaggedEstimator kFold must be a safe integer >= 2')
    if (isinstance(n_repeats, bool) or not isinstance(n_repeats, int) or
            n_repeats < 1 or n_repeats > (1 << 53) - 1):
        raise ValidationError(
            'BaggedEstimator nRepeats must be a safe integer >= 1')
    count = k_fold if isinstance(k_fold, int) else len(k_fold.folds if isinstance(k_fold, ResamplingPlan) else k_fold)
    if count * n_repeats > (1 << 53) - 1:
        raise ValidationError(
            'BaggedEstimator fold model count exceeds the safe integer range')
    if (isinstance(seed, bool) or not isinstance(seed, int) or
            abs(seed) > (1 << 53) - 1 or
            abs(seed + n_repeats - 1) > (1 << 53) - 1):
        raise ValidationError(
            'BaggedEstimator seed range must contain only safe integers')
    if (not isinstance(spec, (list, tuple)) or len(spec) < 2 or
            not isinstance(spec[0], str) or not spec[0] or
            (require_constructor and
             not callable(getattr(spec[1], 'create', None)))):
        raise ValidationError(
            'BaggedEstimator requires a valid estimator specification')


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
