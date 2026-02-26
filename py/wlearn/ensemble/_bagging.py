"""BaggedEstimator: K-fold bagging with OOF storage.

Trains K copies of a base model on K folds, stores out-of-fold predictions,
and averages predictions at inference time. Supports multiple repeats with
different fold assignments.

typeIds:
  wlearn.ensemble.bagged.classifier@1
  wlearn.ensemble.bagged.regressor@1
"""

import struct

import numpy as np

from ..errors import ValidationError, NotFittedError, DisposedError
from ..bundle import encode_bundle, decode_bundle
from ..registry import register, load as registry_load
from ..automl._cv import accuracy, r2_score, stratified_k_fold, k_fold

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
        self._oof_counts = None  # per-sample prediction count (uint8)
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
        n = len(X)
        self._n_samples = n

        if self._task == 'classification':
            labels = sorted(set(int(v) for v in y))
            self._classes = np.array(labels, dtype=np.int32)
            self._n_classes = len(self._classes)

        # Initialize OOF accumulation arrays
        if self._task == 'classification':
            self._oof_accum = np.zeros(n * self._n_classes, dtype=np.float64)
        else:
            self._oof_accum = np.zeros(n, dtype=np.float64)
        self._oof_counts = np.zeros(n, dtype=np.uint8)

        name, est_cls, params = self._spec
        self._fold_models = []

        for repeat in range(self._n_repeats):
            repeat_seed = self._seed + repeat

            if self._task == 'classification':
                folds = stratified_k_fold(y, self._k_fold,
                                          do_shuffle=True, seed=repeat_seed)
            else:
                folds = k_fold(n, self._k_fold,
                               do_shuffle=True, seed=repeat_seed)

            for train_idx, val_idx in folds:
                X_train, y_train = X[train_idx], y[train_idx]
                X_val = X[val_idx]

                model = est_cls.create(params or {})
                model.fit(X_train, y_train)

                # Accumulate OOF predictions
                if self._task == 'classification':
                    proba = model.predict_proba(X_val)
                    for i in range(len(val_idx)):
                        row = val_idx[i]
                        for c in range(self._n_classes):
                            self._oof_accum[row * self._n_classes + c] += \
                                proba[i * self._n_classes + c]
                else:
                    preds = model.predict(X_val)
                    for i in range(len(val_idx)):
                        self._oof_accum[val_idx[i]] += float(preds[i])

                for i in range(len(val_idx)):
                    self._oof_counts[val_idx[i]] += 1

                self._fold_models.append(model)

        self._fitted = True
        return self

    def predict(self, X):
        self._ensure_fitted()
        n = len(X)

        if self._task == 'regression':
            out = np.zeros(n, dtype=np.float64)
            for model in self._fold_models:
                preds = model.predict(X)
                for i in range(n):
                    out[i] += float(preds[i])
            n_models = len(self._fold_models)
            for i in range(n):
                out[i] /= n_models
            return out

        # Classification: average probabilities, then argmax
        proba = self.predict_proba(X)
        nc = self._n_classes
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

    def predict_proba(self, X):
        self._ensure_fitted()
        if self._task != 'classification':
            raise ValidationError('predict_proba is only available for classification')

        n = len(X)
        nc = self._n_classes
        out = np.zeros(n * nc, dtype=np.float64)
        n_models = len(self._fold_models)

        for model in self._fold_models:
            proba = model.predict_proba(X)
            for i in range(n * nc):
                out[i] += proba[i]

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

    def save(self):
        self._ensure_fitted()
        type_id = TYPE_ID_CLS if self._task == 'classification' else TYPE_ID_REG

        manifest = {
            'typeId': type_id,
            'params': {
                'task': self._task,
                'kFold': self._k_fold,
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

        return encode_bundle(manifest, artifacts)

    @classmethod
    def load(cls, data):
        manifest, toc, blobs = decode_bundle(data)
        return cls._load_from_parts(manifest, toc, blobs)

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        if self._fold_models:
            for m in self._fold_models:
                m.dispose()
        self._fold_models = None
        self._oof_accum = None
        self._oof_counts = None

    def get_params(self):
        return {
            'task': self._task,
            'kFold': self._k_fold,
            'nRepeats': self._n_repeats,
            'seed': self._seed,
            'estimatorName': self._spec[0] if self._spec else None,
        }

    def set_params(self, p):
        self._ensure_alive()
        if 'kFold' in p:
            self._k_fold = p['kFold']
        if 'nRepeats' in p:
            self._n_repeats = p['nRepeats']
        if 'seed' in p:
            self._seed = p['seed']
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

        def loader(manifest, toc, blobs):
            return BaggedEstimator._load_from_parts(manifest, toc, blobs)

        register(TYPE_ID_CLS, loader)
        register(TYPE_ID_REG, loader)

    @staticmethod
    def _load_from_parts(manifest, toc, blobs):
        p = manifest['params']
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
        n_fold_models = bag._k_fold * bag._n_repeats
        bag._fold_models = []
        for i in range(n_fold_models):
            fold_id = f'fold_{i}'
            entry = next((t for t in toc if t['id'] == fold_id), None)
            if entry is None:
                raise ValidationError(f'No artifact for "{fold_id}"')
            blob = bytes(blobs[entry['offset']:entry['offset'] + entry['length']])
            bag._fold_models.append(registry_load(blob))

        # Load OOF data
        oof_entry = next((t for t in toc if t['id'] == 'oof'), None)
        if oof_entry is not None:
            oof_blob = bytes(blobs[oof_entry['offset']:oof_entry['offset'] + oof_entry['length']])
            oof = np.frombuffer(oof_blob, dtype='<f8').copy()
            # Store as accum with counts=1 so oof_predictions property works
            bag._oof_accum = oof
            bag._oof_counts = np.ones(bag._n_samples, dtype=np.uint8)
        else:
            # No OOF stored (loaded from older format)
            if bag._task == 'classification':
                bag._oof_accum = np.zeros(bag._n_samples * bag._n_classes, dtype=np.float64)
            else:
                bag._oof_accum = np.zeros(bag._n_samples, dtype=np.float64)
            bag._oof_counts = np.zeros(bag._n_samples, dtype=np.uint8)

        bag._fitted = True
        return bag
