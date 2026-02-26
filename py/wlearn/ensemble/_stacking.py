"""StackingEnsemble matching JS @wlearn/ensemble/stacking.js."""

import numpy as np

from ..errors import ValidationError, NotFittedError, DisposedError
from ..bundle import encode_bundle, decode_bundle
from ..registry import register, load as registry_load
from ..automl._cv import accuracy, r2_score, stratified_k_fold, k_fold

TYPE_ID_CLS = 'wlearn.ensemble.stacking.classifier@1'
TYPE_ID_REG = 'wlearn.ensemble.stacking.regressor@1'
_registered = False


class StackingEnsemble:
    def __init__(self, estimators=None, final_estimator=None, cv=5,
                 task='classification', passthrough=False, seed=42):
        """
        Args:
            estimators: list of (name, cls, params) tuples OR (name, fitted_model) tuples.
                If a tuple has 2 elements and the second is a fitted BaggedEstimator,
                its stored OOF predictions are used directly (no retraining).
            final_estimator: (name, cls, params) tuple for meta-model
            cv: number of folds
            task: 'classification' or 'regression'
            passthrough: include original features in meta features
            seed: random seed
        """
        self._base_specs = estimators or []
        self._meta_spec = final_estimator
        self._cv = cv
        self._task = task
        self._passthrough = passthrough
        self._seed = seed
        self._base_models = None
        self._meta_model = None
        self._classes = None
        self._n_classes = 0
        self._n_meta_cols = 0
        self._fitted = False
        self._disposed = False
        StackingEnsemble._register()

    @classmethod
    def create(cls, estimators=None, final_estimator=None, cv=5,
               task='classification', passthrough=False, seed=42):
        return cls(estimators=estimators, final_estimator=final_estimator,
                   cv=cv, task=task, passthrough=passthrough, seed=seed)

    def _ensure_alive(self):
        if self._disposed:
            raise DisposedError('StackingEnsemble has been disposed.')

    def _ensure_fitted(self):
        self._ensure_alive()
        if not self._fitted:
            raise NotFittedError('StackingEnsemble is not fitted. Call fit() first.')

    def fit(self, X, y):
        self._ensure_alive()
        if self._meta_spec is None:
            raise ValidationError('StackingEnsemble requires a finalEstimator')

        n = len(X)
        n_features = X.shape[1] if hasattr(X, 'shape') else len(X[0])

        if self._task == 'classification':
            labels = sorted(set(int(v) for v in y))
            self._classes = np.array(labels, dtype=np.int32)
            self._n_classes = len(self._classes)

        # Generate folds
        if self._task == 'classification':
            folds = stratified_k_fold(y, self._cv, do_shuffle=True, seed=self._seed)
        else:
            folds = k_fold(n, self._cv, do_shuffle=True, seed=self._seed)

        # Classify base specs: pre-fitted BaggedEstimator vs regular (name, cls, params)
        from ._bagging import BaggedEstimator
        bagged_bases = []  # (index, name, fitted_model) for pre-fitted BaggedEstimators
        spec_bases = []    # (index, name, cls, params) for regular specs
        for b, entry in enumerate(self._base_specs):
            if len(entry) == 2:
                name, model = entry
                if hasattr(model, 'oof_predictions') and model.is_fitted:
                    bagged_bases.append((b, name, model))
                else:
                    raise ValidationError(
                        f'Base estimator "{name}" is a 2-tuple but not a fitted '
                        f'BaggedEstimator with oof_predictions.'
                    )
            else:
                name, est_cls, params = entry
                spec_bases.append((b, name, est_cls, params))

        # Step 1: Generate OOF predictions
        n_base = len(self._base_specs)
        cols_per_model = self._n_classes if self._task == 'classification' else 1
        oof_cols = n_base * cols_per_model
        oof_data = np.zeros(n * oof_cols, dtype=np.float64)

        # Fill OOF from pre-fitted BaggedEstimators
        for b, name, model in bagged_bases:
            oof = model.oof_predictions
            if self._task == 'classification':
                nc = self._n_classes
                for i in range(n):
                    for c in range(nc):
                        oof_data[i * oof_cols + b * cols_per_model + c] = oof[i * nc + c]
            else:
                for i in range(n):
                    oof_data[i * oof_cols + b] = float(oof[i])

        # Generate OOF from regular specs via fold training
        for b, name, est_cls, params in spec_bases:
            for train, test in folds:
                X_train, y_train = X[train], y[train]
                X_test = X[test]

                model = est_cls.create(params or {})
                try:
                    model.fit(X_train, y_train)
                    if self._task == 'classification':
                        proba = model.predict_proba(X_test)
                        for i in range(len(test)):
                            row = test[i]
                            for c in range(self._n_classes):
                                oof_data[row * oof_cols + b * cols_per_model + c] = \
                                    proba[i * self._n_classes + c]
                    else:
                        preds = model.predict(X_test)
                        for i in range(len(test)):
                            oof_data[test[i] * oof_cols + b] = float(preds[i])
                finally:
                    model.dispose()

        # Step 2: Build meta-feature matrix
        if self._passthrough:
            self._n_meta_cols = oof_cols + n_features
            meta_data = np.zeros(n * self._n_meta_cols, dtype=np.float64)
            for i in range(n):
                meta_data[i * self._n_meta_cols:i * self._n_meta_cols + oof_cols] = \
                    oof_data[i * oof_cols:(i + 1) * oof_cols]
                meta_data[i * self._n_meta_cols + oof_cols:
                          i * self._n_meta_cols + oof_cols + n_features] = X[i]
            meta_X = meta_data.reshape(n, self._n_meta_cols)
        else:
            self._n_meta_cols = oof_cols
            meta_X = oof_data.reshape(n, oof_cols)

        # Step 3: Train base models on full data (or use pre-fitted BaggedEstimators)
        self._base_models = [None] * n_base
        for b, name, model in bagged_bases:
            self._base_models[b] = model
        for b, name, est_cls, params in spec_bases:
            model = est_cls.create(params or {})
            model.fit(X, y)
            self._base_models[b] = model

        # Step 4: Train meta-model on OOF features
        _, meta_cls, meta_params = self._meta_spec
        self._meta_model = meta_cls.create(meta_params or {})
        self._meta_model.fit(meta_X, y)

        self._fitted = True
        return self

    def predict(self, X):
        self._ensure_fitted()
        meta_X = self._build_meta_features(X)
        return self._meta_model.predict(meta_X)

    def predict_proba(self, X):
        self._ensure_fitted()
        if self._task != 'classification':
            raise ValidationError('predict_proba is only available for classification')
        if not hasattr(self._meta_model, 'predict_proba'):
            raise ValidationError('Meta-model does not support predict_proba')
        meta_X = self._build_meta_features(X)
        return self._meta_model.predict_proba(meta_X)

    def score(self, X, y):
        self._ensure_fitted()
        preds = self.predict(X)
        if self._task == 'classification':
            return accuracy(y, preds)
        return r2_score(y, preds)

    def save(self):
        self._ensure_fitted()
        type_id = TYPE_ID_CLS if self._task == 'classification' else TYPE_ID_REG
        manifest = {
            'typeId': type_id,
            'params': {
                'task': self._task,
                'cv': self._cv,
                'passthrough': self._passthrough,
                'seed': self._seed,
                'estimatorNames': [s[0] for s in self._base_specs],
                'metaName': self._meta_spec[0],
                'classes': list(self._classes) if self._classes is not None else None,
                'nMetaCols': self._n_meta_cols,
            },
        }
        artifacts = [
            {
                'id': self._base_specs[i][0],
                'data': self._base_models[i].save(),
                'mediaType': 'application/x-wlearn-bundle',
            }
            for i in range(len(self._base_models))
        ]
        artifacts.append({
            'id': self._meta_spec[0],
            'data': self._meta_model.save(),
            'mediaType': 'application/x-wlearn-bundle',
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
        if self._base_models:
            for m in self._base_models:
                m.dispose()
        if self._meta_model:
            self._meta_model.dispose()

    def get_params(self):
        return {
            'task': self._task,
            'cv': self._cv,
            'passthrough': self._passthrough,
            'seed': self._seed,
            'estimatorNames': [s[0] for s in self._base_specs],
            'metaName': self._meta_spec[0] if self._meta_spec else None,
        }

    def set_params(self, p):
        self._ensure_alive()
        if 'cv' in p:
            self._cv = p['cv']
        if 'passthrough' in p:
            self._passthrough = p['passthrough']
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
        return self._fitted

    @property
    def classes(self):
        return self._classes

    def _build_meta_features(self, X):
        n = len(X)
        n_features = X.shape[1] if hasattr(X, 'shape') else len(X[0])
        n_base = len(self._base_models)
        cols_per_model = self._n_classes if self._task == 'classification' else 1
        oof_cols = n_base * cols_per_model

        meta_data = np.zeros(n * self._n_meta_cols, dtype=np.float64)
        for b in range(n_base):
            if self._task == 'classification':
                proba = self._base_models[b].predict_proba(X)
                for i in range(n):
                    for c in range(self._n_classes):
                        meta_data[i * self._n_meta_cols + b * cols_per_model + c] = \
                            proba[i * self._n_classes + c]
            else:
                preds = self._base_models[b].predict(X)
                for i in range(n):
                    meta_data[i * self._n_meta_cols + b] = float(preds[i])

        if self._passthrough:
            for i in range(n):
                for j in range(n_features):
                    meta_data[i * self._n_meta_cols + oof_cols + j] = X[i][j]

        return meta_data.reshape(n, self._n_meta_cols)

    @staticmethod
    def _register():
        global _registered
        if _registered:
            return
        _registered = True

        def loader(manifest, toc, blobs):
            return StackingEnsemble._load_from_parts(manifest, toc, blobs)

        register(TYPE_ID_CLS, loader)
        register(TYPE_ID_REG, loader)

    @staticmethod
    def _load_from_parts(manifest, toc, blobs):
        p = manifest['params']
        ens = StackingEnsemble(
            task=p['task'],
            cv=p.get('cv', 5),
            passthrough=p.get('passthrough', False),
            seed=p.get('seed', 42),
        )
        ens._classes = np.array(p['classes'], dtype=np.int32) if p.get('classes') else None
        ens._n_classes = len(ens._classes) if ens._classes is not None else 0
        ens._n_meta_cols = p.get('nMetaCols', 0)
        ens._base_specs = [(name, None, None) for name in p['estimatorNames']]
        ens._meta_spec = (p['metaName'], None, None)

        ens._base_models = []
        for name in p['estimatorNames']:
            entry = next((t for t in toc if t['id'] == name), None)
            if entry is None:
                raise ValidationError(f'No artifact for base estimator "{name}"')
            blob = bytes(blobs[entry['offset']:entry['offset'] + entry['length']])
            ens._base_models.append(registry_load(blob))

        meta_entry = next((t for t in toc if t['id'] == p['metaName']), None)
        if meta_entry is None:
            raise ValidationError(f'No artifact for meta estimator "{p["metaName"]}"')
        meta_blob = bytes(blobs[meta_entry['offset']:meta_entry['offset'] + meta_entry['length']])
        ens._meta_model = registry_load(meta_blob)

        ens._fitted = True
        return ens
