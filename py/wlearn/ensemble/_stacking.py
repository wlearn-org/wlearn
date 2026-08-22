"""StackingEnsemble matching JS @wlearn/ensemble/stacking.js."""

import numpy as np

from ..errors import ValidationError, NotFittedError, DisposedError
from ..bundle import encode_bundle, validate_bundle, write_bundle_output
from ..registry import (
    register, load as registry_load, _load_with_context,
    assert_required_loaders,
)
from ..automl._cv import accuracy, r2_score, stratified_k_fold, k_fold
from ._manifest import validate_stacking_manifest
from ._class_order import (
    class_column_map, require_probability_model, validate_label_output,
    validate_probability_output, validate_regression_output,
)

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
        _validate_stacking_config(
            self._base_specs, self._meta_spec, self._cv, self._task,
            self._passthrough, self._seed)

        n = len(X)
        n_features = X.shape[1] if hasattr(X, 'shape') else len(X[0])

        classes = None
        n_classes = 0
        if self._task == 'classification':
            labels = sorted(set(int(v) for v in y))
            classes = np.array(labels, dtype=np.int32)
            n_classes = len(classes)

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
                if (hasattr(type(model), 'oof_predictions') and
                        model.is_fitted):
                    bagged_bases.append((b, name, model))
                else:
                    raise ValidationError(
                        f'Base estimator "{name}" is a 2-tuple but not a fitted '
                        f'BaggedEstimator with oof_predictions.'
                    )
            else:
                name, est_cls = entry[:2]
                params = entry[2] if len(entry) > 2 else None
                spec_bases.append((b, name, est_cls, params))

        # Step 1: Generate OOF predictions
        n_base = len(self._base_specs)
        cols_per_model = n_classes if self._task == 'classification' else 1
        oof_cols = n_base * cols_per_model
        oof_data = np.zeros(n * oof_cols, dtype=np.float64)

        # Fill OOF from pre-fitted BaggedEstimators
        for b, name, model in bagged_bases:
            bagged_params = model.get_params()
            if bagged_params.get('task') != self._task:
                raise ValidationError(
                    f'Pre-fitted BaggedEstimator "{name}" task does not '
                    'match stacking task')
            oof = _validate_prefitted_oof(
                model.oof_predictions, n * cols_per_model,
                f'Pre-fitted BaggedEstimator "{name}"')
            if self._task == 'classification':
                columns = class_column_map(
                    model, classes,
                    f'Pre-fitted BaggedEstimator "{name}"')
                for row in range(n):
                    for column in range(cols_per_model):
                        oof_data[
                            row * oof_cols + b * cols_per_model + column
                        ] = oof[row * cols_per_model + columns[column]]
                continue
            for i in range(n):
                oof_data[i * oof_cols + b] = float(oof[i])

        # Generate OOF from regular specs via fold training
        for b, name, est_cls, params in spec_bases:
            for train, test in folds:
                X_train, y_train = X[train], y[train]
                X_test = X[test]

                model = est_cls.create(params or {})
                operation_error = None
                try:
                    model.fit(X_train, y_train)
                    if self._task == 'classification':
                        label = (
                            f'StackingEnsemble base estimator '
                            f'"{self._base_specs[b][0]}"')
                        require_probability_model(model, label)
                        columns = class_column_map(model, classes, label)
                        proba = validate_probability_output(
                            model.predict_proba(X_test), len(test),
                            n_classes, label)
                        for i in range(len(test)):
                            row = test[i]
                            for c in range(n_classes):
                                oof_data[row * oof_cols + b * cols_per_model + c] = \
                                    proba[i * n_classes + columns[c]]
                    else:
                        preds = validate_regression_output(
                            model.predict(X_test), len(test),
                            f'StackingEnsemble base estimator "{name}"')
                        for i in range(len(test)):
                            oof_data[test[i] * oof_cols + b] = float(preds[i])
                except Exception as exc:
                    operation_error = exc
                    raise
                finally:
                    _dispose_owned([model], operation_error)

        # Step 2: Build meta-feature matrix
        if self._passthrough:
            n_meta_cols = oof_cols + n_features
            meta_data = np.zeros(n * n_meta_cols, dtype=np.float64)
            for i in range(n):
                meta_data[i * n_meta_cols:i * n_meta_cols + oof_cols] = \
                    oof_data[i * oof_cols:(i + 1) * oof_cols]
                meta_data[i * n_meta_cols + oof_cols:
                          i * n_meta_cols + oof_cols + n_features] = X[i]
            meta_X = meta_data.reshape(n, n_meta_cols)
        else:
            n_meta_cols = oof_cols
            meta_X = oof_data.reshape(n, oof_cols)

        # Steps 3-4 build replacement state transactionally. Pre-fitted bagged
        # inputs transfer to the ensemble only when the complete fit commits.
        base_models = [None] * n_base
        for b, name, model in bagged_bases:
            base_models[b] = model
        created_models = []
        meta_model = None
        try:
            for b, _name, est_cls, params in spec_bases:
                model = est_cls.create(params or {})
                created_models.append(model)
                base_models[b] = model
                model.fit(X, y)
                if self._task == 'classification':
                    label = (
                        f'StackingEnsemble base estimator '
                        f'"{self._base_specs[b][0]}"')
                    require_probability_model(model, label)
                    class_column_map(
                        model, classes, label)

            meta_cls = self._meta_spec[1]
            meta_params = (
                self._meta_spec[2] if len(self._meta_spec) > 2 else None)
            meta_model = meta_cls.create(meta_params or {})
            meta_model.fit(meta_X, y)
            if self._task == 'classification':
                class_column_map(
                    meta_model, classes,
                    f'StackingEnsemble meta estimator "{self._meta_spec[0]}"')
        except Exception as exc:
            _dispose_owned([*created_models, meta_model], exc)
            raise

        previous = [*(self._base_models or []), self._meta_model]
        self._base_models = base_models
        self._meta_model = meta_model
        self._classes = classes
        self._n_classes = n_classes
        self._n_meta_cols = n_meta_cols
        self._fitted = True
        retained = {id(model) for model in [*base_models, meta_model]
                    if model is not None}
        _dispose_replaced([
            model for model in previous
            if model is not None and id(model) not in retained
        ])
        return self

    def predict(self, X):
        self._ensure_fitted()
        meta_X = self._build_meta_features(X)
        output = self._meta_model.predict(meta_X)
        if self._task != 'classification':
            return validate_regression_output(
                output, len(meta_X),
                f'StackingEnsemble meta estimator "{self._meta_spec[0]}"')
        return validate_label_output(
            output, len(meta_X), self._classes,
            f'StackingEnsemble meta estimator "{self._meta_spec[0]}"')

    def predict_proba(self, X):
        self._ensure_fitted()
        if self._task != 'classification':
            raise ValidationError('predict_proba is only available for classification')
        if not _supports_predict_proba(self._meta_model):
            raise ValidationError('Meta-model does not support predict_proba')
        meta_X = self._build_meta_features(X)
        label = f'StackingEnsemble meta estimator "{self._meta_spec[0]}"'
        proba = validate_probability_output(
            self._meta_model.predict_proba(meta_X), len(X),
            self._n_classes, label)
        columns = class_column_map(
            self._meta_model, self._classes, label)
        aligned = np.empty_like(proba)
        for row in range(len(X)):
            for column in range(self._n_classes):
                aligned[row * self._n_classes + column] = \
                    proba[row * self._n_classes + columns[column]]
        return aligned

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
                'cv': self._cv,
                'passthrough': self._passthrough,
                'seed': self._seed,
                'estimatorNames': [s[0] for s in self._base_specs],
                'metaName': self._meta_spec[0],
                'classes': (
                    [int(value) for value in self._classes]
                    if self._classes is not None else None),
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
        return write_bundle_output(encode_bundle(manifest, artifacts), path)

    @classmethod
    def load(cls, data, *, loader_options=None):
        manifest, _, _ = validate_bundle(data)
        if manifest.get('typeId') not in (TYPE_ID_CLS, TYPE_ID_REG):
            raise ValidationError(
                f'StackingEnsemble.load expected typeId "{TYPE_ID_CLS}" or '
                f'"{TYPE_ID_REG}", '
                f'got "{manifest.get("typeId")}"')
        cls._register()
        return registry_load(data, loader_options=loader_options)

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        try:
            _dispose_owned([*(self._base_models or []), self._meta_model])
        finally:
            self._base_models = None
            self._meta_model = None
            self._fitted = False

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
        if not isinstance(p, dict):
            raise ValidationError('StackingEnsemble params must be a dict')
        unknown = set(p).difference(('cv', 'passthrough', 'seed'))
        if unknown:
            raise ValidationError(
                f'Unknown StackingEnsemble parameter "{next(iter(unknown))}"')
        cv = p.get('cv', self._cv)
        passthrough = p.get('passthrough', self._passthrough)
        seed = p.get('seed', self._seed)
        _validate_stacking_config(
            self._base_specs, self._meta_spec, cv, self._task,
            passthrough, seed, require_constructors=False)
        if any(name in p for name in ('cv', 'passthrough', 'seed')):
            self._fitted = False
        self._cv = cv
        self._passthrough = passthrough
        self._seed = seed
        return self

    @property
    def capabilities(self):
        return {
            'classifier': self._task == 'classification',
            'regressor': self._task == 'regression',
            'predictProba': (
                self._task == 'classification' and
                _supports_predict_proba(self._meta_model)),
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

    def _build_meta_features(self, X):
        n = len(X)
        n_features = X.shape[1] if hasattr(X, 'shape') else len(X[0])
        n_base = len(self._base_models)
        cols_per_model = self._n_classes if self._task == 'classification' else 1
        oof_cols = n_base * cols_per_model

        meta_data = np.zeros(n * self._n_meta_cols, dtype=np.float64)
        for b in range(n_base):
            if self._task == 'classification':
                label = (
                    f'StackingEnsemble base estimator '
                    f'"{self._base_specs[b][0]}"')
                proba = validate_probability_output(
                    self._base_models[b].predict_proba(X), n,
                    self._n_classes, label)
                columns = class_column_map(
                    self._base_models[b], self._classes, label)
                for i in range(n):
                    for c in range(self._n_classes):
                        meta_data[i * self._n_meta_cols + b * cols_per_model + c] = \
                            proba[i * self._n_classes + columns[c]]
            else:
                preds = validate_regression_output(
                    self._base_models[b].predict(X), n,
                    f'StackingEnsemble base estimator '
                    f'"{self._base_specs[b][0]}"')
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

        def loader(manifest, toc, blobs, context):
            return StackingEnsemble._load_from_parts(
                manifest, toc, blobs, context)

        register(TYPE_ID_CLS, loader, accepts_context=True)
        register(TYPE_ID_REG, loader, accepts_context=True)

    @staticmethod
    def _load_from_parts(manifest, toc, blobs, context):
        p = validate_stacking_manifest(
            manifest, toc, TYPE_ID_CLS, TYPE_ID_REG)
        assert_required_loaders(manifest)
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
        try:
            for name in p['estimatorNames']:
                entry = next((t for t in toc if t['id'] == name), None)
                if entry is None:
                    raise ValidationError(
                        f'No artifact for base estimator "{name}"')
                blob = bytes(
                    blobs[entry['offset']:entry['offset'] + entry['length']])
                model = _load_with_context(blob, context)
                ens._base_models.append(model)
                if ens._task == 'classification':
                    label = f'StackingEnsemble base estimator "{name}"'
                    require_probability_model(model, label)
                    class_column_map(
                        model, ens._classes, label)

            meta_entry = next(
                (t for t in toc if t['id'] == p['metaName']), None)
            if meta_entry is None:
                raise ValidationError(
                    f'No artifact for meta estimator "{p["metaName"]}"')
            meta_blob = bytes(
                blobs[meta_entry['offset']:
                      meta_entry['offset'] + meta_entry['length']])
            ens._meta_model = _load_with_context(meta_blob, context)
            if ens._task == 'classification':
                class_column_map(
                    ens._meta_model, ens._classes,
                    f'StackingEnsemble meta estimator "{p["metaName"]}"')

            ens._fitted = True
            return ens
        except Exception:
            _dispose_loaded([
                *ens._base_models,
                ens._meta_model,
            ])
            raise


def _validate_prefitted_oof(value, expected_length, label):
    try:
        output = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(
            f'{label} OOF predictions must be numeric') from exc
    if output.size != expected_length:
        raise ValidationError(
            f'{label} OOF shape does not match the stacking data')
    if not np.all(np.isfinite(output)):
        raise ValidationError(f'{label} OOF predictions must be finite')
    return output


def _supports_predict_proba(model):
    capabilities = getattr(model, 'capabilities', None)
    return (callable(getattr(model, 'predict_proba', None)) and
            isinstance(capabilities, dict) and
            capabilities.get('predictProba') is True)


def _validate_stacking_config(
        base_specs, meta_spec, cv, task, passthrough, seed, *,
        require_constructors=True):
    if task not in ('classification', 'regression'):
        raise ValidationError(
            'StackingEnsemble task must be "classification" or "regression"')
    if (isinstance(cv, bool) or not isinstance(cv, int) or cv < 2 or
            cv > (1 << 53) - 1):
        raise ValidationError(
            'StackingEnsemble cv must be a safe integer >= 2')
    if not isinstance(passthrough, bool):
        raise ValidationError(
            'StackingEnsemble passthrough must be a boolean')
    if (isinstance(seed, bool) or not isinstance(seed, int) or
            abs(seed) > (1 << 53) - 1):
        raise ValidationError(
            'StackingEnsemble seed must be a safe integer')
    if not isinstance(base_specs, (list, tuple)) or not base_specs:
        raise ValidationError(
            'StackingEnsemble estimators must be a nonempty sequence')

    names = set()
    for index, spec in enumerate(base_specs):
        if (not isinstance(spec, (list, tuple)) or len(spec) < 2 or
                not isinstance(spec[0], str) or not spec[0]):
            raise ValidationError(
                f'StackingEnsemble base estimator {index} has an invalid '
                'specification')
        if require_constructors:
            fitted_bag = (
                len(spec) == 2 and bool(getattr(spec[1], 'is_fitted', False))
                and hasattr(type(spec[1]), 'oof_predictions'))
            if (not fitted_bag and
                    (len(spec) < 3 or
                     not callable(getattr(spec[1], 'create', None)))):
                raise ValidationError(
                    f'StackingEnsemble base estimator {index} has an '
                    'invalid specification')
        if spec[0] in names:
            raise ValidationError(
                'StackingEnsemble estimator names must be unique')
        names.add(spec[0])

    if (not isinstance(meta_spec, (list, tuple)) or len(meta_spec) < 2 or
            not isinstance(meta_spec[0], str) or not meta_spec[0] or
            (require_constructors and
             not callable(getattr(meta_spec[1], 'create', None)))):
        raise ValidationError(
            'StackingEnsemble requires a valid finalEstimator')
    if meta_spec[0] in names:
        raise ValidationError(
            'StackingEnsemble finalEstimator name must differ from base '
            'estimator names')


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
