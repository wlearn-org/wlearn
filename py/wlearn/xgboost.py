"""Python wrapper for @wlearn/xgboost bundles.

Loads WLRN bundles produced by JS @wlearn/xgboost, predicts using
native Python xgboost, and saves back to WLRN bundles that JS can load.
"""

import tempfile
import os
import json
from numbers import Integral

import numpy as np
import xgboost as xgb

from .errors import NotFittedError, DisposedError
from ._capabilities import estimator_capabilities
from .bundle import encode_bundle, write_bundle_output
from .registry import register

CLASSIFIER_OBJECTIVES = frozenset([
    'binary:logistic', 'binary:logitraw', 'binary:hinge',
    'multi:softmax', 'multi:softprob',
])

PROBA_OBJECTIVES = frozenset(['binary:logistic', 'multi:softprob'])

WLEARN_PARAMS = frozenset(['numRound', 'coerce', 'task'])


def _validate_unified_objective(objective):
    if not isinstance(objective, str) or not objective:
        raise ValueError('objective must be a non-empty XGBoost objective string')
    if objective.startswith(('rank:', 'survival:')):
        raise ValueError(
            f'The high-level XGBModel does not implement ranking groups or '
            f'survival metrics for objective {objective!r}; use the native '
            'low-level xgboost.Booster API.')


def _resolve_fit_params(params, y):
    resolved = dict(params)
    objective = resolved.get('objective')
    task = resolved.get('task')
    if objective is not None:
        _validate_unified_objective(objective)
        return resolved
    if task is None:
        try:
            numeric = np.asarray(y, dtype=np.float64)
        except (TypeError, ValueError):
            task = 'classification'
        else:
            integral = (numeric.ndim == 1 and np.isfinite(numeric).all() and
                        np.equal(numeric, np.floor(numeric)).all())
            task = ('classification'
                    if integral and np.unique(numeric).size <= 20
                    else 'regression')
        resolved['task'] = task
    if task == 'classification':
        try:
            n_classes = np.unique(np.asarray(y, dtype=np.float64)).size
        except (TypeError, ValueError):
            n_classes = 0
        if n_classes > 2:
            resolved['objective'] = 'multi:softprob'
            resolved.setdefault('num_class', n_classes)
        else:
            resolved['objective'] = 'binary:logistic'
    elif task == 'regression':
        resolved['objective'] = 'reg:squarederror'
    else:
        raise ValueError(
            f'Unknown task: {task!r}. Use "classification" or "regression".')
    return resolved


def _booster_identity(booster):
    try:
        config = json.loads(booster.save_config())
        learner = config['learner']
        model = learner['learner_model_param']
        objective = learner['objective']['name']
        return {
            'objective': objective,
            'nFeatures': int(model['num_feature']),
            'nClasses': int(model['num_class']),
            'nTargets': int(model['num_target']),
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError('XGBoost model has an unsupported identity config') \
            from error


class XGBModel:
    def __init__(self, booster, params, nr_class=0, classes=None):
        self._booster = booster
        self._loaded_model_bytes = None
        self._params = dict(params)
        self._nr_class = nr_class
        self._classes = np.array(classes, dtype=np.int32) if classes else np.array([], dtype=np.int32)
        self._fitted = True
        self._disposed = False

    @classmethod
    def create(cls, params=None):
        """Create an unfitted XGBoost model."""
        obj = cls.__new__(cls)
        obj._booster = None
        obj._loaded_model_bytes = None
        obj._params = dict(params) if params else {}
        obj._nr_class = 0
        obj._classes = np.array([], dtype=np.int32)
        obj._fitted = False
        obj._disposed = False
        return obj

    def fit(self, X, y):
        """Train an XGBoost model.

        Params are passed directly to xgboost.train(). The wlearn-only
        param ``numRound`` controls the number of boosting rounds (default 100).
        """
        if self._disposed:
            raise DisposedError('XGBModel has been disposed.')

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError(
                f'y length ({y.size}) does not match X rows ({len(X)})')

        fit_params = _resolve_fit_params(self._params, y)
        obj = fit_params['objective']
        num_round = fit_params.get('numRound', 100)
        if (isinstance(num_round, bool) or not isinstance(num_round, Integral) or
                num_round < 1 or num_round > 2 ** 53 - 1):
            raise ValueError('numRound must be a positive safe integer')
        num_round = int(num_round)

        # Build xgboost params (exclude wlearn-only params)
        xgb_params = {k: v for k, v in fit_params.items()
                      if k not in WLEARN_PARAMS}
        xgb_params.setdefault('objective', obj)
        xgb_params.setdefault('verbosity', 0)

        # Detect classes for classification
        if obj in CLASSIFIER_OBJECTIVES:
            try:
                y_numeric = y.astype(np.float64)
            except (TypeError, ValueError) as error:
                raise ValueError('Classifier labels must be int32 values') \
                    from error
            if (not np.isfinite(y_numeric).all() or
                    not np.equal(y_numeric, np.floor(y_numeric)).all() or
                    np.any(y_numeric < np.iinfo(np.int32).min) or
                    np.any(y_numeric > np.iinfo(np.int32).max)):
                raise ValueError('Classifier labels must be int32 values')
            classes = np.unique(y_numeric).astype(np.int32)
            nr_class = len(classes)
            if nr_class < 2:
                raise ValueError(
                    f'Classification requires at least 2 classes, got '
                    f'{nr_class}')
            if obj.startswith('binary:') and nr_class != 2:
                raise ValueError(
                    f'Binary objective requires exactly 2 classes, got '
                    f'{nr_class}')

            # Remap to 0-based contiguous for multi:softmax/softprob
            if obj in ('multi:softmax', 'multi:softprob'):
                requested = xgb_params.get('num_class')
                if requested is not None and requested != nr_class:
                    raise ValueError(
                        f'num_class ({requested}) does not match fitted '
                        f'classes ({nr_class})')
                xgb_params['num_class'] = nr_class
                class_map = {int(c): i for i, c in enumerate(classes)}
                y_train = np.array(
                    [class_map[int(v)] for v in y_numeric], dtype=np.float32)
            else:
                # Binary: remap to 0/1
                class_map = {int(c): i for i, c in enumerate(classes)}
                y_train = np.array(
                    [class_map[int(v)] for v in y_numeric], dtype=np.float32)
        else:
            try:
                y_numeric = y.astype(np.float64)
            except (TypeError, ValueError) as error:
                raise ValueError('Regression labels must be finite numbers') \
                    from error
            if not np.isfinite(y_numeric).all():
                raise ValueError('Regression labels must be finite numbers')
            classes = np.array([], dtype=np.int32)
            nr_class = 0
            y_train = y_numeric.astype(np.float32)

        dtrain = xgb.DMatrix(X, label=y_train)
        booster = xgb.train(xgb_params, dtrain, num_boost_round=num_round)
        self._booster = booster
        self._loaded_model_bytes = None
        self._params = fit_params
        self._classes = classes
        self._nr_class = nr_class
        self._fitted = True
        return self

    @staticmethod
    def _from_bundle(manifest, toc, blobs):
        type_id = manifest.get('typeId')
        classifier = type_id == 'wlearn.xgboost.classifier@1'
        regressor = type_id == 'wlearn.xgboost.regressor@1'
        if not classifier and not regressor:
            raise ValueError(f'Unsupported XGBoost bundle typeId: {type_id}')

        params = manifest.get('params', {})
        objective = params.get('objective', 'reg:squarederror')
        _validate_unified_objective(objective)
        meta = manifest.get('metadata', {})
        if meta.get('objective') != objective:
            raise ValueError(
                f'{type_id} objective metadata does not match params')
        if classifier:
            classes = meta.get('classes')
            nr_class = meta.get('nrClass')
            if (objective not in CLASSIFIER_OBJECTIVES or
                    isinstance(nr_class, bool) or
                    not isinstance(nr_class, int) or nr_class < 2 or
                    not isinstance(classes, list) or
                    len(classes) != nr_class or
                    any(isinstance(value, bool) or
                        not isinstance(value, int) or
                        value < np.iinfo(np.int32).min or
                        value > np.iinfo(np.int32).max
                        for value in classes) or
                    any(classes[index] <= classes[index - 1]
                        for index in range(1, len(classes)))):
                raise ValueError(f'{type_id} has invalid classifier metadata')
        elif (objective in CLASSIFIER_OBJECTIVES or
              meta.get('nrClass') != 0 or
              meta.get('classes') not in (None, [])):
            raise ValueError(f'{type_id} has invalid regressor metadata')

        if (not isinstance(toc, list) or len(toc) != 1 or
                toc[0].get('id') != 'model' or
                toc[0].get('mediaType') != 'application/octet-stream'):
            raise ValueError(
                'XGBoost bundle must contain exactly one model artifact')
        entry = toc[0]

        model_bytes = bytes(blobs[entry['offset']:entry['offset'] + entry['length']])

        fd, path = tempfile.mkstemp(suffix='.ubj')
        try:
            with os.fdopen(fd, 'wb') as file:
                file.write(model_bytes)
            booster = xgb.Booster()
            booster.load_model(path)
        finally:
            os.unlink(path)

        identity = _booster_identity(booster)
        if identity['objective'] != objective:
            raise ValueError(f'{type_id} model objective does not match manifest')
        if identity['nTargets'] != 1:
            raise ValueError(f'{type_id} model must contain exactly one target')
        if (classifier and objective.startswith('multi:') and
                identity['nClasses'] != meta['nrClass']):
            raise ValueError(f'{type_id} model class count does not match manifest')
        n_features = meta.get('nFeatures')
        if n_features is not None and (
                isinstance(n_features, bool) or
                not isinstance(n_features, int) or n_features < 1 or
                n_features != identity['nFeatures']):
            raise ValueError(
                f'{type_id} model feature count does not match manifest')

        model = XGBModel(
            booster, params,
            nr_class=meta.get('nrClass', 0),
            classes=meta.get('classes'),
        )
        # Native XGBoost can rewrite UBJ version metadata even without training.
        # Preserve the validated loaded artifact until a successful fit replaces it.
        model._loaded_model_bytes = model_bytes
        return model

    def predict(self, X):
        self._ensure_fitted()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        dm = xgb.DMatrix(X)
        raw = self._booster.predict(dm)
        obj = self._params.get('objective', 'reg:squarederror')

        if obj not in CLASSIFIER_OBJECTIVES:
            return raw.astype(np.float64)

        rows = X.shape[0]
        result = np.empty(rows, dtype=np.int32)

        if obj == 'binary:logistic':
            idx = (raw > 0.5).astype(int)
            for i in range(rows):
                result[i] = self._classes[idx[i]]
        elif obj == 'multi:softprob':
            nc = self._nr_class
            reshaped = raw.reshape(rows, nc)
            best = reshaped.argmax(axis=1)
            for i in range(rows):
                result[i] = self._classes[best[i]]
        elif obj == 'multi:softmax':
            for i in range(rows):
                idx = int(round(raw[i]))
                if idx < 0 or idx >= len(self._classes):
                    raise ValueError(
                        f'XGBoost returned invalid class index {raw[i]} '
                        f'at row {i}')
                result[i] = self._classes[idx]
        else:
            # binary:logitraw, binary:hinge
            idx = (raw > 0).astype(int)
            for i in range(rows):
                result[i] = self._classes[idx[i]]

        return result

    def predict_proba(self, X):
        self._ensure_fitted()
        obj = self._params.get('objective', 'reg:squarederror')
        if obj not in PROBA_OBJECTIVES:
            raise ValueError(
                f'predict_proba requires binary:logistic or multi:softprob, got "{obj}"')

        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        dm = xgb.DMatrix(X)
        raw = self._booster.predict(dm)
        rows = X.shape[0]

        if obj == 'binary:logistic':
            if raw.size != rows:
                raise ValueError(
                    f'XGBoost returned {raw.size} binary probabilities for '
                    f'{rows} rows')
            raw = raw.reshape(-1)
            result = np.empty(rows * 2, dtype=np.float64)
            for i in range(rows):
                result[i * 2] = 1 - raw[i]
                result[i * 2 + 1] = raw[i]
            return result

        # multi:softprob
        expected = rows * self._nr_class
        if raw.size != expected:
            raise ValueError(
                f'XGBoost returned {raw.size} multiclass probabilities; '
                f'expected {expected} for {rows} rows and '
                f'{self._nr_class} classes')
        return raw.astype(np.float64, copy=False).reshape(-1)

    def score(self, X, y):
        preds = self.predict(X)
        y = np.asarray(y)
        if y.size != preds.size:
            raise ValueError(
                f'y length ({y.size}) does not match predictions '
                f'({preds.size})')
        obj = self._params.get('objective', 'reg:squarederror')

        if obj in CLASSIFIER_OBJECTIVES:
            return float(np.mean(preds == y))

        # R-squared
        y = y.astype(np.float64)
        y_mean = y.mean()
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        return 0.0 if ss_tot == 0 else float(1 - ss_res / ss_tot)

    def save(self, path=None):
        self._ensure_fitted()
        output_path = path
        model_bytes = self._loaded_model_bytes
        if model_bytes is None:
            fd, tmp_path = tempfile.mkstemp(suffix='.ubj')
            try:
                os.close(fd)
                self._booster.save_model(tmp_path)
                with open(tmp_path, 'rb') as f:
                    model_bytes = f.read()
            finally:
                os.unlink(tmp_path)

        obj = self._params.get('objective', 'reg:squarederror')
        type_id = ('wlearn.xgboost.classifier@1'
                   if obj in CLASSIFIER_OBJECTIVES
                   else 'wlearn.xgboost.regressor@1')

        bundle = encode_bundle(
            {
                'typeId': type_id,
                'params': self.get_params(),
                'metadata': {
                    'nrClass': self._nr_class,
                    'classes': self._classes.tolist(),
                    'objective': obj,
                    'nFeatures': _booster_identity(self._booster)['nFeatures'],
                },
            },
            [{'id': 'model', 'data': model_bytes}],
        )
        return write_bundle_output(bundle, output_path)

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        self._booster = None
        self._loaded_model_bytes = None
        self._fitted = False

    def get_params(self):
        return dict(self._params)

    def set_params(self, p):
        if ('task' in p and p.get('task') != self._params.get('task') and
                'objective' not in p):
            self._params.pop('objective', None)
        self._params.update(p)
        return self

    @property
    def classes(self):
        self._ensure_fitted()
        return self._classes.copy()

    @property
    def capabilities(self):
        objective = self._params.get('objective', 'reg:squarederror')
        classifier = objective in CLASSIFIER_OBJECTIVES
        return estimator_capabilities(
            classifier=classifier,
            regressor=not classifier,
            predict_proba=objective in PROBA_OBJECTIVES,
        )

    @property
    def is_fitted(self):
        return self._fitted and not self._disposed

    def _ensure_fitted(self):
        if self._disposed:
            raise DisposedError('XGBModel has been disposed.')
        if not self._fitted:
            raise NotFittedError('XGBModel is not fitted.')


    @classmethod
    def default_search_space(cls):
        return {
            'objective': {'type': 'categorical', 'values': ['binary:logistic', 'reg:squarederror']},
            'max_depth': {'type': 'int_uniform', 'low': 3, 'high': 10},
            'eta': {'type': 'log_uniform', 'low': 0.01, 'high': 0.3},
            'numRound': {'type': 'int_uniform', 'low': 50, 'high': 500},
            'subsample': {'type': 'uniform', 'low': 0.5, 'high': 1.0},
            'colsample_bytree': {'type': 'uniform', 'low': 0.5, 'high': 1.0},
            'min_child_weight': {'type': 'log_uniform', 'low': 1, 'high': 10},
            'lambda': {'type': 'log_uniform', 'low': 1e-3, 'high': 10},
            'alpha': {'type': 'log_uniform', 'low': 1e-3, 'high': 10},
        }


register('wlearn.xgboost.classifier@1', XGBModel._from_bundle)
register('wlearn.xgboost.regressor@1', XGBModel._from_bundle)
