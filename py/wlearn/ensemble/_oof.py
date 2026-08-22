"""Out-of-fold predictions matching JS @wlearn/ensemble/oof.js."""

import numpy as np

from ..automl._cv import stratified_k_fold, k_fold
from ._class_order import (
    class_column_map, require_probability_model,
    validate_probability_output, validate_regression_output,
)


def get_oof_predictions(estimator_specs, X, y, cv=5, seed=42, task='classification'):
    """Generate out-of-fold predictions for ensemble building.

    Args:
        estimator_specs: list of (name, cls, params) tuples
        X: np.ndarray feature matrix
        y: np.ndarray labels
        cv: number of folds
        seed: random seed
        task: 'classification' or 'regression'

    Returns:
        dict with:
            'oofPreds': list of np.ndarray (one per estimator)
            'classes': np.ndarray or None
    """
    n = len(X)

    if task == 'classification':
        folds = stratified_k_fold(y, cv, do_shuffle=True, seed=seed)
    else:
        folds = k_fold(n, cv, do_shuffle=True, seed=seed)

    classes = None
    n_classes = 0
    if task == 'classification':
        labels = sorted(set(int(v) for v in y))
        classes = np.array(labels, dtype=np.int32)
        n_classes = len(classes)

    oof_preds = []

    for name, est_cls, params in estimator_specs:
        if task == 'classification':
            oof = np.zeros(n * n_classes, dtype=np.float64)
        else:
            oof = np.zeros(n, dtype=np.float64)

        for train, test in folds:
            X_train, y_train = X[train], y[train]
            X_test = X[test]

            model = est_cls.create(params or {})
            operation_error = None
            try:
                model.fit(X_train, y_train)
                if task == 'classification':
                    label = f'OOF estimator "{name}"'
                    require_probability_model(model, label)
                    columns = class_column_map(model, classes, label)
                    proba = validate_probability_output(
                        model.predict_proba(X_test), len(test),
                        n_classes, label)
                    for i in range(len(test)):
                        row = test[i]
                        for c in range(n_classes):
                            oof[row * n_classes + c] = \
                                proba[i * n_classes + columns[c]]
                else:
                    preds = validate_regression_output(
                        model.predict(X_test), len(test),
                        f'OOF estimator "{name}"')
                    for i in range(len(test)):
                        oof[test[i]] = float(preds[i])
            except Exception as error:
                operation_error = error
                raise
            finally:
                try:
                    model.dispose()
                except Exception:
                    if operation_error is None:
                        raise

        oof_preds.append(oof)

    return {'oofPreds': oof_preds, 'classes': classes}
