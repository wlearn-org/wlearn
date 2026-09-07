"""Cross-validation utilities matching JS @wlearn/core/cv.js."""

import numpy as np

from .errors import ValidationError
from .rng import make_lcg, shuffle


from .measure import get_scorer, score_estimator
from .task import infer_task_kind, task_params, validate_estimator_task

accuracy = get_scorer('accuracy')
r2_score = get_scorer('r2')
neg_mse = get_scorer('neg_mse')
neg_mae = get_scorer('neg_mae')


def neg_logloss(y_true, proba_flat, n_classes=None, classes=None):
    labels = np.unique(y_true) if classes is None else classes
    if n_classes is not None and n_classes != len(labels):
        raise ValidationError('neg_logloss: classes are required when observed labels do not cover probability columns')
    if not set(y_true).issubset(set(labels)):
        raise ValidationError('neg_logloss: target label missing from classes')
    return -get_scorer('log_loss')(y_true, proba_flat, classes=labels)


# --- Fold generators ---

def k_fold(n, k=5, do_shuffle=True, seed=42):
    """Generate k-fold CV splits matching JS kFold.

    Returns list of (train_indices, test_indices) as np.int32 arrays.
    """
    if not isinstance(n, int) or n < 2:
        raise ValidationError('kFold: n must be an integer >= 2')
    if not isinstance(k, int) or k < 2:
        raise ValidationError('kFold: k must be an integer >= 2')
    if n < k:
        raise ValidationError(f'kFold: n ({n}) must be >= k ({k})')

    indices = np.arange(n, dtype=np.int32)
    if do_shuffle:
        rng = make_lcg(seed)
        shuffle(indices, rng)

    fold_size = n // k
    remainder = n % k
    folds = []
    offset = 0

    for f in range(k):
        size = fold_size + (1 if f < remainder else 0)
        test_idx = indices[offset:offset + size].copy()
        train_parts = []
        if offset > 0:
            train_parts.append(indices[:offset])
        if offset + size < n:
            train_parts.append(indices[offset + size:])
        if train_parts:
            train_idx = np.concatenate(train_parts).astype(np.int32)
        else:
            train_idx = np.array([], dtype=np.int32)
        folds.append((train_idx, test_idx))
        offset += size

    return folds


def stratified_k_fold(y, k=5, do_shuffle=True, seed=42):
    """Generate stratified k-fold CV splits matching JS stratifiedKFold.

    Returns list of (train_indices, test_indices) as np.int32 arrays.
    """
    n = len(y)
    if not isinstance(k, int) or k < 2:
        raise ValidationError('stratifiedKFold: k must be an integer >= 2')
    if n < k:
        raise ValidationError(f'stratifiedKFold: n ({n}) must be >= k ({k})')

    # Group indices by class
    class_map = {}
    for i in range(n):
        label = y[i].item() if hasattr(y[i], 'item') else y[i]
        if label not in class_map:
            class_map[label] = []
        class_map[label].append(i)
    for label, indices in class_map.items():
        if len(indices) < k:
            raise ValidationError(
                f'stratifiedKFold: class "{label}" has only {len(indices)} samples, less than k ({k})'
            )

    if do_shuffle:
        rng = make_lcg(seed)
        for label in class_map.keys():
            indices = class_map[label]
            shuffle(indices, rng)

    # Assign each class's samples round-robin to folds
    fold_tests = [[] for _ in range(k)]
    for label in class_map.keys():
        indices = class_map[label]
        for i, idx in enumerate(indices):
            fold_tests[i % k].append(idx)

    all_indices = np.arange(n, dtype=np.int32)
    folds = []
    for f in range(k):
        test_set = set(fold_tests[f])
        test = np.array(fold_tests[f], dtype=np.int32)
        train = np.array([i for i in all_indices if i not in test_set], dtype=np.int32)
        folds.append((train, test))

    return folds


def cross_val_score(cls, X, y, cv=5, scoring='accuracy', seed=42, params=None, task=None):
    """Run cross-validation and return fold scores.

    Args:
        cls: model class with create(params), fit(X, y), predict(X), dispose()
        X: feature matrix (np.ndarray)
        y: labels (np.ndarray)
        cv: number of folds
        scoring: scorer name or function
        seed: random seed for fold generation
        params: model hyperparameters

    Returns:
        np.ndarray of fold scores (float64)
    """
    if params is None:
        params = {}

    scorer_fn = get_scorer(scoring)

    from .resampling import resolve_cv
    task = task or params.get('task') or infer_task_kind(y)
    folds = resolve_cv(cv, y, task=task, seed=seed)

    scores = np.zeros(len(folds), dtype=np.float64)

    for f, (train, test) in enumerate(folds):
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]

        model = cls.create(task_params(params, task))
        try:
            model.fit(X_train, y_train)
            validate_estimator_task(model, task)
            scores[f] = score_estimator(model, X_test, y_test, scorer_fn)
        finally:
            model.dispose()

    return scores
