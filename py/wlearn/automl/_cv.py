"""Cross-validation utilities matching JS @wlearn/core/cv.js."""

import numpy as np

from ..errors import ValidationError
from ._rng import make_lcg, shuffle


# --- Scorer registry ---

def accuracy(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true)


def r2_score(y_true, y_pred):
    n = len(y_true)
    mean = sum(float(y_true[i]) for i in range(n)) / n
    ss_tot = sum((float(y_true[i]) - mean) ** 2 for i in range(n))
    ss_res = sum((float(y_true[i]) - float(y_pred[i])) ** 2 for i in range(n))
    if ss_tot == 0:
        return 0.0
    return 1 - ss_res / ss_tot


def neg_mse(y_true, y_pred):
    n = len(y_true)
    s = sum((float(y_true[i]) - float(y_pred[i])) ** 2 for i in range(n))
    return -s / n


def neg_mae(y_true, y_pred):
    n = len(y_true)
    s = sum(abs(float(y_true[i]) - float(y_pred[i])) for i in range(n))
    return -s / n


def neg_logloss(y_true, proba_flat, n_classes=None, classes=None):
    """Negative log-loss for classification.

    Args:
        y_true: np.ndarray of true class labels (int)
        proba_flat: np.ndarray of shape (n * n_classes,) flat row-major probabilities
        n_classes: number of classes (inferred from data if None)
        classes: probability-column labels. Inferred from ``y_true`` only when
            every class is present; otherwise it must be supplied explicitly.

    Returns:
        float: negative log-loss (higher is better)
    """
    n = len(y_true)
    if n == 0:
        raise ValidationError('neg_logloss: y_true must be non-empty')
    probabilities = np.asarray(proba_flat, dtype=np.float64).reshape(-1)
    if n_classes is None:
        if probabilities.size % n != 0:
            raise ValidationError(
                'neg_logloss: probability length must be divisible by rows')
        n_classes = probabilities.size // n
    if (isinstance(n_classes, bool) or not isinstance(n_classes, int)
            or n_classes < 2):
        raise ValidationError('neg_logloss: n_classes must be an integer >= 2')
    if probabilities.size != n * n_classes:
        raise ValidationError(
            'neg_logloss: probability length must equal rows * n_classes')
    if not np.all(np.isfinite(probabilities)):
        raise ValidationError('neg_logloss: probabilities must be finite')
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValidationError('neg_logloss: probabilities must be in [0, 1]')
    if classes is None:
        classes = sorted(set(np.asarray(y_true).tolist()))
        if len(classes) != n_classes:
            raise ValidationError(
                'neg_logloss: classes are required when y_true does not '
                'contain every probability column label')
    else:
        classes = list(np.asarray(classes).tolist())
    if len(classes) != n_classes or len(set(classes)) != n_classes:
        raise ValidationError(
            'neg_logloss: classes must contain one unique label per column')
    class_map = {label: index for index, label in enumerate(classes)}
    eps = 1e-15
    loss = 0.0
    for i, label in enumerate(np.asarray(y_true).tolist()):
        if label not in class_map:
            raise ValidationError(
                f'neg_logloss: class "{label}" is missing from classes')
        p = float(probabilities[i * n_classes + class_map[label]])
        p = min(1 - eps, max(eps, p))
        loss -= np.log(p)
    return -loss / n


_SCORERS = {
    'accuracy': accuracy,
    'r2': r2_score,
    'neg_mse': neg_mse,
    'neg_mae': neg_mae,
}


def get_scorer(scoring):
    """Get a scorer function by name or pass through a callable."""
    if callable(scoring):
        return scoring
    fn = _SCORERS.get(scoring)
    if fn is None:
        raise ValidationError(
            f'Unknown scoring: "{scoring}". Available: {", ".join(_SCORERS.keys())}'
        )
    return fn


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


def cross_val_score(cls, X, y, cv=5, scoring='accuracy', seed=42, params=None):
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

    # Detect task from y for fold generation
    from ._common import detect_task
    task = detect_task(y)

    if task == 'classification':
        folds = stratified_k_fold(y, cv, do_shuffle=True, seed=seed)
    else:
        folds = k_fold(len(y), cv, do_shuffle=True, seed=seed)

    scores = np.zeros(len(folds), dtype=np.float64)

    for f, (train, test) in enumerate(folds):
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]

        model = cls.create(params)
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            scores[f] = scorer_fn(y_test, preds)
        finally:
            model.dispose()

    return scores
