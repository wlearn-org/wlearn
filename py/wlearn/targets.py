"""Target shape and row ownership shared by tasks, CV and composites."""
import numpy as np

from .errors import ValidationError


def normalize_targets(y, kind=None):
    try:
        a = np.asarray(y)
    except (TypeError, ValueError) as error:
        raise ValidationError('Targets must be a numeric vector or matrix') from error
    expected = 2 if kind in ('multioutput', 'multilabel') else 1
    if a.ndim != expected:
        raise ValidationError('Matrix targets require an explicit multioutput or multilabel task')
    if a.size == 0 or a.dtype.kind not in 'fiu' or not np.all(np.isfinite(a)):
        raise ValidationError('Targets must be non-empty and finite')
    if kind == 'multilabel' and np.any((a != 0) & (a != 1)):
        raise ValidationError('Multilabel targets must be 0 or 1')
    return np.ascontiguousarray(a)


def target_rows(y):
    try:
        a = np.asarray(y)
    except (ValueError, TypeError) as error:
        raise ValidationError('Targets must be a numeric vector or matrix') from error
    if a.ndim not in (1, 2) or a.shape[0] < 1:
        raise ValidationError('Targets require at least one row')
    return a.shape[0]


def validate_sample_weight(weights, rows):
    a = np.asarray(weights)
    if a.ndim != 1 or len(a) != rows or rows < 1 or a.dtype.kind not in 'fiu':
        raise ValidationError('sample_weight must have one numeric entry per target row')
    total = np.sum(a, dtype=np.float64)
    if not np.all(np.isfinite(a)) or np.any(a < 0) or not np.isfinite(total) or total <= 0:
        raise ValidationError('sample_weight must be finite, nonnegative and have positive finite sum')
    return np.asarray(a, dtype=np.float64)
