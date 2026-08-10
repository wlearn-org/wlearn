"""Shared utilities matching JS automl/common.js."""

import time

import numpy as np

from ._candidate import make_candidate_id, seed_for


def detect_task(y):
    """Detect task type from labels.

    Classification if: integer dtype, or all values are integers and <= 20 unique.
    """
    if hasattr(y, 'dtype') and np.issubdtype(y.dtype, np.integer):
        return 'classification'
    unique = set()
    for v in y:
        v_float = float(v)
        if v_float != round(v_float):
            return 'regression'
        unique.add(v_float)
    return 'classification' if len(unique) <= 20 else 'regression'


def partial_shuffle(indices, k, rng):
    """Partial Fisher-Yates: shuffle only first k positions.

    O(k) time, mutates indices in-place. Returns indices[:k].
    """
    n = len(indices)
    m = min(k, n)
    for i in range(m):
        j = i + int(rng() * (n - i))
        indices[i], indices[j] = indices[j], indices[i]
    return indices[:m]


def scorer_greater_is_better(scoring):
    """All built-in scorers are greater-is-better."""
    return True


def now():
    """High-resolution timer in milliseconds."""
    return time.perf_counter() * 1000
