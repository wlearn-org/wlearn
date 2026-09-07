"""Shared utilities matching JS automl/common.js."""

import inspect
import time

from ._candidate import make_candidate_id, seed_for
from ..task import infer_task_kind as detect_task
from ..measure import get_scorer


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
    return getattr(get_scorer(scoring), 'direction', 'maximize') != 'minimize'


def now():
    """High-resolution timer in milliseconds."""
    return time.perf_counter() * 1000


def default_search_space(model):
    """Keep explicit spaces intact; pass task to model-owned methods that accept it."""
    if model.get('searchSpace') is not None:
        return model['searchSpace']
    cls = model['cls']
    method = getattr(cls, 'default_search_space', None) or getattr(cls, 'defaultSearchSpace', None)
    if method is None:
        return {}
    if 'task' in inspect.signature(method).parameters:
        return method(task=model.get('params', {}).get('task'))
    return method()
