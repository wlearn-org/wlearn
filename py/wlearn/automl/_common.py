"""Shared utilities matching JS automl/common.js."""

import json
import time

import numpy as np


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


def _stable_stringify(obj):
    """Stable JSON stringify with sorted keys matching JS stableStringify."""
    if obj is None:
        return 'null'
    if isinstance(obj, bool):
        return 'true' if obj else 'false'
    if isinstance(obj, (int, float)):
        return str(obj)
    if isinstance(obj, str):
        return json.dumps(obj)
    if isinstance(obj, (list, tuple)):
        return '[' + ','.join(_stable_stringify(v) for v in obj) + ']'
    if isinstance(obj, dict):
        keys = sorted(obj.keys())
        return '{' + ','.join(
            json.dumps(k) + ':' + _stable_stringify(obj[k]) for k in keys
        ) + '}'
    return str(obj)


def make_candidate_id(model_label, params):
    """Stable candidate ID from model label and params."""
    return model_label + ':' + _stable_stringify(params)


def _hash_string(s):
    """FNV-1a inspired hash matching JS hashString."""
    h = 0x811c9dc5
    for ch in s:
        h ^= ord(ch)
        h = (h * 0x01000193) & 0x7fffffff
    return h


def seed_for(candidate_id, fold_idx, base_seed):
    """Derive a deterministic seed from base seed, candidate ID, and fold index."""
    h = _hash_string(candidate_id)
    s = (base_seed * 2654435761 + h * 40503 + fold_idx * 65537) & 0x7fffffff
    s = (((s >> 16) ^ s) * 0x45d9f3b) & 0x7fffffff
    return s


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
