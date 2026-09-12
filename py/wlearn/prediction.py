"""Prediction arrays use row-major [row, target, level] axes.

Intervals add a final lower/upper axis. Classification sets use [row, level,
class]; multilabel sets use [row, level, label, state]. Samples use [row, draw,
target]. Numerical region owners validate positive definiteness.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from numbers import Integral
from typing import Any

import numpy as np

from .errors import ValidationError

PREDICTION_FIELDS = ('response', 'proba', 'score', 'decision', 'interval', 'quantiles', 'sets', 'samples', 'region')
_ARRAY_FIELDS = ('truth', 'response', 'proba', 'score', 'decision', 'interval', 'quantiles', 'classes', 'sets', 'samples', 'sample_weights', 'quantile_levels', 'coverage_levels')


@dataclass
class Prediction:
    task_id: str | None = None
    task_kind: str | None = None
    rows: int | None = None
    target_count: int = 1
    target_names: list[str] | None = None
    row_ids: Any = None
    truth: np.ndarray | None = None
    response: np.ndarray | None = None
    proba: np.ndarray | None = None
    proba_rows: int | None = None
    score: np.ndarray | None = None
    decision: np.ndarray | None = None
    interval: np.ndarray | None = None
    quantiles: np.ndarray | None = None
    quantile_levels: np.ndarray | None = None
    coverage_levels: np.ndarray | None = None
    classes: np.ndarray | None = None
    sets: np.ndarray | None = None
    samples: np.ndarray | None = None
    sample_count: int | None = None
    sample_kind: str | None = None
    sample_dependence: str | None = None
    sample_weights: np.ndarray | None = None
    region: dict | None = None
    feature_schema_hash: str | None = None
    model_artifact_hash: str | None = None
    warnings: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _array(value, name):
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValidationError(f'Prediction.{name} must be a flat numeric array') from error
    if arr.ndim != 1 or arr.dtype.kind not in 'fiu':
        raise ValidationError(f'Prediction.{name} must be a flat numeric array')
    return arr


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= 2**53 - 1:
        raise ValidationError(f'Prediction.{name} must be a positive safe integer')
    return int(value)


def _levels(value, name, endpoints=False):
    a = np.asarray(_array(value, name), dtype=np.float64)
    valid = np.all((a >= 0) & (a <= 1)) if endpoints else np.all((a > 0) & (a < 1))
    if not len(a) or not valid or not np.all(np.isfinite(a)) or np.any(np.diff(a) <= 0):
        raise ValidationError(f'Prediction.{name} must contain strictly increasing probabilities')
    return len(a)


def _infer_rows(p):
    if p.rows is not None:
        return p.rows
    t = p.target_count
    for key in ('truth', 'response', 'score', 'decision'):
        if getattr(p, key) is not None:
            n, d = len(getattr(p, key)), t
            return n // d if n % d == 0 else None
    if p.row_ids is not None:
        return len(p.row_ids)
    if p.proba_rows is not None:
        return p.proba_rows
    if p.proba is not None:
        d = t if p.task_kind == 'multilabel' else len(p.classes) if p.classes is not None else 0
        return len(p.proba) // d if d and len(p.proba) % d == 0 else None
    for key, level, factor in [('quantiles', 'quantile_levels', 1), ('interval', 'coverage_levels', 2)]:
        a, b = getattr(p, key), getattr(p, level)
        if a is not None and b is not None and len(b):
            d = t * len(b) * factor
            return len(a) // d if len(a) % d == 0 else None
    return None


def create_prediction(**kwargs):
    p = Prediction(**{f.name: kwargs[f.name] for f in fields(Prediction) if f.name in kwargs})
    for name in _ARRAY_FIELDS:
        if getattr(p, name) is not None:
            setattr(p, name, _array(getattr(p, name), name))
    p.warnings, p.metadata = list(p.warnings or []), dict(p.metadata or {})
    validate_prediction(p)
    p.rows = _infer_rows(p)
    return p


def validate_prediction(p):
    if not isinstance(p, Prediction) or not any(getattr(p, f) is not None for f in PREDICTION_FIELDS):
        raise ValidationError('Prediction must contain at least one prediction field')
    t = _positive(p.target_count, 'target_count')
    if p.task_kind is not None and p.task_kind not in ('classification', 'regression', 'multioutput', 'multilabel'):
        raise ValidationError('Prediction.task_kind is unsupported')
    if t > 1 and p.task_kind not in ('multioutput', 'multilabel'):
        raise ValidationError('Multiple targets require task_kind multioutput or multilabel')
    if p.target_names is not None:
        names = p.target_names
        if not isinstance(names, list) or len(names) != t or any(not isinstance(v, str) or not v for v in names) or len(set(names)) != t:
            raise ValidationError('Prediction.target_names must uniquely name every target')
    rows = _positive(_infer_rows(p), 'rows')
    nt = _positive(rows * t, 'size')
    if p.row_ids is not None and len(p.row_ids) != rows:
        raise ValidationError('Prediction.row_ids length must match rows')
    if p.proba_rows is not None and _positive(p.proba_rows, 'proba_rows') != rows:
        raise ValidationError('Prediction.proba_rows must match rows')

    def check(name, n, extended=False):
        value = getattr(p, name)
        if value is None:
            return
        value = _array(value, name)
        if len(value) != _positive(n, 'size'):
            raise ValidationError(f'Prediction.{name} length must equal {n}')
        invalid = np.any(np.isnan(value)) if extended else not np.all(np.isfinite(value))
        if invalid:
            message = 'must not contain NaN' if extended else 'must be finite'
            raise ValidationError(f'Prediction.{name} {message}')

    for name in ('truth', 'response', 'score', 'decision'):
        check(name, nt)
    if p.task_kind == 'multilabel':
        for name in ('truth', 'response'):
            a = getattr(p, name)
            if a is not None and np.any((np.asarray(a) != 0) & (np.asarray(a) != 1)):
                raise ValidationError(f'Multilabel {name} values must be 0 or 1')
        if p.classes is not None:
            raise ValidationError('Multilabel predictions use target axes, not a shared class axis')
    if p.classes is not None:
        a = _array(p.classes, 'classes')
        if not len(a) or len(np.unique(a)) != len(a) or not np.all(np.isfinite(a)):
            raise ValidationError('Prediction.classes must contain unique finite labels')
    if p.proba is not None:
        if p.task_kind == 'multioutput':
            raise ValidationError('Multioutput regression has no class probabilities')
        cols = t if p.task_kind == 'multilabel' else len(p.classes) if p.classes is not None else len(p.proba) // rows
        check('proba', rows * _positive(cols, 'probability columns'))
        a = np.asarray(p.proba).reshape(rows, cols)
        if np.any((a < 0) | (a > 1)):
            raise ValidationError('Prediction.proba values must be in [0, 1]')
        if p.task_kind != 'multilabel' and np.any(np.abs(a.sum(axis=1) - 1) > 1e-6):
            raise ValidationError('Prediction.proba rows must sum to 1')
    if p.quantile_levels is not None:
        _levels(p.quantile_levels, 'quantile_levels', True)
    if p.coverage_levels is not None:
        _levels(p.coverage_levels, 'coverage_levels')
    if p.quantiles is not None:
        q = _levels(p.quantile_levels, 'quantile_levels', True)
        check('quantiles', nt * q, True)
        a = np.asarray(p.quantiles).reshape(nt, q)
        if np.any(a[:, 1:] < a[:, :-1]):
            raise ValidationError('Prediction.quantiles must be nondecreasing within each target')
    if p.interval is not None:
        k = _levels(p.coverage_levels, 'coverage_levels')
        check('interval', nt * k * 2, True)
        a = np.asarray(p.interval).reshape(-1, 2)
        empty = (a[:, 0] == np.inf) & (a[:, 1] == -np.inf)
        if np.any((a[:, 0] > a[:, 1]) & ~empty):
            raise ValidationError('Prediction.interval has reversed bounds')
    if p.sets is not None:
        k = _levels(p.coverage_levels, 'coverage_levels')
        cols = t * 2 if p.task_kind == 'multilabel' else _positive(len(p.classes) if p.classes is not None else 0, 'classes length')
        check('sets', rows * k * cols)
        a = np.asarray(p.sets)
        if np.any((a != 0) & (a != 1)):
            raise ValidationError('Prediction.sets entries must be 0 or 1')
    # Paired target draws are a model declaration, never inferred from array shape.
    if p.sample_dependence is not None and (p.samples is None or p.sample_dependence not in ('joint', 'marginal')):
        raise ValidationError('Prediction.sample_dependence requires samples and must be joint or marginal')
    if p.samples is not None:
        n = _positive(p.sample_count, 'sample_count')
        if p.sample_kind not in ('outcome', 'mean'):
            raise ValidationError('Prediction.sample_kind must be outcome or mean')
        check('samples', rows * n * t)
    if p.sample_weights is not None:
        if p.samples is None:
            raise ValidationError('Prediction.sample_weights requires samples')
        check('sample_weights', p.sample_count)
        weights = np.asarray(p.sample_weights)
        if np.any(weights < 0) or not np.any(weights > 0):
            raise ValidationError('Prediction.sample_weights must be nonnegative relative draw weights with positive total')
    if p.region is not None:
        k = _levels(p.coverage_levels, 'coverage_levels')
        if not isinstance(p.region, dict) or p.region.get('kind') != 'ellipsoid':
            raise ValidationError('Prediction.region kind must be ellipsoid')
        for key, n in [('centers', nt), ('precision', t*t), ('radii', rows*k)]:
            a = _array(p.region.get(key), 'region.'+key)
            valid = not np.any(np.isnan(a) | (a < 0)) if key == 'radii' else np.all(np.isfinite(a))
            if len(a) != n or not valid:
                raise ValidationError(f'Prediction.region.{key} has invalid dimensions or values')
    return p


def prediction_rows(prediction):
    return _infer_rows(validate_prediction(prediction))


def prediction_field(prediction, field_name):
    validate_prediction(prediction)
    if field_name not in PREDICTION_FIELDS and field_name != 'truth':
        raise ValidationError(f'Unknown prediction field "{field_name}"')
    return getattr(prediction, field_name)
