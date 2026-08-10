"""Structured prediction objects for scoring and reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np

from .errors import ValidationError


PREDICTION_FIELDS = ('response', 'proba', 'score', 'decision', 'interval', 'quantiles')


@dataclass
class Prediction:
    task_id: str | None = None
    row_ids: np.ndarray | list[str] | None = None
    truth: np.ndarray | None = None
    response: np.ndarray | None = None
    proba: np.ndarray | None = None
    proba_rows: int | None = None
    score: np.ndarray | None = None
    decision: np.ndarray | None = None
    interval: np.ndarray | None = None
    quantiles: np.ndarray | None = None
    classes: np.ndarray | None = None
    feature_schema_hash: str | None = None
    model_artifact_hash: str | None = None
    warnings: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _labels(x: Any) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValidationError('Prediction labels must be 1D')
    return arr


def _float_array(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValidationError(f'Prediction.{name} must be flat')
    return arr


def _infer_rows(prediction: Prediction) -> int | None:
    if prediction.truth is not None:
        return len(prediction.truth)
    if prediction.response is not None:
        return len(prediction.response)
    if prediction.score is not None:
        return len(prediction.score)
    if prediction.decision is not None:
        return len(prediction.decision)
    if prediction.row_ids is not None:
        return len(prediction.row_ids)
    if prediction.proba_rows is not None:
        return prediction.proba_rows
    if prediction.proba is not None and prediction.classes is not None and len(prediction.classes) > 0:
        rows = len(prediction.proba) / len(prediction.classes)
        return int(rows) if rows.is_integer() else None
    return None


def create_prediction(**kwargs: Any) -> Prediction:
    pred = Prediction(
        task_id=kwargs.get('task_id'),
        row_ids=kwargs.get('row_ids'),
        truth=None if kwargs.get('truth') is None else _labels(kwargs.get('truth')),
        response=None if kwargs.get('response') is None else _labels(kwargs.get('response')),
        proba=None if kwargs.get('proba') is None else _float_array(kwargs.get('proba'), 'proba'),
        proba_rows=kwargs.get('proba_rows'),
        score=None if kwargs.get('score') is None else _float_array(kwargs.get('score'), 'score'),
        decision=None if kwargs.get('decision') is None else _float_array(kwargs.get('decision'), 'decision'),
        interval=None if kwargs.get('interval') is None else _float_array(kwargs.get('interval'), 'interval'),
        quantiles=None if kwargs.get('quantiles') is None else _float_array(kwargs.get('quantiles'), 'quantiles'),
        classes=None if kwargs.get('classes') is None else _labels(kwargs.get('classes')),
        feature_schema_hash=kwargs.get('feature_schema_hash'),
        model_artifact_hash=kwargs.get('model_artifact_hash'),
        warnings=list(kwargs.get('warnings') or []),
        metadata=dict(kwargs.get('metadata') or {}),
    )
    return validate_prediction(pred)


def validate_prediction(prediction: Prediction) -> Prediction:
    if not isinstance(prediction, Prediction):
        raise ValidationError('Prediction must be a Prediction')
    if not any(getattr(prediction, field) is not None for field in PREDICTION_FIELDS):
        raise ValidationError('Prediction must contain at least one prediction field')
    rows = _infer_rows(prediction)
    if rows is None or rows < 1:
        raise ValidationError('Prediction row count could not be inferred')
    if not isinstance(rows, Integral):
        raise ValidationError('Prediction row count must be an integer')

    for field_name in ('truth', 'response', 'score', 'decision'):
        value = getattr(prediction, field_name)
        if value is not None and len(value) != rows:
            raise ValidationError(f'Prediction.{field_name} length must match row count')
    for field_name in ('interval', 'quantiles'):
        value = getattr(prediction, field_name)
        if value is not None and len(value) % rows != 0:
            raise ValidationError(f'Prediction.{field_name} length must be divisible by row count')
    if prediction.row_ids is not None and len(prediction.row_ids) != rows:
        raise ValidationError('Prediction.row_ids length must match row count')

    if prediction.proba is not None:
        if prediction.proba_rows is not None:
            if not isinstance(prediction.proba_rows, Integral) or prediction.proba_rows < 1:
                raise ValidationError('Prediction.proba_rows must be a positive integer')
            if prediction.proba_rows != rows:
                raise ValidationError('Prediction.proba_rows must match row count')
        if prediction.classes is not None:
            if len(prediction.classes) == 0:
                raise ValidationError('Prediction.classes must be non-empty')
            if len(prediction.proba) != rows * len(prediction.classes):
                raise ValidationError('Prediction.proba length must equal rows * classes')
        elif prediction.proba_rows is not None and len(prediction.proba) % prediction.proba_rows != 0:
            raise ValidationError('Prediction.proba length must be divisible by proba_rows')
    return prediction


def prediction_rows(prediction: Prediction) -> int:
    rows = _infer_rows(validate_prediction(prediction))
    assert rows is not None
    return rows


def prediction_field(prediction: Prediction, field_name: str) -> Any:
    validate_prediction(prediction)
    if field_name not in PREDICTION_FIELDS and field_name != 'truth':
        raise ValidationError(f'Unknown prediction field "{field_name}"')
    return getattr(prediction, field_name)
