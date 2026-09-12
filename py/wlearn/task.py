"""Task and feature-schema primitives for wlearn."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import ValidationError
from .targets import normalize_targets


def task_params(params, task):
    if task not in ('classification', 'regression', 'multioutput', 'multilabel'):
        raise ValidationError('Unsupported estimator task')
    params = dict(params or {})
    if params.get('task') is not None and params['task'] != task:
        raise ValidationError(f'Estimator task conflicts with requested task "{task}"')
    return {**params, 'task': task}


def validate_estimator_task(model, task):
    caps = getattr(model, 'capabilities', {}) or {}
    if task in ('multioutput', 'multilabel') and not caps.get(task):
        raise ValidationError(f'Estimator must declare {task} capability')
    if ((task == 'regression' and caps.get('classifier') and not caps.get('regressor'))
            or (task == 'classification' and caps.get('regressor') and not caps.get('classifier'))):
        raise ValidationError(f'Fitted estimator capabilities conflict with requested task "{task}"')
    return model


TASK_KINDS = (
    'classification',
    'regression',
    'clustering',
    'ranking',
    'survival',
    'forecasting',
    'multioutput',
    'multilabel',
    'anomaly',
)


@dataclass
class FeatureDef:
    name: str
    index: int
    type: str = 'numeric'
    role: str = 'feature'


@dataclass
class FeatureSchema:
    rows: int
    cols: int
    features: list[FeatureDef]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    id: str
    kind: str
    X: np.ndarray
    y: np.ndarray | None
    feature_schema: FeatureSchema
    target_schema: dict[str, Any] | None = None
    row_ids: np.ndarray | list[str] | None = None
    groups: np.ndarray | None = None
    weights: np.ndarray | None = None
    row_roles: dict[str, np.ndarray] | None = None
    provenance: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _matrix(X: Any) -> np.ndarray:
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValidationError('X must be a 2D matrix')
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValidationError('X must have positive rows and columns')
    return arr


def _labels(y: Any) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValidationError('labels must be a 1D array')
    return arr


def infer_task_kind(y: Any | None) -> str:
    if y is None:
        return 'clustering'
    arr = _labels(y)
    if np.issubdtype(arr.dtype, np.integer):
        return 'classification'
    unique = np.unique(arr)
    if np.all(np.equal(unique, np.floor(unique))) and 1 < len(unique) <= 20:
        return 'classification'
    return 'regression'


def create_feature_schema(
    X: Any,
    names: list[str] | None = None,
    types: list[str] | None = None,
    roles: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> FeatureSchema:
    Xn = _matrix(X)
    rows, cols = Xn.shape
    for label, value in [('names', names), ('types', types), ('roles', roles)]:
        if value is not None and len(value) != cols:
            raise ValidationError(
                f'FeatureSchema {label} length ({len(value)}) must equal matrix cols ({cols})'
            )
    features = [
        FeatureDef(
            name=str(names[i]) if names else f'x{i}',
            index=i,
            type=str(types[i]) if types else 'numeric',
            role=str(roles[i]) if roles else 'feature',
        )
        for i in range(cols)
    ]
    return FeatureSchema(rows=rows, cols=cols, features=features, metadata=dict(metadata or {}))


def validate_feature_schema(schema: FeatureSchema, cols: int | None = None, rows: int | None = None) -> FeatureSchema:
    if not isinstance(schema, FeatureSchema):
        raise ValidationError('FeatureSchema must be a FeatureSchema')
    if schema.rows < 1:
        raise ValidationError('FeatureSchema.rows must be positive')
    if schema.cols < 1:
        raise ValidationError('FeatureSchema.cols must be positive')
    if rows is not None and schema.rows != rows:
        raise ValidationError(f'FeatureSchema.rows ({schema.rows}) does not match matrix rows ({rows})')
    if cols is not None and schema.cols != cols:
        raise ValidationError(f'FeatureSchema.cols ({schema.cols}) does not match matrix cols ({cols})')
    if len(schema.features) != schema.cols:
        raise ValidationError('FeatureSchema.features length must equal FeatureSchema.cols')
    seen = set()
    for i, feature in enumerate(schema.features):
        if feature.index != i:
            raise ValidationError(f'FeatureSchema.features[{i}].index must equal {i}')
        if not feature.name:
            raise ValidationError(f'FeatureSchema.features[{i}].name must be non-empty')
        if feature.name in seen:
            raise ValidationError(f'FeatureSchema contains duplicate feature name "{feature.name}"')
        seen.add(feature.name)
        if not feature.type or not feature.role:
            raise ValidationError(f'FeatureSchema.features[{i}] must have type and role')
    return schema


def validate_row_roles(row_roles: dict[str, np.ndarray], rows: int) -> dict[str, np.ndarray]:
    if not isinstance(row_roles, dict):
        raise ValidationError('Task.row_roles must be a dict')
    for role, indices in row_roles.items():
        arr = np.asarray(indices, dtype=np.int32)
        if np.any(arr < 0) or np.any(arr >= rows):
            raise ValidationError(f'Task.row_roles.{role} contains out-of-range row index')
        row_roles[role] = arr
    return row_roles


def create_task(
    *,
    X: Any,
    id: str | None = None,
    kind: str | None = None,
    y: Any | None = None,
    feature_schema: FeatureSchema | None = None,
    target_schema: dict[str, Any] | None = None,
    row_ids: Any | None = None,
    groups: Any | None = None,
    weights: Any | None = None,
    row_roles: dict[str, np.ndarray] | None = None,
    provenance: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Task:
    Xn = _matrix(X)
    yn = None if y is None else normalize_targets(y, kind)
    task_kind = kind or infer_task_kind(yn)
    if task_kind not in TASK_KINDS:
        raise ValidationError(f'Unsupported task kind "{task_kind}"')
    if yn is not None and len(yn) != Xn.shape[0]:
        raise ValidationError(f'Task y length ({len(yn)}) must match X rows ({Xn.shape[0]})')

    group_values = None if groups is None else _labels(groups)
    if group_values is not None and len(group_values) != Xn.shape[0]:
        raise ValidationError('Task groups length must match X rows')
    weight_values = None if weights is None else _labels(weights)
    if weight_values is not None and len(weight_values) != Xn.shape[0]:
        raise ValidationError('Task weights length must match X rows')
    if row_ids is not None and len(row_ids) != Xn.shape[0]:
        raise ValidationError('Task row_ids length must match X rows')

    schema = feature_schema or create_feature_schema(Xn)
    validate_feature_schema(schema, cols=Xn.shape[1], rows=Xn.shape[0])
    if row_roles is not None:
        row_roles = validate_row_roles(dict(row_roles), Xn.shape[0])

    return validate_task(Task(
        id=id or f'{task_kind}-{Xn.shape[0]}x{Xn.shape[1]}',
        kind=task_kind,
        X=Xn,
        y=yn,
        feature_schema=schema,
        target_schema=dict(target_schema) if target_schema else None,
        row_ids=row_ids,
        groups=group_values,
        weights=weight_values,
        row_roles=row_roles,
        provenance=dict(provenance) if provenance else None,
        metadata=dict(metadata or {}),
    ))


def validate_task(task: Task) -> Task:
    if not isinstance(task, Task):
        raise ValidationError('Task must be a Task')
    if not task.id:
        raise ValidationError('Task.id must be non-empty')
    if task.kind not in TASK_KINDS:
        raise ValidationError(f'Unsupported task kind "{task.kind}"')
    Xn = _matrix(task.X)
    validate_feature_schema(task.feature_schema, cols=Xn.shape[1], rows=Xn.shape[0])
    if task.y is not None and len(normalize_targets(task.y, task.kind)) != Xn.shape[0]:
        raise ValidationError('Task.y length must match Task.X rows')
    if task.groups is not None and len(task.groups) != Xn.shape[0]:
        raise ValidationError('Task.groups length must match Task.X rows')
    if task.weights is not None and len(task.weights) != Xn.shape[0]:
        raise ValidationError('Task.weights length must match Task.X rows')
    if task.row_ids is not None and len(task.row_ids) != Xn.shape[0]:
        raise ValidationError('Task.row_ids length must match Task.X rows')
    if task.row_roles is not None:
        validate_row_roles(task.row_roles, Xn.shape[0])
    return task


def task_rows(task: Task) -> int:
    return int(validate_task(task).X.shape[0])
