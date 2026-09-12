"""Serializable resampling plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .cv import k_fold, stratified_k_fold
from .rng import make_lcg, shuffle as lcg_shuffle
from .errors import ValidationError


RESAMPLING_STRATEGIES = (
    'holdout',
    'kfold',
    'stratified_kfold',
    'repeated_kfold',
    'group_kfold',
    'time_series',
    'sliding_window',
    'sliding_index',
    'sliding_period',
)


@dataclass
class ResamplingFold:
    fold_id: str
    train: np.ndarray
    test: np.ndarray
    validate: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResamplingPlan:
    id: str
    strategy: str
    n: int
    folds: list[ResamplingFold]
    task_id: str | None = None
    seed: int | None = 42
    constraints: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def serialize_cv(cv):
    if isinstance(cv, int):
        return cv
    source = cv.folds if isinstance(cv, ResamplingPlan) else cv
    result = []
    for fold in source:
        if isinstance(fold, ResamplingFold):
            result.append({'train': fold.train.tolist(), 'test': fold.test.tolist(),
                           **({'validate': fold.validate.tolist()} if fold.validate is not None else {})})
        elif isinstance(fold, dict):
            result.append({**fold, 'train': np.asarray(fold['train']).tolist(), 'test': np.asarray(fold['test']).tolist(),
                           **({'validate': np.asarray(fold['validate']).tolist()} if fold.get('validate') is not None else {})})
        else:
            result.append({'train': np.asarray(fold[0]).tolist(), 'test': np.asarray(fold[1]).tolist()})
    return result


def resolve_cv(cv, y, *, task=None, seed=42, require_complete=False):
    from .task import infer_task_kind
    row_count = isinstance(y, (int, np.integer)) and not isinstance(y, (bool, np.bool_))
    task = task or ('regression' if row_count else infer_task_kind(y))
    n = int(y) if row_count else len(y)
    if not 1 <= n <= 2**53-1:
        raise ValidationError('CV requires a positive row count')
    if row_count and task == 'classification':
        raise ValidationError('Stratified CV requires class labels')
    if isinstance(cv, int) and not isinstance(cv, bool):
        source = (stratified_k_fold(y, cv, seed=seed) if task == 'classification'
                  else k_fold(n, cv, seed=seed))
    elif isinstance(cv, (list, tuple)):
        source = cv
    else:
        validate_resampling_plan(cv)
        if cv.n != n:
            raise ValidationError('CV plan row count must match y')
        source = cv.folds
    if not source:
        raise ValidationError('CV folds must be non-empty')
    folds = []
    counts = np.zeros(n, dtype=np.int32) if require_complete else None
    for index, fold in enumerate(source):
        validate = None
        if isinstance(fold, ResamplingFold):
            train, test, validate = fold.train, fold.test, fold.validate
        elif isinstance(fold, dict):
            train, test, validate = fold.get('train'), fold.get('test'), fold.get('validate')
        elif isinstance(fold, (list, tuple)) and len(fold) == 2:
            train, test = fold
        else:
            raise ValidationError('CV fold must contain train and test indices')
        arrays = []
        for values in ((train, test) if validate is None else (train, test, validate)):
            arr = np.asarray(values)
            if (arr.ndim != 1 or arr.dtype.kind not in 'iu' or arr.size == 0
                    or np.any(arr < 0) or np.any(arr >= n)):
                raise ValidationError('CV folds must contain valid integer row indices')
            arrays.append(arr.astype(np.int32, copy=True))
        normalized = ResamplingFold(str(index), *arrays)
        _validate_fold(normalized, n)
        if counts is not None:
            counts[normalized.test] += 1
        folds.append((normalized.train, normalized.test))
    if require_complete and np.any(counts != 1):
        raise ValidationError('OOF requires every row in test folds exactly once')
    return folds


def create_resampling_plan(
    *,
    id: str | None = None,
    strategy: str = 'kfold',
    n: int | None = None,
    y: Any | None = None,
    groups: Any | None = None,
    k: int = 5,
    repeats: int = 1,
    test_size: float = 0.2,
    initial_window: int | None = None,
    horizon: int = 1,
    lookback: int | float | None = None,
    assess_start: int | float | None = None,
    assess_stop: int | float | None = None,
    complete: bool = True,
    index: Any | None = None,
    period: str | float = 'day',
    skip: int = 0,
    step: int = 1,
    shuffle: bool = True,
    seed: int = 42,
    folds: list[ResamplingFold] | None = None,
    task_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ResamplingPlan:
    if strategy not in RESAMPLING_STRATEGIES:
        raise ValidationError(f'Unsupported resampling strategy "{strategy}"')
    _validate_k(k)
    if repeats < 1:
        raise ValidationError('ResamplingPlan repeats must be >= 1')
    if test_size <= 0 or test_size >= 1:
        raise ValidationError('ResamplingPlan test_size must be in (0, 1)')
    if folds is not None:
        if n is None:
            raise ValidationError('ResamplingPlan n is required when folds are supplied')
        return validate_resampling_plan(ResamplingPlan(
            id=id or f'{strategy}-{n}',
            strategy=strategy,
            n=n,
            folds=folds,
            task_id=task_id,
            seed=seed,
            metadata=dict(metadata or {}),
        ))
    if n is None and index is not None:
        n = len(index)
    if n is None or n < 2:
        raise ValidationError('ResamplingPlan requires n >= 2')

    constraints = {}
    if strategy == 'holdout':
        indices = np.arange(n, dtype=np.int32)
        if shuffle:
            lcg_shuffle(indices, make_lcg(seed))
        n_test = max(1, int(np.floor(n * test_size + 0.5)))
        n_train = n - n_test
        plan_folds = [ResamplingFold('holdout-0', indices[:n_train].copy(), indices[n_train:].copy())]
    elif strategy == 'kfold':
        plan_folds = [_fold(train, test, f'fold-{i}') for i, (train, test) in enumerate(k_fold(n, k, shuffle, seed))]
    elif strategy == 'stratified_kfold':
        if y is None:
            raise ValidationError('stratified_kfold requires y')
        constraints['stratified'] = True
        plan_folds = [_fold(train, test, f'fold-{i}') for i, (train, test) in enumerate(stratified_k_fold(np.asarray(y), k, shuffle, seed))]
    elif strategy == 'repeated_kfold':
        plan_folds = []
        for repeat in range(repeats):
            for i, (train, test) in enumerate(k_fold(n, k, shuffle, seed + repeat)):
                plan_folds.append(_fold(train, test, f'repeat-{repeat}-fold-{i}', {'repeat': repeat}))
    elif strategy == 'group_kfold':
        if groups is None:
            raise ValidationError('group_kfold requires groups')
        constraints['grouped'] = True
        plan_folds = group_k_fold(np.asarray(groups), k, shuffle=shuffle, seed=seed)
    elif strategy == 'time_series':
        constraints['timeOrdered'] = True
        plan_folds = time_series_split(n, initial_window=initial_window, horizon=horizon, step=step)
    elif strategy == 'sliding_window':
        constraints['timeOrdered'] = True
        plan_folds = sliding_window_split(
            n,
            lookback=lookback if lookback is not None else initial_window,
            assess_start=assess_start,
            assess_stop=assess_stop,
            horizon=horizon,
            complete=complete,
            step=step,
            skip=skip,
        )
    elif strategy == 'sliding_index':
        if index is None:
            raise ValidationError('sliding_index requires index')
        constraints['timeOrdered'] = True
        plan_folds = sliding_index_split(
            index,
            lookback=lookback,
            assess_start=assess_start,
            assess_stop=assess_stop,
            horizon=horizon,
            complete=complete,
            step=step,
            skip=skip,
        )
    elif strategy == 'sliding_period':
        if index is None:
            raise ValidationError('sliding_period requires index')
        constraints['timeOrdered'] = True
        plan_folds = sliding_period_split(
            index,
            period=period,
            lookback=lookback,
            assess_start=assess_start,
            assess_stop=assess_stop,
            horizon=horizon,
            complete=complete,
            step=step,
            skip=skip,
        )
    else:
        raise ValidationError(f'Unsupported resampling strategy "{strategy}"')

    plan_metadata = dict(metadata or {})
    if strategy.startswith('sliding_'):
        plan_metadata = {
            **_sliding_metadata(
                lookback=lookback if strategy != 'sliding_window' else (lookback if lookback is not None else initial_window),
                assess_start=assess_start,
                assess_stop=assess_stop,
                horizon=horizon,
                complete=complete,
                step=step,
                skip=skip,
                period=period if strategy == 'sliding_period' else None,
                n=n if strategy == 'sliding_window' else None,
            ),
            **plan_metadata,
        }

    return validate_resampling_plan(ResamplingPlan(
        id=id or f'{strategy}-{n}',
        strategy=strategy,
        n=n,
        folds=plan_folds,
        task_id=task_id,
        seed=seed,
        constraints=constraints,
        metadata=plan_metadata,
    ))


def group_k_fold(groups: np.ndarray, k: int = 5, *, shuffle: bool = True, seed: int = 42) -> list[ResamplingFold]:
    _validate_k(k)
    groups = np.asarray(groups)
    if len(groups) < k:
        raise ValidationError(f'groupKFold: n ({len(groups)}) must be >= k ({k})')
    group_map: dict[Any, list[int]] = {}
    for i, group in enumerate(groups.tolist()):
        group_map.setdefault(group, []).append(i)
    group_keys = list(group_map)
    if len(group_keys) < k:
        raise ValidationError(f'groupKFold: number of groups ({len(group_keys)}) must be >= k ({k})')
    if shuffle:
        lcg_shuffle(group_keys, make_lcg(seed))
    fold_tests = [[] for _ in range(k)]
    fold_sizes = [0] * k
    for key in group_keys:
        best = min(range(k), key=lambda idx: fold_sizes[idx])
        rows = group_map[key]
        fold_tests[best].extend(rows)
        fold_sizes[best] += len(rows)
    folds = []
    for fold_id, test_rows in enumerate(fold_tests):
        test_set = set(test_rows)
        train = [i for i in range(len(groups)) if i not in test_set]
        folds.append(ResamplingFold(f'fold-{fold_id}', np.asarray(train, dtype=np.int32), np.asarray(test_rows, dtype=np.int32)))
    return folds


def time_series_split(n: int, *, initial_window: int | None = None, horizon: int = 1, step: int = 1) -> list[ResamplingFold]:
    if n < 2:
        raise ValidationError('timeSeriesSplit: n must be >= 2')
    if horizon < 1:
        raise ValidationError('timeSeriesSplit: horizon must be >= 1')
    if step < 1:
        raise ValidationError('timeSeriesSplit: step must be >= 1')
    start = initial_window if initial_window is not None else max(1, n // 2)
    if start < 1 or start >= n:
        raise ValidationError('timeSeriesSplit: initial_window must be between 1 and n - 1')
    folds = []
    fold_id = 0
    train_end = start
    while train_end + horizon <= n:
        train = np.arange(train_end, dtype=np.int32)
        test = np.arange(train_end, train_end + horizon, dtype=np.int32)
        folds.append(ResamplingFold(f'fold-{fold_id}', train, test))
        fold_id += 1
        train_end += step
    if not folds:
        raise ValidationError('timeSeriesSplit: no folds could be generated')
    return folds


def sliding_window_split(
    n: int,
    *,
    lookback: int | float | None = None,
    assess_start: int | float | None = None,
    assess_stop: int | float | None = None,
    horizon: int | float = 1,
    complete: bool = True,
    step: int = 1,
    skip: int = 0,
) -> list[ResamplingFold]:
    if n < 2:
        raise ValidationError('sliding_window_split: n must be >= 2')
    cfg = _normalize_sliding_opts(
        lookback=lookback if lookback is not None else max(1, n // 2),
        assess_start=assess_start,
        assess_stop=assess_stop,
        horizon=horizon,
        complete=complete,
        step=step,
        skip=skip,
        value_window=False,
    )
    folds = []
    anchor = int(cfg['lookback']) - 1 if cfg['complete'] else 0
    fold_id = 0
    while anchor < n:
        test_start = anchor + int(cfg['assess_start'])
        full_test_end = anchor + int(cfg['assess_stop'])
        if test_start >= n:
            break
        if cfg['complete'] and full_test_end >= n:
            break
        train_start = max(0, anchor - int(cfg['lookback']) + 1)
        if cfg['complete'] and anchor - int(cfg['lookback']) + 1 < 0:
            anchor += int(cfg['stride'])
            continue
        test_end = min(n - 1, full_test_end)
        folds.append(ResamplingFold(
            f'fold-{fold_id}',
            np.arange(train_start, anchor + 1, dtype=np.int32),
            np.arange(test_start, test_end + 1, dtype=np.int32),
            metadata={
                'anchor': anchor,
                'trainStart': train_start,
                'trainEnd': anchor,
                'testStart': test_start,
                'testEnd': test_end,
            },
        ))
        fold_id += 1
        anchor += int(cfg['stride'])
    if not folds:
        raise ValidationError('sliding_window_split: no folds could be generated')
    return folds


def sliding_index_split(
    index: Any,
    *,
    lookback: int | float | None = None,
    assess_start: int | float | None = None,
    assess_stop: int | float | None = None,
    horizon: int | float = 1,
    complete: bool = True,
    step: int = 1,
    skip: int = 0,
) -> list[ResamplingFold]:
    values = _coerce_index(index, 'sliding_index_split')
    cfg = _normalize_sliding_opts(
        lookback=1 if lookback is None else lookback,
        assess_start=assess_start,
        assess_stop=assess_stop,
        horizon=horizon,
        complete=complete,
        step=step,
        skip=skip,
        value_window=True,
    )
    return _sliding_value_split(values, cfg, 'sliding_index_split')


def sliding_period_split(
    index: Any,
    *,
    period: str | float = 'day',
    lookback: int | float | None = None,
    assess_start: int | float | None = None,
    assess_stop: int | float | None = None,
    horizon: int | float = 1,
    complete: bool = True,
    step: int = 1,
    skip: int = 0,
) -> list[ResamplingFold]:
    values = _period_ordinals(index, period)
    cfg = _normalize_sliding_opts(
        lookback=1 if lookback is None else lookback,
        assess_start=assess_start,
        assess_stop=assess_stop,
        horizon=horizon,
        complete=complete,
        step=step,
        skip=skip,
        value_window=True,
    )
    return _sliding_value_split(values, cfg, 'sliding_period_split')


def _sliding_value_split(values: np.ndarray, cfg: dict[str, Any], name: str) -> list[ResamplingFold]:
    if values.size < 2:
        raise ValidationError(f'{name}: index must have length >= 2')
    if np.any(values[1:] < values[:-1]):
        raise ValidationError(f'{name}: index must be sorted ascending')
    folds = []
    seen_anchors = set()
    anchor_idx = 0
    fold_id = 0
    while anchor_idx < len(values):
        anchor_value = float(values[anchor_idx])
        if anchor_value in seen_anchors:
            anchor_idx += int(cfg['stride'])
            continue
        seen_anchors.add(anchor_value)
        train_min = anchor_value - float(cfg['lookback'])
        test_min = anchor_value + float(cfg['assess_start'])
        test_max = anchor_value + float(cfg['assess_stop'])
        if cfg['complete'] and train_min < float(values[0]):
            anchor_idx += int(cfg['stride'])
            continue
        if test_min > float(values[-1]):
            break
        if cfg['complete'] and test_max > float(values[-1]):
            break
        train_floor = max(float(values[0]), train_min)
        test_ceiling = min(float(values[-1]), test_max)
        train = np.where((values >= train_floor) & (values <= anchor_value))[0].astype(np.int32)
        test = np.where((values >= test_min) & (values <= test_ceiling))[0].astype(np.int32)
        if train.size and test.size:
            folds.append(ResamplingFold(
                f'fold-{fold_id}',
                train,
                test,
                metadata={
                    'anchorIndex': anchor_idx,
                    'anchorValue': anchor_value,
                    'trainStart': train_floor,
                    'trainEnd': anchor_value,
                    'testStart': test_min,
                    'testEnd': test_ceiling,
                },
            ))
            fold_id += 1
        anchor_idx += int(cfg['stride'])
    if not folds:
        raise ValidationError(f'{name}: no folds could be generated')
    return folds


def _normalize_sliding_opts(
    *,
    lookback: int | float,
    assess_start: int | float | None,
    assess_stop: int | float | None,
    horizon: int | float,
    complete: bool,
    step: int,
    skip: int,
    value_window: bool,
) -> dict[str, Any]:
    assess_start = 1 if assess_start is None else assess_start
    assess_stop = horizon if assess_stop is None else assess_stop
    if not np.isfinite(lookback) or lookback <= 0:
        raise ValidationError('sliding split lookback must be positive')
    if not np.isfinite(assess_start) or assess_start <= 0:
        raise ValidationError('sliding split assess_start must be positive')
    if not np.isfinite(assess_stop) or assess_stop < assess_start:
        raise ValidationError('sliding split assess_stop must be >= assess_start')
    if not isinstance(step, int) or step < 1:
        raise ValidationError('sliding split step must be >= 1')
    if not isinstance(skip, int) or skip < 0:
        raise ValidationError('sliding split skip must be >= 0')
    if not value_window:
        for value, name in ((lookback, 'lookback'), (assess_start, 'assess_start'), (assess_stop, 'assess_stop')):
            if int(value) != value:
                raise ValidationError(f'sliding_window_split {name} must be an integer')
    return {
        'lookback': lookback,
        'assess_start': assess_start,
        'assess_stop': assess_stop,
        'complete': bool(complete),
        'step': step,
        'skip': skip,
        'stride': step + skip,
    }


def _sliding_metadata(
    *,
    lookback: int | float | None,
    assess_start: int | float | None,
    assess_stop: int | float | None,
    horizon: int | float,
    complete: bool,
    step: int,
    skip: int,
    period: str | float | None,
    n: int | None,
) -> dict[str, Any]:
    cfg = _normalize_sliding_opts(
        lookback=lookback if lookback is not None else (max(1, n // 2) if n is not None else 1),
        assess_start=assess_start,
        assess_stop=assess_stop,
        horizon=horizon,
        complete=complete,
        step=step,
        skip=skip,
        value_window=n is None,
    )
    out = {
        'lookback': cfg['lookback'],
        'assessStart': cfg['assess_start'],
        'assessStop': cfg['assess_stop'],
        'complete': cfg['complete'],
        'step': cfg['step'],
        'skip': cfg['skip'],
    }
    if period is not None:
        out['period'] = period
    return out


def _coerce_index(index: Any, name: str) -> np.ndarray:
    if index is None or len(index) < 2:
        raise ValidationError(f'{name}: index must have length >= 2')
    values = np.asarray([_coerce_index_value(value) for value in index], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValidationError(f'{name}: index values must be finite')
    return values


def _coerce_index_value(value: Any) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return float(dt.timestamp() * 1000.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    text = str(value)
    if text.endswith('Z'):
        text = text[:-1] + '+00:00'
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return float('nan')
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return float(dt.timestamp() * 1000.0)


def _period_ordinals(index: Any, period: str | float) -> np.ndarray:
    if isinstance(period, (int, float, np.integer, np.floating)):
        period_value = float(period)
        if not np.isfinite(period_value) or period_value <= 0:
            raise ValidationError('sliding_period_split: numeric period must be positive')
        raw = _coerce_index(index, 'sliding_period_split')
        return np.floor(raw / period_value).astype(np.float64)
    if index is None or len(index) < 2:
        raise ValidationError('sliding_period_split: index must have length >= 2')
    values = np.asarray([_period_ordinal(value, str(period)) for value in index], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValidationError('sliding_period_split: index values must be finite dates')
    return values


def _period_ordinal(value: Any, period: str) -> float:
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    else:
        text = str(value)
        if text.endswith('Z'):
            text = text[:-1] + '+00:00'
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return float('nan')
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    unix_day = int(dt.timestamp() // 86400)
    if period == 'day':
        return float(unix_day)
    if period == 'week':
        return float(unix_day // 7)
    if period == 'month':
        return float(dt.year * 12 + dt.month - 1)
    if period == 'quarter':
        return float(dt.year * 4 + (dt.month - 1) // 3)
    if period == 'year':
        return float(dt.year)
    raise ValidationError(f'sliding_period_split: unsupported period "{period}"')


def validate_resampling_plan(plan: ResamplingPlan) -> ResamplingPlan:
    if not isinstance(plan, ResamplingPlan):
        raise ValidationError('ResamplingPlan must be a ResamplingPlan')
    if not plan.id:
        raise ValidationError('ResamplingPlan.id must be non-empty')
    if plan.strategy not in RESAMPLING_STRATEGIES:
        raise ValidationError(f'Unsupported resampling strategy "{plan.strategy}"')
    if plan.n < 2:
        raise ValidationError('ResamplingPlan.n must be >= 2')
    if not plan.folds:
        raise ValidationError('ResamplingPlan.folds must be non-empty')
    for fold in plan.folds:
        _validate_fold(fold, plan.n)
    return plan


def serialize_resampling_plan(plan: ResamplingPlan) -> dict[str, Any]:
    validate_resampling_plan(plan)
    return {
        'id': plan.id,
        'strategy': plan.strategy,
        'n': plan.n,
        'taskId': plan.task_id,
        'seed': plan.seed,
        'constraints': dict(plan.constraints),
        'metadata': dict(plan.metadata),
        'folds': [
            {
                'foldId': fold.fold_id,
                'train': fold.train.astype(np.int32).tolist(),
                'test': fold.test.astype(np.int32).tolist(),
                'validate': None if fold.validate is None else fold.validate.astype(np.int32).tolist(),
                'metadata': dict(fold.metadata),
            }
            for fold in plan.folds
        ],
    }


def deserialize_resampling_plan(data: dict[str, Any]) -> ResamplingPlan:
    folds = [
        ResamplingFold(
            item['foldId'],
            np.asarray(item['train'], dtype=np.int32),
            np.asarray(item['test'], dtype=np.int32),
            None if item.get('validate') is None else np.asarray(item['validate'], dtype=np.int32),
            dict(item.get('metadata') or {}),
        )
        for item in data['folds']
    ]
    return validate_resampling_plan(ResamplingPlan(
        id=data['id'],
        strategy=data['strategy'],
        n=int(data['n']),
        folds=folds,
        task_id=data.get('taskId'),
        seed=data.get('seed'),
        constraints=dict(data.get('constraints') or {}),
        metadata=dict(data.get('metadata') or {}),
    ))


def _fold(train: np.ndarray, test: np.ndarray, fold_id: str, metadata: dict[str, Any] | None = None) -> ResamplingFold:
    return ResamplingFold(fold_id, np.asarray(train, dtype=np.int32), np.asarray(test, dtype=np.int32), metadata=dict(metadata or {}))


def _validate_fold(fold: ResamplingFold, n: int) -> None:
    if not isinstance(fold, ResamplingFold):
        raise ValidationError('CV fold must be a ResamplingFold')
    if not fold.fold_id:
        raise ValidationError('CV fold must have a non-empty fold_id')
    for name in ('train', 'test'):
        arr = np.asarray(getattr(fold, name), dtype=np.int32)
        if arr.size == 0:
            raise ValidationError(f'CV fold {fold.fold_id}.{name} must be non-empty')
        if np.any(arr < 0) or np.any(arr >= n):
            raise ValidationError(f'CV fold {fold.fold_id}.{name} contains out-of-range index')
        if len(set(arr.tolist())) != len(arr):
            raise ValidationError(f'CV fold {fold.fold_id}.{name} contains duplicate indices')
        setattr(fold, name, arr)
    if set(fold.train.tolist()) & set(fold.test.tolist()):
        raise ValidationError(f'CV fold {fold.fold_id} has overlapping train/test indices')
    if fold.validate is not None:
        validate = np.asarray(fold.validate, dtype=np.int32)
        if validate.size == 0:
            raise ValidationError(f'CV fold {fold.fold_id}.validate must be non-empty')
        if np.any(validate < 0) or np.any(validate >= n):
            raise ValidationError(f'CV fold {fold.fold_id}.validate contains out-of-range index')
        if len(set(validate.tolist())) != len(validate):
            raise ValidationError(f'CV fold {fold.fold_id}.validate contains duplicate indices')
        validate_set = set(validate.tolist())
        if validate_set & set(fold.train.tolist()) or validate_set & set(fold.test.tolist()):
            raise ValidationError(f'CV fold {fold.fold_id} has overlapping validate indices')
        fold.validate = validate


def _validate_k(k: int) -> None:
    if not isinstance(k, int) or k < 2:
        raise ValidationError('ResamplingPlan k must be >= 2')
