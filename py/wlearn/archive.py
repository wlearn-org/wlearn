"""Trial archive primitives for tuning, benchmarking, and AutoML outputs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import ValidationError


TRIAL_STATUSES = ('pending', 'running', 'ok', 'failed', 'pruned', 'timeout')


@dataclass
class TrialError:
    name: str
    message: str
    stack_hash: str | None = None
    phase: str = 'fit'


@dataclass
class TrialRecord:
    trial_id: str
    candidate_id: str
    seed: int = 42
    status: str = 'pending'
    params: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    learner_spec: dict[str, Any] | None = None
    pipeline_spec: dict[str, Any] | None = None
    budget: dict[str, float] | None = None
    batch: int | None = None
    uhash: str | None = None
    fold_id: str | None = None
    scores: dict[str, float] | None = None
    primary_score: float | None = None
    error: TrialError | None = None
    memory: dict[str, float] | None = None
    artifact_hash: str | None = None
    prediction_hash: str | None = None
    resample_result_hash: str | None = None
    warnings: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LeaderboardRow:
    candidate_id: str
    metric: str | None
    mean_score: float
    std_score: float
    n: int
    trial_ids: list[str]
    rank: int
    learner_spec: dict[str, Any] | None = None
    pipeline_spec: dict[str, Any] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, float] | None = None


class Archive:
    def __init__(
        self,
        *,
        id: str = 'archive',
        task_id: str | None = None,
        measures: list[str] | None = None,
        primary_measure: str | None = None,
        direction: str = 'maximize',
        metadata: dict[str, Any] | None = None,
        records: list[TrialRecord | dict[str, Any]] | None = None,
    ):
        if direction not in ('maximize', 'minimize'):
            raise ValidationError('Archive.direction must be "maximize" or "minimize"')
        self.id = id
        self.task_id = task_id
        self.measures = list(measures or [])
        self.primary_measure = primary_measure or (self.measures[0] if self.measures else None)
        self.direction = direction
        self.metadata = dict(metadata or {})
        self._records: list[TrialRecord] = []
        for record in records or []:
            self.add(record)

    def add(self, record: TrialRecord | dict[str, Any]) -> TrialRecord:
        normalized = create_trial_record(record)
        if any(existing.trial_id == normalized.trial_id for existing in self._records):
            raise ValidationError(f'Archive already contains trial_id "{normalized.trial_id}"')
        self._records.append(normalized)
        return deepcopy(normalized)

    def start(self, record: TrialRecord | dict[str, Any] | None = None) -> TrialRecord:
        payload = _record_dict(record or {})
        payload.setdefault('status', 'running')
        return self.add(payload)

    def finish(self, trial_id: str, patch: dict[str, Any] | None = None) -> TrialRecord:
        payload = dict(patch or {})
        payload.setdefault('status', 'ok')
        return self.update(trial_id, payload)

    def fail(self, record: TrialRecord | dict[str, Any], error: Any, phase: str = 'fit') -> TrialRecord:
        payload = _record_dict(record)
        payload['status'] = 'failed'
        payload['error'] = normalize_trial_error(error, phase)
        return self.add(payload)

    def update(self, trial_id: str, patch: dict[str, Any]) -> TrialRecord:
        for i, record in enumerate(self._records):
            if record.trial_id == trial_id:
                payload = _record_to_payload(record)
                if patch.get('trial_id') and patch['trial_id'] != trial_id:
                    raise ValidationError('Archive.update cannot change trial_id')
                if patch.get('trialId') and patch['trialId'] != trial_id:
                    raise ValidationError('Archive.update cannot change trial_id')
                payload.update(patch)
                next_record = validate_trial_record(create_trial_record(payload))
                self._records[i] = next_record
                return deepcopy(next_record)
        raise ValidationError(f'Archive record "{trial_id}" not found')

    def records(self, **filter: Any) -> list[TrialRecord]:
        records = self._records
        if filter:
            records = [
                record for record in records
                if all(getattr(record, key) == value for key, value in filter.items())
            ]
        return deepcopy(records)

    def leaderboard(
        self,
        *,
        metric: str | None = None,
        direction: str | None = None,
    ) -> list[LeaderboardRow]:
        metric = metric or self.primary_measure
        direction = direction or self.direction
        if direction not in ('maximize', 'minimize'):
            raise ValidationError('Archive.leaderboard direction must be "maximize" or "minimize"')
        groups: dict[str, dict[str, Any]] = {}
        for record in self._records:
            if record.status != 'ok':
                continue
            value = _score_value(record, metric)
            if value is None or np.isnan(value):
                continue
            group = groups.setdefault(record.candidate_id, {
                'candidate_id': record.candidate_id,
                'learner_spec': record.learner_spec,
                'pipeline_spec': record.pipeline_spec,
                'params': record.params,
                'budget': record.budget,
                'values': [],
                'trial_ids': [],
            })
            group['values'].append(float(value))
            group['trial_ids'].append(record.trial_id)

        rows = []
        for group in groups.values():
            values = np.asarray(group['values'], dtype=np.float64)
            rows.append(LeaderboardRow(
                candidate_id=group['candidate_id'],
                learner_spec=deepcopy(group['learner_spec']),
                pipeline_spec=deepcopy(group['pipeline_spec']),
                params=deepcopy(group['params']),
                budget=deepcopy(group['budget']),
                metric=metric,
                mean_score=float(np.mean(values)),
                std_score=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                n=int(len(values)),
                trial_ids=list(group['trial_ids']),
                rank=0,
            ))
        rows.sort(key=lambda row: row.mean_score, reverse=direction == 'maximize')
        for i, row in enumerate(rows):
            row.rank = i + 1
        return deepcopy(rows)

    def to_json(self) -> dict[str, Any]:
        return {
            'id': self.id,
            'taskId': self.task_id,
            'measures': list(self.measures),
            'primaryMeasure': self.primary_measure,
            'direction': self.direction,
            'metadata': deepcopy(self.metadata),
            'records': [_record_to_json(record) for record in self._records],
        }

    @property
    def size(self) -> int:
        return len(self._records)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> 'Archive':
        return cls(
            id=data.get('id', 'archive'),
            task_id=data.get('taskId') or data.get('task_id'),
            measures=list(data.get('measures') or []),
            primary_measure=data.get('primaryMeasure') or data.get('primary_measure'),
            direction=data.get('direction', 'maximize'),
            metadata=dict(data.get('metadata') or {}),
            records=list(data.get('records') or []),
        )


def create_trial_record(record: TrialRecord | dict[str, Any] | None = None) -> TrialRecord:
    if isinstance(record, TrialRecord):
        return validate_trial_record(deepcopy(record))
    record = dict(record or {})
    candidate_id = _get(record, 'candidate_id', 'candidateId', 'candidate')
    fold_id = _get(record, 'fold_id', 'foldId')
    seed = _get(record, 'seed', 'seed', 42)
    trial_id = _get(record, 'trial_id', 'trialId') or _default_trial_id(candidate_id, fold_id, seed)
    error = _get(record, 'error', 'error')
    return validate_trial_record(TrialRecord(
        trial_id=str(trial_id),
        candidate_id=str(candidate_id),
        seed=int(seed),
        status=str(_get(record, 'status', 'status', 'pending')),
        params=deepcopy(_get(record, 'params', 'params', {}) or {}),
        timings=deepcopy(_get(record, 'timings', 'timings', {}) or {}),
        learner_spec=deepcopy(_get(record, 'learner_spec', 'learnerSpec')),
        pipeline_spec=deepcopy(_get(record, 'pipeline_spec', 'pipelineSpec')),
        budget=deepcopy(_get(record, 'budget', 'budget')),
        batch=_get(record, 'batch', 'batch'),
        uhash=_get(record, 'uhash', 'uhash'),
        fold_id=fold_id,
        scores=deepcopy(_get(record, 'scores', 'scores')),
        primary_score=_get(record, 'primary_score', 'primaryScore'),
        error=None if error is None else normalize_trial_error(error, getattr(error, 'phase', None)),
        memory=deepcopy(_get(record, 'memory', 'memory')),
        artifact_hash=_get(record, 'artifact_hash', 'artifactHash'),
        prediction_hash=_get(record, 'prediction_hash', 'predictionHash'),
        resample_result_hash=_get(record, 'resample_result_hash', 'resampleResultHash'),
        warnings=deepcopy(_get(record, 'warnings', 'warnings', []) or []),
        metadata=deepcopy(_get(record, 'metadata', 'metadata', {}) or {}),
    ))


def validate_trial_record(record: TrialRecord) -> TrialRecord:
    if not isinstance(record, TrialRecord):
        raise ValidationError('TrialRecord must be a TrialRecord')
    if not record.trial_id:
        raise ValidationError('TrialRecord.trial_id must be non-empty')
    if not record.candidate_id:
        raise ValidationError('TrialRecord.candidate_id must be non-empty')
    if record.status not in TRIAL_STATUSES:
        raise ValidationError(f'TrialRecord.status "{record.status}" is invalid')
    if record.scores is not None:
        for key, value in record.scores.items():
            if not isinstance(value, (int, float)) or np.isnan(value):
                raise ValidationError(f'TrialRecord.scores.{key} must be a number')
    if record.primary_score is not None and (
        not isinstance(record.primary_score, (int, float)) or np.isnan(record.primary_score)
    ):
        raise ValidationError('TrialRecord.primary_score must be a number')
    if record.batch is not None and (not isinstance(record.batch, int) or record.batch < 1):
        raise ValidationError('TrialRecord.batch must be a positive integer')
    if record.error is not None:
        record.error = normalize_trial_error(record.error, record.error.phase)
    return record


def normalize_trial_error(error: Any, phase: str | None = 'fit') -> TrialError:
    if isinstance(error, TrialError):
        out = deepcopy(error)
        if phase:
            out.phase = phase
        return out
    if isinstance(error, dict):
        return TrialError(
            name=str(error.get('name') or 'Error'),
            message=str(error.get('message') or error),
            stack_hash=error.get('stackHash') or error.get('stack_hash'),
            phase=phase or error.get('phase') or 'fit',
        )
    return TrialError(
        name=getattr(error, '__class__', type(error)).__name__,
        message=str(error),
        stack_hash=getattr(error, 'stack_hash', None),
        phase=phase or getattr(error, 'phase', None) or 'fit',
    )


def _score_value(record: TrialRecord, metric: str | None) -> float | None:
    if metric and record.scores and metric in record.scores:
        return float(record.scores[metric])
    if record.primary_score is not None:
        return float(record.primary_score)
    return None


def _default_trial_id(candidate_id: str, fold_id: str | None, seed: int) -> str:
    return f'{candidate_id}-{fold_id or "fold"}-{seed}'


def _get(record: dict[str, Any], snake: str, camel: str, default: Any = None) -> Any:
    if snake in record:
        return record[snake]
    if camel in record:
        return record[camel]
    return default


def _record_dict(record: TrialRecord | dict[str, Any]) -> dict[str, Any]:
    if isinstance(record, TrialRecord):
        return _record_to_payload(record)
    return dict(record)


def _record_to_payload(record: TrialRecord) -> dict[str, Any]:
    return {
        'trial_id': record.trial_id,
        'candidate_id': record.candidate_id,
        'seed': record.seed,
        'status': record.status,
        'params': deepcopy(record.params),
        'timings': deepcopy(record.timings),
        'learner_spec': deepcopy(record.learner_spec),
        'pipeline_spec': deepcopy(record.pipeline_spec),
        'budget': deepcopy(record.budget),
        'batch': record.batch,
        'uhash': record.uhash,
        'fold_id': record.fold_id,
        'scores': deepcopy(record.scores),
        'primary_score': record.primary_score,
        'error': deepcopy(record.error),
        'memory': deepcopy(record.memory),
        'artifact_hash': record.artifact_hash,
        'prediction_hash': record.prediction_hash,
        'resample_result_hash': record.resample_result_hash,
        'warnings': list(record.warnings),
        'metadata': deepcopy(record.metadata),
    }


def _record_to_json(record: TrialRecord) -> dict[str, Any]:
    data = {
        'trialId': record.trial_id,
        'candidateId': record.candidate_id,
        'seed': record.seed,
        'status': record.status,
        'params': deepcopy(record.params),
        'timings': deepcopy(record.timings),
    }
    optional = {
        'learnerSpec': record.learner_spec,
        'pipelineSpec': record.pipeline_spec,
        'budget': record.budget,
        'batch': record.batch,
        'uhash': record.uhash,
        'foldId': record.fold_id,
        'scores': record.scores,
        'primaryScore': record.primary_score,
        'error': None if record.error is None else {
            'name': record.error.name,
            'message': record.error.message,
            'stackHash': record.error.stack_hash,
            'phase': record.error.phase,
        },
        'memory': record.memory,
        'artifactHash': record.artifact_hash,
        'predictionHash': record.prediction_hash,
        'resampleResultHash': record.resample_result_hash,
        'warnings': record.warnings if record.warnings else None,
        'metadata': record.metadata if record.metadata else None,
    }
    for key, value in optional.items():
        if value is not None:
            data[key] = deepcopy(value)
    return data
