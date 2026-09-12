"""Measure registry and sklearn/yardstick-style metric primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .errors import ValidationError
from .prediction import Prediction, create_prediction, validate_prediction
from .targets import normalize_targets


MEASURE_DIRECTIONS = ('maximize', 'minimize')
MEASURE_RESPONSES = ('response', 'proba', 'score', 'decision', 'distribution', 'quantiles', 'interval', 'sets', 'region', 'samples')


@dataclass
class MeasureDef:
    id: str
    task_kinds: list[str]
    direction: str
    response: str
    fn: Callable[..., float]
    label: str | None = None
    range: tuple[float, float] = (-float('inf'), float('inf'))
    average: str = 'macro'
    na_value: float | None = None
    requires_truth: bool = True
    supports_sample_weight: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    aggregator: Callable[[Any], float] | None = None


_REGISTRY: dict[str, MeasureDef] = {}


def define_measure(defn: MeasureDef | dict[str, Any]) -> MeasureDef:
    if isinstance(defn, dict):
        defn = MeasureDef(**defn)
    if not defn.id:
        raise ValidationError('Measure.id must be non-empty')
    if not defn.task_kinds:
        raise ValidationError(f'Measure "{defn.id}" must declare task_kinds')
    if defn.direction not in MEASURE_DIRECTIONS:
        raise ValidationError(f'Measure "{defn.id}" has invalid direction')
    if defn.response not in MEASURE_RESPONSES:
        raise ValidationError(f'Measure "{defn.id}" has invalid response')
    if not callable(defn.fn):
        raise ValidationError(f'Measure "{defn.id}" must provide fn')
    if defn.aggregator is None:
        defn.aggregator = mean_aggregator
    if defn.label is None:
        defn.label = defn.id
    return defn


def register_measure(defn: MeasureDef | dict[str, Any]) -> MeasureDef:
    measure = define_measure(defn)
    _REGISTRY[measure.id] = measure
    return measure


def get_measure_def(id: str) -> MeasureDef:
    try:
        return _REGISTRY[id]
    except KeyError as exc:
        raise ValidationError(f'Unknown measure "{id}". Available: {", ".join(list_measures())}') from exc


def list_measures() -> list[str]:
    return sorted(_REGISTRY)


def evaluate_measure(measure_or_id: str | MeasureDef, prediction: Prediction, **opts: Any) -> float:
    measure = get_measure_def(measure_or_id) if isinstance(measure_or_id, str) else define_measure(measure_or_id)
    validate_prediction(prediction)
    if prediction.task_kind is not None and prediction.task_kind not in measure.task_kinds:
        raise ValidationError(f'Measure {measure.id} does not support {prediction.task_kind}')
    truth = opts.get('truth', prediction.truth)
    if measure.requires_truth and truth is None:
        raise ValidationError(f'Measure "{measure.id}" requires truth labels')
    value = measure.fn(
        truth=truth,
        response=prediction.response,
        proba=prediction.proba,
        score=prediction.score,
        decision=prediction.decision,
        prediction=prediction,
        opts=opts,
    )
    if not isinstance(value, (int, float, np.floating)) or (np.isnan(value) and not _allows_nan(opts)):
        if measure.na_value is not None:
            return float(measure.na_value)
        raise ValidationError(f'Measure "{measure.id}" returned a non-number')
    return float(value)


def evaluate_metric_set(measures: list[str | MeasureDef], prediction: Prediction, **opts: Any) -> dict[str, float]:
    result = {}
    for measure in measures:
        key = measure if isinstance(measure, str) else measure.id
        per_measure_opts = opts.get(key)
        if per_measure_opts is None:
            per_measure_opts = opts
        if not isinstance(per_measure_opts, dict):
            raise ValidationError(f'Measure "{key}" options must be a dict')
        result[key] = evaluate_measure(measure, prediction, **per_measure_opts)
    return result


def aggregate_measure(measure_or_id: str | MeasureDef, values: Any) -> float:
    measure = get_measure_def(measure_or_id) if isinstance(measure_or_id, str) else define_measure(measure_or_id)
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValidationError(f'Measure "{measure.id}" cannot aggregate empty values')
    assert measure.aggregator is not None
    return float(measure.aggregator(arr))


def mean_aggregator(values: Any) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    return float(np.mean(arr)) if arr.size else float('nan')


def _require(value: Any, measure_id: str, field_name: str) -> Any:
    if value is None:
        raise ValidationError(f'Measure "{measure_id}" requires prediction.{field_name}')
    return value


def _allows_nan(opts: dict[str, Any]) -> bool:
    return (
        opts.get('allow_nan') is True
        or opts.get('allowNaN') is True
        or opts.get('undefined_value') in ('nan', 'warn')
        or opts.get('undefinedValue') in ('nan', 'warn')
        or opts.get('zero_division') == 'nan'
        or opts.get('zeroDivision') == 'nan'
    )


def _sample_weight(opts: dict[str, Any], n: int, name: str) -> np.ndarray | None:
    weights = opts.get('sample_weight', opts.get('sampleWeight'))
    if weights is None:
        return None
    arr = np.asarray(weights, dtype=np.float64)
    if arr.shape != (n,):
        raise ValidationError(f'{name}: sample_weight length mismatch')
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValidationError(f'{name}: sample_weight must contain finite non-negative values')
    if float(np.sum(arr)) <= 0:
        raise ValidationError(f'{name}: sample_weight sum must be positive')
    return arr


def _weight_at(weights: np.ndarray | None, i: int) -> float:
    return 1.0 if weights is None else float(weights[i])


def _weight_sum(weights: np.ndarray | None, n: int) -> float:
    return float(n) if weights is None else float(np.sum(weights))


def _classes_from_opts(truth: Any, opts: dict[str, Any]) -> list[Any]:
    if opts.get('classes') is not None:
        return list(np.asarray(opts['classes']).tolist())
    labels = sorted(set(np.asarray(truth).tolist()))
    n_classes = opts.get('n_classes', opts.get('nClasses'))
    if n_classes is not None:
        n_classes = int(n_classes)
        if len(labels) == n_classes:
            return labels
        if all(isinstance(label, (int, np.integer)) and 0 <= int(label) < n_classes for label in labels):
            return list(range(n_classes))
    return labels


def _positive_label(labels: list[Any], opts: dict[str, Any]) -> Any:
    if 'positive_label' in opts:
        return opts['positive_label']
    if 'positiveLabel' in opts:
        return opts['positiveLabel']
    return labels[-1]


def _emit_warning(opts: dict[str, Any], metric: str, message: str) -> None:
    warnings = opts.get('warnings')
    if isinstance(warnings, list):
        warnings.append({'type': 'undefined_metric', 'metric': metric, 'message': message})


def _undefined_metric(metric: str, message: str, opts: dict[str, Any], default_value: float = float('nan')) -> float:
    policy = opts.get('undefined_value', opts.get('undefinedValue', opts.get('na_value', opts.get('naValue'))))
    if policy is None or policy == 'error':
        raise ValidationError(f'{metric}: {message}')
    if policy == 'warn':
        _emit_warning(opts, metric, message)
        return default_value
    if policy == 'nan':
        return float('nan')
    if isinstance(policy, (int, float)):
        return float(policy)
    raise ValidationError(f'{metric}: unsupported undefined_value policy "{policy}"')


def _zero_division(metric: str, message: str, opts: dict[str, Any]) -> float:
    policy = opts.get('zero_division', opts.get('zeroDivision', opts.get('undefined_value', opts.get('undefinedValue'))))
    if policy is None:
        return 0.0
    if policy == 'error':
        raise ValidationError(f'{metric}: {message}')
    if policy == 'warn':
        _emit_warning(opts, metric, message)
        return 0.0
    if policy == 'nan':
        return float('nan')
    if isinstance(policy, (int, float)):
        return float(policy)
    raise ValidationError(f'{metric}: unsupported zero_division policy "{policy}"')


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = (weights > 0) & ~np.isnan(values)
    if not np.any(mask):
        return float('nan')
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


def _resolve_average(average: str, n_classes: int) -> str:
    average = average or 'binary'
    if average == 'binary' and n_classes > 2:
        raise ValidationError('average="binary" requires exactly 2 classes')
    if average not in ('binary', 'micro', 'macro', 'weighted', 'macro_weighted'):
        raise ValidationError(f'Unknown averaging method "{average}"')
    return average


def _validate_pair(truth: Any, response: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    truth_arr = np.asarray(truth)
    response_arr = np.asarray(response)
    if truth_arr.size == 0 or response_arr.size == 0:
        raise ValidationError(f'{name}: inputs must be non-empty')
    if len(truth_arr) != len(response_arr):
        raise ValidationError(f'{name}: length mismatch')
    return truth_arr, response_arr


def _classification_counts(truth: Any, response: Any, opts: dict[str, Any], name: str) -> tuple[list[Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    truth_arr, response_arr = _validate_pair(truth, response, name)
    weights = _sample_weight(opts, len(truth_arr), name)
    labels = _classes_from_opts(np.concatenate([truth_arr, response_arr]), opts) if opts.get('classes') is None else list(np.asarray(opts['classes']).tolist())
    label_map = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for i, (t, p) in enumerate(zip(truth_arr.tolist(), response_arr.tolist())):
        if t not in label_map or p not in label_map:
            raise ValidationError(f'{name}: label missing from classes')
        cm[label_map[t], label_map[p]] += _weight_at(weights, i)
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    support = np.sum(cm, axis=1)
    return labels, tp, fp, fn, support


def _accuracy(truth: Any, response: Any, opts: dict[str, Any], **_: Any) -> float:
    truth_arr, response_arr = _validate_pair(truth, _require(response, 'accuracy', 'response'), 'accuracy')
    weights = _sample_weight(opts, len(truth_arr), 'accuracy')
    correct = np.asarray(truth_arr == response_arr, dtype=np.float64)
    if weights is not None:
        correct *= weights
    return float(np.sum(correct) / _weight_sum(weights, len(truth_arr)))


def _precision(truth: Any, response: Any, opts: dict[str, Any], **_: Any) -> float:
    labels, tp, fp, _fn, support = _classification_counts(truth, _require(response, 'precision', 'response'), opts, 'precision')
    average = _resolve_average(str(opts.get('average', 'binary')), len(labels))
    values = np.empty(len(labels), dtype=np.float64)
    for i, label in enumerate(labels):
        denom = tp[i] + fp[i]
        values[i] = _zero_division('precision', f'precision is undefined for class {label}', opts) if denom == 0 else tp[i] / denom
    if average == 'binary':
        return float(values[-1])
    if average == 'micro':
        denom = float(np.sum(tp + fp))
        return _zero_division('precision', 'micro precision is undefined', opts) if denom == 0 else float(np.sum(tp) / denom)
    if average in ('weighted', 'macro_weighted'):
        return _weighted_mean(values, support)
    return float(np.mean(values))


def _recall(truth: Any, response: Any, opts: dict[str, Any], **_: Any) -> float:
    labels, tp, _fp, fn, support = _classification_counts(truth, _require(response, 'recall', 'response'), opts, 'recall')
    average = _resolve_average(str(opts.get('average', 'binary')), len(labels))
    values = np.empty(len(labels), dtype=np.float64)
    for i, label in enumerate(labels):
        denom = tp[i] + fn[i]
        values[i] = _zero_division('recall', f'recall is undefined for class {label}', opts) if denom == 0 else tp[i] / denom
    if average == 'binary':
        return float(values[-1])
    if average == 'micro':
        denom = float(np.sum(tp + fn))
        return _zero_division('recall', 'micro recall is undefined', opts) if denom == 0 else float(np.sum(tp) / denom)
    if average in ('weighted', 'macro_weighted'):
        return _weighted_mean(values, support)
    return float(np.mean(values))


def _f1(truth: Any, response: Any, opts: dict[str, Any], **_: Any) -> float:
    labels, tp, fp, fn, support = _classification_counts(truth, _require(response, 'f1', 'response'), opts, 'f1')
    average = _resolve_average(str(opts.get('average', 'binary')), len(labels))
    values = np.empty(len(labels), dtype=np.float64)
    for i, label in enumerate(labels):
        p_denom = tp[i] + fp[i]
        r_denom = tp[i] + fn[i]
        p = _zero_division('f1', f'precision is undefined for class {label}', opts) if p_denom == 0 else tp[i] / p_denom
        r = _zero_division('f1', f'recall is undefined for class {label}', opts) if r_denom == 0 else tp[i] / r_denom
        values[i] = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    if average == 'binary':
        return float(values[-1])
    if average == 'micro':
        p_denom = float(np.sum(tp + fp))
        r_denom = float(np.sum(tp + fn))
        p = _zero_division('f1', 'micro precision is undefined', opts) if p_denom == 0 else float(np.sum(tp) / p_denom)
        r = _zero_division('f1', 'micro recall is undefined', opts) if r_denom == 0 else float(np.sum(tp) / r_denom)
        return 0.0 if p + r == 0 else float(2 * p * r / (p + r))
    if average in ('weighted', 'macro_weighted'):
        return _weighted_mean(values, support)
    return float(np.mean(values))


def _mse(truth: Any, response: Any, opts: dict[str, Any], **_: Any) -> float:
    truth_arr, response_arr = _validate_pair(truth, _require(response, 'mse', 'response'), 'mse')
    truth_arr = truth_arr.astype(np.float64)
    response_arr = response_arr.astype(np.float64)
    weights = _sample_weight(opts, len(truth_arr), 'mse')
    errors = (truth_arr - response_arr) ** 2
    if weights is not None:
        errors *= weights
    return float(np.sum(errors) / _weight_sum(weights, len(truth_arr)))


def _mae(truth: Any, response: Any, opts: dict[str, Any], **_: Any) -> float:
    truth_arr, response_arr = _validate_pair(truth, _require(response, 'mae', 'response'), 'mae')
    truth_arr = truth_arr.astype(np.float64)
    response_arr = response_arr.astype(np.float64)
    weights = _sample_weight(opts, len(truth_arr), 'mae')
    errors = np.abs(truth_arr - response_arr)
    if weights is not None:
        errors *= weights
    return float(np.sum(errors) / _weight_sum(weights, len(truth_arr)))


def _r2(truth: Any, response: Any, opts: dict[str, Any], **_: Any) -> float:
    truth_arr, response_arr = _validate_pair(truth, _require(response, 'r2', 'response'), 'r2')
    truth_arr = truth_arr.astype(np.float64)
    response_arr = response_arr.astype(np.float64)
    weights = _sample_weight(opts, len(truth_arr), 'r2')
    w_sum = _weight_sum(weights, len(truth_arr))
    if weights is None:
        mean = float(np.mean(truth_arr))
        ss_tot = float(np.sum((truth_arr - mean) ** 2))
        ss_res = float(np.sum((truth_arr - response_arr) ** 2))
    else:
        mean = float(np.sum(truth_arr * weights) / w_sum)
        ss_tot = float(np.sum(weights * (truth_arr - mean) ** 2))
        ss_res = float(np.sum(weights * (truth_arr - response_arr) ** 2))
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _classes_for_prediction(truth: Any, prediction: Prediction, opts: dict[str, Any]) -> list[Any]:
    if opts.get('classes') is not None:
        return list(np.asarray(opts['classes']).tolist())
    if prediction.classes is not None:
        return list(np.asarray(prediction.classes).tolist())
    return _classes_from_opts(truth, opts)


def _log_loss(truth: Any, proba: Any, opts: dict[str, Any], prediction: Prediction, **_: Any) -> float:
    truth_arr = np.asarray(truth)
    if truth_arr.size == 0:
        raise ValidationError('log_loss: truth must be non-empty')
    proba_arr = np.asarray(_require(proba, 'log_loss', 'proba'), dtype=np.float64)
    weights = _sample_weight(opts, len(truth_arr), 'log_loss')
    classes = _classes_for_prediction(truth_arr, prediction, opts)
    n_classes = int(opts.get('n_classes', opts.get('nClasses', len(classes))))
    if len(classes) != n_classes:
        raise ValidationError('log_loss: classes length must match n_classes')
    if proba_arr.size != len(truth_arr) * n_classes:
        raise ValidationError('log_loss: proba length mismatch')
    if not np.all(np.isfinite(proba_arr)):
        raise ValidationError('log_loss: probabilities must be finite')
    class_map = {label: i for i, label in enumerate(classes)}
    eps = float(opts.get('eps', 1e-15))
    loss = 0.0
    for i, label in enumerate(truth_arr.tolist()):
        if label not in class_map:
            raise ValidationError(f'log_loss: class "{label}" missing from prediction.classes')
        p = float(proba_arr[i * n_classes + class_map[label]])
        p = min(1 - eps, max(eps, p))
        loss -= _weight_at(weights, i) * np.log(p)
    return float(loss / _weight_sum(weights, len(truth_arr)))


def _binary_score(score: Any, proba: Any, truth: Any, prediction: Prediction, opts: dict[str, Any]) -> np.ndarray | None:
    truth_arr = np.asarray(truth)
    if score is not None:
        return np.asarray(score, dtype=np.float64)
    if proba is None:
        return None
    proba_arr = np.asarray(proba, dtype=np.float64)
    if proba_arr.size == len(truth_arr):
        return proba_arr
    if proba_arr.size == len(truth_arr) * 2:
        labels = _classes_for_prediction(truth_arr, prediction, opts)
        if len(labels) != 2:
            raise ValidationError('roc_auc: requires exactly 2 classes')
        positive = _positive_label(labels, opts)
        if positive not in labels:
            raise ValidationError('roc_auc: positive class missing from classes')
        return proba_arr.reshape((len(truth_arr), 2))[:, labels.index(positive)]
    return None


def _binary_roc_auc(truth: Any, score: Any, opts: dict[str, Any]) -> float:
    truth_arr = np.asarray(truth)
    score_arr = np.asarray(score, dtype=np.float64)
    if truth_arr.size == 0:
        raise ValidationError('roc_auc: truth must be non-empty')
    if len(truth_arr) != len(score_arr):
        raise ValidationError('roc_auc: length mismatch')
    if not np.all(np.isfinite(score_arr)):
        raise ValidationError('roc_auc: scores must be finite')
    weights = _sample_weight(opts, len(truth_arr), 'roc_auc')
    present = sorted(set(truth_arr.tolist()))
    if len(present) != 2:
        return _undefined_metric('roc_auc', 'requires exactly 2 classes with positive support', opts)
    labels = _classes_from_opts(truth_arr, opts)
    positive = _positive_label(labels, opts)
    w_pos = 0.0
    w_neg = 0.0
    for i, label in enumerate(truth_arr.tolist()):
        if label == positive:
            w_pos += _weight_at(weights, i)
        else:
            w_neg += _weight_at(weights, i)
    if w_pos <= 0 or w_neg <= 0:
        return _undefined_metric('roc_auc', 'requires positive total weight for both classes', opts)

    order = np.argsort(score_arr, kind='mergesort')
    neg_before = 0.0
    u = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        tied_score = score_arr[order[start]]
        while end < len(order) and score_arr[order[end]] == tied_score:
            end += 1
        group_neg = 0.0
        for idx in order[start:end]:
            if truth_arr[idx] != positive:
                group_neg += _weight_at(weights, int(idx))
        for idx in order[start:end]:
            if truth_arr[idx] == positive:
                u += _weight_at(weights, int(idx)) * (neg_before + group_neg / 2.0)
        neg_before += group_neg
        start = end
    return float(u / (w_pos * w_neg))


def _score_column(score: np.ndarray, n: int, n_classes: int, col: int) -> np.ndarray:
    return score.reshape((n, n_classes))[:, col]


def _multiclass_ovr_auc(truth: np.ndarray, score: np.ndarray, classes: list[Any], opts: dict[str, Any]) -> float:
    n = len(truth)
    n_classes = len(classes)
    weights = _sample_weight(opts, n, 'roc_auc') if opts.get('sample_weight') is not None or opts.get('sampleWeight') is not None else None
    values: list[float] = []
    supports: list[float] = []
    for c, label in enumerate(classes):
        binary_truth = np.asarray([1 if item == label else 0 for item in truth.tolist()], dtype=np.int32)
        support = sum(_weight_at(weights, i) for i, item in enumerate(truth.tolist()) if item == label)
        auc = _binary_roc_auc(
            binary_truth,
            _score_column(score, n, n_classes, c),
            {**opts, 'classes': [0, 1], 'positive_label': 1},
        )
        values.append(auc)
        supports.append(support)
    values_arr = np.asarray(values, dtype=np.float64)
    supports_arr = np.asarray(supports, dtype=np.float64)
    if opts.get('average', 'macro') in ('weighted', 'macro_weighted'):
        return _weighted_mean(values_arr, supports_arr)
    return float(np.mean(values_arr))


def _multiclass_ovo_auc(truth: np.ndarray, score: np.ndarray, classes: list[Any], opts: dict[str, Any]) -> float:
    n = len(truth)
    n_classes = len(classes)
    weights = _sample_weight(opts, n, 'roc_auc') if opts.get('sample_weight') is not None or opts.get('sampleWeight') is not None else None
    values: list[float] = []
    supports: list[float] = []
    for a in range(n_classes):
        for b in range(a + 1, n_classes):
            label_a = classes[a]
            label_b = classes[b]
            pair_truth = []
            score_a = []
            score_b = []
            pair_weights = []
            support = 0.0
            for i, label in enumerate(truth.tolist()):
                if label != label_a and label != label_b:
                    continue
                w = _weight_at(weights, i)
                pair_truth.append(label)
                score_a.append(score[i * n_classes + a])
                score_b.append(score[i * n_classes + b])
                pair_weights.append(w)
                support += w
            auc_a = _binary_roc_auc(
                np.asarray(pair_truth),
                np.asarray(score_a, dtype=np.float64),
                {**opts, 'classes': [label_a, label_b], 'positive_label': label_a, 'sample_weight': np.asarray(pair_weights, dtype=np.float64)},
            )
            auc_b = _binary_roc_auc(
                np.asarray(pair_truth),
                np.asarray(score_b, dtype=np.float64),
                {**opts, 'classes': [label_a, label_b], 'positive_label': label_b, 'sample_weight': np.asarray(pair_weights, dtype=np.float64)},
            )
            values.append((auc_a + auc_b) / 2.0)
            supports.append(support)
    values_arr = np.asarray(values, dtype=np.float64)
    supports_arr = np.asarray(supports, dtype=np.float64)
    if opts.get('average', 'macro') in ('weighted', 'macro_weighted'):
        return _weighted_mean(values_arr, supports_arr)
    return float(np.mean(values_arr))


def _roc_auc(truth: Any, score: Any, proba: Any, opts: dict[str, Any], prediction: Prediction, **_: Any) -> float:
    truth_arr = np.asarray(truth)
    score_values = _binary_score(score, proba, truth_arr, prediction, opts)
    if score_values is None:
        score_values = np.asarray(_require(score if score is not None else proba, 'roc_auc', 'score'), dtype=np.float64)
    else:
        score_values = np.asarray(score_values, dtype=np.float64)
    classes = _classes_for_prediction(truth_arr, prediction, opts)
    n_classes = len(classes)
    if score_values.size == len(truth_arr):
        return _binary_roc_auc(truth_arr, score_values, {**opts, 'classes': classes})
    if score_values.size != len(truth_arr) * n_classes:
        raise ValidationError('roc_auc: score length mismatch')
    if not np.all(np.isfinite(score_values)):
        raise ValidationError('roc_auc: scores must be finite')
    multi_class = opts.get('multi_class', opts.get('multiClass', 'raise'))
    if n_classes == 2 and multi_class == 'raise':
        positive = _positive_label(classes, opts)
        return _binary_roc_auc(
            truth_arr,
            _score_column(score_values, len(truth_arr), n_classes, classes.index(positive)),
            {**opts, 'classes': classes},
        )
    if multi_class == 'raise':
        raise ValidationError('roc_auc: multiclass scores require multi_class="ovr" or "ovo"')
    average = opts.get('average', 'macro')
    if average not in ('macro', 'weighted', 'macro_weighted'):
        raise ValidationError('roc_auc: multiclass average must be "macro" or "weighted"')
    mc_opts = {**opts, 'average': average}
    if multi_class == 'ovr':
        return _multiclass_ovr_auc(truth_arr, score_values, classes, mc_opts)
    if multi_class == 'ovo':
        return _multiclass_ovo_auc(truth_arr, score_values, classes, mc_opts)
    raise ValidationError(f'roc_auc: unsupported multi_class "{multi_class}"')


def _roc_auc_ovr(**kw: Any) -> float:
    kw = dict(kw)
    kw['opts'] = {**kw['opts'], 'multi_class': 'ovr'}
    return _roc_auc(**kw)


def _roc_auc_ovo(**kw: Any) -> float:
    kw = dict(kw)
    kw['opts'] = {**kw['opts'], 'multi_class': 'ovo'}
    return _roc_auc(**kw)


def get_scorer(scoring):
    """Resolve scoring through Measure; plain two-array callables maximize."""
    if callable(scoring):
        return scoring
    measure = get_measure_def(scoring) if isinstance(scoring, str) else define_measure(scoring)

    def scorer(truth, values, **opts):
        yn = normalize_targets(truth, opts.get('task_kind'))
        shape = dict(rows=len(yn), target_count=yn.shape[1], task_kind=opts.get('task_kind')) if yn.ndim == 2 else dict(rows=len(yn))
        field_name = measure.response
        classes = opts.get('classes')
        if isinstance(values, Prediction):
            data = vars(values).copy()
        else:
            values = np.asarray(values)
            if values.ndim == 2 and (len(values) != len(yn) or yn.ndim == 2 and values.shape[1] != yn.shape[1]):
                raise ValidationError('Score prediction target shape mismatch')
            if field_name == 'score' and classes is not None and values.size == len(yn) * len(classes):
                field_name = 'proba'
            data = {field_name: values.reshape(-1)}
        data.update(shape)
        data['truth'] = yn.reshape(-1)
        if classes is not None:
            data['classes'] = classes
        for key in ('quantile_levels', 'coverage_levels', 'task_kind'):
            if opts.get(key) is not None:
                data[key] = opts[key]
        value = evaluate_measure(measure, create_prediction(**data), **opts)
        if not np.isfinite(value):
            raise ValidationError(f'Scorer "{measure.id}" must return a finite number')
        return value

    scorer.measure = measure
    scorer.direction = measure.direction
    scorer.response = measure.response
    return scorer


def score_estimator(model, X, y, scoring):
    scorer = get_scorer(scoring)
    response = getattr(scorer, 'response', 'response')
    method = {'response': 'predict', 'proba': 'predict_proba',
              'score': 'decision_function', 'decision': 'decision_function',
              'quantiles': 'predict_quantiles', 'interval': 'predict_interval',
              'sets': 'predict_set', 'region': 'predict_region',
              'samples': 'predict_distribution', 'distribution': 'predict_distribution'}.get(response)
    if response == 'score' and not callable(getattr(model, method, None)):
        method = 'predict_proba'
    if method is None or not callable(getattr(model, method, None)):
        raise ValidationError(f'Scoring response "{response}" requires {method}')
    args = getattr(scorer, 'measure', None)
    args = args.metadata.get('predictionArgs', []) if args is not None else []
    values = getattr(model, method)(X, *args)
    caps = getattr(model, 'capabilities', {})
    task_kind = 'multilabel' if caps.get('multilabel') else 'multioutput' if caps.get('multioutput') else None
    opts = {'task_kind': task_kind}
    if method == 'predict_proba' and task_kind != 'multilabel':
        classes = getattr(model, 'classes', None)
        if callable(classes):
            classes = classes()
        if classes is None:
            raise ValidationError('Probability scoring requires the model class order')
        opts['classes'] = classes
    value = scorer(y, values, **opts)
    if not np.isscalar(value) or not np.isfinite(value):
        raise ValidationError('Scorer must return a finite number')
    return float(value)


def _target_mean(fn, **kw):
    p = kw['prediction']
    t = p.target_count
    if t == 1:
        return fn(**kw)
    truth = np.asarray(kw['truth']).reshape(-1, t)
    if kw['response'] is None:
        raise ValidationError('Regression scoring requires response')
    response = np.asarray(kw['response']).reshape(-1, t)
    return float(np.mean([fn(truth=truth[:, c], response=response[:, c], opts=kw['opts']) for c in range(t)]))


def _multilabel_score(kind, truth, response, proba, prediction, opts, **_):
    if prediction.task_kind != 'multilabel':
        raise ValidationError('Multilabel scoring requires explicit task and target axes')
    t = prediction.target_count
    truth = np.asarray(truth).reshape(-1, t)
    a = proba if kind == 'log_loss' else response
    if a is None:
        raise ValidationError('Missing multilabel prediction field')
    a = np.asarray(a).reshape(truth.shape)
    if kind == 'log_loss':
        a = np.clip(a, 1e-15, 1 - 1e-15)
        values = -np.mean(np.where(truth, np.log(a), np.log1p(-a)), axis=1)
    elif kind == 'subset_accuracy':
        values = np.all(truth == a, axis=1).astype(float)
    else:
        values = np.mean(truth != a, axis=1)
    return _mae(np.zeros(len(values)), values, opts)


def register_builtin_measures() -> None:
    if _REGISTRY:
        return
    for id, kind, response, direction in [
        ('subset_accuracy', 'subset_accuracy', 'response', 'maximize'),
        ('hamming_loss', 'hamming_loss', 'response', 'minimize'),
        ('multilabel_log_loss', 'log_loss', 'proba', 'minimize'),
    ]:
        register_measure(MeasureDef(id, ['multilabel'], direction, response,
                                   lambda _kind=kind, **kw: _multilabel_score(_kind, **kw), supports_sample_weight=True))
    register_measure(MeasureDef('accuracy', ['classification'], 'maximize', 'response', _accuracy, supports_sample_weight=True))
    register_measure(MeasureDef('precision', ['classification'], 'maximize', 'response', _precision, supports_sample_weight=True))
    register_measure(MeasureDef('recall', ['classification'], 'maximize', 'response', _recall, supports_sample_weight=True))
    register_measure(MeasureDef('f1', ['classification'], 'maximize', 'response', _f1, supports_sample_weight=True))
    register_measure(MeasureDef('log_loss', ['classification'], 'minimize', 'proba', _log_loss, supports_sample_weight=True))
    register_measure(MeasureDef('roc_auc', ['classification'], 'maximize', 'score', _roc_auc, supports_sample_weight=True))
    register_measure(MeasureDef(
        'roc_auc_ovr',
        ['classification'],
        'maximize',
        'proba',
        _roc_auc_ovr,
        supports_sample_weight=True,
    ))
    register_measure(MeasureDef(
        'roc_auc_ovo',
        ['classification'],
        'maximize',
        'proba',
        _roc_auc_ovo,
        supports_sample_weight=True,
    ))
    register_measure(MeasureDef('r2', ['regression', 'multioutput'], 'maximize', 'response', lambda **kw: _target_mean(_r2, **kw), supports_sample_weight=True))
    register_measure(MeasureDef('mse', ['regression', 'multioutput'], 'minimize', 'response', lambda **kw: _target_mean(_mse, **kw), supports_sample_weight=True))
    register_measure(MeasureDef('neg_mse', ['regression', 'multioutput'], 'maximize', 'response', lambda **kw: -_target_mean(_mse, **kw), supports_sample_weight=True))
    register_measure(MeasureDef('mae', ['regression', 'multioutput'], 'minimize', 'response', lambda **kw: _target_mean(_mae, **kw), supports_sample_weight=True))
    register_measure(MeasureDef('neg_mae', ['regression', 'multioutput'], 'maximize', 'response', lambda **kw: -_target_mean(_mae, **kw), supports_sample_weight=True))


register_builtin_measures()
