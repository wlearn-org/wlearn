"""Class-order and probability-shape checks shared by ensembles."""

import numpy as np

from ..errors import ValidationError
from ..prediction import create_prediction


def normalize_class_order(value, expected_length, label):
    """Return a validated unique int32 probability-column order."""
    try:
        values = np.asarray(value, dtype=object)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(
            f'{label}: classes must be an array of unique int32 values') from exc
    if values.ndim != 1 or values.size != expected_length:
        raise ValidationError(
            f'{label}: classes must contain one unique int32 label per '
            'probability column')
    output = np.empty(expected_length, dtype=np.int32)
    known = set()
    for index, item in enumerate(values):
        if (isinstance(item, (bool, np.bool_)) or
                not isinstance(item, (int, np.integer)) or
                int(item) < -(1 << 31) or int(item) > (1 << 31) - 1 or
                int(item) in known):
            raise ValidationError(
                f'{label}: classes must contain one unique int32 label per '
                'probability column')
        known.add(int(item))
        output[index] = int(item)
    return output


def class_column_map(model, expected_classes, label):
    """Map ensemble class columns to a fitted child's declared order."""
    actual = getattr(model, 'classes', None)
    if callable(actual):
        actual = actual()
    if actual is None:
        raise ValidationError(f'{label} must expose its fitted classes')
    values = np.asarray(actual, dtype=object)
    if values.ndim != 1 or len(values) != len(expected_classes):
        raise ValidationError(
            f'{label} classes do not match the ensemble classes')

    local_columns = {}
    for index, value in enumerate(values):
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or
                int(value) < -(1 << 31) or int(value) > (1 << 31) - 1 or
                int(value) in local_columns):
            raise ValidationError(
                f'{label} classes must be unique int32 values')
        local_columns[int(value)] = index

    mapping = np.empty(len(expected_classes), dtype=np.intp)
    for index, value in enumerate(expected_classes):
        local = local_columns.get(int(value))
        if local is None:
            raise ValidationError(
                f'{label} classes do not match the ensemble classes')
        mapping[index] = local
    return mapping


def require_probability_model(model, label):
    """Require an explicit probability capability and callable API."""
    capabilities = getattr(model, 'capabilities', None)
    if (not isinstance(capabilities, dict) or
            capabilities.get('predictProba') is not True):
        raise ValidationError(
            f'{label} must declare predictProba capability for probability '
            'aggregation')
    if not callable(getattr(model, 'predict_proba', None)):
        raise ValidationError(f'{label} must implement predict_proba')
    return model


def validate_probability_output(value, rows, class_count, label):
    """Return a flat finite probability array with the required shape."""
    try:
        output = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(
            f'{label} predict_proba output must be numeric') from exc
    if output.size != rows * class_count:
        raise ValidationError(
            f'{label} predict_proba output has the wrong shape')
    if not np.all(np.isfinite(output)):
        raise ValidationError(
            f'{label} predict_proba output must contain finite numbers')
    create_prediction(proba=output, proba_rows=rows)
    return output


def validate_label_output(value, rows, classes, label):
    """Return canonical int32 labels after exact class-domain validation."""
    try:
        values = np.asarray(value, dtype=object)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(
            f'{label} predict output must be an array') from exc
    if values.ndim != 1 or values.size != rows:
        raise ValidationError(
            f'{label} predict output has the wrong shape')
    allowed = {int(item) for item in classes}
    output = np.empty(rows, dtype=np.int32)
    for index, item in enumerate(values):
        if (isinstance(item, (bool, np.bool_)) or
                not isinstance(item, (int, np.integer, float, np.floating)) or
                not np.isfinite(item) or int(item) != item or
                int(item) < -(1 << 31) or int(item) > (1 << 31) - 1 or
                int(item) not in allowed):
            raise ValidationError(
                f'{label} predict output must contain declared int32 '
                'class labels')
        output[index] = int(item)
    return output


def validate_regression_output(value, rows, label):
    """Return an exact-length finite float64 regression vector."""
    try:
        output = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(
            f'{label} predict output must be numeric') from exc
    if output.ndim != 1 or output.size != rows:
        raise ValidationError(f'{label} predict output has the wrong shape')
    if not np.all(np.isfinite(output)):
        raise ValidationError(
            f'{label} predict output must contain finite numbers')
    return output
