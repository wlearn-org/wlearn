"""Semantic validation for composite ensemble WLRN manifests."""

import math

from ..errors import ValidationError
from ..resampling import resolve_cv


_NESTED_MEDIA_TYPE = 'application/x-wlearn-bundle'
_OOF_MEDIA_TYPE = 'application/octet-stream'


def validate_voting_manifest(
        manifest, toc, classifier_type, regressor_type):
    params = _validate_common(
        manifest, classifier_type, regressor_type, 'VotingEnsemble')
    names = _validate_names(
        params.get('estimatorNames'), 'VotingEnsemble estimatorNames')
    weights = params.get('weights')
    if (not isinstance(weights, list) or len(weights) != len(names) or
            any(isinstance(value, bool) or
                not isinstance(value, (int, float)) or
                not math.isfinite(value) or value < 0 for value in weights) or
            not math.isfinite(sum(weights)) or sum(weights) <= 0):
        raise ValidationError(
            'VotingEnsemble weights must be a nonnegative finite number '
            'array with a positive sum matching estimatorNames')
    if params.get('voting') not in ('soft', 'hard'):
        raise ValidationError(
            'VotingEnsemble voting must be "soft" or "hard"')
    _validate_classes(
        params, manifest['typeId'] == classifier_type, 'VotingEnsemble')
    _validate_artifacts(
        toc, [(name, _NESTED_MEDIA_TYPE) for name in names],
        'VotingEnsemble')
    return params


def validate_stacking_manifest(
        manifest, toc, classifier_type, regressor_type):
    params = _validate_common(
        manifest, classifier_type, regressor_type, 'StackingEnsemble')
    names = _validate_names(
        params.get('estimatorNames'), 'StackingEnsemble estimatorNames')
    meta_name = _validate_name(
        params.get('metaName'), 'StackingEnsemble metaName')
    if meta_name in names:
        raise ValidationError(
            'StackingEnsemble metaName must differ from every base '
            'estimator name')
    _validate_cv(params.get('cv'), 'StackingEnsemble cv')
    if not isinstance(params.get('passthrough'), bool):
        raise ValidationError(
            'StackingEnsemble passthrough must be a boolean')
    _assert_integer(
        params.get('seed'), -(1 << 53) + 1, 'StackingEnsemble seed',
        enforce_minimum=False)
    classes = _validate_classes(
        params, manifest['typeId'] == classifier_type, 'StackingEnsemble')
    n_meta_cols = params.get('nMetaCols')
    _assert_integer(n_meta_cols, 1, 'StackingEnsemble nMetaCols')
    learned_columns = len(names) * (1 if classes is None else len(classes))
    if ((not params['passthrough'] and n_meta_cols != learned_columns) or
            (params['passthrough'] and n_meta_cols < learned_columns)):
        raise ValidationError(
            'StackingEnsemble nMetaCols is inconsistent with its base '
            'estimators and classes')
    _validate_artifacts(
        toc,
        [(name, _NESTED_MEDIA_TYPE) for name in [*names, meta_name]],
        'StackingEnsemble')
    return params


def validate_bagging_manifest(
        manifest, toc, classifier_type, regressor_type):
    params = _validate_common(
        manifest, classifier_type, regressor_type, 'BaggedEstimator')
    k_fold = params.get('kFold')
    n_repeats = params.get('nRepeats')
    fold_count = _validate_cv(k_fold, 'BaggedEstimator kFold', params.get('nSamples'))
    _assert_integer(n_repeats, 1, 'BaggedEstimator nRepeats')
    model_count = fold_count * n_repeats
    if model_count > (1 << 53) - 1:
        raise ValidationError(
            'BaggedEstimator fold model count exceeds the safe integer range')
    _assert_integer(
        params.get('seed'), -(1 << 53) + 1, 'BaggedEstimator seed',
        enforce_minimum=False)
    _validate_name(
        params.get('estimatorName'), 'BaggedEstimator estimatorName')
    classes = _validate_classes(
        params, manifest['typeId'] == classifier_type, 'BaggedEstimator')
    n_samples = params.get('nSamples')
    _assert_integer(n_samples, 1, 'BaggedEstimator nSamples')
    expected_classes = 0 if classes is None else len(classes)
    n_classes = params.get('nClasses')
    _assert_integer(n_classes, 0, 'BaggedEstimator nClasses')
    if n_classes != expected_classes:
        raise ValidationError(
            'BaggedEstimator nClasses is inconsistent with classes')

    has_oof = any(entry['id'] == 'oof' for entry in toc)
    if len(toc) != model_count + (1 if has_oof else 0):
        raise ValidationError(
            'BaggedEstimator artifact count is inconsistent with its params')

    expected = [
        (f'fold_{index}', _NESTED_MEDIA_TYPE)
        for index in range(model_count)
    ]
    oof_entry = next((entry for entry in toc if entry['id'] == 'oof'), None)
    if oof_entry is not None:
        value_count = n_samples * (1 if classes is None else len(classes))
        if (value_count > ((1 << 53) - 1) // 8 or
                oof_entry['length'] != value_count * 8):
            raise ValidationError(
                'BaggedEstimator OOF artifact length is inconsistent with '
                'params')
        expected.append(('oof', _OOF_MEDIA_TYPE))
    _validate_artifacts(toc, expected, 'BaggedEstimator')
    return params


def _validate_cv(cv, label, rows=None):
    if isinstance(cv, int) and not isinstance(cv, bool):
        _assert_integer(cv, 2, label)
        return cv
    if not isinstance(cv, list) or not cv:
        raise ValidationError(f'{label} must be an integer or fold array')
    if rows is None:
        rows = 0
        for fold in cv:
            for field in ('train', 'test'):
                if not isinstance(fold, dict) or not isinstance(fold.get(field), list):
                    raise ValidationError(f'{label} has invalid folds')
                for index in fold[field]:
                    if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= 2147483647:
                        raise ValidationError(f'{label} has invalid row indices')
                    rows = max(rows, index + 1)
    _assert_integer(rows, 2, f'{label} rows')
    # range carries the row count without materializing untrusted manifest sizes.
    resolve_cv(cv, range(rows), task='regression')
    return len(cv)


def _validate_common(
        manifest, classifier_type, regressor_type, label):
    type_id = manifest.get('typeId')
    if type_id not in (classifier_type, regressor_type):
        raise ValidationError(
            f'{label}.load expected typeId "{classifier_type}" or '
            f'"{regressor_type}", got "{type_id}"')
    params = manifest.get('params')
    if not isinstance(params, dict):
        raise ValidationError(f'{label} manifest params must be an object')
    expected_task = (
        'regression' if type_id == regressor_type else 'classification')
    if params.get('task') != expected_task:
        raise ValidationError(f'{label} task is inconsistent with its typeId')
    return params


def _validate_classes(params, classification, label):
    classes = params.get('classes')
    if not classification:
        if classes is not None:
            raise ValidationError(
                f'{label} regression manifest must not declare classes')
        return None
    if not isinstance(classes, list) or not classes:
        raise ValidationError(
            f'{label} classification manifest must declare classes')
    if (any(isinstance(value, bool) or not isinstance(value, int) or
            value < -(1 << 31) or value > (1 << 31) - 1
            for value in classes) or len(set(classes)) != len(classes)):
        raise ValidationError(
            f'{label} classes must be unique int32 values')
    return classes


def _validate_names(value, label):
    if not isinstance(value, list) or not value:
        raise ValidationError(f'{label} must be a nonempty array')
    names = [_validate_name(name, f'{label} entry') for name in value]
    if len(set(names)) != len(names):
        raise ValidationError(f'{label} must contain unique names')
    return names


def _validate_name(value, label):
    if not isinstance(value, str) or not value:
        raise ValidationError(f'{label} must be a nonempty string')
    return value


def _assert_integer(
        value, minimum, label, *, enforce_minimum=True):
    safe = ((isinstance(value, int) and not isinstance(value, bool)) and
            abs(value) <= (1 << 53) - 1)
    if not safe or (enforce_minimum and value < minimum):
        suffix = f' >= {minimum}' if enforce_minimum else ''
        raise ValidationError(f'{label} must be a safe integer{suffix}')


def _validate_artifacts(toc, expected, label):
    if len(toc) != len(expected):
        raise ValidationError(
            f'{label} artifact count is inconsistent with its params')
    actual = {entry['id']: entry for entry in toc}
    for artifact_id, media_type in expected:
        entry = actual.get(artifact_id)
        if entry is None or entry.get('mediaType') != media_type:
            raise ValidationError(
                f'{label} artifact "{artifact_id}" is missing or has the '
                'wrong media type')
