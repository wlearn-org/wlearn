"""Portable fitted tabular preprocessing over Tranfi prepared transforms."""

from array import array
from collections.abc import Mapping
import hashlib
import importlib
import json
import math
import numbers
import re
import struct

import numpy as np

from .bundle import encode_bundle, validate_bundle, write_bundle_output
from .errors import (
    BackendError, BundleError, CancelledError, DisposedError,
    NotFittedError, ResourceLimitError, ValidationError,
)
from .registry import register


TYPE_ID = 'wlearn.preprocess.tabular@1'
PLAN_MEDIA_TYPE = 'application/x-tranfi-transform-plan'
_HEX_DIGITS = frozenset('0123456789abcdef')
_MAX_SAFE = (1 << 53) - 1
_UNSET = object()
_TRANFI = None

_RESOLVED_KEYS = frozenset((
    'allMissing', 'encode', 'impute', 'maxCategories',
    'maxOutputColumns', 'maxOutputElements', 'policyVersion', 'scale',
    'unknownCategory',
))
_REQUEST_KEYS = frozenset((
    'allMissing', 'encode', 'impute', 'maxCategories',
    'maxOutputColumns', 'maxOutputElements', 'scale', 'unknownCategory', 'columns',
))
_DEFAULTS = {
    'impute': 'auto',
    'encode': 'auto',
    'scale': False,
    'maxCategories': 20,
    'maxOutputColumns': 65536,
    'maxOutputElements': 100000000,
}
_LIMIT_NAMES = {
    'maxRecipeBytes': 'max_recipe_bytes',
    'maxPlanBytes': 'max_plan_bytes',
    'maxJsonDepth': 'max_json_depth',
    'maxObjectKeys': 'max_object_keys',
    'maxSteps': 'max_steps',
    'maxInputColumns': 'max_input_columns',
    'maxOutputColumns': 'max_output_columns',
    'maxCategoriesPerColumn': 'max_categories_per_column',
    'maxTotalCategories': 'max_total_categories',
    'maxStringBytes': 'max_string_bytes',
    'maxDecodedStringBytes': 'max_decoded_string_bytes',
    'maxAnalyzerRows': 'max_analyzer_rows',
    'maxAnalyzerInputBytes': 'max_analyzer_input_bytes',
    'maxResidentStateBytes': 'max_resident_state_bytes',
    'maxSpillBytes': 'max_spill_bytes',
    'maxApplyRows': 'max_apply_rows',
    'maxApplyInputBytes': 'max_apply_input_bytes',
    'maxOutputElementsPerCall': 'max_output_elements_per_call',
    'maxAllocationBytes': 'max_allocation_bytes',
    'maxAllocationsPerSession': 'max_allocations_per_session',
    'maxLiveHandles': 'max_live_handles',
    'maxRetiredHandleSlots': 'max_retired_handle_slots',
}


def resolve_preprocess_config(config=None):
    """Resolve a preprocessing request without importing or initializing Tranfi."""
    return _clone_json(_resolve_config(config))


class Preprocessor:
    """A wlearn Transformer backed by one immutable Tranfi plan."""

    def __init__(
            self, config=None, *, runtime_options=None,
            impute=_UNSET, encode=_UNSET, scale=_UNSET,
            max_categories=_UNSET, unknown_category=_UNSET,
            all_missing=_UNSET, max_output_columns=_UNSET,
            max_output_elements=_UNSET, columns=_UNSET):
        keyword_values = {
            'impute': impute,
            'encode': encode,
            'scale': scale,
            'maxCategories': max_categories,
            'unknownCategory': unknown_category,
            'allMissing': all_missing,
            'maxOutputColumns': max_output_columns,
            'maxOutputElements': max_output_elements,
            'columns': columns,
        }
        supplied = {key: value for key, value in keyword_values.items()
                    if value is not _UNSET}
        if config is not None and supplied:
            raise ValidationError(
                'Pass either config or preprocessing keyword arguments, not both.')
        if config is None:
            config = supplied
        self._backend_module = _load_tranfi()
        self._config = _resolve_config(config)
        self._runtime_options, self._effective_limits = (
            _normalize_runtime_options(
                self._backend_module, runtime_options or {}))
        _assert_config_within_host_limits(
            self._config, self._effective_limits)
        self._plan = None
        self._input_schema = None
        self._output_schema = None
        self._recipe_sha256 = None
        self._disposed = False

    def fit(self, X, y=None):
        del y
        self._ensure_alive()
        matrix = _normalize_matrix(
            X, operation='fit', allow_zero_rows=False,
            expected_cols=None, limits=self._effective_limits,
            runtime=self._runtime_options)
        schema = _input_schema(matrix['cols'])
        recipe_spec = _build_recipe(self._config, matrix['cols'])
        recipe = analyzer = plan = None
        try:
            recipe = self._backend_module.TransformRecipe.from_json(
                _canonical_json(recipe_spec),
                limits=self._runtime_options.get('limits'))
            analyzer = recipe.analyzer(schema, **self._runtime_options)
            analyzer.push({
                'rows': matrix['rows'],
                'columns': matrix['columns'],
            })
            plan = analyzer.finalize()
            input_schema = _read_plan_schema(
                plan, 'input', self._runtime_options.get('limits'))
            output_schema = _read_plan_schema(
                plan, 'output', self._runtime_options.get('limits'))
            fingerprint = plan.recipe_sha256(
                limits=self._runtime_options.get('limits'))
            _validate_plan_identity(
                self._config, input_schema, output_schema,
                fingerprint, recipe_spec)
            previous = self._plan
            self._plan = plan
            plan = None
            self._input_schema = input_schema
            self._output_schema = output_schema
            self._recipe_sha256 = fingerprint
            _close_quietly(previous)
            return self
        except Exception as error:
            _raise_mapped(error, 'fit')
        finally:
            _close_quietly(plan)
            _close_quietly(analyzer)
            _close_quietly(recipe)

    def transform(self, X):
        self._ensure_fitted()
        matrix = _normalize_matrix(
            X, operation='transform', allow_zero_rows=True,
            expected_cols=len(self._input_schema),
            limits=self._effective_limits, runtime=self._runtime_options)
        apply = None
        try:
            apply = self._plan.apply(
                self._input_schema, **self._runtime_options)
            result = apply.run({
                'rows': matrix['rows'],
                'columns': matrix['columns'],
            })
            expected_size = result.rows * result.columns
            if (result.rows != matrix['rows'] or
                    result.columns != len(self._output_schema) or
                    len(result.data) != expected_size or
                    result.dtype != 'float64'):
                raise BackendError(
                    'Tranfi returned an invalid dense transform result.')
            # NumPy retains the owned Python buffer after the apply session closes.
            return np.asarray(result.data, dtype=np.float64).reshape(
                result.rows, result.columns)
        except Exception as error:
            _raise_mapped(error, 'apply')
        finally:
            _close_quietly(apply)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_params(self):
        self._ensure_alive()
        return _clone_json(self._config)

    def set_params(self, params):
        self._ensure_alive()
        resolved = _resolve_set_params(self._config, params)
        _assert_config_within_host_limits(
            resolved, self._effective_limits)
        old_plan = self._plan
        self._config = resolved
        self._plan = None
        self._input_schema = None
        self._output_schema = None
        self._recipe_sha256 = None
        _close_quietly(old_plan)
        return self

    def save(self, path=None):
        self._ensure_fitted()
        try:
            limits = self._runtime_options.get('limits')
            input_schema = _read_plan_schema(self._plan, 'input', limits)
            output_schema = _read_plan_schema(self._plan, 'output', limits)
            fingerprint = self._plan.recipe_sha256(limits=limits)
            recipe = _build_recipe(self._config, len(input_schema))
            _validate_plan_identity(
                self._config, input_schema, output_schema,
                fingerprint, recipe)
            if (input_schema != self._input_schema or
                    output_schema != self._output_schema or
                    fingerprint != self._recipe_sha256):
                raise BackendError(
                    'Fitted Tranfi plan identity changed unexpectedly.')
            plan_bytes = self._plan.to_bytes(limits=limits)
        except Exception as error:
            _raise_mapped(error, 'export')
        bundle = encode_bundle({
            'typeId': TYPE_ID,
            'requires': [],
            'params': _clone_json(self._config),
            'metadata': {
                'inputSchema': _clone_json(input_schema),
                'outputSchema': _clone_json(output_schema),
                'tranfi': {
                    'abiVersion': 1,
                    'planFormatVersion': 1,
                    'recipeSha256': fingerprint,
                },
            },
        }, [{
            'id': 'plan',
            'mediaType': PLAN_MEDIA_TYPE,
            'data': plan_bytes,
        }])
        return write_bundle_output(bundle, path)

    @classmethod
    def load(cls, data, *, runtime_options=None):
        manifest, toc, blobs = validate_bundle(data)
        actual = manifest.get('typeId')
        if actual != TYPE_ID:
            raise ValidationError(
                f'Preprocessor.load expected typeId "{TYPE_ID}", got "{actual}"')
        return cls._load_from_parts(
            manifest, toc, blobs, runtime_options or {})

    @classmethod
    def _load_from_parts(cls, manifest, toc, blobs, runtime_options):
        _validate_manifest_shape(manifest, toc)
        instance = cls(
            config=_resolve_config(manifest['params'], require_resolved=True),
            runtime_options=runtime_options)
        input_schema = _validate_schema(
            manifest['metadata']['inputSchema'], 'input')
        output_schema = _validate_schema(
            manifest['metadata']['outputSchema'], 'output')
        expected_input = _input_schema(len(input_schema))
        if input_schema != expected_input:
            raise BundleError(
                'Preprocessor input schema is not the canonical x0..xN '
                'float64 schema.')
        recipe = _build_recipe(instance._config, len(input_schema))
        expected_fingerprint = _recipe_fingerprint(expected_input, recipe)
        if (manifest['metadata']['tranfi']['recipeSha256'] !=
                expected_fingerprint):
            raise BundleError(
                'Preprocessor params do not match the stored recipe fingerprint.')
        entry = toc[0]
        plan_bytes = bytes(
            blobs[entry['offset']:entry['offset'] + entry['length']])
        plan = None
        try:
            plan = instance._backend_module.TransformPlan.from_bytes(
                plan_bytes, **instance._runtime_options)
            limits = instance._runtime_options.get('limits')
            plan_input = _read_plan_schema(plan, 'input', limits)
            plan_output = _read_plan_schema(plan, 'output', limits)
            fingerprint = plan.recipe_sha256(limits=limits)
            if plan_input != input_schema or plan_output != output_schema:
                raise BundleError(
                    'Preprocessor manifest schemas do not match the Tranfi plan.')
            if fingerprint != expected_fingerprint:
                raise BundleError(
                    'Preprocessor manifest params do not match the Tranfi plan recipe.')
            instance._plan = plan
            plan = None
            instance._input_schema = plan_input
            instance._output_schema = plan_output
            instance._recipe_sha256 = fingerprint
            return instance
        except Exception as error:
            _close_quietly(plan)
            instance.dispose()
            _raise_mapped(error, 'import')

    def dispose(self):
        if self._disposed:
            return
        self._disposed = True
        plan = self._plan
        self._plan = None
        self._input_schema = None
        self._output_schema = None
        self._recipe_sha256 = None
        _close_quietly(plan)

    @property
    def capabilities(self):
        return {'transformer': True}

    @property
    def is_fitted(self):
        return not self._disposed and self._plan is not None

    @property
    def backend(self):
        return 'native'

    @property
    def input_schema(self):
        self._ensure_fitted()
        return _clone_json(self._input_schema)

    @property
    def output_schema(self):
        self._ensure_fitted()
        return _clone_json(self._output_schema)

    @property
    def output_cols(self):
        self._ensure_fitted()
        return len(self._output_schema)

    def _ensure_alive(self):
        if self._disposed:
            raise DisposedError('Preprocessor has been disposed.')

    def _ensure_fitted(self):
        self._ensure_alive()
        if self._plan is None:
            raise NotFittedError(
                'Preprocessor is not fitted. Call fit() first.')

    # JS contract spellings are aliases; Python code should prefer snake_case.
    fitTransform = fit_transform
    getParams = get_params
    setParams = set_params

    @property
    def isFitted(self):
        return self.is_fitted


def _load_tranfi():
    global _TRANFI
    if _TRANFI is not None:
        return _TRANFI
    try:
        module = importlib.import_module('tranfi')
    except ImportError as error:
        mapped = BackendError(
            'wlearn preprocessing requires the optional Tranfi dependency. '
            'Install wlearn[preprocess].')
        mapped.engine = 'tranfi'
        mapped.engineCode = None
        raise mapped from error
    required = ('TransformRecipe', 'TransformPlan', 'TransformLimits')
    if any(not hasattr(module, name) for name in required):
        mapped = BackendError(
            'Installed Tranfi does not expose the prepared-transform API. '
            'Install a compatible tranfi>=0.2,<0.3 release.')
        mapped.engine = 'tranfi'
        mapped.engineCode = None
        raise mapped
    _TRANFI = module
    return module


def _normalize_runtime_options(backend, options):
    if not isinstance(options, Mapping):
        raise ValidationError('runtime_options must be a mapping.')
    unknown = set(options).difference(('limits', 'cancel_token'))
    if unknown:
        raise ValidationError(
            f'Unknown runtime option "{sorted(unknown)[0]}".')
    limits = options.get('limits')
    if limits is None:
        limits = backend.TransformLimits()
    elif isinstance(limits, Mapping):
        normalized = {}
        for name, value in limits.items():
            snake = _LIMIT_NAMES.get(name, name)
            if snake in normalized:
                raise ValidationError(
                    f'Duplicate runtime limit alias for "{snake}".')
            normalized[snake] = value
        try:
            limits = backend.TransformLimits(**normalized)
        except Exception as error:
            _raise_mapped(error, 'configuration')
    elif not isinstance(limits, backend.TransformLimits):
        raise ValidationError(
            'runtime_options.limits must be a dict or TransformLimits.')
    cancel_token = options.get('cancel_token')
    if (cancel_token is not None and
            not isinstance(cancel_token, backend.TransformCancelToken)):
        raise ValidationError(
            'runtime_options.cancel_token must be a TransformCancelToken.')
    runtime = {'limits': limits}
    if cancel_token is not None:
        runtime['cancel_token'] = cancel_token
    return runtime, limits.to_dict()


def _resolve_config(config, require_resolved=False):
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValidationError('preprocess config must be a dict.')
    is_resolved = isinstance(config.get('impute'), dict)
    if require_resolved or is_resolved or 'policyVersion' in config:
        _assert_exact_keys(config, _RESOLVED_KEYS | ({'columns'} if 'columns' in config else set()),
                           'resolved preprocess config')
        return _validate_resolved_config(config)
    unknown = set(config).difference(_REQUEST_KEYS)
    if unknown:
        raise ValidationError(
            f'Unknown preprocess config key "{sorted(unknown)[0]}".')
    request = dict(_DEFAULTS)
    request.update(config)
    resolved = _resolve_request_config(request, config)
    if 'columns' in config:
        resolved['columns'] = config['columns']
    return _validate_resolved_config(resolved)


def _resolve_set_params(current, patch):
    if not isinstance(patch, dict):
        raise ValidationError('preprocess params must be a dict.')
    if isinstance(patch.get('impute'), dict) or 'policyVersion' in patch:
        return _resolve_config(patch, require_resolved=True)
    unknown = set(patch).difference(_REQUEST_KEYS)
    if unknown:
        raise ValidationError(
            f'Unknown preprocess config key "{sorted(unknown)[0]}".')
    merged = _clone_json(current)
    if 'impute' in patch:
        value = patch['impute']
        if not (value is False or value in ('auto', 'mean', 'median', 'zero')):
            raise ValidationError(
                'impute must be auto, mean, median, zero, or false.')
        merged['impute'] = (
            {'numeric': False, 'categorical': False}
            if value is False else
            {'numeric': 'mean' if value == 'auto' else value,
             'categorical': 'mode'})
        if value is False:
            if 'allMissing' in patch:
                raise ValidationError(
                    'allMissing is invalid when imputation is disabled.')
            merged['allMissing'] = None
        elif 'allMissing' not in patch:
            merged['allMissing'] = 'zero'
    if 'encode' in patch:
        value = 'onehot' if patch['encode'] == 'auto' else patch['encode']
        if not (value is False or value in ('onehot', 'label')):
            raise ValidationError(
                'encode must be auto, onehot, label, or false.')
        merged['encode'] = value
        if 'unknownCategory' not in patch:
            merged['unknownCategory'] = (
                None if value is False else
                'all_zero' if value == 'onehot' else 'sentinel')
    for key, value in patch.items():
        if key not in ('impute', 'encode'):
            merged[key] = value
    return _validate_resolved_config(merged)


def _resolve_request_config(request, supplied):
    impute = request['impute']
    if not (impute is False or impute in ('auto', 'mean', 'median', 'zero')):
        raise ValidationError(
            'impute must be auto, mean, median, zero, or false.')
    encode = 'onehot' if request['encode'] == 'auto' else request['encode']
    if not (encode is False or encode in ('onehot', 'label')):
        raise ValidationError(
            'encode must be auto, onehot, label, or false.')
    if not (request['scale'] is False or
            request['scale'] in ('standard', 'minmax')):
        raise ValidationError('scale must be standard, minmax, or false.')
    _assert_positive_safe_integer(
        request['maxCategories'], 'maxCategories', minimum=2)
    _assert_positive_safe_integer(
        request['maxOutputColumns'], 'maxOutputColumns')
    _assert_positive_safe_integer(
        request['maxOutputElements'], 'maxOutputElements')
    resolved_impute = (
        {'numeric': False, 'categorical': False}
        if impute is False else
        {'numeric': 'mean' if impute == 'auto' else impute,
         'categorical': 'mode'})
    if impute is False and 'allMissing' in supplied:
        raise ValidationError(
            'allMissing is invalid when imputation is disabled.')
    all_missing = None if impute is False else request.get(
        'allMissing', 'zero')
    if all_missing is not None and all_missing not in ('error', 'zero'):
        raise ValidationError('allMissing must be error or zero.')
    if encode is False and 'unknownCategory' in supplied:
        raise ValidationError(
            'unknownCategory is invalid when encoding is disabled.')
    unknown = (
        None if encode is False else
        request.get(
            'unknownCategory',
            'all_zero' if encode == 'onehot' else 'sentinel'))
    _validate_unknown_policy(encode, unknown)
    return _validate_resolved_config({
        'impute': resolved_impute,
        'encode': encode,
        'scale': request['scale'],
        'maxCategories': request['maxCategories'],
        'unknownCategory': unknown,
        'allMissing': all_missing,
        'maxOutputColumns': request['maxOutputColumns'],
        'maxOutputElements': request['maxOutputElements'],
        'policyVersion': 1,
    })


def _validate_resolved_config(config):
    _assert_exact_keys(config, _RESOLVED_KEYS | ({'columns'} if 'columns' in config else set()),
                           'resolved preprocess config')
    impute = config['impute']
    if not isinstance(impute, dict):
        raise ValidationError(
            'resolved preprocess config.impute must be a dict.')
    _assert_exact_keys(
        impute, frozenset(('categorical', 'numeric')),
        'resolved preprocess config.impute')
    numeric = impute['numeric']
    categorical = impute['categorical']
    if not (numeric is False or numeric in ('mean', 'median', 'zero')):
        raise ValidationError('resolved numeric imputation is invalid.')
    if (not (categorical is False or categorical == 'mode') or
            (numeric is False) != (categorical is False)):
        raise ValidationError(
            'resolved categorical imputation is invalid or inconsistent.')
    if not (config['encode'] is False or
            config['encode'] in ('onehot', 'label')):
        raise ValidationError('resolved encoding is invalid.')
    if not (config['scale'] is False or
            config['scale'] in ('standard', 'minmax')):
        raise ValidationError('resolved scaling is invalid.')
    _assert_positive_safe_integer(
        config['maxCategories'], 'maxCategories', minimum=2)
    _assert_positive_safe_integer(
        config['maxOutputColumns'], 'maxOutputColumns')
    _assert_positive_safe_integer(
        config['maxOutputElements'], 'maxOutputElements')
    if (isinstance(config['policyVersion'], bool) or
            config['policyVersion'] != 1):
        raise ValidationError('policyVersion must be 1.')
    if numeric is False:
        if config['allMissing'] is not None:
            raise ValidationError(
                'allMissing must be null when imputation is disabled.')
    elif config['allMissing'] not in ('error', 'zero'):
        raise ValidationError(
            'resolved allMissing must be error or zero.')
    _validate_unknown_policy(
        config['encode'], config['unknownCategory'])
    result = _clone_json({key: value for key, value in config.items() if key != 'columns'})
    if 'columns' in config:
        columns = _resolve_columns(result, config['columns'])
        if columns:
            result['columns'] = columns
    return result


def _resolve_columns(base, columns):
    if not isinstance(columns, dict):
        raise ValidationError('columns must be a dict keyed by x0, x1, etc.')
    # Store explicit overrides so later global updates still apply to inherited options.
    result = {}
    allowed = {'kind', 'categories', 'impute', 'encode', 'scale',
               'allMissing', 'unknownCategory', 'maxCategories'}
    for source, patch in columns.items():
        if (not isinstance(source, str) or not re.fullmatch(r'x(?:0|[1-9][0-9]*)', source)
                or len(source) > 17 or int(source[1:]) > _MAX_SAFE):
            raise ValidationError('columns keys must be canonical x0, x1, etc.')
        if not isinstance(patch, dict) or set(patch).difference(allowed):
            raise ValidationError(f'Invalid column policy for {source}.')
        kind = patch.get('kind', 'categorical' if 'categories' in patch else 'infer')
        if kind not in ('numeric', 'categorical', 'infer'):
            raise ValidationError(f'Invalid kind for {source}.')
        operations = {key: value for key, value in patch.items()
                      if key not in ('kind', 'categories')}
        local = _resolve_set_params(base, operations)
        normalized = dict(operations, kind=kind)
        if 'categories' in patch:
            values = patch['categories']
            if (kind != 'categorical' or not isinstance(values, list) or not values
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           for v in values)):
                raise ValidationError(f'{source} categories must be finite numbers for a categorical column.')
            try:
                values = [0.0 if v == 0 else float(v) for v in values]
            except OverflowError as error:
                raise ValidationError(f'{source} categories must fit finite float64.') from error
            if not all(math.isfinite(value) for value in values):
                raise ValidationError(f'{source} categories must fit finite float64.')
            values.sort()
            if any(a == b for a, b in zip(values, values[1:])):
                raise ValidationError(f'{source} categories must be unique.')
            if local['encode'] is False and local['impute']['categorical'] is False:
                raise ValidationError('Fixed categories require encoding or mode imputation.')
            normalized['categories'] = values
        result[source] = normalized
    return result


def _validate_unknown_policy(encode, unknown):
    if encode is False and unknown is not None:
        raise ValidationError(
            'unknownCategory must be null when encoding is disabled.')
    if encode == 'onehot' and unknown not in ('error', 'all_zero'):
        raise ValidationError(
            'onehot encoding requires unknownCategory error or all_zero.')
    if encode == 'label' and unknown not in ('error', 'sentinel'):
        raise ValidationError(
            'label encoding requires unknownCategory error or sentinel.')


def _recipe_column(config, index, policy):
    numeric = config['impute']['numeric']
    if numeric is False:
        numeric_impute = {
            'op': 'none', 'constant': None, 'allMissing': None}
    elif numeric == 'zero':
        numeric_impute = {
            'op': 'zero', 'constant': None, 'allMissing': None}
    else:
        numeric_impute = {
            'op': numeric, 'constant': None,
            'allMissing': config['allMissing']}
    normalize = (
        {'op': 'none', 'ddof': None}
        if config['scale'] is False else
        {'op': config['scale'],
         'ddof': 0 if config['scale'] == 'standard' else None})
    categorical = config['impute']['categorical']
    categorical_impute = (
        {'op': 'none', 'constant': None, 'allMissing': None}
        if categorical is False else
        {'op': 'mode', 'constant': None,
         'allMissing': config['allMissing']})
    if config['encode'] is False:
        categorical_encode = {
            'op': 'none',
            'categories': None if categorical is False else 'discover',
            'unknown': None,
            'sentinelLabel': None,
        }
    else:
        categorical_encode = {
            'op': config['encode'],
            'categories': 'discover',
            'unknown': config['unknownCategory'],
            'sentinelLabel': (
                -1 if config['unknownCategory'] == 'sentinel' else None),
        }
    kind = policy.get('kind', 'infer')
    if 'categories' in policy:
        categorical_encode['categories'] = [
            {'t': 'f64', 'v': struct.pack('>d', value).hex()}
            for value in policy['categories']]
    return {
        'sourceId': f'x{index}',
        'kind': ({'op': 'infer', 'value': None,
                  'rule': 'finite-integer-cardinality-v1',
                  'maxCategories': config['maxCategories']}
                 if kind == 'infer' else
                 {'op': 'declared', 'value': kind, 'rule': None, 'maxCategories': None}),
        'numeric': None if kind == 'categorical' else {
            'impute': numeric_impute, 'normalize': normalize},
        'categorical': None if kind == 'numeric' else {
            'impute': categorical_impute, 'encode': categorical_encode},
    }


def _build_recipe(config, columns):
    policies = config.get('columns', {})
    for source in policies:
        if int(source[1:]) >= columns:
            raise ValidationError(f'Column policy {source} is outside the input width.')
    base = {key: value for key, value in config.items() if key != 'columns'}
    fields = []
    for index in range(columns):
        policy = policies.get(f'x{index}', {})
        local = _resolve_set_params(base, {
            key: value for key, value in policy.items() if key not in ('kind', 'categories')
        }) if policy else base
        fields.append(_recipe_column(local, index, policy))
    return {
        'format': 'tranfi.transform-recipe',
        'version': 1,
        'policyVersion': 1,
        'outputDtype': 'float64',
        'semanticLimits': {
            'maxOutputColumns': config['maxOutputColumns'],
            'maxOutputElementsPerApply': config['maxOutputElements'],
        },
        'columns': fields,
    }


def _check_cancelled(runtime):
    token = runtime.get('cancel_token')
    if token is not None and token.requested:
        _raise_mapped(_load_tranfi().TranfiTransformError(
            109, 'Tranfi preprocessing cancelled during input conversion.'),
            'transport')


def _normalize_matrix(X, *, operation, allow_zero_rows,
                      expected_cols, limits, runtime=None):
    runtime = runtime or {}
    _check_cancelled(runtime)
    if isinstance(X, np.ndarray):
        if X.ndim != 2 or X.dtype not in (np.dtype('float32'),
                                          np.dtype('float64')):
            raise ValidationError(
                'NumPy X must be a two-dimensional float32 or float64 array.')
        rows, cols = X.shape
        source = X
    elif isinstance(X, (list, tuple)):
        rows = len(X)
        if rows == 0:
            raise ValidationError(
                'A zero-row matrix must declare its fitted width.')
        if not isinstance(X[0], (list, tuple)) or len(X[0]) == 0:
            raise ValidationError('Matrix rows must be nonempty sequences.')
        cols = len(X[0])
        for index, row in enumerate(X):
            if index % 8192 == 0:
                _check_cancelled(runtime)
            if not isinstance(row, (list, tuple)) or len(row) != cols:
                raise ValidationError('Matrix must be rectangular.')
        source = X
    else:
        raise ValidationError(
            'X must be a float32/float64 NumPy array or number matrix.')
    _assert_nonnegative_safe_integer(rows, 'rows')
    _assert_nonnegative_safe_integer(cols, 'cols')
    if cols == 0 or (not allow_zero_rows and rows == 0):
        raise ValidationError(
            f'{operation} requires positive rows and columns.')
    if expected_cols is not None and cols != expected_cols:
        raise ValidationError(
            f'Transform expected {expected_cols} columns, got {cols}.')
    elements = _checked_product(rows, cols, 'rows * cols')
    byte_length = _checked_product(elements, 8, 'matrix byte length')
    row_limit = limits[
        'max_analyzer_rows' if operation == 'fit' else 'max_apply_rows']
    byte_limit = limits[
        'max_analyzer_input_bytes'
        if operation == 'fit' else 'max_apply_input_bytes']
    if (cols > limits['max_input_columns'] or rows > row_limit or
            byte_length > byte_limit):
        raise ResourceLimitError(
            f'{operation} input exceeds Tranfi runtime limits.')
    column_bytes = _checked_product(rows, 8, 'column byte length')
    if column_bytes > limits['max_allocation_bytes']:
        raise ResourceLimitError(
            f'{operation} column exceeds Tranfi allocation limit.')
    columns = []
    if isinstance(source, np.ndarray):
        source = np.asarray(source)
        for col in range(cols):
            _check_cancelled(runtime)
            values = np.empty(rows, dtype=np.float64)
            # Bounded blocks keep validation temporaries small and let a
            # cancellation request interrupt large/strided host copies.
            for start in range(0, rows, 8192):
                _check_cancelled(runtime)
                block = source[start:start + 8192, col]
                if np.isinf(block).any():
                    raise ValidationError(
                        'Matrix values must be finite numbers or NaN.')
                values[start:start + 8192] = block
            columns.append(values)
    else:
        for _ in range(cols):
            _check_cancelled(runtime)
            columns.append(array('d'))
        for row in range(rows):
            for col in range(cols):
                if (row * cols + col) % 8192 == 0:
                    _check_cancelled(runtime)
                value = source[row][col]
                if (isinstance(value, (bool, np.bool_)) or
                        not isinstance(value, numbers.Real)):
                    raise ValidationError(
                        'Matrix values must be finite numbers or NaN.')
                value = float(value)
                if math.isinf(value):
                    raise ValidationError(
                        'Matrix values must be finite numbers or NaN.')
                columns[col].append(value)
    _check_cancelled(runtime)
    return {'rows': rows, 'cols': cols, 'columns': columns}


def _input_schema(columns):
    return [
        {'dtype': 'float64', 'id': f'x{index}', 'name': f'x{index}'}
        for index in range(columns)
    ]


def _read_plan_schema(plan, which, limits):
    try:
        value = json.loads(plan.schema_json(which, limits=limits))
    except Exception as error:
        if isinstance(getattr(error, 'code', None), int):
            raise
        raise BackendError(
            f'Tranfi returned invalid {which} schema JSON.') from error
    return _validate_schema(value, which)


def _validate_schema(schema, which):
    if not isinstance(schema, list) or not schema:
        raise BundleError(
            f'Preprocessor {which} schema must be a nonempty list.')
    expected_keys = (
        frozenset(('dtype', 'id', 'name')) if which == 'input' else
        frozenset(('category', 'dtype', 'id', 'name', 'role', 'sourceId')))
    result = []
    ids = set()
    for index, field in enumerate(schema):
        if not isinstance(field, dict):
            raise BundleError(
                f'{which} schema field {index} must be a dict.')
        _assert_exact_keys(
            field, expected_keys, f'{which} schema field {index}',
            error_class=BundleError)
        if (field['dtype'] != 'float64' or
                not isinstance(field['id'], str) or not field['id'] or
                not isinstance(field['name'], str) or not field['name']):
            raise BundleError(
                f'Preprocessor {which} schema field {index} is invalid.')
        if field['id'] in ids:
            raise BundleError(
                f'Duplicate {which} schema id "{field["id"]}".')
        ids.add(field['id'])
        if which == 'output':
            if (not isinstance(field['sourceId'], str) or
                    not field['sourceId'] or
                    field['role'] not in ('value', 'label', 'onehot')):
                raise BundleError(
                    f'Preprocessor output schema field {index} metadata '
                    'is invalid.')
            _validate_category_tag(field['category'], index)
        result.append(_clone_json(field))
    return result


def _validate_category_tag(category, index):
    if category is None:
        return
    if not isinstance(category, dict):
        raise BundleError(
            f'output schema field {index} category must be a dict.')
    if category.get('t') == 'other':
        _assert_exact_keys(
            category, frozenset(('t',)),
            f'output schema field {index} category',
            error_class=BundleError)
        return
    _assert_exact_keys(
        category, frozenset(('t', 'v')),
        f'output schema field {index} category',
        error_class=BundleError)
    tag = category['t']
    value = category['v']
    if (tag not in ('f32', 'f64') or not isinstance(value, str) or
            len(value) != (8 if tag == 'f32' else 16) or
            any(char not in _HEX_DIGITS for char in value)):
        raise BundleError(
            f'Preprocessor output schema field {index} category is invalid.')


def _validate_manifest_shape(manifest, toc):
    if not isinstance(manifest, dict):
        raise BundleError('preprocessor manifest must be a dict.')
    _assert_exact_keys(
        manifest,
        frozenset(('artifacts', 'bundleVersion', 'metadata', 'params',
                   'requires', 'typeId')),
        'preprocessor manifest', error_class=BundleError)
    if (manifest['typeId'] != TYPE_ID or manifest['bundleVersion'] != 1 or
            manifest['requires'] != [] or
            not isinstance(manifest['artifacts'], list) or
            len(manifest['artifacts']) != 1 or
            not isinstance(toc, list) or len(toc) != 1):
        raise BundleError(
            'Preprocessor bundle has an invalid top-level shape.')
    declaration = manifest['artifacts'][0]
    entry = toc[0]
    for value in (declaration, entry):
        if (value.get('id') != 'plan' or
                value.get('mediaType') != PLAN_MEDIA_TYPE):
            raise BundleError(
                'Preprocessor bundle must contain exactly one Tranfi plan '
                'artifact.')
    if (declaration['length'] != entry['length'] or
            declaration['sha256'] != entry['sha256']):
        raise BundleError(
            'Preprocessor artifact declaration and TOC disagree.')
    metadata = manifest['metadata']
    if not isinstance(metadata, dict):
        raise BundleError('preprocessor metadata must be a dict.')
    _assert_exact_keys(
        metadata, frozenset(('inputSchema', 'outputSchema', 'tranfi')),
        'preprocessor metadata', error_class=BundleError)
    info = metadata['tranfi']
    if not isinstance(info, dict):
        raise BundleError('preprocessor metadata.tranfi must be a dict.')
    _assert_exact_keys(
        info, frozenset(('abiVersion', 'planFormatVersion', 'recipeSha256')),
        'preprocessor metadata.tranfi', error_class=BundleError)
    fingerprint = info['recipeSha256']
    if (isinstance(info['abiVersion'], bool) or
            info['abiVersion'] != 1 or
            isinstance(info['planFormatVersion'], bool) or
            info['planFormatVersion'] != 1 or
            not isinstance(fingerprint, str) or len(fingerprint) != 64 or
            any(char not in _HEX_DIGITS for char in fingerprint)):
        raise BundleError(
            'Preprocessor Tranfi metadata is invalid or unsupported.')


def _validate_plan_identity(config, input_schema, output_schema,
                            fingerprint, recipe):
    expected_input = _input_schema(len(input_schema))
    if input_schema != expected_input:
        raise BackendError(
            'Tranfi plan input schema differs from the canonical wlearn schema.')
    if len(output_schema) > config['maxOutputColumns']:
        raise ResourceLimitError(
            'Fitted output schema exceeds maxOutputColumns.')
    if fingerprint != _recipe_fingerprint(expected_input, recipe):
        raise BackendError(
            'Tranfi plan recipe fingerprint does not match the wlearn config.')


def _recipe_fingerprint(schema, recipe):
    value = {
        'inputSchema': schema,
        'policyVersion': 1,
        'recipe': recipe,
    }
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _canonical_json(value):
    return json.dumps(
        value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode('utf-8')


def _assert_config_within_host_limits(config, limits):
    if (config['maxCategories'] > limits['max_categories_per_column'] or
            config['maxOutputColumns'] > limits['max_output_columns'] or
            config['maxOutputElements'] >
            limits['max_output_elements_per_call']):
        raise ResourceLimitError(
            'Preprocess semantic limits exceed the active Tranfi host profile.')
    total_categories = 0
    for policy in config.get('columns', {}).values():
        count = len(policy.get('categories', []))
        total_categories += count
        if (count > limits['max_categories_per_column'] or
                total_categories > limits['max_total_categories'] or
                (policy.get('kind') == 'infer' and
                 policy.get('maxCategories', config['maxCategories']) >
                 min(limits['max_categories_per_column'], limits['max_total_categories']))):
            raise ResourceLimitError('Column policy exceeds the active Tranfi category limits.')


def _raise_mapped(error, phase):
    if isinstance(error, (
            ValidationError, ResourceLimitError, CancelledError, BundleError,
            BackendError, DisposedError, NotFittedError)):
        raise error
    code = getattr(error, 'code', None)
    if not isinstance(code, int):
        mapped = BackendError(
            f'Tranfi backend operation failed: {error}')
        mapped.engine = 'tranfi'
        mapped.engineCode = None
        raise mapped from error
    if 100 <= code <= 103 or code in (107, 108):
        mapped = ValidationError(str(error))
    elif code == 104:
        mapped = ResourceLimitError(str(error))
    elif code == 109:
        mapped = CancelledError(str(error))
    elif phase == 'import' and code in (105, 106):
        mapped = BundleError(str(error))
    else:
        mapped = BackendError(str(error))
    mapped.engine = 'tranfi'
    mapped.engineCode = code
    raise mapped from error


def _preprocessor_loader(manifest, toc, blobs, context):
    options = context['loaderOptions'].get(TYPE_ID, {})
    return Preprocessor._load_from_parts(
        manifest, toc, blobs, options)


def _close_quietly(value):
    if value is None:
        return
    try:
        if hasattr(value, 'close'):
            value.close()
        elif hasattr(value, 'dispose'):
            value.dispose()
    except Exception:
        pass


def _assert_exact_keys(value, expected, label,
                       error_class=ValidationError):
    if set(value) != set(expected):
        raise error_class(
            f'{label} must contain exactly: {", ".join(sorted(expected))}.')


def _assert_positive_safe_integer(value, label, minimum=1):
    if (isinstance(value, bool) or not isinstance(value, int) or
            value < minimum or value > _MAX_SAFE):
        raise ValidationError(
            f'{label} must be a safe integer >= {minimum}.')


def _assert_nonnegative_safe_integer(value, label):
    if (isinstance(value, bool) or not isinstance(value, int) or
            value < 0 or value > _MAX_SAFE):
        raise ValidationError(
            f'{label} must be a nonnegative safe integer.')


def _checked_product(left, right, label):
    if left and right > _MAX_SAFE // left:
        raise ResourceLimitError(
            f'{label} exceeds the portable safe integer range.')
    return left * right


def _clone_json(value):
    if isinstance(value, dict):
        return {key: _clone_json(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_clone_json(child) for child in value]
    return value


register(TYPE_ID, _preprocessor_loader, accepts_context=True)
