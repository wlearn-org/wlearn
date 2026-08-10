"""Portable structured AutoML candidate identity shared with JavaScript."""

from copy import deepcopy
import hashlib
import math
import struct

from ..errors import ValidationError
from ..preprocess import resolve_preprocess_config


PREPROCESS_TYPE_ID = 'wlearn.preprocess.tabular@1'
MAX_SAFE_INTEGER = (1 << 53) - 1
UINT32_MAX = (1 << 32) - 1


def create_candidate(model, params, preprocess=None):
    if not isinstance(model, dict):
        raise ValidationError('candidate model must be a dict.')
    display_name = _assert_string(
        model.get('displayName', model.get('name')), 'model.displayName')
    class_id = _assert_string(model.get('classId'), 'model.classId')
    if not isinstance(params, dict):
        raise ValidationError('candidate model.params must be a dict.')
    record = {
        'model': {
            'displayName': display_name,
            'classId': class_id,
            'params': _clone_domain(params),
        },
        'preprocess': (
            None if preprocess is None else _normalize_preprocess(preprocess)),
    }
    _encode_tagged(record)
    return _freeze_domain(record)


def _normalize_preprocess(preprocess):
    if not isinstance(preprocess, dict):
        raise ValidationError('candidate preprocess must be null or a dict.')
    template_id = _assert_string(
        preprocess.get('templateId'), 'preprocess.templateId')
    if preprocess.get('typeId') != PREPROCESS_TYPE_ID:
        raise ValidationError(
            f'preprocess.typeId must be "{PREPROCESS_TYPE_ID}".')
    if 'resolvedParams' not in preprocess:
        raise ValidationError('preprocess.resolvedParams is required.')
    return {
        'templateId': template_id,
        'typeId': PREPROCESS_TYPE_ID,
        'resolvedParams': _clone_domain(resolve_preprocess_config(
            preprocess['resolvedParams'])),
    }


def candidate_canonical_bytes(candidate):
    normalized = create_candidate(
        candidate['model'], candidate['model']['params'],
        candidate.get('preprocess'))
    identity = {
        'version': 1,
        'model': {
            'classId': normalized['model']['classId'],
            'params': normalized['model']['params'],
        },
        'preprocess': normalized['preprocess'],
    }
    return _write_canonical(_encode_tagged(identity)).encode('utf-8')


def candidate_hash(candidate):
    return hashlib.sha256(candidate_canonical_bytes(candidate)).hexdigest()


def make_candidate_id(candidate):
    return f'wlc1_{candidate_hash(candidate)}'


def seed_for(candidate, fold_idx, base_seed):
    _assert_uint32(base_seed, 'base_seed')
    if (isinstance(fold_idx, bool) or not isinstance(fold_idx, int) or
            fold_idx < 0 or fold_idx >= UINT32_MAX):
        raise ValidationError(
            'fold_idx must be an integer from 0 through 4294967294.')
    digest = bytes.fromhex(candidate_hash(candidate))
    hash_word = int.from_bytes(digest[:4], byteorder='little')
    fold_mix = ((fold_idx + 1) * 0x9e3779b9) & UINT32_MAX
    return (base_seed ^ hash_word ^ fold_mix) & UINT32_MAX


def register_candidate(candidate, seen):
    canonical = candidate_canonical_bytes(candidate)
    candidate_id = f'wlc1_{hashlib.sha256(canonical).hexdigest()}'
    previous = seen.get(candidate_id)
    if previous is not None and previous != canonical:
        raise ValidationError(
            f'candidate identity collision for {candidate_id}.')
    seen[candidate_id] = canonical
    return candidate_id


def preprocess_choices(model):
    if 'preprocessChoices' not in model:
        return [None]
    choices = model['preprocessChoices']
    if not isinstance(choices, (list, tuple)) or not choices:
        raise ValidationError(
            'model preprocessChoices must be a nonempty list.')
    return list(choices)


def create_candidate_task(model, params, preprocess, seen):
    candidate = create_candidate(model, params, preprocess)
    candidate_id = register_candidate(candidate, seen)
    cls = model['cls']
    if preprocess is not None:
        factory = model.get('createCandidateClass')
        if not callable(factory):
            raise ValidationError(
                'a non-null preprocessing candidate requires a '
                'candidate class factory.')
        cls = factory(candidate)
    return {
        'candidateId': candidate_id,
        'candidate': candidate,
        'cls': cls,
        'params': candidate['model']['params'],
    }


def class_for_candidate(model, candidate):
    if candidate.get('preprocess') is None:
        return model['cls']
    factory = model.get('createCandidateClass')
    if not callable(factory):
        raise ValidationError(
            'a non-null preprocessing candidate requires a '
            'candidate class factory.')
    return factory(candidate)


def normalize_model_specs(models, label='models'):
    if not isinstance(models, (list, tuple)) or not models:
        raise ValidationError(f'{label} must be a nonempty list.')
    result = []
    class_ids = set()
    for index, item in enumerate(models):
        if isinstance(item, (list, tuple)):
            if len(item) not in (2, 3):
                raise ValidationError(
                    f'{label}[{index}] tuple must contain name, class, '
                    'and optional params.')
            spec = {
                'name': item[0], 'cls': item[1],
                'params': item[2] if len(item) == 3 else {},
            }
        elif isinstance(item, dict):
            spec = dict(item)
        else:
            raise ValidationError(
                f'{label}[{index}] must be a model spec or tuple.')
        cls = spec.get('cls')
        if cls is None or not callable(getattr(cls, 'create', None)):
            raise ValidationError(
                f'{label}[{index}].cls must expose create().')
        display_name = _assert_string(
            spec.get('displayName', spec.get('name')),
            f'{label}[{index}].name')
        class_id = _assert_string(
            spec.get('classId', getattr(cls, 'class_id', None)),
            f'{label}[{index}].classId')
        if class_id in class_ids:
            raise ValidationError(f'duplicate model classId "{class_id}".')
        class_ids.add(class_id)
        params = spec['params'] if 'params' in spec else {}
        if not isinstance(params, dict):
            raise ValidationError(
                f'{label}[{index}].params must be a dict.')
        if 'preprocessChoices' in spec:
            preprocess_choices(spec)
        spec.update({
            'name': display_name,
            'displayName': display_name,
            'classId': class_id,
            'cls': cls,
            'params': _clone_domain(params),
        })
        result.append(spec)
    return result


def _encode_tagged(value, stack=None):
    if stack is None:
        stack = set()
    if value is None:
        return ['null']
    if isinstance(value, bool):
        return ['boolean', value]
    if isinstance(value, str):
        _validate_unicode(value, 'candidate string')
        return ['string', value]
    if isinstance(value, int):
        if abs(value) > MAX_SAFE_INTEGER:
            raise ValidationError(
                'candidate integers must be within the JavaScript '
                'safe-integer domain.')
        return ['number', struct.pack('>d', float(value)).hex()]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError('candidate numbers must be finite.')
        if value.is_integer() and abs(value) > MAX_SAFE_INTEGER:
            raise ValidationError(
                'candidate integers must be within the JavaScript '
                'safe-integer domain.')
        canonical = 0.0 if value == 0.0 else value
        return ['number', struct.pack('>d', canonical).hex()]
    if not isinstance(value, (list, tuple, dict)):
        raise ValidationError(
            'candidate values must use the portable JSON domain.')
    identity = id(value)
    if identity in stack:
        raise ValidationError('candidate values must not be cyclic.')
    stack.add(identity)
    try:
        if isinstance(value, (list, tuple)):
            return ['array', [_encode_tagged(item, stack) for item in value]]
        for key in value:
            if not isinstance(key, str):
                raise ValidationError('candidate object keys must be strings.')
            _validate_unicode(key, 'candidate object key')
        keys = sorted(value, key=lambda key: key.encode('utf-8'))
        return ['object', [
            [key, _encode_tagged(value[key], stack)] for key in keys
        ]]
    finally:
        stack.remove(identity)


def _write_canonical(value):
    if value is None:
        return 'null'
    if value is True:
        return 'true'
    if value is False:
        return 'false'
    if isinstance(value, str):
        return _quote_canonical(value)
    if isinstance(value, list):
        return '[' + ','.join(_write_canonical(item) for item in value) + ']'
    raise ValidationError('internal candidate canonicalization error.')


def _quote_canonical(value):
    _validate_unicode(value, 'candidate string')
    output = ['"']
    for char in value:
        code = ord(char)
        if code == 0x22:
            output.append('\\"')
        elif code == 0x5c:
            output.append('\\\\')
        elif code <= 0x1f:
            output.append(f'\\u00{code:02x}')
        else:
            output.append(char)
    output.append('"')
    return ''.join(output)


def _validate_unicode(value, label):
    if any(0xd800 <= ord(char) <= 0xdfff for char in value):
        raise ValidationError(f'{label} contains a surrogate code point.')


def _assert_string(value, label):
    if not isinstance(value, str) or not value:
        raise ValidationError(f'{label} must be a nonempty string.')
    _validate_unicode(value, label)
    return value


def _assert_uint32(value, label):
    if (isinstance(value, bool) or not isinstance(value, int) or
            value < 0 or value > UINT32_MAX):
        raise ValidationError(
            f'{label} must be an unsigned 32-bit integer.')


def _clone_domain(value):
    # Validation happens through the tagged encoder; deepcopy prevents callers
    # from mutating the retained candidate through the original input object.
    copied = deepcopy(value)
    _encode_tagged(copied)
    return copied


class _FrozenDict(dict):
    def _immutable(self, *_args, **_kwargs):
        raise TypeError('candidate records are immutable')

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __deepcopy__(self, _memo):
        return self


class _FrozenList(list):
    def _immutable(self, *_args, **_kwargs):
        raise TypeError('candidate records are immutable')

    __setitem__ = _immutable
    __delitem__ = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable

    def __deepcopy__(self, _memo):
        return self


def _freeze_domain(value):
    if isinstance(value, dict):
        return _FrozenDict({
            key: _freeze_domain(item) for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return _FrozenList(_freeze_domain(item) for item in value)
    return value
