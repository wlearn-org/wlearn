import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct

from .errors import BundleError

BUNDLE_MAGIC = b'WLRN'
BUNDLE_VERSION = 1
HEADER_SIZE = 16
DEFAULT_BUNDLE_LIMITS = {
    'max_bundle_bytes': 1024 * 1024 * 1024,
    'max_manifest_bytes': 16 * 1024 * 1024,
    'max_toc_bytes': 16 * 1024 * 1024,
    'max_artifacts': 10000,
    'max_artifact_bytes': 1024 * 1024 * 1024,
    'max_nesting_depth': 32,
    'max_decoded_bytes': 4 * 1024 * 1024 * 1024,
}

_TYPE_ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*@[0-9]+$')
_SHA256_RE = re.compile(r'^[0-9a-f]{64}$')
_MAX_SAFE_INTEGER = 9007199254740991
_TOC_ENTRY_KEYS = frozenset(('id', 'offset', 'length', 'sha256', 'mediaType'))
_ARTIFACT_DECLARATION_KEYS = frozenset(
    ('id', 'length', 'sha256', 'mediaType'))


def _limits(limits=None):
    resolved = dict(DEFAULT_BUNDLE_LIMITS)
    if limits is not None:
        if not isinstance(limits, dict):
            raise BundleError('limits must be a dict')
        for name in DEFAULT_BUNDLE_LIMITS:
            if name in limits:
                resolved[name] = limits[name]
    for name, value in resolved.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BundleError(f'Invalid bundle limit {name}: {value}')
    return resolved


def _validate_type_id(type_id, field='manifest.typeId'):
    if not isinstance(type_id, str) or _TYPE_ID_RE.fullmatch(type_id) is None:
        raise BundleError(
            f'{field} must be a versioned typeId such as "wlearn.model@1"')


def _validate_artifact_id(artifact_id, field='artifact.id'):
    if (not isinstance(artifact_id, str) or not artifact_id or
            len(artifact_id) > 1024):
        raise BundleError(
            f'{field} must be a non-empty string of at most 1024 characters')


def _validate_json_value(value, path='value', ancestors=None):
    if ancestors is None:
        ancestors = set()
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        if abs(value) > _MAX_SAFE_INTEGER:
            raise BundleError(
                f'{path} contains an integer outside the portable safe range')
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BundleError(f'{path} must contain only finite numbers')
        if value.is_integer() and abs(value) > _MAX_SAFE_INTEGER:
            raise BundleError(
                f'{path} contains an integer outside the portable safe range')
        return
    if not isinstance(value, (dict, list)):
        raise BundleError(
            f'{path} contains a value that is not representable in portable JSON')
    identity = id(value)
    if identity in ancestors:
        raise BundleError(f'{path} contains a circular reference')
    ancestors.add(identity)
    try:
        if isinstance(value, list):
            for index, child in enumerate(value):
                _validate_json_value(child, f'{path}[{index}]', ancestors)
        else:
            for key, child in value.items():
                if not isinstance(key, str):
                    raise BundleError(f'{path} object keys must be strings')
                _validate_json_value(child, f'{path}.{key}', ancestors)
    finally:
        ancestors.remove(identity)


def _stable_json(obj):
    """Deterministic JSON: sorted keys, no whitespace."""
    _validate_json_value(obj)
    try:
        return json.dumps(
            obj, sort_keys=True, separators=(',', ':'),
            allow_nan=False,
        ).encode('utf-8')
    except (TypeError, ValueError) as exc:
        raise BundleError(f'Value is not valid deterministic JSON: {exc}') from exc


def _reject_json_constant(value):
    raise ValueError(f'invalid JSON constant {value}')


def read_bundle_input(data, max_bytes=None):
    """Return bundle bytes from bytes-like input, str path, or PathLike path."""
    if isinstance(data, (str, os.PathLike)):
        path = Path(data)
        if max_bytes is not None and path.stat().st_size > max_bytes:
            raise BundleError(f'Bundle exceeds maximum size {max_bytes}')
        with path.open('rb') as handle:
            result = handle.read(None if max_bytes is None else max_bytes + 1)
        if max_bytes is not None and len(result) > max_bytes:
            raise BundleError(f'Bundle exceeds maximum size {max_bytes}')
        return result
    return data


def write_bundle_output(data, path=None):
    """Optionally write bundle bytes to a str/PathLike path, then return bytes."""
    if path is not None:
        Path(path).write_bytes(data)
    return data


def encode_bundle(manifest, artifacts, *, limits=None):
    """Encode a canonical wlearn bundle with bounded metadata and blobs."""
    bounds = _limits(limits)
    if not isinstance(manifest, dict):
        raise BundleError('manifest must be a dict')
    _validate_type_id(manifest.get('typeId'))
    if ('params' in manifest and
            (not isinstance(manifest['params'], dict))):
        raise BundleError('manifest.params must be a dict')
    if 'requires' in manifest and not isinstance(manifest['requires'], list):
        raise BundleError('manifest.requires must be a list')
    if not isinstance(artifacts, list):
        raise BundleError('artifacts must be a list')
    if len(artifacts) > bounds['max_artifacts']:
        raise BundleError(
            f'Too many artifacts: {len(artifacts)} '
            f'(maximum {bounds["max_artifacts"]})')

    for art in artifacts:
        if not isinstance(art, dict):
            raise BundleError('artifact must be a dict')
        _validate_artifact_id(art.get('id'))

    sorted_arts = sorted(artifacts, key=lambda art: art['id'])
    derived_requires = set(manifest.get('requires', []))
    for type_id in derived_requires:
        _validate_type_id(type_id, 'manifest.requires entry')

    blob_offset = 0
    toc = []
    blobs = []
    artifact_ids = set()
    nested_validation_state = {'decoded_bytes': 0}

    for art in sorted_arts:
        artifact_id = art['id']
        if artifact_id in artifact_ids:
            raise BundleError(f'Duplicate artifact id "{artifact_id}"')
        artifact_ids.add(artifact_id)

        data = art.get('data')
        if isinstance(data, memoryview):
            data = data.tobytes()
        elif isinstance(data, bytearray):
            data = bytes(data)
        if not isinstance(data, bytes):
            raise BundleError(
                f'Artifact "{artifact_id}" data must be bytes-like')
        if len(data) > bounds['max_artifact_bytes']:
            raise BundleError(
                f'Artifact "{artifact_id}" exceeds maximum size '
                f'{bounds["max_artifact_bytes"]}')

        media_type = art.get('mediaType', 'application/octet-stream')
        if not isinstance(media_type, str) or not media_type:
            raise BundleError(
                f'Artifact "{artifact_id}" mediaType must be a non-empty string')
        sha = hashlib.sha256(data).hexdigest()
        entry = {
            'id': artifact_id,
            'offset': blob_offset,
            'length': len(data),
            'sha256': sha,
            'mediaType': media_type,
        }
        toc.append(entry)
        blobs.append(data)
        blob_offset += len(data)

        if media_type == 'application/x-wlearn-bundle':
            nested_manifest, _, _ = validate_bundle(
                data,
                limits=limits,
                allow_legacy_manifest=False,
                _context={
                    'state': nested_validation_state,
                    'depth': 1,
                },
            )
            derived_requires.add(nested_manifest['typeId'])
            derived_requires.update(nested_manifest.get('requires', []))

    declarations = [
        {key: entry[key] for key in ('id', 'length', 'sha256', 'mediaType')}
        for entry in toc
    ]
    full_manifest = {
        **manifest,
        'typeId': manifest['typeId'],
        'bundleVersion': BUNDLE_VERSION,
        'requires': sorted(derived_requires),
        'artifacts': declarations,
        'params': manifest.get('params', {}),
    }
    _validate_json_value(full_manifest, 'manifest')
    manifest_bytes = _stable_json(full_manifest)
    toc_bytes = _stable_json(toc)

    if len(manifest_bytes) > bounds['max_manifest_bytes']:
        raise BundleError(
            f'Manifest exceeds maximum size {bounds["max_manifest_bytes"]}')
    if len(toc_bytes) > bounds['max_toc_bytes']:
        raise BundleError(f'TOC exceeds maximum size {bounds["max_toc_bytes"]}')
    total_len = HEADER_SIZE + len(manifest_bytes) + len(toc_bytes) + blob_offset
    if total_len > bounds['max_bundle_bytes']:
        raise BundleError(
            f'Bundle exceeds maximum size {bounds["max_bundle_bytes"]}')
    if (nested_validation_state['decoded_bytes'] + total_len >
            bounds['max_decoded_bytes']):
        raise BundleError(
            'Decoded nested bundle bytes exceed maximum '
            f'{bounds["max_decoded_bytes"]}')

    header = struct.pack(
        '<4sIII', BUNDLE_MAGIC, BUNDLE_VERSION,
        len(manifest_bytes), len(toc_bytes),
    )
    return b''.join([header, manifest_bytes, toc_bytes, *blobs])


def decode_bundle(data, *, limits=None, allow_legacy_manifest=True):
    """Decode and structurally validate a wlearn bundle."""
    bounds = _limits(limits)
    data = read_bundle_input(data, bounds['max_bundle_bytes'])
    try:
        buf = data if isinstance(data, memoryview) else memoryview(data)
    except TypeError as exc:
        raise BundleError('Bundle input must be bytes-like or a path') from exc

    if len(buf) > bounds['max_bundle_bytes']:
        raise BundleError(
            f'Bundle exceeds maximum size {bounds["max_bundle_bytes"]}')
    if len(buf) < HEADER_SIZE:
        raise BundleError(
            f'Bundle too small: {len(buf)} bytes (minimum {HEADER_SIZE})')
    if bytes(buf[:4]) != BUNDLE_MAGIC:
        raise BundleError('Invalid bundle magic (expected WLRN)')

    version, manifest_len, toc_len = struct.unpack_from('<III', buf, 4)
    if version != BUNDLE_VERSION:
        raise BundleError(
            f'Unsupported bundle version: {version} (expected {BUNDLE_VERSION})')
    if manifest_len > bounds['max_manifest_bytes']:
        raise BundleError(
            f'Manifest exceeds maximum size {bounds["max_manifest_bytes"]}')
    if toc_len > bounds['max_toc_bytes']:
        raise BundleError(f'TOC exceeds maximum size {bounds["max_toc_bytes"]}')

    metadata_end = HEADER_SIZE + manifest_len + toc_len
    if metadata_end > len(buf):
        raise BundleError(
            f'Bundle truncated: header declares {metadata_end} bytes '
            f'but got {len(buf)}')

    try:
        manifest = json.loads(
            bytes(buf[HEADER_SIZE:HEADER_SIZE + manifest_len]).decode('utf-8'),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise BundleError(f'Invalid manifest JSON: {exc}') from exc
    try:
        toc = json.loads(
            bytes(buf[HEADER_SIZE + manifest_len:metadata_end]).decode('utf-8'),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise BundleError(f'Invalid TOC JSON: {exc}') from exc

    if not isinstance(manifest, dict):
        raise BundleError('Manifest JSON must be an object')
    _validate_type_id(manifest.get('typeId'))
    if (isinstance(manifest.get('bundleVersion'), bool) or
            manifest.get('bundleVersion') != BUNDLE_VERSION):
        raise BundleError(
            f'Manifest bundleVersion must be {BUNDLE_VERSION}')
    requires = manifest.get('requires')
    if requires is not None:
        if not isinstance(requires, list):
            raise BundleError('manifest.requires must be a list')
        seen_requires = set()
        for type_id in requires:
            _validate_type_id(type_id, 'manifest.requires entry')
            if type_id in seen_requires:
                raise BundleError(
                    f'Duplicate manifest requirement "{type_id}"')
            seen_requires.add(type_id)
    if 'params' in manifest and not isinstance(manifest['params'], dict):
        raise BundleError('manifest.params must be a dict')
    seed = manifest.get('seed')
    if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int) or
            abs(seed) > _MAX_SAFE_INTEGER):
        raise BundleError('manifest.seed must be a safe integer')
    if 'metadata' in manifest and not isinstance(manifest['metadata'], dict):
        raise BundleError('manifest.metadata must be a dict')
    _validate_json_value(manifest, 'manifest')
    if not isinstance(toc, list):
        raise BundleError('TOC JSON must be an array')
    _validate_json_value(toc, 'toc')
    if len(toc) > bounds['max_artifacts']:
        raise BundleError(
            f'Too many artifacts: {len(toc)} '
            f'(maximum {bounds["max_artifacts"]})')

    blob_region_len = len(buf) - metadata_end
    artifact_ids = set()
    for index, entry in enumerate(toc):
        if not isinstance(entry, dict):
            raise BundleError(f'TOC entry {index} must be an object')
        if not allow_legacy_manifest and set(entry) != _TOC_ENTRY_KEYS:
            raise BundleError(
                f'TOC entry {index} must contain exactly: '
                f'{", ".join(sorted(_TOC_ENTRY_KEYS))}')
        artifact_id = entry.get('id')
        _validate_artifact_id(artifact_id, f'TOC entry {index}.id')
        if artifact_id in artifact_ids:
            raise BundleError(f'Duplicate artifact id "{artifact_id}"')
        artifact_ids.add(artifact_id)
        offset = entry.get('offset')
        length = entry.get('length')
        valid_bounds = (
            isinstance(offset, int) and not isinstance(offset, bool) and
            isinstance(length, int) and not isinstance(length, bool) and
            offset >= 0 and length >= 0 and
            length <= bounds['max_artifact_bytes'] and
            offset + length <= blob_region_len
        )
        if not valid_bounds:
            raise BundleError(
                f'TOC entry "{artifact_id}" out of bounds: '
                f'offset={offset}, length={length}, '
                f'blobRegion={blob_region_len}')
        sha = entry.get('sha256')
        if not isinstance(sha, str) or _SHA256_RE.fullmatch(sha) is None:
            raise BundleError(
                f'TOC entry "{artifact_id}" has invalid SHA-256')
        media_type = entry.get('mediaType')
        if media_type is not None and (
                not isinstance(media_type, str) or not media_type):
            raise BundleError(
                f'TOC entry "{artifact_id}" has invalid mediaType')

    by_offset = sorted(toc, key=lambda entry: (entry['offset'], entry['length']))
    covered_bytes = 0
    for entry in by_offset:
        if entry['offset'] < covered_bytes:
            raise BundleError(
                f'TOC entry "{entry["id"]}" overlaps a previous artifact')
        if entry['offset'] > covered_bytes:
            raise BundleError(
                f'Unreferenced blob gap before artifact "{entry["id"]}"')
        covered_bytes = entry['offset'] + entry['length']
    if covered_bytes != blob_region_len:
        raise BundleError(
            f'Unreferenced trailing blob bytes: '
            f'{blob_region_len - covered_bytes}')

    declarations = manifest.get('artifacts')
    if declarations is not None:
        if not isinstance(declarations, list):
            raise BundleError('manifest.artifacts must be a list')
        if len(declarations) != len(toc):
            raise BundleError('manifest.artifacts length does not match TOC')
        keys = ('id', 'length', 'sha256', 'mediaType')
        for declaration, entry in zip(declarations, toc):
            if (not allow_legacy_manifest and
                    isinstance(declaration, dict) and
                    set(declaration) != _ARTIFACT_DECLARATION_KEYS):
                raise BundleError(
                    'manifest artifact declaration must contain exactly: '
                    f'{", ".join(sorted(_ARTIFACT_DECLARATION_KEYS))}')
            if (not isinstance(declaration, dict) or
                    any(declaration.get(key) != entry.get(key) for key in keys)):
                raise BundleError(
                    'manifest artifact declaration does not match TOC entry '
                    f'"{entry["id"]}"')
    elif not allow_legacy_manifest:
        raise BundleError('manifest.artifacts is required')

    if not allow_legacy_manifest:
        if 'requires' not in manifest:
            raise BundleError('manifest.requires is required')
        if 'params' not in manifest:
            raise BundleError('manifest.params is required')
        for index, entry in enumerate(toc):
            if 'mediaType' not in entry:
                raise BundleError(
                    f'TOC entry "{entry["id"]}" mediaType is required')
            if index > 0 and toc[index - 1]['id'] >= entry['id']:
                raise BundleError(
                    'TOC entries must be ordered by unique artifact id')

    return manifest, toc, buf[metadata_end:]


def validate_bundle(data, *, limits=None, allow_legacy_manifest=True,
                    _context=None):
    """Decode a bundle and verify SHA-256 hashes of all blobs."""
    manifest, toc, blobs = decode_bundle(
        data, limits=limits, allow_legacy_manifest=allow_legacy_manifest)
    bounds = _limits(limits)
    if _context is None:
        state = {'decoded_bytes': 0}
        depth = 0
    else:
        state = _context['state']
        depth = _context['depth']
    if depth > bounds['max_nesting_depth']:
        raise BundleError(
            f'Nested bundle depth {depth} exceeds maximum '
            f'{bounds["max_nesting_depth"]}')
    input_length = HEADER_SIZE + len(blobs)
    # Include metadata bytes, which are omitted from the blobs memoryview length.
    if isinstance(data, (str, os.PathLike)):
        input_length = Path(data).stat().st_size
    else:
        input_length = len(data)
    state['decoded_bytes'] += input_length
    if state['decoded_bytes'] > bounds['max_decoded_bytes']:
        raise BundleError(
            f'Decoded nested bundle bytes exceed maximum '
            f'{bounds["max_decoded_bytes"]}')

    nested_requires = set()
    for entry in toc:
        blob = bytes(blobs[entry['offset']:entry['offset'] + entry['length']])
        actual = hashlib.sha256(blob).hexdigest()
        if actual != entry['sha256']:
            raise BundleError(
                f'SHA-256 mismatch for "{entry["id"]}": '
                f'expected {entry["sha256"]}, got {actual}')
        if entry.get('mediaType') == 'application/x-wlearn-bundle':
            nested_manifest, _, _ = validate_bundle(
                blob,
                limits=limits,
                allow_legacy_manifest=allow_legacy_manifest,
                _context={'state': state, 'depth': depth + 1},
            )
            nested_requires.add(nested_manifest['typeId'])
            nested_requires.update(nested_manifest.get('requires', []))

    if 'requires' in manifest:
        missing = nested_requires.difference(manifest['requires'])
        if missing:
            type_id = sorted(missing)[0]
            raise BundleError(
                f'manifest.requires is missing nested loader "{type_id}"')
    return manifest, toc, blobs
