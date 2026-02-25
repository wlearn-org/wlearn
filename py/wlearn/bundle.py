import hashlib
import json
import struct

from .errors import BundleError

BUNDLE_MAGIC = b'WLRN'
BUNDLE_VERSION = 1
HEADER_SIZE = 16


def _stable_json(obj):
    """Deterministic JSON: sorted keys, no whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(',', ':')).encode('utf-8')


def encode_bundle(manifest, artifacts):
    """Encode a wlearn bundle.

    Args:
        manifest: dict with at least 'typeId'
        artifacts: list of {'id': str, 'data': bytes, 'mediaType'?: str}

    Returns:
        bytes
    """
    sorted_arts = sorted(artifacts, key=lambda a: a['id'])

    blob_offset = 0
    toc = []
    blobs = []

    for art in sorted_arts:
        data = art['data']
        if isinstance(data, memoryview):
            data = bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        entry = {
            'id': art['id'],
            'offset': blob_offset,
            'length': len(data),
            'sha256': sha,
        }
        if 'mediaType' in art:
            entry['mediaType'] = art['mediaType']
        toc.append(entry)
        blobs.append(data)
        blob_offset += len(data)

    full_manifest = {**manifest, 'bundleVersion': BUNDLE_VERSION}
    manifest_bytes = _stable_json(full_manifest)
    toc_bytes = _stable_json(toc)

    header = struct.pack('<4sIII', BUNDLE_MAGIC, BUNDLE_VERSION,
                         len(manifest_bytes), len(toc_bytes))

    parts = [header, manifest_bytes, toc_bytes]
    parts.extend(blobs)
    return b''.join(parts)


def decode_bundle(data):
    """Decode a wlearn bundle.

    Args:
        data: bytes or memoryview

    Returns:
        (manifest: dict, toc: list, blobs: memoryview)
    """
    if isinstance(data, memoryview):
        buf = data
    else:
        buf = memoryview(data)

    if len(buf) < HEADER_SIZE:
        raise BundleError(
            f'Bundle too small: {len(buf)} bytes (minimum {HEADER_SIZE})')

    magic = bytes(buf[:4])
    if magic != BUNDLE_MAGIC:
        raise BundleError('Invalid bundle magic (expected WLRN)')

    version, manifest_len, toc_len = struct.unpack_from('<III', buf, 4)
    if version != BUNDLE_VERSION:
        raise BundleError(
            f'Unsupported bundle version: {version} (expected {BUNDLE_VERSION})')

    if HEADER_SIZE + manifest_len + toc_len > len(buf):
        raise BundleError(
            f'Bundle truncated: header declares {HEADER_SIZE + manifest_len + toc_len} '
            f'bytes but got {len(buf)}')

    try:
        manifest = json.loads(bytes(buf[HEADER_SIZE:HEADER_SIZE + manifest_len]))
    except json.JSONDecodeError as e:
        raise BundleError(f'Invalid manifest JSON: {e}') from e

    try:
        toc = json.loads(bytes(
            buf[HEADER_SIZE + manifest_len:HEADER_SIZE + manifest_len + toc_len]))
    except json.JSONDecodeError as e:
        raise BundleError(f'Invalid TOC JSON: {e}') from e

    blob_start = HEADER_SIZE + manifest_len + toc_len
    blob_region_len = len(buf) - blob_start

    # validate TOC entries
    for i, entry in enumerate(toc):
        offset = entry['offset']
        length = entry['length']
        if offset < 0 or length < 0 or offset + length > blob_region_len:
            raise BundleError(
                f'TOC entry "{entry["id"]}" out of bounds: '
                f'offset={offset}, length={length}, blobRegion={blob_region_len}')
        for j in range(i + 1, len(toc)):
            other = toc[j]
            a_start, a_end = offset, offset + length
            b_start, b_end = other['offset'], other['offset'] + other['length']
            if a_start < b_end and b_start < a_end and length > 0 and other['length'] > 0:
                raise BundleError(
                    f'TOC entries "{entry["id"]}" and "{other["id"]}" overlap')

    blobs = buf[blob_start:]
    return manifest, toc, blobs


def validate_bundle(data):
    """Decode and verify SHA-256 hashes of all blobs.

    Returns:
        (manifest, toc, blobs)
    """
    manifest, toc, blobs = decode_bundle(data)

    for entry in toc:
        blob = bytes(blobs[entry['offset']:entry['offset'] + entry['length']])
        actual = hashlib.sha256(blob).hexdigest()
        if actual != entry['sha256']:
            raise BundleError(
                f'SHA-256 mismatch for "{entry["id"]}": '
                f'expected {entry["sha256"]}, got {actual}')

    return manifest, toc, blobs
