import hashlib
import struct

import pytest

from wlearn.bundle import encode_bundle, decode_bundle, validate_bundle, HEADER_SIZE
from wlearn.errors import BundleError


def test_round_trip():
    manifest = {'typeId': 'test.model@1', 'params': {'C': 1.0}}
    artifacts = [
        {'id': 'weights', 'data': b'\x01\x02\x03'},
        {'id': 'bias', 'data': b'\x04\x05'},
    ]
    bundle = encode_bundle(manifest, artifacts)
    m, toc, blobs = decode_bundle(bundle)

    assert m['typeId'] == 'test.model@1'
    assert m['bundleVersion'] == 1
    assert m['params'] == {'C': 1.0}
    assert len(toc) == 2
    # artifacts sorted by id: bias before weights
    assert toc[0]['id'] == 'bias'
    assert toc[1]['id'] == 'weights'
    assert bytes(blobs[toc[0]['offset']:toc[0]['offset'] + toc[0]['length']]) == b'\x04\x05'
    assert bytes(blobs[toc[1]['offset']:toc[1]['offset'] + toc[1]['length']]) == b'\x01\x02\x03'


def test_empty_artifacts():
    manifest = {'typeId': 'test.empty@1'}
    bundle = encode_bundle(manifest, [])
    m, toc, blobs = decode_bundle(bundle)
    assert m['typeId'] == 'test.empty@1'
    assert len(toc) == 0


def test_header_magic_and_version():
    manifest = {'typeId': 'test.hdr@1'}
    bundle = encode_bundle(manifest, [{'id': 'x', 'data': b'\x00'}])
    assert bundle[:4] == b'WLRN'
    version = struct.unpack_from('<I', bundle, 4)[0]
    assert version == 1


def test_determinism():
    manifest = {'typeId': 'test.det@1', 'b': 2, 'a': 1}
    arts = [{'id': 'z', 'data': b'\x01'}, {'id': 'a', 'data': b'\x02'}]
    b1 = encode_bundle(manifest, arts)
    b2 = encode_bundle(manifest, arts)
    assert b1 == b2


def test_validate_passes():
    manifest = {'typeId': 'test.val@1'}
    arts = [{'id': 'model', 'data': b'hello world'}]
    bundle = encode_bundle(manifest, arts)
    m, toc, blobs = validate_bundle(bundle)
    assert m['typeId'] == 'test.val@1'
    expected_hash = hashlib.sha256(b'hello world').hexdigest()
    assert toc[0]['sha256'] == expected_hash


def test_validate_corrupted_blob():
    manifest = {'typeId': 'test.corrupt@1'}
    arts = [{'id': 'model', 'data': b'hello world'}]
    bundle = bytearray(encode_bundle(manifest, arts))
    # corrupt last byte of blob
    bundle[-1] ^= 0xFF
    with pytest.raises(BundleError, match='SHA-256 mismatch'):
        validate_bundle(bytes(bundle))


def test_reject_truncated_header():
    with pytest.raises(BundleError, match='too small'):
        decode_bundle(b'WLR')


def test_reject_bad_magic():
    buf = bytearray(HEADER_SIZE)
    buf[:4] = b'NOPE'
    with pytest.raises(BundleError, match='Invalid bundle magic'):
        decode_bundle(bytes(buf))


def test_reject_bad_version():
    buf = bytearray(HEADER_SIZE)
    buf[:4] = b'WLRN'
    struct.pack_into('<I', buf, 4, 99)
    struct.pack_into('<I', buf, 8, 0)
    struct.pack_into('<I', buf, 12, 0)
    with pytest.raises(BundleError, match='Unsupported bundle version'):
        decode_bundle(bytes(buf))


def test_reject_truncated_manifest():
    buf = bytearray(HEADER_SIZE)
    buf[:4] = b'WLRN'
    struct.pack_into('<I', buf, 4, 1)
    struct.pack_into('<I', buf, 8, 9999)  # manifest way too long
    struct.pack_into('<I', buf, 12, 0)
    with pytest.raises(BundleError, match='truncated'):
        decode_bundle(bytes(buf))


def test_reject_overlapping_toc():
    """Build a bundle with manually overlapping TOC entries."""
    import json
    manifest_json = json.dumps(
        {'typeId': 'test.overlap@1', 'bundleVersion': 1},
        sort_keys=True, separators=(',', ':')).encode()
    toc_json = json.dumps([
        {'id': 'a', 'offset': 0, 'length': 10, 'sha256': 'x'},
        {'id': 'b', 'offset': 5, 'length': 10, 'sha256': 'y'},
    ], sort_keys=True, separators=(',', ':')).encode()

    header = struct.pack('<4sIII', b'WLRN', 1, len(manifest_json), len(toc_json))
    blob_data = b'\x00' * 20
    bundle = header + manifest_json + toc_json + blob_data

    with pytest.raises(BundleError, match='overlap'):
        decode_bundle(bundle)


def test_reject_toc_out_of_bounds():
    """Build a bundle with a TOC entry pointing past the blob region."""
    import json
    manifest_json = json.dumps(
        {'typeId': 'test.oob@1', 'bundleVersion': 1},
        sort_keys=True, separators=(',', ':')).encode()
    toc_json = json.dumps([
        {'id': 'a', 'offset': 0, 'length': 9999, 'sha256': 'x'},
    ], sort_keys=True, separators=(',', ':')).encode()

    header = struct.pack('<4sIII', b'WLRN', 1, len(manifest_json), len(toc_json))
    blob_data = b'\x00' * 5
    bundle = header + manifest_json + toc_json + blob_data

    with pytest.raises(BundleError, match='out of bounds'):
        decode_bundle(bundle)
