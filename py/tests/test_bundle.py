import base64
import hashlib
import struct

import pytest

from wlearn.bundle import (
    encode_bundle, decode_bundle, validate_bundle, write_bundle_output,
    DEFAULT_BUNDLE_LIMITS, HEADER_SIZE,
)
from wlearn.errors import BundleError


# Exact published-era fixture bytes from git 96a57f9:
# fixtures/liblinear-classifier.wlrn. Keep immutable as a compatibility lock.
HISTORICAL_PRE_CANONICAL_BUNDLE = base64.b64decode(
    'V0xSTgEAAABjAAAAdAAAAHsiYnVuZGxlVmVyc2lvbiI6MSwicGFyYW1zIjp7IkMiOjEs'
    'ImVwcyI6MC4wMSwic29sdmVyIjowfSwidHlwZUlkIjoid2xlYXJuLmxpYmxpbmVhci5j'
    'bGFzc2lmaWVyQDEifVt7ImlkIjoibW9kZWwiLCJsZW5ndGgiOjEwNywib2Zmc2V0Ijow'
    'LCJzaGEyNTYiOiJhMDdiZDAwNDI5M2Y5OTUxZjY2MjI5OWUyMDU5ZTdhYThhNDNmNWI1'
    'ZWQ4ZGMyNDIwN2U4MGU1YzUyNzg5ZTA1In1dc29sdmVyX3R5cGUgTDJSX0xSCm5yX2Ns'
    'YXNzIDIKbGFiZWwgMCAxCm5yX2ZlYXR1cmUgMgpiaWFzIC0xCncKLTAuODczNjI4NzEw'
    'MTc2MTExOTkgCi0wLjk1NTUzNzYxNTE4MDUyMDA1IAo='
)


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
    assert m['requires'] == []
    assert len(toc) == 2
    assert m['artifacts'] == [
        {key: entry[key] for key in ('id', 'length', 'sha256', 'mediaType')}
        for entry in toc
    ]
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


def test_nested_bundle_derives_requirements():
    nested = encode_bundle({
        'typeId': 'wlearn.child@1',
        'requires': ['wlearn.dependency@1'],
    }, [])
    bundle = encode_bundle({'typeId': 'wlearn.parent@1'}, [{
        'id': 'child',
        'mediaType': 'application/x-wlearn-bundle',
        'data': nested,
    }])
    manifest, _, _ = decode_bundle(bundle)
    assert manifest['requires'] == [
        'wlearn.child@1', 'wlearn.dependency@1']
    validate_bundle(bundle, allow_legacy_manifest=False)


def test_writer_rejects_legacy_nested_bundle():
    legacy = _raw_bundle({
        'typeId': 'wlearn.legacy-child@1', 'bundleVersion': 1,
    })
    with pytest.raises(BundleError, match='manifest.artifacts is required'):
        encode_bundle({'typeId': 'wlearn.parent@1'}, [{
            'id': 'child',
            'mediaType': 'application/x-wlearn-bundle',
            'data': legacy,
        }])


@pytest.mark.parametrize('manifest,artifacts', [
    ({}, []),
    ({'typeId': 'unversioned'}, []),
    ({'typeId': 'wlearn.test@1', 'params': []}, []),
    ({'typeId': 'wlearn.test@1', 'requires': 'bad'}, []),
    ({'typeId': 'wlearn.test@1'}, [None]),
])
def test_encode_rejects_malformed_inputs(manifest, artifacts):
    with pytest.raises(BundleError):
        encode_bundle(manifest, artifacts)


def test_encode_rejects_duplicate_artifact_ids():
    artifacts = [
        {'id': 'same', 'data': b''},
        {'id': 'same', 'data': b''},
    ]
    with pytest.raises(BundleError, match='Duplicate artifact'):
        encode_bundle({'typeId': 'wlearn.test@1'}, artifacts)


def test_encode_rejects_nonportable_json_values():
    with pytest.raises(BundleError, match='finite'):
        encode_bundle({
            'typeId': 'wlearn.test@1', 'metadata': {'value': float('nan')}
        }, [])
    with pytest.raises(BundleError, match='safe range'):
        encode_bundle({
            'typeId': 'wlearn.test@1',
            'metadata': {'value': 9007199254740992},
        }, [])
    with pytest.raises(BundleError, match='safe range'):
        encode_bundle({
            'typeId': 'wlearn.test@1',
            'metadata': {'value': 9007199254740992.0},
        }, [])
    with pytest.raises(BundleError, match='portable JSON'):
        encode_bundle({
            'typeId': 'wlearn.test@1', 'metadata': {'value': (1, 2)}
        }, [])
    circular = {}
    circular['self'] = circular
    with pytest.raises(BundleError, match='circular'):
        encode_bundle({'typeId': 'wlearn.test@1', 'metadata': circular}, [])


def test_encode_enforces_configurable_limits():
    with pytest.raises(BundleError, match='Too many artifacts'):
        encode_bundle(
            {'typeId': 'wlearn.test@1'},
            [{'id': 'a', 'data': b''}],
            limits={'max_artifacts': 0},
        )
    with pytest.raises(BundleError, match='maximum size'):
        encode_bundle(
            {'typeId': 'wlearn.test@1'},
            [{'id': 'a', 'data': b'12'}],
            limits={'max_artifact_bytes': 1},
        )
    child = encode_bundle({'typeId': 'wlearn.child@1'}, [])
    nested_artifact = [{
        'id': 'child', 'data': child,
        'mediaType': 'application/x-wlearn-bundle',
    }]
    with pytest.raises(BundleError, match='depth'):
        encode_bundle(
            {'typeId': 'wlearn.parent@1'}, nested_artifact,
            limits={'max_nesting_depth': 0},
        )
    with pytest.raises(BundleError, match='Decoded nested bundle bytes'):
        encode_bundle(
            {'typeId': 'wlearn.parent@1'}, nested_artifact,
            limits={'max_decoded_bytes': len(child)},
        )
    assert DEFAULT_BUNDLE_LIMITS['max_artifacts'] == 10000


def test_validate_passes():
    manifest = {'typeId': 'test.val@1'}
    arts = [{'id': 'model', 'data': b'hello world'}]
    bundle = encode_bundle(manifest, arts)
    m, toc, blobs = validate_bundle(bundle)
    assert m['typeId'] == 'test.val@1'
    expected_hash = hashlib.sha256(b'hello world').hexdigest()
    assert toc[0]['sha256'] == expected_hash


def test_path_input_and_output(tmp_path):
    manifest = {'typeId': 'test.path@1'}
    bundle = encode_bundle(manifest, [{'id': 'model', 'data': b'abc'}])
    path = tmp_path / 'model.wlrn'

    returned = write_bundle_output(bundle, path)
    assert returned == bundle
    assert path.read_bytes() == bundle

    m1, toc1, blobs1 = decode_bundle(path)
    m2, toc2, blobs2 = decode_bundle(str(path))

    assert m1['typeId'] == 'test.path@1'
    assert m2 == m1
    assert toc2 == toc1
    assert bytes(blobs1[toc1[0]['offset']:toc1[0]['offset'] + toc1[0]['length']]) == b'abc'
    assert bytes(blobs2[toc2[0]['offset']:toc2[0]['offset'] + toc2[0]['length']]) == b'abc'


def test_validate_corrupted_blob():
    manifest = {'typeId': 'test.corrupt@1'}
    arts = [{'id': 'model', 'data': b'hello world'}]
    canonical = encode_bundle(manifest, arts)
    for allow_legacy_manifest in (True, False):
        bundle = bytearray(canonical)
        bundle[-1] ^= 0xFF
        with pytest.raises(BundleError, match='SHA-256 mismatch'):
            validate_bundle(
                bytes(bundle),
                allow_legacy_manifest=allow_legacy_manifest,
            )


def test_validate_recurses_into_nested_bundle_hashes():
    child = bytearray(encode_bundle(
        {'typeId': 'wlearn.child@1'},
        [{'id': 'data', 'data': b'abc'}],
    ))
    child[-1] ^= 0xff
    child = bytes(child)
    child_hash = hashlib.sha256(child).hexdigest()
    toc = [{
        'id': 'child', 'offset': 0, 'length': len(child),
        'sha256': child_hash,
        'mediaType': 'application/x-wlearn-bundle',
    }]
    manifest = {
        'typeId': 'wlearn.parent@1', 'bundleVersion': 1,
        'requires': ['wlearn.child@1'], 'params': {},
        'artifacts': [{
            'id': 'child', 'length': len(child), 'sha256': child_hash,
            'mediaType': 'application/x-wlearn-bundle',
        }],
    }
    with pytest.raises(BundleError, match='SHA-256 mismatch for "data"'):
        validate_bundle(_raw_bundle(manifest, toc, child))


def test_validate_enforces_nested_depth_and_decoded_byte_budgets():
    child = encode_bundle({'typeId': 'wlearn.child@1'}, [])
    parent = encode_bundle({'typeId': 'wlearn.parent@1'}, [{
        'id': 'child', 'data': child,
        'mediaType': 'application/x-wlearn-bundle',
    }])
    with pytest.raises(BundleError, match='depth'):
        validate_bundle(parent, limits={'max_nesting_depth': 0})
    with pytest.raises(BundleError, match='Decoded nested bundle bytes'):
        validate_bundle(parent, limits={
            'max_decoded_bytes': len(parent) + len(child) - 1,
        })


def test_validate_requires_nested_loader_declarations():
    child = encode_bundle({'typeId': 'wlearn.child@1'}, [])
    child_hash = hashlib.sha256(child).hexdigest()
    toc = [{
        'id': 'child', 'offset': 0, 'length': len(child),
        'sha256': child_hash,
        'mediaType': 'application/x-wlearn-bundle',
    }]
    manifest = {
        'typeId': 'wlearn.parent@1', 'bundleVersion': 1,
        'requires': [], 'params': {},
        'artifacts': [{
            'id': 'child', 'length': len(child), 'sha256': child_hash,
            'mediaType': 'application/x-wlearn-bundle',
        }],
    }
    with pytest.raises(BundleError, match='missing nested loader'):
        validate_bundle(
            _raw_bundle(manifest, toc, child),
            allow_legacy_manifest=False,
        )


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
        {'id': 'a', 'offset': 0, 'length': 10, 'sha256': 'a' * 64},
        {'id': 'b', 'offset': 5, 'length': 10, 'sha256': 'b' * 64},
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
        {'id': 'a', 'offset': 0, 'length': 9999, 'sha256': 'a' * 64},
    ], sort_keys=True, separators=(',', ':')).encode()

    header = struct.pack('<4sIII', b'WLRN', 1, len(manifest_json), len(toc_json))
    blob_data = b'\x00' * 5
    bundle = header + manifest_json + toc_json + blob_data

    with pytest.raises(BundleError, match='out of bounds'):
        decode_bundle(bundle)


def _raw_bundle(manifest, toc=None, blob=b'', manifest_bytes=None):
    import json
    if manifest_bytes is None:
        manifest_bytes = json.dumps(manifest).encode()
    toc_bytes = json.dumps(toc or []).encode()
    header = struct.pack(
        '<4sIII', b'WLRN', 1, len(manifest_bytes), len(toc_bytes))
    return header + manifest_bytes + toc_bytes + blob


def test_reject_duplicate_ids_and_malformed_toc_fields():
    manifest = {'typeId': 'wlearn.test@1', 'bundleVersion': 1}
    valid = {'id': 'a', 'offset': 0, 'length': 0, 'sha256': 'a' * 64}
    with pytest.raises(BundleError, match='Duplicate'):
        decode_bundle(_raw_bundle(manifest, [valid, valid]))
    with pytest.raises(BundleError, match='out of bounds'):
        decode_bundle(_raw_bundle(manifest, [{**valid, 'offset': 0.5}]))
    with pytest.raises(BundleError, match='invalid SHA-256'):
        decode_bundle(_raw_bundle(
            manifest, [{**valid, 'sha256': 'not-a-hash'}]))


def test_reject_non_utf8_manifest():
    with pytest.raises(BundleError, match='Invalid manifest JSON'):
        decode_bundle(_raw_bundle(None, manifest_bytes=b'\xff'))


def test_reject_nonstandard_json_constants_and_boolean_version():
    with pytest.raises(BundleError, match='Invalid manifest JSON'):
        decode_bundle(_raw_bundle(
            None,
            manifest_bytes=(
                b'{"typeId":"wlearn.test@1","bundleVersion":1,'
                b'"metadata":{"value":NaN}}'
            ),
        ))
    with pytest.raises(BundleError, match='bundleVersion'):
        decode_bundle(_raw_bundle({
            'typeId': 'wlearn.test@1', 'bundleVersion': True,
        }))


def test_manifest_artifact_declarations_must_match_toc():
    entry = {
        'id': 'a', 'offset': 0, 'length': 0,
        'sha256': 'a' * 64, 'mediaType': 'x/test',
    }
    declaration = {
        'id': 'a', 'length': 1,
        'sha256': entry['sha256'], 'mediaType': entry['mediaType'],
    }
    manifest = {
        'typeId': 'wlearn.test@1', 'bundleVersion': 1,
        'artifacts': [declaration],
    }
    with pytest.raises(BundleError, match='does not match'):
        decode_bundle(_raw_bundle(manifest, [entry]))


def test_legacy_manifest_default_and_strict_modes():
    legacy = _raw_bundle({
        'typeId': 'wlearn.legacy@1', 'bundleVersion': 1})
    assert decode_bundle(legacy)[0]['typeId'] == 'wlearn.legacy@1'
    with pytest.raises(BundleError, match='manifest.artifacts is required'):
        decode_bundle(legacy, allow_legacy_manifest=False)
    artifacts_only = _raw_bundle({
        'typeId': 'wlearn.incomplete@1',
        'bundleVersion': 1,
        'artifacts': [],
    })
    with pytest.raises(BundleError, match='manifest.requires is required'):
        decode_bundle(artifacts_only, allow_legacy_manifest=False)


def test_immutable_precanonical_published_artifact_remains_readable():
    manifest, _, _ = validate_bundle(HISTORICAL_PRE_CANONICAL_BUNDLE)
    assert manifest['typeId'] == 'wlearn.liblinear.classifier@1'
    with pytest.raises(
            BundleError,
            match='must contain exactly|manifest.artifacts is required'):
        validate_bundle(
            HISTORICAL_PRE_CANONICAL_BUNDLE,
            allow_legacy_manifest=False,
        )


def test_strict_mode_rejects_extensions_to_fixed_records():
    sha = hashlib.sha256(b'').hexdigest()
    entry = {
        'id': 'a', 'offset': 0, 'length': 0, 'sha256': sha,
        'mediaType': 'application/octet-stream',
    }
    declaration = {
        'id': 'a', 'length': 0, 'sha256': sha,
        'mediaType': 'application/octet-stream',
    }
    manifest = {
        'typeId': 'wlearn.fixed-records@1', 'bundleVersion': 1,
        'requires': [], 'params': {}, 'artifacts': [declaration],
    }

    toc_extension = _raw_bundle(
        manifest, [{**entry, 'extension': 'legacy'}])
    decode_bundle(toc_extension)
    with pytest.raises(BundleError, match='must contain exactly'):
        decode_bundle(toc_extension, allow_legacy_manifest=False)

    declaration_extension = _raw_bundle({
        **manifest,
        'artifacts': [{**declaration, 'extension': 'legacy'}],
    }, [entry])
    decode_bundle(declaration_extension)
    with pytest.raises(BundleError, match='must contain exactly'):
        decode_bundle(declaration_extension, allow_legacy_manifest=False)


def test_decode_validates_portable_json_domain_for_entire_toc():
    sha = hashlib.sha256(b'').hexdigest()
    entry = {
        'id': 'a', 'offset': 0, 'length': 0, 'sha256': sha,
        'mediaType': 'application/octet-stream',
        'extension': 9007199254740992.0,
    }
    with pytest.raises(BundleError, match='portable safe range'):
        decode_bundle(_raw_bundle({
            'typeId': 'wlearn.toc-json@1', 'bundleVersion': 1,
        }, [entry]))


def test_decode_enforces_configurable_limits_before_parsing():
    bundle = encode_bundle({'typeId': 'wlearn.test@1'}, [])
    with pytest.raises(BundleError, match='Bundle exceeds'):
        decode_bundle(bundle, limits={'max_bundle_bytes': len(bundle) - 1})
    with pytest.raises(BundleError, match='Manifest exceeds'):
        decode_bundle(bundle, limits={'max_manifest_bytes': 1})


def test_decode_checks_path_size_before_reading(tmp_path):
    path = tmp_path / 'oversized.wlrn'
    with path.open('wb') as handle:
        handle.seek(1024 * 1024)
        handle.write(b'\x00')
    with pytest.raises(BundleError, match='Bundle exceeds'):
        decode_bundle(path, limits={'max_bundle_bytes': 1024})


def test_reject_unreferenced_gaps_and_trailing_bytes():
    bundle = encode_bundle({'typeId': 'wlearn.test@1'}, []) + b'\xaa'
    with pytest.raises(BundleError, match='trailing blob bytes'):
        validate_bundle(bundle)

    gap = _raw_bundle(
        {'typeId': 'wlearn.test@1', 'bundleVersion': 1},
        [{
            'id': 'empty', 'offset': 1, 'length': 0,
            'sha256': 'a' * 64,
        }],
        b'\x00',
    )
    with pytest.raises(BundleError, match='blob gap'):
        decode_bundle(gap)


def test_lightgbm_loaded_bytes_are_preserved_without_native_reserialization(tmp_path):
    """Artifact preservation does not require native LightGBM to be installed."""
    import numpy as np
    from wlearn.lightgbm import LGBModel

    model = LGBModel.__new__(LGBModel)
    model._booster = object()
    model._params = {'objective': 'binary', 'numRound': 3}
    model._nr_class = 2
    model._classes = np.array([0, 1], dtype=np.int32)
    model._model_bytes = b'canonical-lightgbm-model-bytes\n'
    model._fitted = True
    model._disposed = False

    output = tmp_path / 'model.wlrn'
    bundle = model.save(output)
    assert output.read_bytes() == bundle
    _, toc, blobs = validate_bundle(bundle)
    entry = toc[0]
    actual = bytes(blobs[entry['offset']:entry['offset'] + entry['length']])
    assert actual == model._model_bytes


def test_direct_pipeline_load_verifies_outer_artifact_hash():
    from wlearn.pipeline import Pipeline
    from wlearn.registry import register

    class FakeModel:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return [0] * len(X)

        def save(self):
            return encode_bundle(
                {'typeId': 'wlearn.fake.pipeline-model@1'},
                [{'id': 'state', 'data': b'fake'}],
            )

        def get_params(self):
            return {}

        def dispose(self):
            pass

    register('wlearn.fake.pipeline-model@1', lambda manifest, toc, blobs: FakeModel())
    pipeline = Pipeline([('model', FakeModel())])
    pipeline.fit([[1], [2]], [0, 1])
    bundle = pipeline.save()
    _, toc, _ = decode_bundle(bundle)
    original_hash = toc[0]['sha256'].encode()
    corrupted = bundle.replace(original_hash, b'0' * 64)
    assert corrupted.count(b'0' * 64) >= 2
    with pytest.raises(BundleError, match='SHA-256 mismatch'):
        Pipeline.load(corrupted)
