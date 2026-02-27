"""Cross-language compatibility tests.

Part 1: Bundle format tests (no model packages needed)
  - Decode JS fixtures, validate hashes, check manifest, re-encode round-trip

Part 2: Prediction tests (requires model packages)
  - Load JS fixtures via registry, predict, compare to sidecar predictions

Part 3: JS -> Py -> JS round-trip
  - Load JS fixture, save from Python, reload, predict, compare
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from wlearn.bundle import decode_bundle, validate_bundle, encode_bundle
from wlearn.registry import load as registry_load

# Import model wrappers to register loaders
import wlearn.xgboost    # noqa: F401
import wlearn.liblinear  # noqa: F401
import wlearn.libsvm     # noqa: F401
import wlearn.nanoflann  # noqa: F401
import wlearn.ebm        # noqa: F401
import wlearn.lightgbm   # noqa: F401
import wlearn.stochtree  # noqa: F401

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / 'fixtures'


def fixture_names():
    if not FIXTURES_DIR.exists():
        return []
    return sorted(p.stem for p in FIXTURES_DIR.glob('*.wlrn'))


def model_fixture_names():
    """Fixtures that have a direct model loader (not pipeline)."""
    return [n for n in fixture_names() if not n.startswith('pipeline')]


@pytest.fixture(params=fixture_names(), ids=fixture_names())
def fixture(request):
    name = request.param
    wlrn = (FIXTURES_DIR / f'{name}.wlrn').read_bytes()
    sidecar = json.loads((FIXTURES_DIR / f'{name}.json').read_text())
    return name, wlrn, sidecar


@pytest.fixture(params=model_fixture_names(), ids=model_fixture_names())
def model_fixture(request):
    name = request.param
    wlrn = (FIXTURES_DIR / f'{name}.wlrn').read_bytes()
    sidecar = json.loads((FIXTURES_DIR / f'{name}.json').read_text())
    return name, wlrn, sidecar


# --- Part 1: Bundle format tests ---


class TestBundleFormat:
    def test_decode(self, fixture):
        name, wlrn, sidecar = fixture
        manifest, toc, blobs = decode_bundle(wlrn)
        assert manifest is not None
        assert isinstance(toc, list)

    def test_validate_hashes(self, fixture):
        name, wlrn, sidecar = fixture
        validate_bundle(wlrn)

    def test_manifest_type_id(self, fixture):
        name, wlrn, sidecar = fixture
        manifest, _, _ = decode_bundle(wlrn)
        assert manifest['typeId'] == sidecar['typeId']

    def test_manifest_params(self, fixture):
        name, wlrn, sidecar = fixture
        manifest, _, _ = decode_bundle(wlrn)
        if manifest['typeId'] == 'wlearn.pipeline@1':
            return
        if 'params' in sidecar and sidecar['params']:
            assert manifest.get('params') == sidecar['params']

    def test_toc_entries(self, fixture):
        name, wlrn, sidecar = fixture
        _, toc, _ = decode_bundle(wlrn)
        assert len(toc) == len(sidecar['toc'])
        for actual, expected in zip(toc, sidecar['toc']):
            assert actual['id'] == expected['id']
            assert actual['length'] == expected['length']
            assert actual['sha256'] == expected['sha256']

    def test_blob_integrity(self, fixture):
        name, wlrn, sidecar = fixture
        _, toc, blobs = decode_bundle(wlrn)
        for entry in toc:
            blob = bytes(blobs[entry['offset']:entry['offset'] + entry['length']])
            actual_hash = hashlib.sha256(blob).hexdigest()
            assert actual_hash == entry['sha256']

    def test_reencode_round_trip(self, fixture):
        name, wlrn, sidecar = fixture
        manifest, toc, blobs = decode_bundle(wlrn)

        artifacts = []
        for entry in toc:
            blob = bytes(blobs[entry['offset']:entry['offset'] + entry['length']])
            art = {'id': entry['id'], 'data': blob}
            if 'mediaType' in entry:
                art['mediaType'] = entry['mediaType']
            artifacts.append(art)

        manifest_clean = {k: v for k, v in manifest.items() if k != 'bundleVersion'}
        reencoded = encode_bundle(manifest_clean, artifacts)
        m2, toc2, blobs2 = decode_bundle(reencoded)

        assert m2['typeId'] == manifest['typeId']
        assert m2['bundleVersion'] == manifest['bundleVersion']
        assert len(toc2) == len(toc)

        for orig, re in zip(toc, toc2):
            assert orig['id'] == re['id']
            assert orig['length'] == re['length']
            assert orig['sha256'] == re['sha256']

        for entry in toc2:
            orig_blob = bytes(blobs[entry['offset']:entry['offset'] + entry['length']])
            re_blob = bytes(blobs2[entry['offset']:entry['offset'] + entry['length']])
            assert orig_blob == re_blob


# --- Part 2: Prediction tests ---


class TestPredictions:
    def test_load_and_predict(self, model_fixture):
        """Load JS fixture via registry, predict on X, match sidecar."""
        name, wlrn, sidecar = model_fixture
        model = registry_load(wlrn)

        preds = model.predict(sidecar['X'])
        expected = np.array(sidecar['predictions'], dtype=np.float64)

        assert len(preds) == len(expected)
        np.testing.assert_allclose(preds, expected, atol=1e-5,
                                   err_msg=f'{name}: predictions differ')
        model.dispose()

    def test_score(self, model_fixture):
        """Score on training data should be reasonable."""
        name, wlrn, sidecar = model_fixture
        model = registry_load(wlrn)
        s = model.score(sidecar['X'], sidecar['y'])
        assert s > 0.5, f'{name}: score too low ({s})'
        model.dispose()


# --- Part 3: JS -> Py -> JS round-trip ---


class TestRoundTrip:
    def test_save_reload_predict(self, model_fixture):
        """Load JS fixture -> save from Python -> reload -> predict -> compare."""
        name, wlrn, sidecar = model_fixture

        model = registry_load(wlrn)
        py_bundle = model.save()

        validate_bundle(py_bundle)

        model2 = registry_load(py_bundle)

        preds1 = model.predict(sidecar['X'])
        preds2 = model2.predict(sidecar['X'])
        expected = np.array(sidecar['predictions'], dtype=np.float64)

        np.testing.assert_allclose(preds2, expected, atol=1e-5,
                                   err_msg=f'{name}: round-trip predictions differ')
        np.testing.assert_allclose(preds1, preds2, atol=1e-10,
                                   err_msg=f'{name}: direct vs round-trip differ')

        model.dispose()
        model2.dispose()

    def test_manifest_preserved(self, model_fixture):
        """Manifest typeId and params survive round-trip."""
        name, wlrn, sidecar = model_fixture

        model = registry_load(wlrn)
        py_bundle = model.save()

        manifest_orig, _, _ = decode_bundle(wlrn)
        manifest_new, _, _ = decode_bundle(py_bundle)

        assert manifest_new['typeId'] == manifest_orig['typeId']
        assert manifest_new.get('params') == manifest_orig.get('params')

        if 'metadata' in manifest_orig:
            assert manifest_new.get('metadata', {}) == manifest_orig['metadata']

        model.dispose()

    # LightGBM text model format is not byte-stable across save APIs:
    # C API (WASM) vs Python booster.save_model() produce slightly different
    # output (extra metadata fields). Predictions still match.
    BLOB_IDENTICAL_SKIP = frozenset([
        'lightgbm-binary', 'lightgbm-multiclass', 'lightgbm-regressor',
    ])

    def test_blob_identical(self, model_fixture):
        """Model blob bytes should be identical after Py save."""
        name, wlrn, sidecar = model_fixture

        if name in self.BLOB_IDENTICAL_SKIP:
            pytest.skip(f'{name}: text format not byte-stable across save APIs')

        _, toc_orig, blobs_orig = decode_bundle(wlrn)

        model = registry_load(wlrn)
        py_bundle = model.save()

        _, toc_new, blobs_new = decode_bundle(py_bundle)

        entry_orig = next(e for e in toc_orig if e['id'] == 'model')
        entry_new = next(e for e in toc_new if e['id'] == 'model')

        blob_orig = bytes(blobs_orig[entry_orig['offset']:entry_orig['offset'] + entry_orig['length']])
        blob_new = bytes(blobs_new[entry_new['offset']:entry_new['offset'] + entry_new['length']])

        assert blob_orig == blob_new, (
            f'{name}: model blob differs after round-trip '
            f'(orig={len(blob_orig)}, new={len(blob_new)})')

        model.dispose()
