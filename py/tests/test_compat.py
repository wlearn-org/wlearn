"""Cross-language compatibility tests.

Part 1: Bundle format tests (no model packages needed)
  - Decode JS fixtures, validate hashes, check manifest, re-encode round-trip

Part 2: Prediction tests (requires model packages)
  - Load JS fixtures via registry, predict, compare to sidecar predictions

Part 3: JS -> Py -> JS round-trip
  - Load JS fixture, save from Python, reload, predict, compare
"""

import hashlib
import importlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from wlearn.bundle import decode_bundle, validate_bundle, encode_bundle
from wlearn.registry import get_registry, load as registry_load

EXPECTED_FIXTURES = (
    'ebm-classifier',
    'ebm-regressor',
    'liblinear-classifier',
    'libsvm-classifier',
    'lightgbm-binary',
    'lightgbm-multiclass',
    'lightgbm-regressor',
    'nanoflann-classifier',
    'nanoflann-regressor',
    'pipeline-preprocess-liblinear',
    'pipeline-single',
    'preprocess-tabular',
    'stochtree-classifier',
    'stochtree-regressor',
    'xgboost-binary',
    'xgboost-multiclass',
    'xgboost-regressor',
    'xlearn-classifier',
    'xlearn-regressor',
)

LOADER_MODULES = {
    'xgboost': 'wlearn.xgboost',
    'liblinear': 'wlearn.liblinear',
    'libsvm': 'wlearn.libsvm',
    'nanoflann': 'wlearn.nanoflann',
    'ebm': 'wlearn.ebm',
    'lightgbm': 'wlearn.lightgbm',
    'stochtree': 'wlearn.stochtree',
    'xlearn': 'wlearn.xlearn',
    'preprocess': 'wlearn.preprocess',
}

LOADER_RUNTIME_MODULES = {
    'preprocess': 'tranfi',
}

AVAILABLE_LOADERS = set()
LOADER_IMPORT_ERRORS = {}
for loader_name, module_name in LOADER_MODULES.items():
    try:
        importlib.import_module(module_name)
        runtime_module = LOADER_RUNTIME_MODULES.get(loader_name)
        if runtime_module is not None:
            importlib.import_module(runtime_module)
    except Exception as error:
        LOADER_IMPORT_ERRORS[loader_name] = (
            f'{type(error).__name__}: {error}')
    else:
        prefix = f'wlearn.{loader_name}.'
        if loader_name == 'xlearn':
            prefix = 'wlearn.xlearn.'
        if not any(type_id.startswith(prefix) for type_id in get_registry()):
            continue
        AVAILABLE_LOADERS.add(loader_name)

FIXTURE_LOADER_PREFIXES = (
    ('xgboost-', 'xgboost'),
    ('liblinear-', 'liblinear'),
    ('libsvm-', 'libsvm'),
    ('nanoflann-', 'nanoflann'),
    ('ebm-', 'ebm'),
    ('lightgbm-', 'lightgbm'),
    ('stochtree-', 'stochtree'),
    ('xlearn-', 'xlearn'),
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / 'fixtures'
PY_PRODUCED_DIR = None
PRODUCED_BUNDLES = set()
REQUIRE_ALL_BACKENDS = os.environ.get('WLEARN_INTEROP_REQUIRE_ALL') == '1'


def test_roundtrip_output_is_run_owned():
    configured = os.environ.get('WLEARN_INTEROP_OUTPUT_DIR')
    if configured:
        assert PY_PRODUCED_DIR.resolve() == Path(configured).resolve()
    assert PY_PRODUCED_DIR.resolve() != (FIXTURES_DIR / 'py-produced').resolve()


def assert_prediction_parity(actual, expected, *, atol, message):
    actual_array = np.asarray(actual, dtype=np.float64).reshape(-1)
    expected_array = np.asarray(expected, dtype=np.float64).reshape(-1)
    assert actual_array.size > 0, f'{message}: actual predictions are empty'
    assert expected_array.size > 0, f'{message}: expected predictions are empty'
    assert actual_array.size == expected_array.size, (
        f'{message}: length differs '
        f'({actual_array.size} != {expected_array.size})')
    assert np.isfinite(actual_array).all(), (
        f'{message}: actual predictions contain NaN or infinity')
    assert np.isfinite(expected_array).all(), (
        f'{message}: expected predictions contain NaN or infinity')
    np.testing.assert_allclose(
        actual_array, expected_array, atol=atol, err_msg=message)


def fixture_names():
    return list(EXPECTED_FIXTURES)


def model_fixture_names():
    """Fixtures with executable Python loaders in the current package."""
    return fixture_names()


def _prepare_output_dir(directory):
    assert directory != (FIXTURES_DIR / 'py-produced').resolve(), (
        'Use a fresh WLEARN_INTEROP_OUTPUT_DIR, not the shared legacy directory')
    directory.mkdir(parents=True, exist_ok=True)
    # Never erase another run's output, including when a caller reuses a path.
    assert not any(directory.iterdir()), 'Interop output directory must be empty'
    # Reserve before yielding: concurrent callers can observe the same empty
    # directory, but only one may become its writer. Keep the marker for audit.
    with (directory / '.wlearn-interop-owner').open('x') as marker:
        marker.write(str(os.getpid()))


@pytest.fixture(scope='session', autouse=True)
def roundtrip_output_index(tmp_path_factory):
    """Declare exactly which Python round-trip bundles this environment produced."""
    global PY_PRODUCED_DIR
    if REQUIRE_ALL_BACKENDS:
        missing = sorted(set(LOADER_MODULES) - AVAILABLE_LOADERS)
        assert not missing, (
            'Full interop lane requires every Python backend; missing: '
            f'{", ".join(missing)}; import errors: '
            + '; '.join(
                f'{name}={LOADER_IMPORT_ERRORS[name]}'
                for name in missing if name in LOADER_IMPORT_ERRORS))

    configured = os.environ.get('WLEARN_INTEROP_OUTPUT_DIR')
    PY_PRODUCED_DIR = (Path(configured).resolve() if configured
                       else tmp_path_factory.mktemp('wlearn-interop'))
    _prepare_output_dir(PY_PRODUCED_DIR)
    index_path = PY_PRODUCED_DIR / 'index.json'

    yield

    if REQUIRE_ALL_BACKENDS:
        expected = sorted(model_fixture_names())
    else:
        expected = sorted(
            name for name in model_fixture_names()
            if required_loaders(name) <= AVAILABLE_LOADERS
        )
    skipped = sorted(set(model_fixture_names()) - set(expected))
    index_path.write_text(json.dumps({
        'mode': 'full' if REQUIRE_ALL_BACKENDS else 'minimal',
        'expected': expected,
        'produced': sorted(PRODUCED_BUNDLES),
        'skipped': skipped,
    }, indent=2) + '\n')


def required_loaders(name):
    if name == 'preprocess-tabular':
        return {'preprocess'}
    if name == 'pipeline-preprocess-liblinear':
        return {'preprocess', 'liblinear'}
    if name == 'pipeline-single':
        return {'xgboost'}
    for prefix, loader_name in FIXTURE_LOADER_PREFIXES:
        if name.startswith(prefix):
            return {loader_name}
    return set()


def require_loader_for_fixture(name):
    missing = sorted(required_loaders(name) - AVAILABLE_LOADERS)
    if missing:
        if REQUIRE_ALL_BACKENDS:
            pytest.fail(
                f'{name}: required Python backends are not installed: '
                f'{", ".join(missing)}')
        pytest.skip(
            f'{name}: Python backends are not installed: {", ".join(missing)}')


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
    def test_expected_fixture_corpus_is_complete(self):
        missing = []
        for name in EXPECTED_FIXTURES:
            for suffix in ('.wlrn', '.json'):
                if not (FIXTURES_DIR / f'{name}{suffix}').is_file():
                    missing.append(f'{name}{suffix}')
        assert not missing, f'Missing canonical fixtures: {", ".join(missing)}'

        actual = sorted(path.stem for path in FIXTURES_DIR.glob('*.wlrn'))
        assert actual == sorted(EXPECTED_FIXTURES)

    def test_decode(self, fixture):
        name, wlrn, sidecar = fixture
        manifest, toc, blobs = decode_bundle(wlrn)
        assert manifest is not None
        assert isinstance(toc, list)

    def test_validate_hashes(self, fixture):
        name, wlrn, sidecar = fixture
        validate_bundle(wlrn, allow_legacy_manifest=False)

    def test_manifest_type_id(self, fixture):
        name, wlrn, sidecar = fixture
        manifest, _, _ = decode_bundle(wlrn)
        assert manifest['typeId'] == sidecar['typeId']
        assert manifest.get('requires', []) == sidecar.get('requires', [])

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
        require_loader_for_fixture(name)
        model = registry_load(wlrn)
        try:
            if sidecar.get('operation') == 'transform':
                result = model.transform(sidecar['X'])
                preds = np.asarray(result).reshape(-1)
                assert list(result.shape) == sidecar['outputShape']
            else:
                preds = model.predict(sidecar['X'])
            expected = np.array(sidecar['predictions'], dtype=np.float64)
            assert_prediction_parity(
                preds, expected, atol=1e-5,
                message=f'{name}: predictions differ')
            expected_classes = (
                sidecar.get('classes') or
                sidecar.get('metadata', {}).get('classes'))
            if expected_classes:
                classes = model.classes
                if callable(classes):
                    classes = classes()
                assert list(np.asarray(classes).reshape(-1)) == expected_classes
        finally:
            model.dispose()

    def test_score(self, model_fixture):
        """Score on training data should be reasonable."""
        name, wlrn, sidecar = model_fixture
        require_loader_for_fixture(name)
        model = registry_load(wlrn)
        try:
            if sidecar.get('operation') == 'transform':
                transformed = model.transform(sidecar['X'])
                assert np.isfinite(transformed).all()
                return
            s = model.score(sidecar['X'], sidecar['y'])
            assert np.isfinite(s), f'{name}: score must be finite ({s})'
            assert s > 0.5, f'{name}: score too low ({s})'
        finally:
            model.dispose()


# --- Part 3: JS -> Py -> JS round-trip ---


class TestRoundTrip:
    def test_save_reload_predict(self, model_fixture):
        """Load JS fixture -> save from Python -> reload -> predict -> compare."""
        name, wlrn, sidecar = model_fixture
        require_loader_for_fixture(name)

        model = registry_load(wlrn)
        py_bundle = model.save()

        validate_bundle(py_bundle, allow_legacy_manifest=False)
        (PY_PRODUCED_DIR / f'{name}.wlrn').write_bytes(py_bundle)
        PRODUCED_BUNDLES.add(name)

        model2 = registry_load(py_bundle)
        try:
            if sidecar.get('operation') == 'transform':
                preds1 = np.asarray(model.transform(sidecar['X'])).reshape(-1)
                preds2 = np.asarray(model2.transform(sidecar['X'])).reshape(-1)
            else:
                preds1 = model.predict(sidecar['X'])
                preds2 = model2.predict(sidecar['X'])
            expected = np.array(sidecar['predictions'], dtype=np.float64)

            assert_prediction_parity(
                preds2, expected, atol=1e-5,
                message=f'{name}: round-trip predictions differ')
            assert_prediction_parity(
                preds1, preds2, atol=1e-10,
                message=f'{name}: direct vs round-trip differ')
        finally:
            model.dispose()
            model2.dispose()

    def test_manifest_preserved(self, model_fixture):
        """Manifest typeId and params survive round-trip."""
        name, wlrn, sidecar = model_fixture
        require_loader_for_fixture(name)

        model = registry_load(wlrn)
        py_bundle = model.save()

        manifest_orig, _, _ = decode_bundle(wlrn)
        manifest_new, _, _ = decode_bundle(py_bundle)

        assert manifest_new['typeId'] == manifest_orig['typeId']
        assert manifest_new.get('params') == manifest_orig.get('params')

        if 'metadata' in manifest_orig:
            assert manifest_new.get('metadata', {}) == manifest_orig['metadata']

        model.dispose()

    def test_blob_identical(self, model_fixture):
        """Every artifact blob should be identical after Python save."""
        name, wlrn, sidecar = model_fixture
        require_loader_for_fixture(name)

        _, toc_orig, blobs_orig = decode_bundle(wlrn)

        model = registry_load(wlrn)
        py_bundle = model.save()

        _, toc_new, blobs_new = decode_bundle(py_bundle)

        assert toc_new == toc_orig, f'{name}: TOC differs after round-trip'
        for entry_orig, entry_new in zip(toc_orig, toc_new):
            blob_orig = bytes(
                blobs_orig[entry_orig['offset']:
                           entry_orig['offset'] + entry_orig['length']])
            blob_new = bytes(
                blobs_new[entry_new['offset']:
                          entry_new['offset'] + entry_new['length']])
            assert blob_orig == blob_new, (
                f'{name}: {entry_orig["id"]} blob differs after round-trip '
                f'(orig={len(blob_orig)}, new={len(blob_new)})')

        model.dispose()


def test_prediction_parity_rejects_nonfinite_and_empty_values():
    with pytest.raises(AssertionError, match='empty'):
        assert_prediction_parity([], [], atol=1e-5, message='empty')
    with pytest.raises(AssertionError, match='NaN or infinity'):
        assert_prediction_parity(
            [float('nan')], [float('nan')], atol=1e-5, message='nan')


def test_output_reservation_rejects_overlapping_runs(tmp_path):
    directory = tmp_path / 'shared'
    _prepare_output_dir(directory)
    with pytest.raises((AssertionError, FileExistsError)):
        _prepare_output_dir(directory)


def test_output_reservation_preserves_existing_data(tmp_path):
    directory = tmp_path / 'existing'
    directory.mkdir()
    sentinel = directory / 'sentinel'
    sentinel.write_text('owned by another run')
    with pytest.raises(AssertionError, match='empty'):
        _prepare_output_dir(directory)
    assert sentinel.read_text() == 'owned by another run'
