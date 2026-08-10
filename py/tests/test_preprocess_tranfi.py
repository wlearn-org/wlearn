import hashlib
import math

import numpy as np
import pytest

from wlearn import (
    BackendError, BundleError, CancelledError, DisposedError, NotFittedError,
    ResourceLimitError, Pipeline, ValidationError, encode_bundle, load,
    register, validate_bundle,
)
from wlearn.preprocess import PLAN_MEDIA_TYPE, TYPE_ID, Preprocessor


tranfi = pytest.importorskip('tranfi')
FINAL_TYPE_ID = 'wlearn.test.preprocess-final@1'


class FinalModel:
    def __init__(self, cols=None):
        self.cols = cols
        self.fitted = cols is not None
        self.disposed = False

    def fit(self, X, _y):
        self.cols = X.shape[1]
        self.fitted = True
        return self

    def predict(self, X):
        if not self.fitted or self.disposed:
            raise RuntimeError('final model is unavailable')
        return np.full(X.shape[0], self.cols, dtype=np.float64)

    def save(self, path=None):
        data = encode_bundle({
            'typeId': FINAL_TYPE_ID,
            'params': {'cols': self.cols},
        }, [])
        if path is not None:
            path.write_bytes(data)
        return data

    def get_params(self):
        return {'cols': self.cols}

    def set_params(self, params):
        self.cols = params['cols']
        return self

    def dispose(self):
        self.disposed = True


register(
    FINAL_TYPE_ID,
    lambda manifest, _toc, _blobs: FinalModel(manifest['params']['cols']))


def test_mixed_defaults_fit_transform_and_zero_rows():
    preprocessor = Preprocessor()
    assert preprocessor.get_params() == {
        'impute': {'numeric': 'mean', 'categorical': 'mode'},
        'encode': 'onehot',
        'scale': False,
        'maxCategories': 20,
        'unknownCategory': 'all_zero',
        'allMissing': 'zero',
        'maxOutputColumns': 65536,
        'maxOutputElements': 100000000,
        'policyVersion': 1,
    }
    output = preprocessor.fit_transform([
        [1, 10.5],
        [2, 20.5],
        [1, float('nan')],
        [2, 40.5],
    ])
    assert output.dtype == np.float64
    assert output.shape == (4, 3)
    np.testing.assert_array_equal(output[:, :2], [
        [1, 0], [0, 1], [1, 0], [0, 1],
    ])
    np.testing.assert_allclose(
        output[:, 2], [10.5, 20.5, 23.833333333333336, 40.5])
    assert [field['role'] for field in preprocessor.output_schema] == [
        'onehot', 'onehot', 'value']
    np.testing.assert_array_equal(
        preprocessor.transform([[3, 50.5]]), [[0, 0, 50.5]])
    assert preprocessor.transform(
        np.empty((0, 2), dtype=np.float32)).shape == (0, 3)
    preprocessor.dispose()


def test_impute_false_nan_and_label_sentinel():
    passthrough = Preprocessor(impute=False, encode=False)
    passthrough.fit([[1.5], [2.5]])
    assert math.isnan(passthrough.transform([[float('nan')]])[0, 0])
    passthrough.dispose()

    label = Preprocessor(impute=False, encode='label')
    label.fit([[1], [2], [1]])
    np.testing.assert_array_equal(
        label.transform([[2], [3], [float('nan')]]), [[1], [-1], [-1]])
    label.dispose()


def test_save_direct_and_registry_load_round_trip(tmp_path):
    preprocessor = Preprocessor(scale='standard')
    preprocessor.fit([[1, 10.5], [2, 20.5], [1, 30.5], [2, 40.5]])
    path = tmp_path / 'preprocess.wlrn'
    bundle = preprocessor.save(path)
    assert path.read_bytes() == bundle
    manifest, _, _ = validate_bundle(bundle)
    assert manifest['typeId'] == TYPE_ID
    assert len(manifest['metadata']['tranfi']['recipeSha256']) == 64

    direct = Preprocessor.load(path)
    generic = load(
        bundle, loader_options={TYPE_ID: {}})
    for restored in (direct, generic):
        assert restored.get_params() == preprocessor.get_params()
        np.testing.assert_array_equal(
            restored.transform([[1, 25.5]]),
            preprocessor.transform([[1, 25.5]]))
        restored.dispose()
    preprocessor.dispose()


def test_load_rejects_wrong_type_metadata_disagreement_and_corrupt_plan():
    wrong = encode_bundle(
        {'typeId': 'wlearn.test.other@1'},
        [{'id': 'state', 'data': b'1'}])
    with pytest.raises(ValidationError, match='expected typeId'):
        Preprocessor.load(wrong)

    preprocessor = Preprocessor()
    preprocessor.fit([[1], [2], [1]])
    manifest, toc, blobs = validate_bundle(preprocessor.save())
    plan = bytes(blobs)
    metadata = _deep_copy(manifest['metadata'])
    metadata['outputSchema'][0]['name'] = 'tampered'
    mismatched = encode_bundle({
        'typeId': TYPE_ID,
        'requires': [],
        'params': manifest['params'],
        'metadata': metadata,
    }, [{
        'id': 'plan', 'mediaType': PLAN_MEDIA_TYPE, 'data': plan,
    }])
    with pytest.raises(BundleError, match='schemas do not match'):
        Preprocessor.load(mismatched)

    corrupt_plan = bytearray(plan)
    corrupt_plan[-1] ^= 1
    corrupt = encode_bundle({
        'typeId': TYPE_ID,
        'requires': [],
        'params': manifest['params'],
        'metadata': manifest['metadata'],
    }, [{
        'id': 'plan', 'mediaType': PLAN_MEDIA_TYPE,
        'data': bytes(corrupt_plan),
    }])
    with pytest.raises(BundleError) as caught:
        Preprocessor.load(corrupt)
    assert caught.value.engine == 'tranfi'
    assert caught.value.engineCode == 106
    assert caught.value.__cause__ is not None
    preprocessor.dispose()


def test_load_rejects_boolean_plan_and_policy_versions_before_import(
        monkeypatch):
    preprocessor = Preprocessor()
    preprocessor.fit([[1], [2], [1]])
    manifest, toc, blobs = validate_bundle(preprocessor.save())
    plan = bytes(blobs)

    def unexpected_import(*_args, **_kwargs):
        raise AssertionError('Tranfi plan import must not run')

    with monkeypatch.context() as patcher:
        patcher.setattr(
            tranfi.TransformPlan, 'from_bytes', staticmethod(unexpected_import))
        for field in ('policyVersion', 'abiVersion', 'planFormatVersion'):
            params = _deep_copy(manifest['params'])
            metadata = _deep_copy(manifest['metadata'])
            if field == 'policyVersion':
                params[field] = True
                expected_error = ValidationError
            else:
                metadata['tranfi'][field] = True
                expected_error = BundleError
            mutated = encode_bundle({
                'typeId': TYPE_ID,
                'requires': [],
                'params': params,
                'metadata': metadata,
            }, [{
                'id': 'plan', 'mediaType': toc[0]['mediaType'], 'data': plan,
            }])
            with pytest.raises(expected_error):
                Preprocessor.load(mutated)

    restored = Preprocessor.load(preprocessor.save())
    np.testing.assert_array_equal(restored.transform([[2]]), [[0, 1]])
    restored.dispose()
    preprocessor.dispose()


def test_validation_resources_transactionality_params_and_disposal():
    for value in (0, True, '1'):
        with pytest.raises(ValidationError):
            Preprocessor(impute=value)
    with pytest.raises(ValidationError):
        Preprocessor(max_categories=1)
    with pytest.raises(ResourceLimitError):
        Preprocessor(
            max_categories=3,
            runtime_options={
                'limits': {'max_categories_per_column': 2},
            })

    preprocessor = Preprocessor(encode=False)
    preprocessor.fit([[1.5], [2.5]])
    with pytest.raises(ValidationError, match='finite numbers or NaN'):
        preprocessor.fit([[float('inf')]])
    np.testing.assert_array_equal(preprocessor.transform([[3.5]]), [[3.5]])
    with pytest.raises(ValidationError):
        preprocessor.transform(np.array([[1]], dtype=np.int64))
    with pytest.raises(ValidationError, match='declare its fitted width'):
        preprocessor.transform([])
    preprocessor.set_params({'scale': 'minmax'})
    assert not preprocessor.is_fitted
    with pytest.raises(NotFittedError):
        preprocessor.transform([[1.5]])
    preprocessor.dispose()
    preprocessor.dispose()
    with pytest.raises(DisposedError):
        preprocessor.get_params()


def test_fitted_artifact_plan_bytes_match_engine_identity():
    preprocessor = Preprocessor(
        impute='median', encode='onehot', scale='minmax')
    preprocessor.fit([[1, 4.5], [2, 1.5], [1, float('nan')], [2, 9.5]])
    manifest, toc, blobs = validate_bundle(preprocessor.save())
    entry = toc[0]
    plan_bytes = bytes(blobs[entry['offset']:entry['offset'] + entry['length']])
    assert hashlib.sha256(plan_bytes).hexdigest() == (
        '28224e12cc56847f8cc2876cc96d38c5'
        '61777b4831c3f6e3c13b5c0ff6cd8da2')
    assert manifest['metadata']['tranfi']['recipeSha256'] == (
        '5f151a8c4e97bf7966f8b801dc4d0f5f'
        'fc9445fea1224dd01599c7913cc924d0')
    with tranfi.TransformPlan.from_bytes(plan_bytes) as plan:
        assert plan.recipe_sha256() == (
            manifest['metadata']['tranfi']['recipeSha256'])
        assert plan.schema_json('input') == (
            b'[{"dtype":"float64","id":"x0","name":"x0"},'
            b'{"dtype":"float64","id":"x1","name":"x1"}]')
    preprocessor.dispose()


def test_nested_pipeline_load_forwards_type_keyed_runtime_limits():
    pipeline = Pipeline([
        ('preprocess', Preprocessor(
            impute='median', encode='label', scale='minmax')),
        ('model', FinalModel()),
    ])
    pipeline.fit(
        [[1, 4.5], [2, 1.5], [1, float('nan')], [2, 9.5]],
        np.array([0, 1, 0, 1], dtype=np.float64))
    data = pipeline.save()
    pipeline.dispose()

    restored = load(data, loader_options={
        TYPE_ID: {'limits': {'max_apply_rows': 2}},
    })
    np.testing.assert_array_equal(
        restored.predict([[2, 4.5], [3, 5.5]]), [2, 2])
    with pytest.raises(ResourceLimitError):
        restored.predict([[1, 1], [2, 2], [3, 3]])
    restored.dispose()


def test_cancellation_and_unsupported_runtime_preserve_fitted_plan(monkeypatch):
    token = tranfi.TransformCancelToken()
    preprocessor = Preprocessor(
        encode=False, runtime_options={'cancel_token': token})
    preprocessor.fit([[1.5], [2.5]])
    token.request()
    with pytest.raises(CancelledError) as caught:
        preprocessor.transform([[3.5]])
    assert caught.value.engine == 'tranfi'
    assert caught.value.engineCode == 109
    assert caught.value.__cause__ is not None
    assert preprocessor.is_fitted
    preprocessor.dispose()

    healthy = Preprocessor(encode=False)
    healthy.fit([[1.5], [2.5]])

    def unsupported(*_args, **_kwargs):
        raise tranfi.TranfiTransformError(
            113, 'unsupported floating-point runtime')

    monkeypatch.setattr(tranfi.TransformRecipe, 'from_json', unsupported)
    with pytest.raises(BackendError) as caught:
        healthy.fit([[4.5]])
    assert caught.value.engine == 'tranfi'
    assert caught.value.engineCode == 113
    assert caught.value.__cause__ is not None
    assert healthy.is_fitted
    np.testing.assert_array_equal(healthy.transform([[4.5]]), [[4.5]])
    healthy.dispose()


def _deep_copy(value):
    if isinstance(value, dict):
        return {key: _deep_copy(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_deep_copy(child) for child in value]
    return value
