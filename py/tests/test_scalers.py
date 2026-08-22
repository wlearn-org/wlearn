"""Tests for Python StandardScaler and MinMaxScaler."""

import json

import numpy as np
import pytest

from wlearn.scalers import StandardScaler, MinMaxScaler
from wlearn.bundle import decode_bundle, encode_bundle
from wlearn.registry import load as registry_load
from wlearn.errors import NotFittedError, DisposedError, ValidationError


def _artifact_bundle(type_id, artifact):
    return encode_bundle(
        {'typeId': type_id, 'params': {}},
        [{'id': 'params', 'data': json.dumps(artifact).encode(),
          'mediaType': 'application/json'}])


class TestStandardScaler:
    def test_fit_transform_zero_mean_unit_var(self):
        X = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=np.float64)
        scaler = StandardScaler()
        result = scaler.fit_transform(X)
        assert result.shape == X.shape
        # Column means should be ~0
        assert np.allclose(result.mean(axis=0), 0, atol=1e-10)
        # Column stds should be ~1 (population std, sklearn-compatible)
        assert np.allclose(result.std(axis=0, ddof=0), 1, atol=1e-10)

    def test_fit_then_transform(self):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        scaler = StandardScaler()
        scaler.fit(X)
        result = scaler.transform(X)
        assert result.shape == X.shape
        assert np.allclose(result.mean(axis=0), 0, atol=1e-10)

    def test_save_load_roundtrip(self):
        X = np.array([[1, 10], [2, 20], [3, 30]], dtype=np.float64)
        scaler = StandardScaler()
        scaler.fit(X)
        result_orig = scaler.transform(X)

        bundle = scaler.save()
        manifest, toc, blobs = decode_bundle(bundle)
        assert manifest['typeId'] == 'wlearn.preprocess.standard_scaler@2'
        loaded = StandardScaler._from_bundle(manifest, toc, blobs)
        result_loaded = loaded.transform(X)
        assert np.allclose(result_orig, result_loaded)

    def test_registry_load(self):
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)
        scaler = StandardScaler()
        scaler.fit(X)
        bundle = scaler.save()

        loaded = registry_load(bundle)
        assert isinstance(loaded, StandardScaler)
        assert loaded.is_fitted

    def test_constant_column(self):
        X = np.array([[5, 1], [5, 2], [5, 3]], dtype=np.float64)
        scaler = StandardScaler()
        result = scaler.fit_transform(X)
        # Constant column should produce all zeros
        assert np.all(result[:, 0] == 0)
        assert scaler.transform(np.array([[10, 4]], dtype=np.float64))[0, 0] == 5

    def test_legacy_v1_constant_column_until_refit(self):
        artifact = b'{"means":[5],"stds":[0]}'
        bundle = encode_bundle(
            {'typeId': 'wlearn.preprocess.standard_scaler@1', 'params': {}},
            [{'id': 'params', 'data': artifact,
              'mediaType': 'application/json'}])
        scaler = registry_load(bundle)
        np.testing.assert_array_equal(scaler.transform([[5], [10]]), [[0], [0]])
        assert decode_bundle(scaler.save())[0]['typeId'] == (
            'wlearn.preprocess.standard_scaler@1')

        scaler.fit([[5], [5]])
        np.testing.assert_array_equal(scaler.transform([[5], [10]]), [[0], [5]])
        assert decode_bundle(scaler.save())[0]['typeId'] == (
            'wlearn.preprocess.standard_scaler@2')

    @pytest.mark.parametrize(
        'type_id',
        ['wlearn.preprocess.standard_scaler@1',
         'wlearn.preprocess.standard_scaler@2'])
    def test_rejects_malformed_artifacts(self, type_id):
        malformed = [
            None,
            {'means': [], 'stds': []},
            {'means': [0, 1], 'stds': [1]},
            {'means': [None], 'stds': [1]},
            {'means': [0], 'stds': [-1]},
            {'stds': [1]},
        ]
        for artifact in malformed:
            with pytest.raises(ValidationError, match='StandardScaler artifact'):
                registry_load(_artifact_bundle(type_id, artifact))

    def test_rejects_nonfinite_fit_without_replacing_fitted_model(self):
        scaler = StandardScaler().fit([[1], [3]])
        with pytest.raises(ValidationError, match='finite numbers'):
            scaler.fit([[1], [np.nan]])
        with pytest.raises(ValidationError, match='zero columns'):
            scaler.fit(np.empty((1, 0)))
        np.testing.assert_array_equal(scaler.transform([[3]]), [[1]])
        loaded = registry_load(scaler.save())
        np.testing.assert_array_equal(loaded.transform([[3]]), [[1]])

    def test_not_fitted_error(self):
        scaler = StandardScaler()
        with pytest.raises(NotFittedError):
            scaler.transform(np.zeros((1, 2)))

    def test_disposed_error(self):
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)
        scaler = StandardScaler()
        scaler.fit(X)
        scaler.dispose()
        with pytest.raises(DisposedError):
            scaler.transform(X)

    def test_column_mismatch(self):
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)
        scaler = StandardScaler()
        scaler.fit(X)
        with pytest.raises(ValidationError, match='columns'):
            scaler.transform(np.zeros((1, 3)))

    def test_get_set_params(self):
        scaler = StandardScaler({'key': 'value'})
        assert scaler.get_params() == {'key': 'value'}
        scaler.set_params({'key': 'new'})
        assert scaler.get_params() == {'key': 'new'}


class TestMinMaxScaler:
    def test_fit_transform_scales_to_unit_range(self):
        X = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=np.float64)
        scaler = MinMaxScaler()
        result = scaler.fit_transform(X)
        assert result.shape == X.shape
        assert np.allclose(result.min(axis=0), 0)
        assert np.allclose(result.max(axis=0), 1)

    def test_fit_then_transform(self):
        X = np.array([[0, 0], [5, 10], [10, 20]], dtype=np.float64)
        scaler = MinMaxScaler()
        scaler.fit(X)
        result = scaler.transform(X)
        expected = np.array([[0, 0], [0.5, 0.5], [1, 1]])
        assert np.allclose(result, expected)

    def test_save_load_roundtrip(self):
        X = np.array([[1, 10], [2, 20], [3, 30]], dtype=np.float64)
        scaler = MinMaxScaler()
        scaler.fit(X)
        result_orig = scaler.transform(X)

        bundle = scaler.save()
        manifest, toc, blobs = decode_bundle(bundle)
        assert manifest['typeId'] == 'wlearn.preprocess.minmax_scaler@2'
        loaded = MinMaxScaler._from_bundle(manifest, toc, blobs)
        result_loaded = loaded.transform(X)
        assert np.allclose(result_orig, result_loaded)

    def test_registry_load(self):
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)
        scaler = MinMaxScaler()
        scaler.fit(X)
        bundle = scaler.save()

        loaded = registry_load(bundle)
        assert isinstance(loaded, MinMaxScaler)
        assert loaded.is_fitted

    def test_constant_column(self):
        X = np.array([[5, 1], [5, 2], [5, 3]], dtype=np.float64)
        scaler = MinMaxScaler()
        result = scaler.fit_transform(X)
        assert np.all(result[:, 0] == 0)
        assert scaler.transform(np.array([[10, 4]], dtype=np.float64))[0, 0] == 5

    def test_legacy_v1_constant_column_until_refit(self):
        artifact = b'{"maxs":[5],"mins":[5]}'
        bundle = encode_bundle(
            {'typeId': 'wlearn.preprocess.minmax_scaler@1', 'params': {}},
            [{'id': 'params', 'data': artifact,
              'mediaType': 'application/json'}])
        scaler = registry_load(bundle)
        np.testing.assert_array_equal(scaler.transform([[5], [10]]), [[0], [0]])
        assert decode_bundle(scaler.save())[0]['typeId'] == (
            'wlearn.preprocess.minmax_scaler@1')

        scaler.fit([[5], [5]])
        np.testing.assert_array_equal(scaler.transform([[5], [10]]), [[0], [5]])
        assert decode_bundle(scaler.save())[0]['typeId'] == (
            'wlearn.preprocess.minmax_scaler@2')

    @pytest.mark.parametrize(
        'type_id',
        ['wlearn.preprocess.minmax_scaler@1',
         'wlearn.preprocess.minmax_scaler@2'])
    def test_rejects_malformed_artifacts(self, type_id):
        malformed = [
            None,
            {'mins': [], 'maxs': []},
            {'mins': [0, 1], 'maxs': [1]},
            {'mins': [None], 'maxs': [1]},
            {'mins': [2], 'maxs': [1]},
            {'maxs': [1]},
        ]
        for artifact in malformed:
            with pytest.raises(ValidationError, match='MinMaxScaler artifact'):
                registry_load(_artifact_bundle(type_id, artifact))

    def test_rejects_nonfinite_fit_without_replacing_fitted_model(self):
        scaler = MinMaxScaler().fit([[1], [3]])
        with pytest.raises(ValidationError, match='finite numbers'):
            scaler.fit([[1], [np.inf]])
        with pytest.raises(ValidationError, match='zero columns'):
            scaler.fit(np.empty((1, 0)))
        np.testing.assert_array_equal(scaler.transform([[3]]), [[1]])
        loaded = registry_load(scaler.save())
        np.testing.assert_array_equal(loaded.transform([[3]]), [[1]])

    def test_unseen_data_outside_range(self):
        X_train = np.array([[0], [10]], dtype=np.float64)
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_test = np.array([[-5], [15]], dtype=np.float64)
        result = scaler.transform(X_test)
        assert result[0, 0] < 0
        assert result[1, 0] > 1

    def test_not_fitted_error(self):
        scaler = MinMaxScaler()
        with pytest.raises(NotFittedError):
            scaler.transform(np.zeros((1, 2)))

    def test_disposed_error(self):
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)
        scaler = MinMaxScaler()
        scaler.fit(X)
        scaler.dispose()
        with pytest.raises(DisposedError):
            scaler.transform(X)
