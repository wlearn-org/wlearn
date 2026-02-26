"""Tests for Python StandardScaler and MinMaxScaler."""

import numpy as np
import pytest

from wlearn.scalers import StandardScaler, MinMaxScaler
from wlearn.bundle import decode_bundle
from wlearn.registry import load as registry_load
from wlearn.errors import NotFittedError, DisposedError, ValidationError


class TestStandardScaler:
    def test_fit_transform_zero_mean_unit_var(self):
        X = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=np.float64)
        scaler = StandardScaler()
        result = scaler.fit_transform(X)
        assert result.shape == X.shape
        # Column means should be ~0
        assert np.allclose(result.mean(axis=0), 0, atol=1e-10)
        # Column stds should be ~1 (sample std, ddof=1)
        assert np.allclose(result.std(axis=0, ddof=1), 1, atol=1e-10)

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
