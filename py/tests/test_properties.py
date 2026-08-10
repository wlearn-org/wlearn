"""Property-style probes for public metric and resampling APIs."""

import numpy as np
import pytest

hypothesis = pytest.importorskip('hypothesis')
from hypothesis import given, settings, strategies as st  # noqa: E402

from wlearn.measure import evaluate_measure
from wlearn.prediction import create_prediction
from wlearn.resampling import (
    sliding_index_split,
    sliding_period_split,
    sliding_window_split,
)


@st.composite
def weighted_classification_cases(draw):
    n = draw(st.integers(min_value=2, max_value=30))
    truth = draw(st.lists(st.integers(min_value=0, max_value=2), min_size=n, max_size=n))
    response = draw(st.lists(st.integers(min_value=0, max_value=2), min_size=n, max_size=n))
    weights = draw(st.lists(st.integers(min_value=1, max_value=4), min_size=n, max_size=n))
    return truth, response, weights


@st.composite
def weighted_regression_cases(draw):
    n = draw(st.integers(min_value=2, max_value=30))
    truth = draw(st.lists(st.floats(min_value=-20, max_value=20, allow_nan=False, allow_infinity=False), min_size=n, max_size=n))
    response = draw(st.lists(st.floats(min_value=-20, max_value=20, allow_nan=False, allow_infinity=False), min_size=n, max_size=n))
    weights = draw(st.lists(st.integers(min_value=1, max_value=4), min_size=n, max_size=n))
    return truth, response, weights


def _repeat(values, weights, dtype):
    out = []
    for value, weight in zip(values, weights):
        out.extend([value] * int(weight))
    return np.asarray(out, dtype=dtype)


@given(weighted_classification_cases())
@settings(max_examples=80, deadline=None)
def test_weighted_classification_measures_match_duplicated_rows(case):
    truth, response, weights = case
    truth_arr = np.asarray(truth, dtype=np.int32)
    response_arr = np.asarray(response, dtype=np.int32)
    weight_arr = np.asarray(weights, dtype=np.float64)
    dup_truth = _repeat(truth, weights, np.int32)
    dup_response = _repeat(response, weights, np.int32)

    weighted_prediction = create_prediction(truth=truth_arr, response=response_arr)
    duplicated_prediction = create_prediction(truth=dup_truth, response=dup_response)

    assert evaluate_measure('accuracy', weighted_prediction, sample_weight=weight_arr) == pytest.approx(
        evaluate_measure('accuracy', duplicated_prediction)
    )
    for average in ('micro', 'macro', 'weighted'):
        assert evaluate_measure('precision', weighted_prediction, average=average, sample_weight=weight_arr) == pytest.approx(
            evaluate_measure('precision', duplicated_prediction, average=average)
        )
        assert evaluate_measure('recall', weighted_prediction, average=average, sample_weight=weight_arr) == pytest.approx(
            evaluate_measure('recall', duplicated_prediction, average=average)
        )
        assert evaluate_measure('f1', weighted_prediction, average=average, sample_weight=weight_arr) == pytest.approx(
            evaluate_measure('f1', duplicated_prediction, average=average)
        )


@given(weighted_regression_cases())
@settings(max_examples=80, deadline=None)
def test_weighted_regression_measures_match_duplicated_rows(case):
    truth, response, weights = case
    truth_arr = np.asarray(truth, dtype=np.float64)
    response_arr = np.asarray(response, dtype=np.float64)
    weight_arr = np.asarray(weights, dtype=np.float64)
    dup_truth = _repeat(truth, weights, np.float64)
    dup_response = _repeat(response, weights, np.float64)

    weighted_prediction = create_prediction(truth=truth_arr, response=response_arr)
    duplicated_prediction = create_prediction(truth=dup_truth, response=dup_response)

    assert evaluate_measure('mse', weighted_prediction, sample_weight=weight_arr) == pytest.approx(
        evaluate_measure('mse', duplicated_prediction)
    )
    assert evaluate_measure('mae', weighted_prediction, sample_weight=weight_arr) == pytest.approx(
        evaluate_measure('mae', duplicated_prediction)
    )


def test_auc_affine_transform_property_probe():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        n = 16 + seed % 8
        truth = np.asarray([i % 2 for i in range(n)], dtype=np.int32)
        score = rng.random(n) + np.arange(n) * 1e-8
        prediction = create_prediction(truth=truth, score=score.astype(np.float64))
        transformed = create_prediction(truth=truth, score=(3.5 + 7 * score).astype(np.float64))
        reversed_prediction = create_prediction(truth=truth, score=(-score).astype(np.float64))
        auc = evaluate_measure('roc_auc', prediction)
        assert evaluate_measure('roc_auc', transformed) == pytest.approx(auc)
        assert auc + evaluate_measure('roc_auc', reversed_prediction) == pytest.approx(1.0)


def _assert_fold_invariants(folds, n):
    assert folds
    for fold in folds:
        train = set(fold.train.tolist())
        test = set(fold.test.tolist())
        assert train
        assert test
        assert len(train) == len(fold.train)
        assert len(test) == len(fold.test)
        assert not (train & test)
        assert min(train) >= 0
        assert min(test) >= 0
        assert max(train) < n
        assert max(test) < n
        assert max(train) < min(test)


def test_sliding_resampling_property_probes():
    for n in range(5, 25):
        for lookback in range(1, min(6, n - 2) + 1):
            horizon = min(3, n - lookback)
            folds = sliding_window_split(n, lookback=lookback, assess_start=1, assess_stop=horizon, step=1 + n % 3)
            _assert_fold_invariants(folds, n)

    index = np.asarray([0, 0, 1, 2, 2, 3, 4, 5, 5, 6], dtype=np.float64)
    _assert_fold_invariants(sliding_index_split(index, lookback=2, assess_start=1, assess_stop=1), len(index))

    dates = [f'2026-01-{day:02d}' for day in range(1, 8)]
    _assert_fold_invariants(sliding_period_split(dates, period='day', lookback=2, assess_start=1, assess_stop=2), len(dates))
