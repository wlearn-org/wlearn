"""Tests for task, prediction, measure, resampling, and archive primitives."""

import numpy as np
import pytest

from wlearn.archive import Archive, create_trial_record
from wlearn.errors import ValidationError
from wlearn.measure import (
    aggregate_measure,
    evaluate_measure,
    evaluate_metric_set,
    list_measures,
)
from wlearn.prediction import create_prediction
from wlearn.resampling import (
    create_resampling_plan,
    deserialize_resampling_plan,
    group_k_fold,
    ResamplingFold,
    serialize_resampling_plan,
    sliding_index_split,
    sliding_period_split,
    sliding_window_split,
    time_series_split,
)
from wlearn.task import create_feature_schema, create_task


X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
y = np.array([0, 1, 0], dtype=np.int32)


@pytest.mark.parametrize('proba', [[np.nan, 1, 0, 1], [-0.1, 1.1, 0, 1], [0.2, 0.2, 0, 1]])
def test_prediction_rejects_invalid_probabilities(proba):
    with pytest.raises(ValidationError):
        create_prediction(truth=[0, 1], proba=proba, classes=[0, 1])


def test_prediction_rejects_duplicate_classes():
    with pytest.raises(ValidationError):
        create_prediction(truth=[0, 1], proba=[0.5, 0.5, 0, 1], classes=[0, 0])


def test_holdout_rounds_half_up_like_javascript():
    plan = create_resampling_plan(strategy='holdout', n=10, test_size=0.25, shuffle=False)
    assert plan.folds[0].test.tolist() == [7, 8, 9]


def test_task_creates_schema_and_infers_kind():
    schema = create_feature_schema(X, names=['a', 'b'], roles=['feature', 'offset'])
    task = create_task(id='toy', X=X, y=y, feature_schema=schema)

    assert task.id == 'toy'
    assert task.kind == 'classification'
    assert task.feature_schema.features[0].name == 'a'
    assert task.feature_schema.features[1].role == 'offset'


def test_task_rejects_target_length_mismatch():
    with pytest.raises(ValidationError):
        create_task(X=X, y=np.array([0, 1], dtype=np.int32))


def test_task_rejects_feature_schema_row_mismatch():
    schema = create_feature_schema(X)
    schema.rows = 999
    with pytest.raises(ValidationError):
        create_task(X=X, y=y, feature_schema=schema)


def test_task_rejects_out_of_range_row_roles():
    with pytest.raises(ValidationError):
        create_task(X=X, y=y, row_roles={'train': np.array([0, 3], dtype=np.int32)})


def test_prediction_scores_builtin_measures():
    prediction = create_prediction(
        truth=y,
        response=np.array([0, 1, 1], dtype=np.int32),
    )

    assert evaluate_measure('accuracy', prediction) == pytest.approx(2 / 3)
    scores = evaluate_metric_set(['accuracy', 'f1'], prediction)
    assert scores['accuracy'] == pytest.approx(2 / 3)
    assert isinstance(scores['f1'], float)


def test_prediction_probability_uses_declared_class_order():
    prediction = create_prediction(
        truth=np.array([0, 1], dtype=np.int32),
        proba=np.array([0.1, 0.9, 0.9, 0.1], dtype=np.float64),
        classes=np.array([1, 0], dtype=np.int32),
    )

    assert evaluate_measure('log_loss', prediction) < 0.2


def test_log_loss_rejects_missing_class_and_non_finite_probability():
    missing_class = create_prediction(
        truth=np.array([0, 2], dtype=np.int32),
        proba=np.array([0.8, 0.2, 0.1, 0.9], dtype=np.float64),
        classes=np.array([0, 1], dtype=np.int32),
    )
    with pytest.raises(ValidationError):
        evaluate_measure('log_loss', missing_class)

    with pytest.raises(ValidationError):
        create_prediction(
            truth=np.array([0, 1], dtype=np.int32),
            proba=np.array([0.8, 0.2, np.nan, 0.9], dtype=np.float64),
            classes=np.array([0, 1], dtype=np.int32),
        )


def test_prediction_rejects_probability_length_mismatch():
    with pytest.raises(ValidationError):
        create_prediction(
            proba=np.array([0.1, 0.9, 0.8], dtype=np.float64),
            classes=np.array([0, 1], dtype=np.int32),
        )


def test_prediction_rejects_conflicting_proba_rows():
    with pytest.raises(ValidationError):
        create_prediction(
            truth=np.array([0, 1], dtype=np.int32),
            proba=np.array([0.7, 0.3, 0.2, 0.8], dtype=np.float64),
            classes=np.array([0, 1], dtype=np.int32),
            proba_rows=3,
        )


def test_prediction_validates_interval_and_quantile_alignment():
    create_prediction(
        row_ids=['a', 'b'],
        interval=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
    )
    with pytest.raises(ValidationError):
        create_prediction(
            row_ids=['a', 'b'],
            quantiles=np.array([0.0, 1.0, 2.0], dtype=np.float64),
        )


def test_roc_auc_uses_average_ranks_for_ties():
    prediction = create_prediction(
        truth=np.array([0, 1], dtype=np.int32),
        score=np.array([0.5, 0.5], dtype=np.float64),
    )
    assert evaluate_measure('roc_auc', prediction) == pytest.approx(0.5)

    prediction = create_prediction(
        truth=np.array([0, 1, 0, 1], dtype=np.int32),
        score=np.array([0.5, 0.5, 0.2, 0.8], dtype=np.float64),
    )
    assert evaluate_measure('roc_auc', prediction) == pytest.approx(0.875)


def test_measure_registry_and_aggregation():
    assert 'accuracy' in list_measures()
    assert aggregate_measure('accuracy', np.array([1.0, 0.5, 0.75])) == pytest.approx(0.75)


def test_measure_supports_sample_weight_and_multiclass_auc():
    prediction = create_prediction(
        truth=np.array([0, 1, 2, 0, 1, 2], dtype=np.int32),
        response=np.array([0, 1, 2, 1, 1, 0], dtype=np.int32),
        proba=np.array([
            0.9, 0.05, 0.05,
            0.05, 0.9, 0.05,
            0.05, 0.05, 0.9,
            0.8, 0.1, 0.1,
            0.1, 0.8, 0.1,
            0.1, 0.1, 0.8,
        ], dtype=np.float64),
        classes=np.array([0, 1, 2], dtype=np.int32),
    )

    assert evaluate_measure(
        'accuracy',
        prediction,
        sample_weight=np.array([1, 1, 1, 3, 1, 1], dtype=np.float64),
    ) == pytest.approx(0.5)
    assert evaluate_measure('roc_auc_ovr', prediction) == pytest.approx(1.0)
    assert evaluate_measure('roc_auc_ovo', prediction) == pytest.approx(1.0)


def test_measure_allows_undefined_auc_warning():
    warnings = []
    prediction = create_prediction(
        truth=np.array([1, 1, 1], dtype=np.int32),
        score=np.array([0.1, 0.5, 0.9], dtype=np.float64),
    )
    value = evaluate_measure('roc_auc', prediction, undefined_value='warn', warnings=warnings)
    assert np.isnan(value)
    assert len(warnings) == 1


def test_resampling_creates_serializable_kfold_plans():
    plan = create_resampling_plan(strategy='kfold', n=10, k=5, seed=7)
    assert len(plan.folds) == 5
    restored = deserialize_resampling_plan(serialize_resampling_plan(plan))
    assert len(restored.folds) == 5
    assert restored.folds[0].train.dtype == np.int32


def test_resampling_rejects_duplicate_indices_in_supplied_folds():
    with pytest.raises(ValidationError):
        create_resampling_plan(
            strategy='kfold',
            n=4,
            folds=[
                ResamplingFold(
                    'fold-0',
                    np.array([0, 0, 1], dtype=np.int32),
                    np.array([2, 3], dtype=np.int32),
                )
            ],
        )


def test_resampling_rejects_invalid_validation_indices():
    with pytest.raises(ValidationError):
        create_resampling_plan(
            strategy='kfold',
            n=5,
            folds=[
                ResamplingFold(
                    'fold-0',
                    np.array([0, 1], dtype=np.int32),
                    np.array([2], dtype=np.int32),
                    np.array([1, 4], dtype=np.int32),
                )
            ],
        )

    with pytest.raises(ValidationError):
        create_resampling_plan(
            strategy='kfold',
            n=5,
            folds=[
                ResamplingFold(
                    'fold-0',
                    np.array([0, 1], dtype=np.int32),
                    np.array([2], dtype=np.int32),
                    np.array([6], dtype=np.int32),
                )
            ],
        )


def test_resampling_rejects_invalid_repeats():
    with pytest.raises(ValidationError):
        create_resampling_plan(strategy='repeated_kfold', n=10, k=5, repeats=0)


def test_stratified_resampling_rejects_class_smaller_than_k():
    with pytest.raises(ValidationError):
        create_resampling_plan(
            strategy='stratified_kfold',
            n=4,
            y=np.array([0, 0, 1, 1], dtype=np.int32),
            k=3,
        )


def test_group_kfold_keeps_groups_together():
    groups = np.array([1, 1, 2, 2, 3, 3], dtype=np.int32)
    folds = group_k_fold(groups, 3, shuffle=False)

    for fold in folds:
        test_groups = {int(groups[i]) for i in fold.test}
        for idx in fold.train:
            assert int(groups[idx]) not in test_groups


def test_time_series_split_is_forward_only():
    folds = time_series_split(8, initial_window=4, horizon=2, step=2)
    assert len(folds) == 2
    for fold in folds:
        assert int(np.max(fold.train)) < int(np.min(fold.test))


def test_sliding_window_split():
    folds = sliding_window_split(8, lookback=3, assess_start=1, assess_stop=2, step=2)
    assert len(folds) == 2
    assert folds[0].train.tolist() == [0, 1, 2]
    assert folds[0].test.tolist() == [3, 4]
    assert folds[1].train.tolist() == [2, 3, 4]
    assert folds[1].test.tolist() == [5, 6]


def test_sliding_index_split():
    folds = sliding_index_split(
        np.array([0, 1, 2, 3, 4, 5], dtype=np.float64),
        lookback=2,
        assess_start=1,
        assess_stop=1,
    )
    assert folds[0].train.tolist() == [0, 1, 2]
    assert folds[0].test.tolist() == [3]
    with pytest.raises(ValidationError):
        sliding_index_split(np.array([0, 2, 1], dtype=np.float64), lookback=1)


def test_sliding_period_plan_serialization():
    index = ['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04', '2026-01-05']
    folds = sliding_period_split(index, period='day', lookback=2, assess_start=1, assess_stop=1)
    assert folds[0].train.tolist() == [0, 1, 2]
    assert folds[0].test.tolist() == [3]

    plan = create_resampling_plan(
        strategy='sliding_period',
        index=index,
        period='day',
        lookback=2,
        assess_start=1,
        assess_stop=1,
    )
    restored = deserialize_resampling_plan(serialize_resampling_plan(plan))
    assert restored.strategy == 'sliding_period'
    assert restored.metadata['period'] == 'day'


def test_archive_records_trials_and_builds_leaderboard():
    archive = Archive(measures=['accuracy'], primary_measure='accuracy')
    archive.add({
        'trial_id': 'a-1',
        'candidate_id': 'a',
        'pipeline_spec': {'nodes': [], 'edges': [], 'endpoints': {}},
        'scores': {'accuracy': 0.8},
        'status': 'ok',
    })
    archive.add({
        'trial_id': 'b-1',
        'candidate_id': 'b',
        'scores': {'accuracy': 0.9},
        'status': 'ok',
    })

    rows = archive.leaderboard()
    assert rows[0].candidate_id == 'b'
    assert rows[0].rank == 1
    assert rows[1].pipeline_spec['nodes'] == []


def test_archive_rejects_duplicate_trial_ids():
    archive = Archive()
    archive.add({'trial_id': 'x', 'candidate_id': 'x'})
    with pytest.raises(ValidationError):
        archive.add({'trial_id': 'x', 'candidate_id': 'x'})


def test_archive_does_not_expose_mutable_records():
    archive = Archive()
    archive.add({'trial_id': 'x', 'candidate_id': 'x', 'scores': {'accuracy': 1.0}, 'status': 'ok'})
    records = archive.records()
    records[0].scores['accuracy'] = 0.0
    assert archive.records()[0].scores['accuracy'] == 1.0


def test_archive_snapshots_nested_payloads_on_add_and_update():
    archive = Archive()
    params = {'nested': {'depth': 3}}
    archive.add({'trial_id': 'x', 'candidate_id': 'x', 'params': params, 'status': 'running'})
    params['nested']['depth'] = 9
    assert archive.records()[0].params['nested']['depth'] == 3

    patch = {'metadata': {'nested': {'phase': 'fit'}}, 'status': 'ok'}
    archive.update('x', patch)
    patch['metadata']['nested']['phase'] = 'predict'
    assert archive.records()[0].metadata['nested']['phase'] == 'fit'


def test_archive_normalizes_failed_trial_records():
    archive = Archive()
    archive.fail({'trial_id': 'bad', 'candidate_id': 'bad'}, RuntimeError('boom'), 'fit')
    assert archive.size == 1
    record = archive.records()[0]
    assert record.status == 'failed'
    assert record.error.phase == 'fit'


def test_archive_validates_trial_records():
    record = create_trial_record({'candidate_id': 'x', 'fold_id': 'fold-1', 'batch': 1})
    assert record.trial_id == 'x-fold-1-42'
    assert record.batch == 1


def test_cv_validates_reserved_validation_rows():
    from wlearn.resampling import resolve_cv
    with pytest.raises(ValidationError, match='overlap'):
        resolve_cv([{'train': [0, 1], 'test': [2, 3], 'validate': [1]}],
                   np.arange(4, dtype=float))
