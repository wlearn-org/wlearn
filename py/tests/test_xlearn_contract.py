import numpy as np
import pytest
import struct

from wlearn.bundle import decode_bundle, encode_bundle
from wlearn.registry import load
from wlearn.xlearn import _parse_model, _serialize_model


def _linear_bundle(type_id, classes, *, label_encoding=None, seed=None):
    raw_model = {
        'score_func': 'linear',
        'loss_func': 'cross-entropy',
        'num_feat': 2,
        'num_field': 0,
        'num_K': 0,
        'aux_size': 2,
        'w': np.array([1.0, 1.0, -1.0, 1.0], dtype=np.float32),
        'b': np.array([0.0, 1.0], dtype=np.float32),
        'v': None,
    }
    metadata = {
        'algo': 'linear',
        'task': 'binary',
        'nFeatures': 2,
        'nClasses': 2,
        'classes': classes,
    }
    if label_encoding is not None:
        metadata['labelEncoding'] = label_encoding
    manifest = {
        'typeId': type_id,
        'params': {},
        'metadata': metadata,
    }
    if seed is not None:
        manifest['seed'] = seed
        manifest['params']['seed'] = seed
    return encode_bundle(
        manifest,
        [{'id': 'model', 'data': _serialize_model(raw_model)}],
    )


def test_v2_classifier_predicts_public_labels_and_exposes_margins():
    bundle = _linear_bundle(
        'wlearn.xlearn.lr.classifier@2', [10, 20],
        label_encoding='sorted-int32-sign-v1', seed=42)
    model = load(bundle)
    try:
        X = [[2.0, 0.0], [0.0, 2.0]]
        np.testing.assert_array_equal(model.predict(X), [20, 10])
        np.testing.assert_allclose(model.decision_function(X), [2.0, -2.0])
        probabilities = model.predict_proba(X).reshape(-1, 2)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        assert model.score(X, [20, 10]) == 1.0
        assert model.capabilities['decisionFunction'] is True
    finally:
        model.dispose()


def test_classifier_probability_is_stable_for_extreme_margins():
    model = load(_linear_bundle(
        'wlearn.xlearn.lr.classifier@2', [10, 20],
        label_encoding='sorted-int32-sign-v1', seed=42))
    try:
        probabilities = model.predict_proba(
            [[1000.0, 0.0], [-1000.0, 0.0]]).reshape(-1, 2)
        np.testing.assert_allclose(probabilities, [[0.0, 1.0], [1.0, 0.0]])
        assert np.all(np.isfinite(probabilities))
    finally:
        model.dispose()


def test_python_xlearn_matrix_boundary_matches_js_contract():
    model = load(_linear_bundle(
        'wlearn.xlearn.lr.classifier@2', [10, 20],
        label_encoding='sorted-int32-sign-v1', seed=42))
    try:
        invalid = [
            [1.0, 2.0],
            [],
            [[1.0], [2.0]],
            [[1.0, 2.0], [3.0]],
            [[np.nan, 0.0]],
            [[np.inf, 0.0]],
        ]
        for X in invalid:
            with pytest.raises(ValueError):
                model.predict(X)
    finally:
        model.dispose()


def test_bundle_rejects_model_loss_that_disagrees_with_task():
    bundle = _linear_bundle(
        'wlearn.xlearn.lr.classifier@2', [10, 20],
        label_encoding='sorted-int32-sign-v1', seed=42)
    manifest, toc, blobs = decode_bundle(bundle)
    entry = next(item for item in toc if item['id'] == 'model')
    raw_model = _parse_model(
        blobs[entry['offset']:entry['offset'] + entry['length']])
    raw_model['loss_func'] = 'squared'
    swapped = encode_bundle(
        manifest, [{'id': 'model', 'data': _serialize_model(raw_model)}])
    with pytest.raises(ValueError, match='model loss'):
        load(swapped)


def test_python_loader_rejects_truncated_and_inconsistent_model_counts():
    bundle = _linear_bundle(
        'wlearn.xlearn.lr.classifier@2', [10, 20],
        label_encoding='sorted-int32-sign-v1', seed=42)
    manifest, toc, blobs = decode_bundle(bundle)
    entry = toc[0]
    raw = bytearray(
        blobs[entry['offset']:entry['offset'] + entry['length']])
    pos = 0
    for _ in range(2):
        length = struct.unpack_from('<I', raw, pos)[0]
        pos += 4 + length
    n_features_end = pos + 4
    n_weights_offset = pos + 16

    changed_count = bytearray(raw)
    count = struct.unpack_from('<I', changed_count, n_weights_offset)[0]
    struct.pack_into('<I', changed_count, n_weights_offset, count + 1)
    cases = [raw[:n_features_end], changed_count, raw + b'\x00']
    for model_bytes in cases:
        malformed = encode_bundle(
            manifest,
            [{'id': 'model', 'data': bytes(model_bytes)}],
        )
        with pytest.raises(ValueError, match='truncated|inconsistent'):
            load(malformed)


def test_python_ffm_loader_rejects_field_ids_outside_model_dimensions():
    raw_model = {
        'score_func': 'ffm',
        'loss_func': 'cross-entropy',
        'num_feat': 2,
        'num_field': 2,
        'num_K': 1,
        'aux_size': 2,
        'w': np.zeros(4, dtype=np.float32),
        'b': np.zeros(2, dtype=np.float32),
        'v': np.zeros(32, dtype=np.float32),
    }
    bundle = encode_bundle({
        'typeId': 'wlearn.xlearn.ffm.classifier@2',
        'params': {'seed': 1},
        'seed': 1,
        'metadata': {
            'algo': 'ffm',
            'task': 'binary',
            'nFeatures': 2,
            'nClasses': 2,
            'classes': [0, 1],
            'labelEncoding': 'sorted-int32-sign-v1',
        },
    }, [
        {'id': 'model', 'data': _serialize_model(raw_model)},
        {'id': 'field_map', 'data': np.array([0, 2], dtype='<i4').tobytes()},
    ])
    with pytest.raises(ValueError, match='invalid field ID'):
        load(bundle)


def test_v2_python_resave_preserves_manifest_and_predictions():
    bundle = _linear_bundle(
        'wlearn.xlearn.lr.classifier@2', [-3, 7],
        label_encoding='sorted-int32-sign-v1', seed=9)
    model = load(bundle)
    try:
        saved = model.save()
        assert decode_bundle(saved)[0] == decode_bundle(bundle)[0]
        reloaded = load(saved)
        try:
            np.testing.assert_array_equal(
                reloaded.predict([[1.0, 0.0], [0.0, 1.0]]), [7, -3])
        finally:
            reloaded.dispose()
    finally:
        model.dispose()


@pytest.mark.parametrize('classes', ([0, 1], [-1, 1]))
def test_legacy_classifier_matches_public_label_contract(classes):
    bundle = _linear_bundle(
        'wlearn.xlearn.lr.classifier@1', classes)
    model = load(bundle)
    try:
        X = [[2.0, 0.0], [0.0, 2.0]]
        expected = [classes[1], classes[0]]
        np.testing.assert_array_equal(model.predict(X), expected)
        np.testing.assert_allclose(model.decision_function(X), [2.0, -2.0])
        assert model.score(X, expected) == 1.0
        reloaded = load(model.save())
        try:
            np.testing.assert_array_equal(reloaded.predict(X), expected)
            np.testing.assert_allclose(
                reloaded.decision_function(X), [2.0, -2.0])
        finally:
            reloaded.dispose()
    finally:
        model.dispose()


def test_legacy_classifier_rejects_ambiguous_same_sign_classes():
    with pytest.raises(ValueError, match='ambiguous same-sign'):
        load(_linear_bundle(
            'wlearn.xlearn.lr.classifier@1', [10, 20]))


@pytest.mark.parametrize(
    ('label_encoding', 'seed', 'message'),
    [
        (None, 42, 'label encoding'),
        ('sorted-int32-sign-v1', None, 'training seed'),
    ],
)
def test_v2_classifier_requires_versioned_contract_metadata(
        label_encoding, seed, message):
    with pytest.raises(ValueError, match=message):
        load(_linear_bundle(
            'wlearn.xlearn.lr.classifier@2', [10, 20],
            label_encoding=label_encoding, seed=seed))
