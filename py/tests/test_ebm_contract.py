import copy
import json

import numpy as np
import pytest

import wlearn.ebm  # noqa: F401 - registers the loader
from wlearn.bundle import decode_bundle, encode_bundle
from wlearn.registry import load
from wlearn.ebm import EBMModel


def _bundle(model_data=None, metadata=None, artifacts=None, params=None):
    data = model_data or {
        'format': 'ebm-json-v1',
        'task': 'classification',
        'nFeatures': 1,
        'nTerms': 1,
        'nScores': 1,
        'intercept': [0.0],
        'features': [{'type': 'continuous', 'cuts': [0.0]}],
        'terms': [{
            'features': [0],
            'binCounts': [3],
            'scores': [-1.0, 1.0, 0.0],
        }],
    }
    manifest = {
        'typeId': 'wlearn.ebm.classifier@1',
        'params': ({'objective': 'classification'}
                   if params is None else params),
        'metadata': metadata or {
            'nClasses': 2,
            'classes': [3, 7],
            'termNames': ['feature_0'],
            'featureNames': ['feature_0'],
        },
    }
    return encode_bundle(
        manifest,
        artifacts or [{
            'id': 'model',
            'data': json.dumps(data, separators=(',', ':')).encode(),
        }],
    )


def test_python_ebm_bundle_and_matrix_boundaries():
    base_bundle = _bundle()
    model = load(base_bundle)
    try:
        np.testing.assert_array_equal(model.predict([[-1.0], [1.0]]), [3, 7])
        for invalid in (
                np.array([1.0]),
                np.zeros((1, 2)),
                np.array([[np.inf]])):
            with pytest.raises(ValueError):
                model.predict(invalid)
    finally:
        model.dispose()

    manifest, toc, blobs = decode_bundle(base_bundle)
    entry = toc[0]
    base_data = json.loads(bytes(
        blobs[entry['offset']:entry['offset'] + entry['length']]))
    mutations = []
    for mutate in (
            lambda data: data.update(nFeatures=2),
            lambda data: data['features'][0].update(cuts=[0.0, 0.0]),
            lambda data: data['terms'][0].update(features=[1]),
            lambda data: data['terms'][0].update(binCounts=[2]),
            lambda data: data['terms'][0].update(scores=[0.0]),
            lambda data: data.update(intercept=[None]),
            lambda data: data.update(nScores=2)):
        candidate = copy.deepcopy(base_data)
        mutate(candidate)
        mutations.append(candidate)

    for candidate in mutations:
        with pytest.raises(ValueError):
            load(_bundle(candidate))

    with pytest.raises(ValueError, match='termNames'):
        load(_bundle(metadata={
            **manifest['metadata'], 'termNames': [],
        }))
    with pytest.raises(ValueError, match='exactly one'):
        load(_bundle(artifacts=[
            {'id': 'model', 'data': json.dumps(base_data).encode()},
            {'id': 'extra', 'data': b'x'},
        ]))

    for key in ('objective', 'task'):
        with pytest.raises(ValueError, match='does not match fitted'):
            load(_bundle(params={key: 'regression'}))

    loaded = load(base_bundle)
    try:
        loaded.set_params({'objective': 'regression'})
        with pytest.raises(ValueError, match='does not match fitted'):
            loaded.save()
    finally:
        loaded.dispose()


def test_python_ebm_rejects_unknown_objective_before_optional_backend_import():
    model = EBMModel.create({'objective': 'clustering'})
    with pytest.raises(ValueError, match='objective'):
        model.fit([[0.0], [1.0]], [0, 1])
    assert not model.is_fitted

    model = EBMModel.create({'task': 'clustering'})
    with pytest.raises(ValueError, match='task'):
        model.fit([[0.0], [1.0]], [0, 1])
    assert not model.is_fitted


def test_python_ebm_nominal_bins_match_port_semantics():
    nominal = {
        'format': 'ebm-json-v1',
        'task': 'classification',
        'nFeatures': 1,
        'nTerms': 1,
        'nScores': 1,
        'intercept': [0.0],
        'features': [{'type': 'nominal', 'nBins': 3}],
        'terms': [{
            'features': [0],
            'binCounts': [3],
            'scores': [-2.0, 2.0, 4.0],
        }],
    }
    model = load(_bundle(nominal))
    try:
        X = [[0.0], [1.0], [2.0], [7.0], [np.nan]]
        np.testing.assert_array_equal(model.predict(X), [3, 7, 7, 7, 7])
        np.testing.assert_allclose(
            model.explain(X)['contributions'],
            [-2.0, 2.0, 4.0, 4.0, 4.0])
    finally:
        model.dispose()

    inconsistent = copy.deepcopy(nominal)
    inconsistent['terms'][0]['binCounts'] = [2]
    inconsistent['terms'][0]['scores'] = [-2.0, 2.0]
    with pytest.raises(ValueError, match='bin count disagrees'):
        load(_bundle(inconsistent))
