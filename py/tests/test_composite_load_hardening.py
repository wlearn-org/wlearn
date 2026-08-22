import numpy as np
import pytest

from wlearn.bundle import encode_bundle
from wlearn.ensemble import (
    BaggedEstimator, StackingEnsemble, VotingEnsemble,
)
from wlearn.errors import RegistryError, ValidationError
from wlearn.pipeline import Pipeline
from wlearn.registry import register


def _child_bundle(type_id):
    return encode_bundle({'typeId': type_id}, [])


def _nested_artifact(artifact_id, type_id):
    return {
        'id': artifact_id,
        'data': _child_bundle(type_id),
        'mediaType': 'application/x-wlearn-bundle',
    }


def _assert_transactional_cleanup(
        load, outer_type_id, manifest_fields, artifact_ids, prefix):
    first_type_id = f'wlearn.test.{prefix}.first@1'
    failing_type_id = f'wlearn.test.{prefix}.failing@1'
    state = {'disposed': 0}

    class LoadedModel:
        classes = np.array([0, 1], dtype=np.int32)
        capabilities = {'predictProba': True}

        def predict_proba(self, X):
            return np.empty(0, dtype=np.float64)

        def dispose(self):
            state['disposed'] += 1

    register(first_type_id, lambda manifest, toc, blobs: LoadedModel())

    def fail_loader(manifest, toc, blobs):
        raise RuntimeError(f'{prefix} loader failure')

    register(failing_type_id, fail_loader)
    artifacts = [
        _nested_artifact(
            artifact_id,
            first_type_id if index == 0 else failing_type_id,
        )
        for index, artifact_id in enumerate(artifact_ids)
    ]
    bundle = encode_bundle({
        'typeId': outer_type_id,
        **manifest_fields,
    }, artifacts)
    with pytest.raises(RuntimeError, match=f'{prefix} loader failure'):
        load(bundle)
    assert state['disposed'] == 1


def _assert_context_forwarding(
        load, outer_type_id, manifest_fields, artifact_ids, prefix):
    child_type_id = f'wlearn.test.{prefix}.context@1'
    contexts = []

    class LoadedModel:
        classes = np.array([0, 1], dtype=np.int32)
        capabilities = {'predictProba': True}

        def predict_proba(self, X):
            return np.empty(0, dtype=np.float64)

        def dispose(self):
            pass

    def loader(manifest, toc, blobs, context):
        contexts.append(context)
        return LoadedModel()

    register(child_type_id, loader, accepts_context=True)
    bundle = encode_bundle({
        'typeId': outer_type_id,
        **manifest_fields,
    }, [
        _nested_artifact(artifact_id, child_type_id)
        for artifact_id in artifact_ids
    ])
    runtime_options = {'limits': {'max_apply_rows': 2}}
    loaded = load(bundle, loader_options={
        'wlearn.preprocess.tabular@1': runtime_options,
    })
    assert len(contexts) == len(artifact_ids)
    assert all(
        context['loaderOptions']['wlearn.preprocess.tabular@1']
        is not runtime_options for context in contexts)
    assert all(
        context['loaderOptions']['wlearn.preprocess.tabular@1']['limits']
        ['max_apply_rows'] == 2 for context in contexts)
    assert all(context is contexts[0] for context in contexts)
    loaded.dispose()


def test_pipeline_direct_load_type_preflight_and_transactional_cleanup():
    wrong_type = encode_bundle({
        'typeId': 'wlearn.test.not-pipeline@1',
        'steps': [{'name': 'model', 'params': {}}],
    }, [])
    with pytest.raises(ValidationError, match='expected typeId'):
        Pipeline.load(wrong_type)

    available_type_id = 'wlearn.test.py-pipeline-preflight.available@1'
    missing_type_id = 'wlearn.test.py-pipeline-preflight.missing@1'
    state = {'dispatched': 0}

    def available_loader(manifest, toc, blobs):
        state['dispatched'] += 1
        return object()

    register(available_type_id, available_loader)
    preflight = encode_bundle({
        'typeId': 'wlearn.pipeline@1',
        'steps': [
            {'name': 'first', 'params': {}},
            {'name': 'second', 'params': {}},
        ],
    }, [
        _nested_artifact('first', available_type_id),
        _nested_artifact('second', missing_type_id),
    ])
    with pytest.raises(RegistryError, match='Missing required loader'):
        Pipeline.load(preflight)
    assert state['dispatched'] == 0

    _assert_transactional_cleanup(
        Pipeline.load,
        'wlearn.pipeline@1',
        {'steps': [
            {'name': 'first', 'params': {}},
            {'name': 'second', 'params': {}},
        ]},
        ['first', 'second'],
        'py-pipeline-transaction',
    )


def test_pipeline_set_params_invalidates_before_child_mutation():
    class Child:
        def __init__(self, fail=False):
            self.fail = fail
            self.params = {}

        def fit(self, _X, _y):
            return self

        def set_params(self, params):
            if self.fail:
                raise RuntimeError('parameter update failed')
            self.params.update(params)
            return self

    child = Child()
    pipeline = Pipeline([('model', child)])
    pipeline.fit([[1.0]], [0])
    pipeline.set_params({'model': {'depth': 2}})
    assert not pipeline.is_fitted
    assert child.params == {'depth': 2}

    failing = Child(fail=True)
    pipeline = Pipeline([('model', failing)])
    pipeline.fit([[1.0]], [0])
    with pytest.raises(RuntimeError, match='parameter update failed'):
        pipeline.set_params({'model': {'depth': 3}})
    assert not pipeline.is_fitted


def test_voting_direct_load_type_and_transactional_cleanup():
    params = {
        'task': 'classification', 'voting': 'soft',
        'weights': [0.5, 0.5],
        'estimatorNames': ['first', 'second'], 'classes': [0, 1],
    }
    wrong_type = encode_bundle({
        'typeId': 'wlearn.test.not-voting@1', 'params': params,
    }, [])
    with pytest.raises(ValidationError, match='expected typeId'):
        VotingEnsemble.load(wrong_type)
    _assert_transactional_cleanup(
        VotingEnsemble.load,
        'wlearn.ensemble.voting.classifier@1',
        {'params': params},
        params['estimatorNames'],
        'py-voting-transaction',
    )


def test_bagging_direct_load_type_and_transactional_cleanup():
    params = {
        'task': 'classification', 'kFold': 2, 'nRepeats': 1, 'seed': 42,
        'estimatorName': 'base', 'classes': [0, 1],
        'nClasses': 2, 'nSamples': 2,
    }
    wrong_type = encode_bundle({
        'typeId': 'wlearn.test.not-bagging@1', 'params': params,
    }, [])
    with pytest.raises(ValidationError, match='expected typeId'):
        BaggedEstimator.load(wrong_type)
    _assert_transactional_cleanup(
        BaggedEstimator.load,
        'wlearn.ensemble.bagged.classifier@1',
        {'params': params},
        ['fold_0', 'fold_1'],
        'py-bagging-transaction',
    )


def test_legacy_bagging_inference_rejects_unavailable_oof_evidence():
    child_type_id = 'wlearn.test.py-legacy-bag-child@1'
    state = {'meta_creates': 0}

    class LoadedModel:
        classes = np.array([0, 1], dtype=np.int32)
        capabilities = {'predictProba': True}

        def predict_proba(self, X):
            return np.tile([0.75, 0.25], len(X))

        def save(self):
            return _child_bundle(child_type_id)

        def dispose(self):
            pass

    register(child_type_id, lambda manifest, toc, blobs: LoadedModel())
    legacy = encode_bundle({
        'typeId': 'wlearn.ensemble.bagged.classifier@1',
        'params': {
            'task': 'classification', 'kFold': 2, 'nRepeats': 1,
            'seed': 42, 'estimatorName': 'base', 'classes': [0, 1],
            'nClasses': 2, 'nSamples': 4,
        },
    }, [
        _nested_artifact('fold_0', child_type_id),
        _nested_artifact('fold_1', child_type_id),
    ])
    bag = BaggedEstimator.load(legacy)
    X = np.arange(4, dtype=np.float64).reshape(4, 1)
    y = np.array([0, 0, 1, 1], dtype=np.int32)

    assert bag.is_fitted
    np.testing.assert_allclose(
        bag.predict_proba(X), np.tile([0.75, 0.25], 4))
    with pytest.raises(ValidationError, match='does not include stored OOF'):
        _ = bag.oof_predictions
    with pytest.raises(ValidationError, match='does not include stored OOF'):
        bag.save()

    class NeverCreatedMeta:
        @classmethod
        def create(cls, params=None):
            state['meta_creates'] += 1
            return cls()

    stacking = StackingEnsemble.create(
        estimators=[('legacy', bag)],
        final_estimator=('meta', NeverCreatedMeta, {}),
        cv=2,
        task='classification',
    )
    with pytest.raises(ValidationError, match='does not include stored OOF'):
        stacking.fit(X, y)
    assert state['meta_creates'] == 0
    assert bag.is_fitted
    stacking.dispose()
    bag.dispose()


def test_stacking_direct_load_type_and_transactional_cleanup():
    params = {
        'task': 'classification', 'cv': 2, 'passthrough': False,
        'seed': 42, 'estimatorNames': ['base'], 'metaName': 'meta',
        'classes': [0, 1], 'nMetaCols': 2,
    }
    wrong_type = encode_bundle({
        'typeId': 'wlearn.test.not-stacking@1', 'params': params,
    }, [])
    with pytest.raises(ValidationError, match='expected typeId'):
        StackingEnsemble.load(wrong_type)
    _assert_transactional_cleanup(
        StackingEnsemble.load,
        'wlearn.ensemble.stacking.classifier@1',
        {'params': params},
        ['base', 'meta'],
        'py-stacking-transaction',
    )


def test_ensemble_loaders_forward_one_recursive_context():
    options = {
        'wlearn.preprocess.tabular@1': {
            'limits': {'max_apply_rows': 2},
        },
    }
    _assert_context_forwarding(
        lambda data, loader_options: VotingEnsemble.load(
            data, loader_options=loader_options),
        'wlearn.ensemble.voting.classifier@1',
        {'params': {
            'task': 'classification', 'voting': 'soft',
            'weights': [0.5, 0.5],
            'estimatorNames': ['first', 'second'], 'classes': [0, 1],
        }},
        ['first', 'second'],
        'py-voting-context',
    )
    _assert_context_forwarding(
        lambda data, loader_options: BaggedEstimator.load(
            data, loader_options=loader_options),
        'wlearn.ensemble.bagged.classifier@1',
        {'params': {
            'task': 'classification', 'kFold': 2, 'nRepeats': 1,
            'seed': 42, 'estimatorName': 'base', 'classes': [0, 1],
            'nClasses': 2, 'nSamples': 2,
        }},
        ['fold_0', 'fold_1'],
        'py-bagging-context',
    )
    _assert_context_forwarding(
        lambda data, loader_options: StackingEnsemble.load(
            data, loader_options=loader_options),
        'wlearn.ensemble.stacking.classifier@1',
        {'params': {
            'task': 'classification', 'cv': 2, 'passthrough': False,
            'seed': 42, 'estimatorNames': ['base'], 'metaName': 'meta',
            'classes': [0, 1], 'nMetaCols': 2,
        }},
        ['base', 'meta'],
        'py-stacking-context',
    )


def test_ensemble_loaders_reject_semantically_inconsistent_manifests():
    child_type_id = 'wlearn.test.py-ensemble-semantic-child@1'
    state = {'dispatched': 0}

    def child_loader(manifest, toc, blobs):
        state['dispatched'] += 1
        return object()

    register(child_type_id, child_loader)
    child = lambda artifact_id: _nested_artifact(artifact_id, child_type_id)

    voting = encode_bundle({
        'typeId': 'wlearn.ensemble.voting.classifier@1',
        'params': {
            'task': 'classification', 'voting': 'soft', 'weights': [1],
            'estimatorNames': ['first', 'second'], 'classes': [0, 1],
        },
    }, [child('first'), child('second')])
    with pytest.raises(ValidationError, match='weights.*estimatorNames'):
        VotingEnsemble.load(voting)

    stacking = encode_bundle({
        'typeId': 'wlearn.ensemble.stacking.classifier@1',
        'params': {
            'task': 'classification', 'cv': 2, 'passthrough': False,
            'seed': 42, 'estimatorNames': ['base'], 'metaName': 'base',
            'classes': [0, 1], 'nMetaCols': 2,
        },
    }, [child('base')])
    with pytest.raises(ValidationError, match='metaName must differ'):
        StackingEnsemble.load(stacking)

    bagging = encode_bundle({
        'typeId': 'wlearn.ensemble.bagged.classifier@1',
        'params': {
            'task': 'classification', 'kFold': 2, 'nRepeats': 1,
            'seed': 42, 'estimatorName': 'base', 'classes': [0, 1],
            'nClasses': 2, 'nSamples': 2,
        },
    }, [
        child('fold_0'), child('fold_1'),
        {'id': 'oof', 'data': bytes(8),
         'mediaType': 'application/octet-stream'},
    ])
    with pytest.raises(ValidationError, match='OOF artifact length'):
        BaggedEstimator.load(bagging)

    boolean_count = encode_bundle({
        'typeId': 'wlearn.ensemble.bagged.classifier@1',
        'params': {
            'task': 'classification', 'kFold': 2, 'nRepeats': 1,
            'seed': 42, 'estimatorName': 'base', 'classes': [0],
            'nClasses': True, 'nSamples': 2,
        },
    }, [child('fold_0'), child('fold_1')])
    with pytest.raises(ValidationError, match='nClasses.*safe integer'):
        BaggedEstimator.load(boolean_count)

    oversized_bagging = encode_bundle({
        'typeId': 'wlearn.ensemble.bagged.classifier@1',
        'params': {
            'task': 'classification', 'kFold': 100000000,
            'nRepeats': 1, 'seed': 42, 'estimatorName': 'base',
            'classes': [0, 1], 'nClasses': 2, 'nSamples': 2,
        },
    }, [])
    with pytest.raises(ValidationError, match='artifact count'):
        BaggedEstimator.load(oversized_bagging)
    assert state['dispatched'] == 0
