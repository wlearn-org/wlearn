import pytest

from wlearn.registry import register, load, get_registry, _load_with_context
from wlearn.bundle import encode_bundle
from wlearn.errors import BundleError, RegistryError


def _make_bundle(type_id):
    return encode_bundle(
        {'typeId': type_id},
        [{'id': 'model', 'data': b'\x01\x02\x03'}],
    )


def test_register_and_load():
    register('test.reg.basic@1', lambda m, t, b: {'loaded': True, 'typeId': m['typeId']})
    bundle = _make_bundle('test.reg.basic@1')
    result = load(bundle)
    assert result['loaded'] is True
    assert result['typeId'] == 'test.reg.basic@1'


def test_load_verifies_hash_before_dispatch():
    called = False

    def loader(manifest, toc, blobs):
        nonlocal called
        called = True

    register('test.reg.corrupt@1', loader)
    bundle = bytearray(_make_bundle('test.reg.corrupt@1'))
    bundle[-1] ^= 0xff
    with pytest.raises(BundleError, match='SHA-256 mismatch'):
        load(bytes(bundle))
    assert called is False


def test_load_fails_before_dispatch_when_required_loader_is_missing():
    called = False

    def loader(manifest, toc, blobs):
        nonlocal called
        called = True

    register('test.reg.requires@1', loader)
    bundle = encode_bundle({
        'typeId': 'test.reg.requires@1',
        'requires': ['test.reg.not-registered@1'],
    }, [])
    with pytest.raises(RegistryError, match='Missing required loader'):
        load(bundle)
    assert called is False


def test_register_invalid_type_id():
    with pytest.raises(RegistryError, match='must contain "@"'):
        register('no-version', lambda m, t, b: None)


def test_register_invalid_loader():
    with pytest.raises(RegistryError, match='callable'):
        register('test.bad@1', 'not a function')


def test_register_invalid_context_metadata():
    with pytest.raises(RegistryError, match='accepts_context'):
        register('test.bad-context@1', lambda m, t, b: None,
                 accepts_context='yes')


def test_load_context_is_opt_in_and_immutable():
    cancel_token = object()
    runtime_options = {
        'limits': {'maxPlanBytes': 123},
        'cancel_token': cancel_token,
    }
    received = []

    def context_loader(manifest, toc, blobs, context):
        received.append(context)
        return context['loaderOptions']['test.reg.context@1']

    register('test.reg.context@1', context_loader, accepts_context=True)
    result = load(
        _make_bundle('test.reg.context@1'),
        loader_options={'test.reg.context@1': runtime_options},
    )
    assert result is not runtime_options
    assert result['limits'] is not runtime_options['limits']
    assert result['limits']['maxPlanBytes'] == 123
    assert result['cancel_token'] is cancel_token
    runtime_options['limits']['maxPlanBytes'] = 1
    assert result['limits']['maxPlanBytes'] == 123
    with pytest.raises(TypeError):
        received[0]['new'] = 'value'
    with pytest.raises(TypeError):
        received[0]['loaderOptions']['new'] = 'value'

    def no_context_loader(*args):
        assert len(args) == 3
        return 'ok'

    register('test.reg.no-context@1', no_context_loader)
    assert load(
        _make_bundle('test.reg.no-context@1'),
        loader_options={'ignored': True},
    ) == 'ok'


def test_recursive_load_forwards_the_same_context():
    contexts = []
    inner_bundle = _make_bundle('test.reg.context-inner@1')

    def inner_loader(manifest, toc, blobs, context):
        contexts.append(context)
        return context['loaderOptions']['wlearn.preprocess.tabular@1']

    def outer_loader(manifest, toc, blobs, context):
        contexts.append(context)
        return _load_with_context(inner_bundle, context)

    register(
        'test.reg.context-inner@1', inner_loader, accepts_context=True)
    register(
        'test.reg.context-outer@1', outer_loader, accepts_context=True)
    runtime_options = {'maxPlanBytes': 123}
    result = load(
        _make_bundle('test.reg.context-outer@1'),
        loader_options={'wlearn.preprocess.tabular@1': runtime_options},
    )
    assert result is not runtime_options
    assert result['maxPlanBytes'] == 123
    assert contexts[0] is contexts[1]


def test_load_missing_type_id():
    bundle = _make_bundle('test.reg.nonexistent@99')
    with pytest.raises(RegistryError, match='No loader registered'):
        load(bundle)


def test_load_error_lists_available():
    register('test.reg.available@1', lambda m, t, b: None)
    bundle = _make_bundle('test.reg.missing@1')
    with pytest.raises(RegistryError, match='test.reg.available@1'):
        load(bundle)


def test_get_registry_returns_copy():
    register('test.reg.copy@1', lambda m, t, b: None)
    reg = get_registry()
    assert 'test.reg.copy@1' in reg
    # mutating the copy should not affect the real registry
    reg.clear()
    assert 'test.reg.copy@1' in get_registry()
