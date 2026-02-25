import pytest

from wlearn.registry import register, load, get_registry
from wlearn.bundle import encode_bundle
from wlearn.errors import RegistryError


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


def test_register_invalid_type_id():
    with pytest.raises(RegistryError, match='must contain "@"'):
        register('no-version', lambda m, t, b: None)


def test_register_invalid_loader():
    with pytest.raises(RegistryError, match='callable'):
        register('test.bad@1', 'not a function')


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
