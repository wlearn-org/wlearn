import pytest

from wlearn import ValidationError, resolve_preprocess_config


def test_resolves_without_importing_optional_tranfi():
    resolved = resolve_preprocess_config({
        'encode': 'label',
        'scale': 'standard',
    })
    assert resolved['encode'] == 'label'
    assert resolved['scale'] == 'standard'
    assert resolved['policyVersion'] == 1

    resolved['impute']['numeric'] = 'zero'
    assert resolve_preprocess_config({})['impute']['numeric'] == 'mean'

    with pytest.raises(ValidationError, match='maxCategories'):
        resolve_preprocess_config({'maxCategories': 1})
