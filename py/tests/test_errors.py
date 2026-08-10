from wlearn import BackendError, CancelledError, ResourceLimitError


def test_backend_facing_error_families_have_stable_codes():
    assert [
        ResourceLimitError().code,
        CancelledError().code,
        BackendError().code,
    ] == [
        'ERR_RESOURCE_LIMIT',
        'ERR_CANCELLED',
        'ERR_BACKEND',
    ]
