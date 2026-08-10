class WlearnError(Exception):
    """Base error for all wlearn exceptions."""

    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code or 'ERR_WLEARN'


class BundleError(WlearnError):
    def __init__(self, message):
        super().__init__(message, 'ERR_BUNDLE')


class RegistryError(WlearnError):
    def __init__(self, message):
        super().__init__(message, 'ERR_REGISTRY')


class ValidationError(WlearnError):
    def __init__(self, message):
        super().__init__(message, 'ERR_VALIDATION')


class NotFittedError(WlearnError):
    def __init__(self, message='Model is not fitted. Call fit() first.'):
        super().__init__(message, 'ERR_NOT_FITTED')


class DisposedError(WlearnError):
    def __init__(self, message='Model has been disposed.'):
        super().__init__(message, 'ERR_DISPOSED')


class ResourceLimitError(WlearnError):
    def __init__(self, message='Resource limit exceeded.'):
        super().__init__(message, 'ERR_RESOURCE_LIMIT')


class CancelledError(WlearnError):
    def __init__(self, message='Operation cancelled.'):
        super().__init__(message, 'ERR_CANCELLED')


class BackendError(WlearnError):
    def __init__(self, message='Backend operation failed.'):
        super().__init__(message, 'ERR_BACKEND')
