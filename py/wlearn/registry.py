from .errors import RegistryError
from .bundle import decode_bundle

_registry = {}


def register(type_id, loader_fn):
    """Register a loader function for a typeId.

    Args:
        type_id: string like 'wlearn.liblinear.classifier@1'
        loader_fn: callable(manifest, toc, blobs) -> model
    """
    if not isinstance(type_id, str) or '@' not in type_id:
        raise RegistryError(
            f'Invalid typeId "{type_id}": must contain "@" '
            f'(e.g. "wlearn.liblinear.classifier@1")')
    if not callable(loader_fn):
        raise RegistryError('loader_fn must be callable')
    _registry[type_id] = loader_fn


def load(data):
    """Decode a bundle and dispatch to the registered loader.

    Args:
        data: bytes

    Returns:
        model instance
    """
    manifest, toc, blobs = decode_bundle(data)
    type_id = manifest.get('typeId')

    if not type_id:
        raise RegistryError('Bundle manifest missing typeId')

    loader_fn = _registry.get(type_id)
    if loader_fn is None:
        available = list(_registry.keys())
        if available:
            lst = f'Registered loaders: {", ".join(available)}'
        else:
            lst = 'No loaders registered'
        raise RegistryError(
            f'No loader registered for typeId "{type_id}". {lst}. '
            f'Install the corresponding wlearn model package and import it '
            f'to register the loader.')

    return loader_fn(manifest, toc, blobs)


def get_registry():
    """Return a copy of the registry dict."""
    return dict(_registry)
