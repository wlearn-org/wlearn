from types import MappingProxyType

from .errors import RegistryError
from .bundle import validate_bundle

_registry = {}


def register(type_id, loader_fn, *, accepts_context=False):
    """Register a loader function for a typeId.

    Args:
        type_id: string like 'wlearn.liblinear.classifier@1'
        loader_fn: callable(manifest, toc, blobs[, context]) -> model
        accepts_context: whether loader_fn accepts the immutable fourth argument
    """
    if not isinstance(type_id, str) or '@' not in type_id:
        raise RegistryError(
            f'Invalid typeId "{type_id}": must contain "@" '
            f'(e.g. "wlearn.liblinear.classifier@1")')
    if not callable(loader_fn):
        raise RegistryError('loader_fn must be callable')
    if not isinstance(accepts_context, bool):
        raise RegistryError('accepts_context must be a boolean')
    _registry[type_id] = {
        'loader_fn': loader_fn,
        'accepts_context': accepts_context,
    }


def load(data, *, loader_options=None):
    """Decode a bundle and dispatch to the registered loader.

    Args:
        data: bytes, memoryview, str path, or PathLike path

    Returns:
        model instance
    """
    return _load_with_context(data, _make_load_context(loader_options))


def _load_with_context(data, context):
    """Dispatch with an already-normalized context for recursive loaders."""
    manifest, toc, blobs = validate_bundle(data)
    type_id = manifest.get('typeId')

    if not type_id:
        raise RegistryError('Bundle manifest missing typeId')

    registration = _registry.get(type_id)
    if registration is None:
        available = list(_registry.keys())
        if available:
            lst = f'Registered loaders: {", ".join(available)}'
        else:
            lst = 'No loaders registered'
        guidance = (
            'Install wlearn[preprocess] and import wlearn.preprocess before '
            'loading.' if type_id == 'wlearn.preprocess.tabular@1' else
            'Install the corresponding wlearn model package and import it '
            'to register the loader.')
        raise RegistryError(
            f'No loader registered for typeId "{type_id}". {lst}. '
            f'{guidance}')

    assert_required_loaders(manifest)
    loader_fn = registration['loader_fn']
    if registration['accepts_context']:
        return loader_fn(manifest, toc, blobs, context)
    return loader_fn(manifest, toc, blobs)


def assert_required_loaders(manifest):
    """Fail before nested dispatch when a declared loader is unavailable."""
    for required_type_id in manifest.get('requires', []):
        if required_type_id not in _registry:
            guidance = (
                'Install wlearn[preprocess] and import wlearn.preprocess '
                'before loading.'
                if required_type_id == 'wlearn.preprocess.tabular@1' else
                'Install and import the corresponding wlearn model package '
                'before loading this bundle.')
            raise RegistryError(
                f'Missing required loader for nested typeId '
                f'"{required_type_id}". {guidance}')


def get_registry():
    """Return a copy of the registry dict."""
    return {
        type_id: registration['loader_fn']
        for type_id, registration in _registry.items()
    }


def _make_load_context(loader_options):
    if loader_options is None:
        loader_options = {}
    if not isinstance(loader_options, dict):
        raise RegistryError('loader_options must be a dict keyed by typeId')
    options = MappingProxyType(dict(loader_options))
    return MappingProxyType({'loaderOptions': options})
