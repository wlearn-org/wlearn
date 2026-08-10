# wlearn -- portable ML computation primitives

__version__ = '0.2.0'

from .errors import (
    WlearnError, BundleError, RegistryError,
    ValidationError, NotFittedError, DisposedError,
    ResourceLimitError, CancelledError, BackendError,
)
from .bundle import (
    encode_bundle, decode_bundle, validate_bundle,
    read_bundle_input, write_bundle_output,
)
from .registry import register, load, get_registry, assert_required_loaders
from .pipeline import Pipeline
from .preprocess import Preprocessor
from .scalers import StandardScaler, MinMaxScaler
from . import automl
from . import ensemble
from . import stochtree
from . import xlearn
from . import rf
