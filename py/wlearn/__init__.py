# wlearn -- portable ML computation primitives

__version__ = '0.3.0'

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
from .targets import normalize_targets, target_rows
from .pipeline import Pipeline
from .preprocess import Preprocessor, resolve_preprocess_config
from .scalers import StandardScaler, MinMaxScaler
from .task import (
    TASK_KINDS, FeatureDef, FeatureSchema, Task,
    infer_task_kind, create_feature_schema, validate_feature_schema,
    validate_row_roles, create_task, validate_task, task_rows,
)
from .prediction import (
    PREDICTION_FIELDS, Prediction,
    create_prediction, validate_prediction, prediction_rows, prediction_field,
)
from .measure import (
    MEASURE_DIRECTIONS, MEASURE_RESPONSES, MeasureDef,
    define_measure, register_measure, get_measure_def, list_measures,
    evaluate_measure, evaluate_metric_set, aggregate_measure,
    mean_aggregator, register_builtin_measures,
)
from .resampling import (
    RESAMPLING_STRATEGIES, ResamplingFold, ResamplingPlan,
    create_resampling_plan, validate_resampling_plan,
    serialize_resampling_plan, deserialize_resampling_plan,
    group_k_fold, time_series_split, sliding_window_split,
    sliding_index_split, sliding_period_split,
)
from .archive import (
    TRIAL_STATUSES, TrialError, TrialRecord, LeaderboardRow, Archive,
    create_trial_record, validate_trial_record, normalize_trial_error,
)
from . import automl
from . import ensemble
