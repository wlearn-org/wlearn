// errors
const {
  WlearnError,
  BundleError,
  RegistryError,
  ValidationError,
  NotFittedError,
  DisposedError,
  ResourceLimitError,
  CancelledError,
  BackendError
} = require('./errors.js')

// matrix
const { normalizeX, normalizeY, makeDense, validateMatrix } = require('./matrix.js')

// hash
const { sha256Sync } = require('./hash.js')

// bundle
const {
  DEFAULT_BUNDLE_LIMITS, encodeBundle, decodeBundle, validateBundle, encodeJSON, decodeJSON
} = require('./bundle.js')

// registry
const {
  register, load, loadSync, getRegistry, assertRequiredLoaders
} = require('./registry.js')

// pipeline
const { Pipeline } = require('./pipeline.js')
const { Step } = require('./step.js')

// preprocessing
const { Preprocessor } = require('./preprocess.js')
const { StandardScaler, MinMaxScaler } = require('./scalers.js')

// rng
const { makeLCG, shuffle } = require('./rng.js')

// metrics
const {
  accuracy, r2Score, meanSquaredError, meanAbsoluteError,
  confusionMatrix, precisionScore, recallScore, f1Score,
  logLoss, rocAuc
} = require('./metrics.js')

// cross-validation
const { kFold, stratifiedKFold, trainTestSplit, crossValScore, getScorer } = require('./cv.js')

// lift (MaybePromise utilities)
const { isPromiseLike, lift } = require('./lift.js')

// model wrapper
const { createModelClass, detectTask } = require('./model.js')

// ecosystem primitives
const {
  TASK_KINDS, inferTaskKind, createFeatureSchema, validateFeatureSchema,
  validateRowRoles, createTask, validateTask, taskRows
} = require('./task.js')
const {
  PREDICTION_FIELDS, createPrediction, validatePrediction,
  predictionRows, predictionField
} = require('./prediction.js')
const {
  MEASURE_DIRECTIONS, MEASURE_RESPONSES, defineMeasure, registerMeasure,
  getMeasureDef, listMeasures, evaluateMeasure, aggregateMeasure,
  evaluateMetricSet, meanAggregator, registerBuiltinMeasures
} = require('./measure.js')
const {
  RESAMPLING_STRATEGIES, createResamplingPlan, validateResamplingPlan,
  serializeResamplingPlan, deserializeResamplingPlan, groupKFold, timeSeriesSplit,
  slidingWindowSplit, slidingIndexSplit, slidingPeriodSplit
} = require('./resampling.js')
const {
  TRIAL_STATUSES, Archive, createTrialRecord, validateTrialRecord,
  normalizeTrialError
} = require('./archive.js')

module.exports = {
  // errors
  WlearnError, BundleError, RegistryError, ValidationError, NotFittedError, DisposedError,
  ResourceLimitError, CancelledError, BackendError,
  // matrix
  normalizeX, normalizeY, makeDense, validateMatrix,
  // hash
  sha256Sync,
  // bundle
  DEFAULT_BUNDLE_LIMITS, encodeBundle, decodeBundle, validateBundle, encodeJSON, decodeJSON,
  // registry
  register, load, loadSync, getRegistry, assertRequiredLoaders,
  // pipeline
  Pipeline, Step,
  // preprocessing
  Preprocessor, StandardScaler, MinMaxScaler,
  // rng
  makeLCG, shuffle,
  // metrics
  accuracy, r2Score, meanSquaredError, meanAbsoluteError,
  confusionMatrix, precisionScore, recallScore, f1Score,
  logLoss, rocAuc,
  // cross-validation
  kFold, stratifiedKFold, trainTestSplit, crossValScore, getScorer,
  // lift
  isPromiseLike, lift,
  // model
  createModelClass, detectTask,
  // task
  TASK_KINDS, inferTaskKind, createFeatureSchema, validateFeatureSchema,
  validateRowRoles, createTask, validateTask, taskRows,
  // prediction
  PREDICTION_FIELDS, createPrediction, validatePrediction, predictionRows, predictionField,
  // measure
  MEASURE_DIRECTIONS, MEASURE_RESPONSES, defineMeasure, registerMeasure,
  getMeasureDef, listMeasures, evaluateMeasure, aggregateMeasure,
  evaluateMetricSet, meanAggregator, registerBuiltinMeasures,
  // resampling
  RESAMPLING_STRATEGIES, createResamplingPlan, validateResamplingPlan,
  serializeResamplingPlan, deserializeResamplingPlan, groupKFold, timeSeriesSplit,
  slidingWindowSplit, slidingIndexSplit, slidingPeriodSplit,
  // archive
  TRIAL_STATUSES, Archive, createTrialRecord, validateTrialRecord, normalizeTrialError
}
