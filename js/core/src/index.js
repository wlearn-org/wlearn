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
const { normalizeX, normalizeY, makeDense, validateMatrix, subsetRows, subsetLabels } = require('./matrix.js')

const { isTargetMatrix, normalizeTargets, targetRows, subsetTargets, validateSampleWeight } = require('./targets.js')

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
  taskParams, validateEstimatorTask,
  TASK_KINDS, inferTaskKind, createFeatureSchema, validateFeatureSchema,
  validateRowRoles, createTask, validateTask, taskRows
} = require('./task.js')
const {
  PREDICTION_FIELDS, createPrediction, validatePrediction,
  predictionRows, predictionField
} = require('./prediction.js')
const {
  scoreEstimator,
  MEASURE_DIRECTIONS, MEASURE_RESPONSES, defineMeasure, registerMeasure,
  getMeasureDef, listMeasures, evaluateMeasure, aggregateMeasure,
  evaluateMetricSet, meanAggregator, registerBuiltinMeasures
} = require('./measure.js')
const {
  resolveCv, serializeCv,
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
  normalizeX, normalizeY, makeDense, validateMatrix, subsetRows, subsetLabels,
  isTargetMatrix, normalizeTargets, targetRows, subsetTargets, validateSampleWeight,
  // hash
  sha256Sync,
  // bundle
  DEFAULT_BUNDLE_LIMITS, encodeBundle, decodeBundle, validateBundle, encodeJSON, decodeJSON,
  // registry
  register, load, loadSync, getRegistry, assertRequiredLoaders,
  // pipeline
  Pipeline, Step,
  // preprocessing
  StandardScaler, MinMaxScaler,
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
  taskParams, validateEstimatorTask,
  TASK_KINDS, inferTaskKind, createFeatureSchema, validateFeatureSchema,
  validateRowRoles, createTask, validateTask, taskRows,
  // prediction
  PREDICTION_FIELDS, createPrediction, validatePrediction, predictionRows, predictionField,
  // measure
  scoreEstimator,
  MEASURE_DIRECTIONS, MEASURE_RESPONSES, defineMeasure, registerMeasure,
  getMeasureDef, listMeasures, evaluateMeasure, aggregateMeasure,
  evaluateMetricSet, meanAggregator, registerBuiltinMeasures,
  // resampling
  resolveCv, serializeCv,
  RESAMPLING_STRATEGIES, createResamplingPlan, validateResamplingPlan,
  serializeResamplingPlan, deserializeResamplingPlan, groupKFold, timeSeriesSplit,
  slidingWindowSplit, slidingIndexSplit, slidingPeriodSplit,
  // archive
  TRIAL_STATUSES, Archive, createTrialRecord, validateTrialRecord, normalizeTrialError
}

// Match CommonJS's single-module identity across independently bundled browser
// entrypoints, including error constructors and custom Measure registrations.
// registry.js rejects a different core version before this API can be reused.
const CORE_API = Symbol.for('wlearn.core.api')
if (globalThis[CORE_API]) module.exports = globalThis[CORE_API]
else Object.defineProperty(globalThis, CORE_API, { value: module.exports })
