const { test } = require('node:test')
const assert = require('node:assert/strict')

const core = require('..')

const EXPECTED_EXPORTS = [
  'Archive', 'BackendError', 'BundleError', 'CancelledError',
  'DEFAULT_BUNDLE_LIMITS', 'DisposedError', 'MEASURE_DIRECTIONS',
  'MEASURE_RESPONSES', 'MinMaxScaler', 'NotFittedError', 'PREDICTION_FIELDS',
  'Pipeline', 'Preprocessor', 'RESAMPLING_STRATEGIES', 'RegistryError',
  'ResourceLimitError', 'StandardScaler', 'Step', 'TASK_KINDS',
  'TRIAL_STATUSES', 'ValidationError', 'WlearnError', 'accuracy',
  'aggregateMeasure', 'assertRequiredLoaders', 'confusionMatrix',
  'createFeatureSchema', 'createModelClass', 'createPrediction',
  'createResamplingPlan', 'createTask', 'createTrialRecord', 'crossValScore',
  'decodeBundle', 'decodeJSON', 'defineMeasure', 'deserializeResamplingPlan',
  'detectTask', 'encodeBundle', 'encodeJSON', 'evaluateMeasure',
  'evaluateMetricSet', 'f1Score', 'getMeasureDef', 'getRegistry', 'getScorer',
  'groupKFold', 'inferTaskKind', 'isPromiseLike', 'kFold', 'lift',
  'listMeasures', 'load', 'loadSync', 'logLoss', 'makeDense', 'makeLCG',
  'meanAbsoluteError', 'meanAggregator', 'meanSquaredError',
  'normalizeTrialError', 'normalizeX', 'normalizeY', 'precisionScore',
  'predictionField', 'predictionRows', 'r2Score', 'recallScore', 'register',
  'registerBuiltinMeasures', 'registerMeasure', 'rocAuc',
  'serializeResamplingPlan', 'sha256Sync', 'shuffle', 'slidingIndexSplit',
  'slidingPeriodSplit', 'slidingWindowSplit', 'stratifiedKFold', 'taskRows',
  'timeSeriesSplit', 'trainTestSplit', 'validateBundle',
  'validateFeatureSchema', 'validateMatrix', 'validatePrediction',
  'validateResamplingPlan', 'validateRowRoles', 'validateTask',
  'validateTrialRecord'
].sort()

test('@wlearn/core runtime export inventory is explicit', () => {
  assert.deepEqual(Object.keys(core).sort(), EXPECTED_EXPORTS)
})
