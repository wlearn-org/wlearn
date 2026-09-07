import {
  Pipeline,
  StandardScaler,
  decodeBundle,
  encodeBundle,
  load,
  loadSync,
  normalizeX,
  register
} from '@wlearn/core'
import * as core from '@wlearn/core'
import type { MaybePromise } from '@wlearn/types'

const matrix = normalizeX([[1], [2]])
const scaler = new StandardScaler().fit(matrix)
const pipeline = new Pipeline([['scale', scaler], ['model', scaler]])
const pipelineFit: MaybePromise<Pipeline> = pipeline.fit(
  matrix, new Int32Array([0, 1])
)
const pipelineClasses: Int32Array | null = pipeline.classes
const bytes = encodeBundle({ typeId: 'wlearn.test.types@1' }, [])
decodeBundle(bytes)
register('wlearn.test.types@1', () => scaler, { sync: true })
const sync = loadSync<typeof scaler>(bytes)
const pending: Promise<typeof scaler> = load<typeof scaler>(bytes)
const exhaustiveRuntimeNames = [
  'subsetRows', 'subsetLabels', 'taskParams', 'validateEstimatorTask', 'scoreEstimator', 'resolveCv', 'serializeCv',
  'Archive', 'BackendError', 'BundleError', 'CancelledError',
  'DEFAULT_BUNDLE_LIMITS', 'DisposedError', 'MEASURE_DIRECTIONS',
  'MEASURE_RESPONSES', 'MinMaxScaler', 'NotFittedError', 'PREDICTION_FIELDS',
  'Pipeline', 'RESAMPLING_STRATEGIES', 'RegistryError',
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
] as const satisfies readonly (keyof typeof core)[]
void pipeline
void pipelineFit
void pipelineClasses
void sync
void pending
void exhaustiveRuntimeNames

// Public scoring accepts a registered name, class order, and caller-owned folds.
const customScorer = core.getScorer('application_metric')
customScorer(new Int32Array([5, 2]), new Float64Array([.9, .1, .1, .9]), {
  classes: new Int32Array([5, 2])
})
const customFolds = core.resolveCv([{ train: [0], test: [1] }], new Float64Array([0, 1]))
core.serializeCv(customFolds)
core.subsetRows(matrix, [1, 0])
core.subsetLabels(new Int32Array([0, 1]), [1, 0])
const sparse = { data: new Float64Array([1, 2]), rows: 2, cols: 1,
  indices: new Int32Array([0, 0]), indptr: new Int32Array([0, 1, 2]) }
// @ts-expect-error Dense-only normalization must not accept CSR descriptors.
core.normalizeX(sparse)
