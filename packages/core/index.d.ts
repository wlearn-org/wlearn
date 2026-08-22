import type {
  BundleManifest,
  BundleTOCEntry,
  Capabilities,
  DenseMatrix,
  Estimator,
  EstimatorClass,
  Labels,
  Matrix,
  PipelineStep,
  Transformer
} from '@wlearn/types'

export type {
  ArchiveJSON,
  BundleArtifactDeclaration,
  BundleManifest,
  BundleTOCEntry,
  Capabilities,
  Classifier,
  CSRMatrix,
  DenseMatrix,
  Dtype,
  Estimator,
  EstimatorClass,
  Labels,
  Matrix,
  MaybePromise,
  PipelineStep,
  Prediction,
  SearchSpace,
  Task,
  TensorRef,
  Transformer
} from '@wlearn/types'

export {
  Archive,
  BackendError,
  BundleError,
  CancelledError,
  DisposedError,
  NotFittedError,
  Pipeline,
  RegistryError,
  ResourceLimitError,
  ValidationError,
  WlearnError,
  TASK_KINDS,
  PREDICTION_FIELDS,
  MEASURE_DIRECTIONS,
  MEASURE_RESPONSES,
  RESAMPLING_STRATEGIES,
  TRIAL_STATUSES,
  accuracy,
  aggregateMeasure,
  confusionMatrix,
  createFeatureSchema,
  createPrediction,
  createResamplingPlan,
  createTask,
  createTrialRecord,
  crossValScore,
  defineMeasure,
  deserializeResamplingPlan,
  evaluateMeasure,
  evaluateMetricSet,
  f1Score,
  getMeasureDef,
  getScorer,
  groupKFold,
  inferTaskKind,
  isPromiseLike,
  kFold,
  lift,
  listMeasures,
  logLoss,
  makeLCG,
  meanAbsoluteError,
  meanAggregator,
  meanSquaredError,
  normalizeTrialError,
  precisionScore,
  predictionField,
  predictionRows,
  r2Score,
  recallScore,
  registerBuiltinMeasures,
  registerMeasure,
  rocAuc,
  serializeResamplingPlan,
  shuffle,
  slidingIndexSplit,
  slidingPeriodSplit,
  slidingWindowSplit,
  stratifiedKFold,
  taskRows,
  timeSeriesSplit,
  trainTestSplit,
  validateFeatureSchema,
  validatePrediction,
  validateResamplingPlan,
  validateRowRoles,
  validateTask,
  validateTrialRecord
} from '@wlearn/types'

export interface BundleLimits {
  maxManifestBytes?: number
  maxTocBytes?: number
  maxArtifacts?: number
  maxArtifactBytes?: number
  maxBundleBytes?: number
  maxNestingDepth?: number
  maxDecodedBytes?: number
}

export interface BundleDecodeOptions extends BundleLimits {
  allowLegacyManifest?: boolean
}

export interface DecodedBundle {
  manifest: BundleManifest & Record<string, unknown>
  toc: BundleTOCEntry[]
  blobs: Uint8Array
}

export interface LoadOptions {
  loaderOptions?: Record<string, unknown>
}

export interface LoaderContext {
  readonly loaderOptions: Readonly<Record<string, unknown>>
}

export type RuntimeLoader<T = Estimator | Transformer> = (
  manifest: BundleManifest,
  toc: BundleTOCEntry[],
  blobs: Uint8Array,
  context?: LoaderContext
) => T | Promise<T>

export declare const DEFAULT_BUNDLE_LIMITS: Readonly<Required<BundleLimits>>

export declare function normalizeX(
  X: Matrix | number[][],
  coerce?: 'auto' | 'warn' | 'error'
): DenseMatrix
export declare function normalizeY(y: Labels | number[]): Labels
export declare function makeDense(
  data: Float32Array | Float64Array,
  rows: number,
  cols: number
): DenseMatrix
export declare function validateMatrix<T extends Matrix>(matrix: T): T
export declare function sha256Sync(bytes: Uint8Array | ArrayBuffer): string

export declare function encodeBundle(
  manifest: Record<string, unknown>,
  artifacts: Array<{ id: string; data: Uint8Array; mediaType?: string }>,
  options?: BundleLimits
): Uint8Array
export declare function decodeBundle(
  bytes: Uint8Array | ArrayBuffer,
  options?: BundleDecodeOptions
): DecodedBundle
export declare function validateBundle(
  bytes: Uint8Array | ArrayBuffer,
  options?: BundleDecodeOptions
): DecodedBundle
export declare function encodeJSON(value: unknown): Uint8Array
export declare function decodeJSON(bytes: Uint8Array | ArrayBuffer): unknown

export declare function register<T = Estimator | Transformer>(
  typeId: string,
  loader: RuntimeLoader<T>,
  options?: { acceptsContext?: boolean; sync?: boolean }
): void
export declare function load<T = Estimator | Transformer>(
  bytes: Uint8Array | ArrayBuffer,
  options?: LoadOptions
): Promise<T>
export declare function loadSync<T = Estimator | Transformer>(
  bytes: Uint8Array | ArrayBuffer,
  options?: LoadOptions
): T
export declare function getRegistry(): Map<string, RuntimeLoader>
export declare function assertRequiredLoaders(manifest: BundleManifest): void

export declare class Step {
  constructor(name: string, estimator: Estimator | Transformer)
  readonly name: string
  readonly estimator: Estimator | Transformer
  readonly isFitted: boolean
  readonly isTransformer: boolean
}

export interface LegacyPreprocessorConfig {
  impute?: 'auto' | 'mean' | 'median' | 'zero' | false
  encode?: 'auto' | 'onehot' | 'label' | false
  scale?: 'standard' | 'minmax' | false
  maxCategories?: number
  [key: string]: unknown
}

export declare class Preprocessor {
  constructor(config?: LegacyPreprocessorConfig)
  fit(X: Matrix | number[][], y?: Labels | number[]): this
  transform(X: Matrix | number[][]): DenseMatrix
  fitTransform(X: Matrix | number[][], y?: Labels | number[]): DenseMatrix
  getState(): Record<string, unknown>
  static fromState(state: Record<string, unknown>): Preprocessor
  getParams(): LegacyPreprocessorConfig
  setParams(params: LegacyPreprocessorConfig): this
  dispose(): void
  readonly isFitted: boolean
  readonly outputCols: number
  readonly capabilities: Readonly<{ transformer: true }>
}

export declare class StandardScaler implements Transformer {
  constructor(params?: Record<string, unknown>)
  fit(X: Matrix | number[][]): this
  transform(X: Matrix | number[][]): DenseMatrix
  fitTransform(X: Matrix | number[][]): DenseMatrix
  save(): Uint8Array
  dispose(): void
  getParams(): Record<string, unknown>
  setParams(params: Record<string, unknown>): this
  readonly capabilities: Readonly<{ transformer: true }>
  readonly isFitted: boolean
}

export declare class MinMaxScaler extends StandardScaler {}

export declare function detectTask(y: Labels | number[]): 'classification' | 'regression'
export declare function createModelClass(
  classifier: EstimatorClass,
  regressor: EstimatorClass,
  options?: { name?: string; load?: () => void | Promise<void> }
): EstimatorClass
