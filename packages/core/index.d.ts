import type {
  ArchiveJSON,
  BundleManifest,
  BundleTOCEntry,
  Capabilities,
  ConfusionMatrixResult,
  CrossValScoreOpts,
  CVFold,
  DataProvenance,
  DenseMatrix,
  Estimator,
  EstimatorClass,
  FeatureSchema,
  Labels,
  Matrix,
  MeasureDef,
  MeasureDirection,
  MetricOpts,
  MaybePromise,
  PipelineStep,
  Prediction,
  ResamplingFold,
  ResamplingPlan,
  RngFn,
  RowRoles,
  ScoringFn,
  ScoringName,
  SerializedResamplingPlan,
  SlidingResamplingOpts,
  TargetSchema,
  Task,
  TaskKind,
  TrialError,
  TrialRecord,
  Transformer
} from '@wlearn/types'

export type * from '@wlearn/types'

export {
  TASK_KINDS,
  PREDICTION_FIELDS,
  MEASURE_DIRECTIONS,
  MEASURE_RESPONSES,
  RESAMPLING_STRATEGIES,
  TRIAL_STATUSES
} from '@wlearn/types'

export declare class WlearnError extends Error {
  readonly code: string
  readonly engine?: string
  readonly engineCode?: number | null
  readonly cause?: unknown
}
export declare class BundleError extends WlearnError {}
export declare class RegistryError extends WlearnError {}
export declare class ValidationError extends WlearnError {}
export declare class NotFittedError extends WlearnError {}
export declare class DisposedError extends WlearnError {}
export declare class ResourceLimitError extends WlearnError {}
export declare class CancelledError extends WlearnError {}
export declare class BackendError extends WlearnError {}

export declare class Pipeline implements Estimator {
  constructor(
    steps: PipelineStep[],
    options?: { provenance?: Record<string, unknown> | null }
  )
  fit(X: Matrix | number[][], y: Labels | number[]): MaybePromise<this>
  predict(X: Matrix | number[][]): MaybePromise<Labels>
  predictProba(X: Matrix | number[][]): MaybePromise<Float64Array>
  score(X: Matrix | number[][], y: Labels | number[]): MaybePromise<number>
  save(): Uint8Array
  dispose(): void
  getParams(): Record<string, unknown>
  setParams(p: Record<string, unknown>): this
  readonly capabilities: Capabilities
  readonly classes: Int32Array | null
  readonly isFitted: boolean
  readonly provenance: Record<string, unknown> | null
  static load(
    bytes: Uint8Array,
    options?: { loaderOptions?: Record<string, unknown> }
  ): Promise<Pipeline>
}

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

export declare function isPromiseLike(x: unknown): x is PromiseLike<unknown>
export declare function lift<T, U>(
  x: MaybePromise<T>,
  f: (value: T) => U
): MaybePromise<U>

export declare function makeLCG(seed?: number): RngFn
export declare function shuffle<
  T extends ArrayLike<number> & { [i: number]: number }
>(arr: T, rng: RngFn): T

export declare function accuracy(
  yTrue: Labels,
  yPred: Labels,
  opts?: MetricOpts
): number
export declare function r2Score(
  yTrue: Labels,
  yPred: Labels,
  opts?: MetricOpts
): number
export declare function meanSquaredError(
  yTrue: Labels,
  yPred: Labels,
  opts?: MetricOpts
): number
export declare function meanAbsoluteError(
  yTrue: Labels,
  yPred: Labels,
  opts?: MetricOpts
): number
export declare function confusionMatrix(
  yTrue: Labels,
  yPred: Labels,
  opts?: MetricOpts
): ConfusionMatrixResult
export declare function precisionScore(
  yTrue: Labels,
  yPred: Labels,
  opts?: MetricOpts
): number
export declare function recallScore(
  yTrue: Labels,
  yPred: Labels,
  opts?: MetricOpts
): number
export declare function f1Score(
  yTrue: Labels,
  yPred: Labels,
  opts?: MetricOpts
): number
export declare function logLoss(
  yTrue: Labels,
  yProba: Float64Array,
  opts?: MetricOpts & { nClasses?: number; n_classes?: number; eps?: number }
): number
export declare function rocAuc(
  yTrue: Labels,
  yScore: Float64Array,
  opts?: MetricOpts & {
    multiClass?: 'raise' | 'ovr' | 'ovo'
    multi_class?: 'raise' | 'ovr' | 'ovo'
  }
): number

export declare function kFold(
  n: number,
  k?: number,
  opts?: { shuffle?: boolean; seed?: number }
): CVFold[]
export declare function stratifiedKFold(
  y: Labels,
  k?: number,
  opts?: { shuffle?: boolean; seed?: number }
): CVFold[]
export declare function trainTestSplit(
  n: number,
  opts?: { testSize?: number; shuffle?: boolean; seed?: number }
): CVFold
export declare function getScorer(scoring: ScoringName | ScoringFn): ScoringFn
export declare function crossValScore(
  EstimatorClass: EstimatorClass,
  X: Matrix | number[][],
  y: Labels | number[],
  opts?: CrossValScoreOpts
): Promise<Float64Array>

export declare function inferTaskKind(y?: Labels | number[] | null): TaskKind
export declare function createFeatureSchema(
  X: Matrix | number[][],
  opts?: {
    names?: string[]
    types?: string[]
    roles?: string[]
    metadata?: Record<string, unknown>
  }
): FeatureSchema
export declare function validateFeatureSchema(
  schema: FeatureSchema,
  cols?: number,
  rows?: number
): FeatureSchema
export declare function validateRowRoles(rowRoles: RowRoles, rows: number): RowRoles
export declare function createTask(opts: {
  id?: string
  kind?: TaskKind
  X: Matrix | number[][]
  y?: Labels | number[]
  featureSchema?: FeatureSchema
  targetSchema?: TargetSchema
  rowIds?: Int32Array | string[]
  groups?: Labels | number[]
  weights?: Labels | number[]
  rowRoles?: RowRoles
  provenance?: DataProvenance
  metadata?: Record<string, unknown>
}): Task
export declare function validateTask(task: Task): Task
export declare function taskRows(task: Task): number

export declare function createPrediction(opts?: Prediction): Prediction
export declare function validatePrediction(prediction: Prediction): Prediction
export declare function predictionRows(prediction: Prediction): number
export declare function predictionField(prediction: Prediction, field: string): unknown

export declare function defineMeasure(def: MeasureDef): MeasureDef
export declare function registerMeasure(def: MeasureDef): MeasureDef
export declare function getMeasureDef(id: string): MeasureDef
export declare function listMeasures(): string[]
export declare function evaluateMeasure(
  measureOrId: string | MeasureDef,
  prediction: Prediction,
  opts?: Record<string, unknown>
): number
export declare function aggregateMeasure(
  measureOrId: string | MeasureDef,
  values: ArrayLike<number>
): number
export declare function evaluateMetricSet(
  measures: (string | MeasureDef)[],
  prediction: Prediction,
  opts?: Record<string, unknown>
): Record<string, number>
export declare function meanAggregator(values: ArrayLike<number>): number
export declare function registerBuiltinMeasures(): void

export declare function createResamplingPlan(opts?: {
  id?: string
  strategy?: string
  n?: number
  y?: Labels | number[]
  groups?: Labels | number[]
  k?: number
  repeats?: number
  testSize?: number
  initialWindow?: number
  horizon?: number
  lookback?: number
  assessStart?: number
  assessStop?: number
  complete?: boolean
  index?: ArrayLike<number | string | Date>
  period?: 'day' | 'week' | 'month' | 'quarter' | 'year' | number
  skip?: number
  step?: number
  shuffle?: boolean
  seed?: number
  folds?: ResamplingFold[]
  taskId?: string
  metadata?: Record<string, unknown>
}): ResamplingPlan
export declare function validateResamplingPlan(plan: ResamplingPlan): ResamplingPlan
export declare function serializeResamplingPlan(
  plan: ResamplingPlan
): SerializedResamplingPlan
export declare function deserializeResamplingPlan(
  plan: SerializedResamplingPlan
): ResamplingPlan
export declare function groupKFold(
  groups: Labels,
  k?: number,
  opts?: { shuffle?: boolean; seed?: number }
): ResamplingFold[]
export declare function timeSeriesSplit(
  n: number,
  opts?: { initialWindow?: number; horizon?: number; step?: number }
): ResamplingFold[]
export declare function slidingWindowSplit(
  n: number,
  opts?: SlidingResamplingOpts
): ResamplingFold[]
export declare function slidingIndexSplit(
  index: ArrayLike<number | string | Date>,
  opts?: SlidingResamplingOpts
): ResamplingFold[]
export declare function slidingPeriodSplit(
  index: ArrayLike<number | string | Date>,
  opts?: SlidingResamplingOpts & {
    period?: 'day' | 'week' | 'month' | 'quarter' | 'year' | number
  }
): ResamplingFold[]

export declare class Archive {
  constructor(opts?: {
    id?: string
    taskId?: string
    measures?: string[]
    primaryMeasure?: string
    direction?: MeasureDirection
    metadata?: Record<string, unknown>
    records?: TrialRecord[]
  })
  id: string
  taskId?: string
  measures: string[]
  primaryMeasure?: string
  direction: MeasureDirection
  metadata: Record<string, unknown>
  readonly size: number
  add(record: Partial<TrialRecord>): TrialRecord
  start(record?: Partial<TrialRecord>): TrialRecord
  finish(trialId: string, patch?: Partial<TrialRecord>): TrialRecord
  fail(
    record: Partial<TrialRecord>,
    error: Error | TrialError | string,
    phase?: string
  ): TrialRecord
  update(trialId: string, patch: Partial<TrialRecord>): TrialRecord
  records(filter?: Record<string, unknown>): TrialRecord[]
  leaderboard(opts?: {
    metric?: string
    direction?: MeasureDirection
  }): import('@wlearn/types').LeaderboardRow[]
  toJSON(): ArchiveJSON
  static fromJSON(json: ArchiveJSON): Archive
}
export declare function createTrialRecord(
  record?: Partial<TrialRecord>
): TrialRecord
export declare function validateTrialRecord(record: TrialRecord): TrialRecord
export declare function normalizeTrialError(
  error: Error | TrialError | string,
  phase?: string
): TrialError | undefined

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
