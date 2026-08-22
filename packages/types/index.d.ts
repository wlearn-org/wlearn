// wlearn bundle format constants
export declare const BUNDLE_MAGIC: Uint8Array
export declare const BUNDLE_VERSION: 1
export declare const HEADER_SIZE: 16
export declare const DTYPE: {
  readonly FLOAT32: 'float32'
  readonly FLOAT64: 'float64'
  readonly INT32: 'int32'
}

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

// Data types
export type Dtype = 'float32' | 'float64' | 'int32'

export interface DenseMatrix {
  dtype?: Dtype
  rows: number
  cols: number
  data: Float32Array | Float64Array
}

export interface CSRMatrix {
  dtype?: Dtype
  rows: number
  cols: number
  data: Float64Array
  indices: Int32Array
  indptr: Int32Array
}

export type Matrix = DenseMatrix | CSRMatrix
export type Labels = Int32Array | Float32Array | Float64Array

export interface TensorRef {
  location: 'host' | 'wasm'
  moduleId?: string
  buffer: ArrayBuffer | SharedArrayBuffer
  byteOffset: number
  byteLength: number
  dtype: Dtype
  shape: number[]
  strides?: number[]
}

// Estimator contract
export interface Capabilities {
  classifier: boolean
  regressor: boolean
  predictProba: boolean
  decisionFunction: boolean
  sampleWeight: boolean
  csr: boolean
  earlyStopping: boolean
  [key: string]: boolean
}

/**
 * MaybePromise<T> allows methods to return T (sync) or Promise<T> (async).
 * Sync WASM models always return T directly. Async models (ONNX, WebGPU) may return Promise<T>.
 */
export type MaybePromise<T> = T | Promise<T>

export interface Estimator {
  fit(X: Matrix | number[][], y: Labels | number[]): MaybePromise<this>
  predict(X: Matrix | number[][]): MaybePromise<Labels>
  score(X: Matrix | number[][], y: Labels | number[]): MaybePromise<number>
  save(): Uint8Array
  dispose(): void
  getParams(): Record<string, unknown>
  setParams(p: Record<string, unknown>): this
  readonly capabilities: Capabilities
  readonly isFitted: boolean
}

export interface Classifier extends Estimator {
  predictProba(X: Matrix | number[][]): MaybePromise<Float64Array>
  readonly classes: Int32Array
}

// Transformer contract
export interface TransformerCapabilities {
  readonly transformer: true
  readonly [key: string]: boolean
}

export interface Transformer {
  fit(X: Matrix | number[][], y?: Labels | number[]): this
  transform(X: Matrix | number[][]): DenseMatrix
  fitTransform(X: Matrix | number[][], y?: Labels | number[]): DenseMatrix
  save(): Uint8Array
  dispose(): void
  getParams(): Record<string, unknown>
  setParams(p: Record<string, unknown>): this
  readonly capabilities: TransformerCapabilities
  readonly isFitted: boolean
}

export type PipelineStep = [name: string, estimator: Estimator | Transformer]

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

export type PreprocessNumericImpute = false | 'mean' | 'median' | 'zero'
export type PreprocessCategoricalImpute = false | 'mode'
export type PreprocessEncode = false | 'onehot' | 'label'
export type PreprocessScale = false | 'standard' | 'minmax'
export type PreprocessUnknownCategory = null | 'error' | 'all_zero' | 'sentinel'
export type PreprocessAllMissing = null | 'error' | 'zero'

export interface PreprocessResolvedConfig {
  impute: {
    numeric: PreprocessNumericImpute
    categorical: PreprocessCategoricalImpute
  }
  encode: PreprocessEncode
  scale: PreprocessScale
  maxCategories: number
  unknownCategory: PreprocessUnknownCategory
  allMissing: PreprocessAllMissing
  maxOutputColumns: number
  maxOutputElements: number
  policyVersion: 1
}

export interface PreprocessConfig {
  impute?: 'auto' | 'mean' | 'median' | 'zero' | false
  encode?: 'auto' | 'onehot' | 'label' | false
  scale?: PreprocessScale
  maxCategories?: number
  unknownCategory?: Exclude<PreprocessUnknownCategory, null>
  allMissing?: Exclude<PreprocessAllMissing, null>
  maxOutputColumns?: number
  maxOutputElements?: number
}

// Search space IR (for AutoML)
export type SearchParam =
  | { type: 'categorical'; values: unknown[] }
  | { type: 'uniform'; low: number; high: number }
  | { type: 'log_uniform'; low: number; high: number }
  | { type: 'int_uniform'; low: number; high: number }
  | { type: 'int_log_uniform'; low: number; high: number }

export type SearchSpace = Record<string, SearchParam & { condition?: Record<string, unknown> }>

export interface PreprocessTemplate {
  templateId: string
  typeId: 'wlearn.preprocess.tabular@1'
  params?: PreprocessConfig | PreprocessResolvedConfig
}

export interface CandidateModel {
  displayName: string
  classId: string
  params: Record<string, unknown>
}

export interface CandidatePreprocess {
  templateId: string
  typeId: 'wlearn.preprocess.tabular@1'
  resolvedParams: PreprocessResolvedConfig
}

export interface CandidateTemplate {
  model: CandidateModel
  preprocess: CandidatePreprocess | null
}

// Bundle format v1
export interface BundleArtifactDeclaration {
  id: string
  length: number
  sha256: string
  mediaType: string
}

export interface BundleManifest {
  typeId: string
  bundleVersion: number
  requires: string[]
  artifacts: BundleArtifactDeclaration[]
  params: Record<string, unknown>
  seed?: number
  metadata?: Record<string, unknown>
}

export interface BundleTOCEntry {
  id: string
  offset: number
  length: number
  sha256: string
  mediaType?: string
}

// Pipeline graph (DAG IR for forward compat)
export interface PipelineNode {
  nodeId: string
  typeId: string
  params: Record<string, unknown>
}

export interface PipelineEdge {
  id: string
  from: { nodeId: string; port: string }
  to: { nodeId: string; port: string }
}

export interface PipelineGraph {
  nodes: PipelineNode[]
  edges: PipelineEdge[]
  endpoints: { predict?: string; transform?: string }
}

// Ecosystem primitives
export type TaskKind =
  | 'classification'
  | 'regression'
  | 'clustering'
  | 'ranking'
  | 'survival'
  | 'forecasting'
  | 'multioutput'
  | 'anomaly'

export interface FeatureDef {
  name: string
  index: number
  type: string
  role: string
  [key: string]: unknown
}

export interface FeatureSchema {
  rows: number
  cols: number
  features: FeatureDef[]
  metadata?: Record<string, unknown>
}

export interface TargetSchema {
  name?: string
  type?: string
  classes?: Labels | string[]
  positiveClass?: number | string
  metadata?: Record<string, unknown>
}

export interface DataProvenance {
  source?: string
  sha256?: string
  license?: string
  createdAt?: string
  metadata?: Record<string, unknown>
}

export interface RowRoles {
  train?: Int32Array
  test?: Int32Array
  validate?: Int32Array
  holdout?: Int32Array
  [key: string]: Int32Array | undefined
}

export interface Task {
  id: string
  kind: TaskKind
  X: Matrix
  y?: Labels
  featureSchema: FeatureSchema
  targetSchema?: TargetSchema
  rowIds?: Int32Array | string[]
  groups?: Labels
  weights?: Labels
  rowRoles?: RowRoles
  provenance?: DataProvenance
  metadata?: Record<string, unknown>
}

export interface Prediction {
  taskId?: string
  rowIds?: Int32Array | string[]
  truth?: Labels
  response?: Labels
  proba?: Float64Array
  probaRows?: number
  score?: Float64Array
  decision?: Float64Array
  interval?: Float64Array
  quantiles?: Float64Array
  classes?: Labels
  featureSchemaHash?: string
  modelArtifactHash?: string
  warnings?: unknown[]
  metadata?: Record<string, unknown>
}

export type MeasureDirection = 'maximize' | 'minimize'
export type MeasureResponse = 'response' | 'proba' | 'score' | 'decision' | 'distribution'

export interface MeasureContext {
  truth?: Labels
  response?: Labels
  proba?: Float64Array
  score?: Float64Array
  decision?: Float64Array
  prediction: Prediction
  opts?: Record<string, unknown>
}

export interface MeasureDef {
  id: string
  label?: string
  taskKinds: TaskKind[]
  direction: MeasureDirection
  response: MeasureResponse
  range?: [number, number]
  average?: 'binary' | 'micro' | 'macro' | 'weighted' | 'macro_weighted' | 'custom'
  naValue?: number
  requiresTruth?: boolean
  supportsSampleWeight?: boolean
  metadata?: Record<string, unknown>
  aggregator?: (values: ArrayLike<number>) => number
  fn: (ctx: MeasureContext) => number
}

export interface ResamplingFold {
  foldId: string
  train: Int32Array
  test: Int32Array
  validate?: Int32Array
  metadata?: Record<string, unknown>
}

export interface ResamplingPlan {
  id: string
  strategy: string
  n: number
  folds: ResamplingFold[]
  taskId?: string
  seed?: number
  constraints?: Record<string, boolean>
  metadata?: Record<string, unknown>
}

export interface SerializedResamplingFold {
  foldId: string
  train: number[]
  test: number[]
  validate?: number[]
  metadata?: Record<string, unknown>
}

export interface SerializedResamplingPlan extends Omit<ResamplingPlan, 'folds'> {
  folds: SerializedResamplingFold[]
}

export interface LearnerSpec {
  id: string
  packageName?: string
  typeIds?: string[]
  taskKinds?: TaskKind[]
  capabilities?: Capabilities
  defaultSearchSpace?: SearchSpace
  license?: string
  backend?: 'wasm' | 'js' | 'onnx' | 'python-native'
  metadata?: Record<string, unknown>
}

export type TrialStatus = 'pending' | 'running' | 'ok' | 'failed' | 'pruned' | 'timeout'

export interface TrialError {
  name: string
  message: string
  stackHash?: string
  phase?: string
}

export interface TrialRecord {
  trialId: string
  candidateId: string
  learnerSpec?: LearnerSpec
  pipelineSpec?: PipelineGraph | Record<string, unknown>
  params: Record<string, unknown>
  budget?: Record<string, number>
  seed: number
  batch?: number
  uhash?: string
  foldId?: string
  scores?: Record<string, number>
  primaryScore?: number
  status: TrialStatus
  error?: TrialError
  timings: Record<string, number>
  memory?: Record<string, number>
  artifactHash?: string
  predictionHash?: string
  resampleResultHash?: string
  warnings?: unknown[]
  metadata?: Record<string, unknown>
}

export interface LeaderboardRow {
  candidateId: string
  learnerSpec?: LearnerSpec
  pipelineSpec?: PipelineGraph | Record<string, unknown>
  params: Record<string, unknown>
  budget?: Record<string, number>
  metric?: string
  meanScore: number
  stdScore: number
  n: number
  trialIds: string[]
  rank: number
}

export interface ArchiveJSON {
  id: string
  taskId?: string
  measures: string[]
  primaryMeasure?: string
  direction: MeasureDirection
  metadata: Record<string, unknown>
  records: TrialRecord[]
}

export declare const TASK_KINDS: readonly TaskKind[]
export declare const PREDICTION_FIELDS: readonly string[]
export declare const MEASURE_DIRECTIONS: readonly MeasureDirection[]
export declare const MEASURE_RESPONSES: readonly MeasureResponse[]
export declare const RESAMPLING_STRATEGIES: readonly string[]
export declare const TRIAL_STATUSES: readonly TrialStatus[]

export declare function inferTaskKind(y?: Labels | number[] | null): TaskKind
export declare function createFeatureSchema(X: Matrix | number[][], opts?: {
  names?: string[]
  types?: string[]
  roles?: string[]
  metadata?: Record<string, unknown>
}): FeatureSchema
export declare function validateFeatureSchema(schema: FeatureSchema, cols?: number, rows?: number): FeatureSchema
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
export declare function evaluateMeasure(measureOrId: string | MeasureDef, prediction: Prediction, opts?: Record<string, unknown>): number
export declare function aggregateMeasure(measureOrId: string | MeasureDef, values: ArrayLike<number>): number
export declare function evaluateMetricSet(measures: (string | MeasureDef)[], prediction: Prediction, opts?: Record<string, unknown>): Record<string, number>
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
export declare function serializeResamplingPlan(plan: ResamplingPlan): SerializedResamplingPlan
export declare function deserializeResamplingPlan(plan: SerializedResamplingPlan): ResamplingPlan
export declare function groupKFold(groups: Labels, k?: number, opts?: { shuffle?: boolean; seed?: number }): ResamplingFold[]
export declare function timeSeriesSplit(n: number, opts?: { initialWindow?: number; horizon?: number; step?: number }): ResamplingFold[]
export declare function slidingWindowSplit(n: number, opts?: SlidingResamplingOpts): ResamplingFold[]
export declare function slidingIndexSplit(index: ArrayLike<number | string | Date>, opts?: SlidingResamplingOpts): ResamplingFold[]
export declare function slidingPeriodSplit(index: ArrayLike<number | string | Date>, opts?: SlidingResamplingOpts & { period?: 'day' | 'week' | 'month' | 'quarter' | 'year' | number }): ResamplingFold[]

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
  fail(record: Partial<TrialRecord>, error: Error | TrialError | string, phase?: string): TrialRecord
  update(trialId: string, patch: Partial<TrialRecord>): TrialRecord
  records(filter?: Record<string, unknown>): TrialRecord[]
  leaderboard(opts?: { metric?: string; direction?: MeasureDirection }): LeaderboardRow[]
  toJSON(): ArchiveJSON
  static fromJSON(json: ArchiveJSON): Archive
}
export declare function createTrialRecord(record?: Partial<TrialRecord>): TrialRecord
export declare function validateTrialRecord(record: TrialRecord): TrialRecord
export declare function normalizeTrialError(error: Error | TrialError | string, phase?: string): TrialError | undefined

// Loader
export type LoaderFn = (
  manifest: BundleManifest,
  toc: BundleTOCEntry[],
  blobs: Uint8Array,
  context?: Readonly<{
    loaderOptions: Readonly<Record<string, unknown>>
  }>
) => Estimator | Transformer | Promise<Estimator | Transformer>

// Promise-lifting utilities
export declare function isPromiseLike(x: unknown): x is PromiseLike<unknown>
export declare function lift<T, U>(x: MaybePromise<T>, f: (value: T) => U): MaybePromise<U>

// RNG
export type RngFn = () => number
export declare function makeLCG(seed?: number): RngFn
export declare function shuffle<T extends ArrayLike<number> & { [i: number]: number }>(arr: T, rng: RngFn): T

// Metrics
export type AveragingMethod = 'binary' | 'micro' | 'macro' | 'weighted' | 'macro_weighted'
export type UndefinedMetricPolicy = 'error' | 'warn' | 'nan' | number
export type ZeroDivisionPolicy = 'error' | 'warn' | 'nan' | 0 | 1 | number

export interface MetricOpts {
  sampleWeight?: ArrayLike<number>
  sample_weight?: ArrayLike<number>
  classes?: ArrayLike<number>
  average?: AveragingMethod
  positiveLabel?: number
  positive_label?: number
  zeroDivision?: ZeroDivisionPolicy
  zero_division?: ZeroDivisionPolicy
  undefinedValue?: UndefinedMetricPolicy
  undefined_value?: UndefinedMetricPolicy
  warnings?: unknown[]
}

export interface SlidingResamplingOpts {
  lookback?: number
  initialWindow?: number
  assessStart?: number
  assessStop?: number
  horizon?: number
  complete?: boolean
  step?: number
  skip?: number
}

export interface ConfusionMatrixResult {
  matrix: Int32Array | Float64Array
  labels: Int32Array
}

export declare function accuracy(yTrue: Labels, yPred: Labels, opts?: MetricOpts): number
export declare function r2Score(yTrue: Labels, yPred: Labels, opts?: MetricOpts): number
export declare function meanSquaredError(yTrue: Labels, yPred: Labels, opts?: MetricOpts): number
export declare function meanAbsoluteError(yTrue: Labels, yPred: Labels, opts?: MetricOpts): number
export declare function confusionMatrix(yTrue: Labels, yPred: Labels, opts?: MetricOpts): ConfusionMatrixResult
export declare function precisionScore(yTrue: Labels, yPred: Labels, opts?: MetricOpts): number
export declare function recallScore(yTrue: Labels, yPred: Labels, opts?: MetricOpts): number
export declare function f1Score(yTrue: Labels, yPred: Labels, opts?: MetricOpts): number
export declare function logLoss(yTrue: Labels, yProba: Float64Array, opts?: MetricOpts & { nClasses?: number; n_classes?: number; eps?: number }): number
export declare function rocAuc(yTrue: Labels, yScore: Float64Array, opts?: MetricOpts & { multiClass?: 'raise' | 'ovr' | 'ovo'; multi_class?: 'raise' | 'ovr' | 'ovo' }): number

// Cross-validation
export interface CVFold {
  train: Int32Array
  test: Int32Array
}

export type ScoringName = 'accuracy' | 'r2' | 'neg_mse' | 'neg_mae'
export type ScoringFn = (yTrue: Labels, yPred: Labels) => number

export declare function kFold(n: number, k?: number, opts?: { shuffle?: boolean; seed?: number }): CVFold[]
export declare function stratifiedKFold(y: Labels, k?: number, opts?: { shuffle?: boolean; seed?: number }): CVFold[]
export declare function trainTestSplit(n: number, opts?: { testSize?: number; shuffle?: boolean; seed?: number }): CVFold
export declare function getScorer(scoring: ScoringName | ScoringFn): ScoringFn

export interface CrossValScoreOpts {
  cv?: number | CVFold[]
  scoring?: ScoringName | ScoringFn
  seed?: number
  params?: Record<string, unknown>
}

export interface EstimatorClass {
  create(params?: Record<string, unknown>): Promise<Estimator>
  readonly classId?: string
  defaultSearchSpace?(task?: TaskType): SearchSpace
  budgetSpec?(): { roundsParam?: string }
}

export declare function crossValScore(
  EstimatorClass: EstimatorClass,
  X: Matrix | number[][],
  y: Labels | number[],
  opts?: CrossValScoreOpts
): Promise<Float64Array>

// Ensemble types
export type TaskType = 'classification' | 'regression'
export type VotingMethod = 'soft' | 'hard'

export type EstimatorSpec = [name: string, cls: EstimatorClass, params?: Record<string, unknown>]

export interface BaggedEstimatorLike {
  predict(X: Matrix | number[][]): MaybePromise<Labels>
  predictProba?(X: Matrix | number[][]): MaybePromise<Float64Array>
  score(X: Matrix | number[][], y: Labels | number[]): MaybePromise<number>
  save(): Uint8Array
  dispose(): void
  getParams(): Record<string, unknown>
  setParams(p: Record<string, unknown>): unknown
  readonly capabilities: Capabilities
  readonly isFitted: boolean
  readonly classes: Int32Array | null
  readonly oofPredictions: Float64Array
}

export type PrefittedBaggedSpec = [name: string, estimator: BaggedEstimatorLike]

export interface AutoMLEstimatorClass extends EstimatorClass {
  readonly classId: string
}

export type AutoMLEstimatorSpec = [
  name: string,
  cls: AutoMLEstimatorClass,
  params?: Record<string, unknown>
]

export interface VotingEnsembleParams {
  estimators?: EstimatorSpec[]
  weights?: number[] | Float64Array
  voting?: VotingMethod
  task?: TaskType
}

export type VotingEnsembleMutableParams = Partial<
  Pick<VotingEnsembleParams, 'voting' | 'weights'>
>

export interface StackingEnsembleParams {
  estimators?: Array<EstimatorSpec | PrefittedBaggedSpec>
  finalEstimator?: EstimatorSpec
  cv?: number
  task?: TaskType
  passthrough?: boolean
  seed?: number
}

export type StackingEnsembleMutableParams = Partial<
  Pick<StackingEnsembleParams, 'cv' | 'passthrough' | 'seed'>
>

export interface CaruanaResult {
  indices: Int32Array
  weights: Float64Array
  scores: Float64Array
}

export interface CaruanaOpts {
  maxSize?: number
  scoring?: ScoringName | ScoringFn
  task?: TaskType
  nClasses?: number
  refineWeights?: boolean
  classes?: ArrayLike<number>
}

export interface BaggedEstimatorParams {
  estimator?: EstimatorSpec
  kFold?: number
  nRepeats?: number
  task?: TaskType
  seed?: number
}

export type BaggedEstimatorMutableParams = Partial<
  Pick<BaggedEstimatorParams, 'kFold' | 'nRepeats' | 'seed'>
>

export interface WeightOptimizationOpts {
  task?: TaskType
  lr?: number
  nIter?: number
  classes?: ArrayLike<number>
}

export declare function projectSimplex(values: ArrayLike<number>): Float64Array
export declare function optimizeWeights(
  oofPredictions: Float64Array[],
  yTrue: Labels | number[],
  initWeights: ArrayLike<number>,
  opts?: WeightOptimizationOpts
): Float64Array

export declare function caruanaSelect(
  oofPredictions: Float64Array[],
  yTrue: Labels,
  opts?: CaruanaOpts
): CaruanaResult

export interface OofOpts {
  cv?: number
  seed?: number
  task?: TaskType
}

export interface OofResult {
  oofPreds: Float64Array[]
  classes: Int32Array | null
}

export declare function getOofPredictions(
  estimatorSpecs: EstimatorSpec[],
  X: Matrix | number[][],
  y: Labels | number[],
  opts?: OofOpts
): Promise<OofResult>

// AutoML types
export interface ModelSpecBase {
  name: string
  displayName?: string
  /** Stable key selecting a built-in zero-shot portfolio family. */
  portfolioKey?: string
  searchSpace?: SearchSpace
  params?: Record<string, unknown>
}

export type ModelSpec = ModelSpecBase & (
  | { classId: string; cls: EstimatorClass }
  | { classId?: string; cls: AutoMLEstimatorClass }
)

export interface CandidateResult {
  id: number
  candidateId: string
  candidate: CandidateTemplate
  modelName: string
  params: Record<string, unknown>
  scores: Float64Array
  /** Run-level seed used as the root for executor-owned deterministic choices. */
  baseSeed: number
  /**
   * Candidate/fold-derived provenance seeds. These drive executor-owned random
   * operations such as budget subsampling; they do not override model params.
   */
  foldSeeds: Uint32Array
  meanScore: number
  stdScore: number
  fitTimeMs: number
  rank: number
}

export interface SearchOpts {
  scoring?: ScoringName | ScoringFn
  cv?: number
  seed?: number
  task?: TaskType
  nIter?: number
  nInitial?: number
  maxTimeMs?: number
}

export interface HalvingOpts extends SearchOpts {
  factor?: number
  minResources?: number
}

export interface AutoFitOpts extends SearchOpts {
  preprocess?: boolean | PreprocessConfig | PreprocessTemplate[]
  ensemble?: boolean
  ensembleSize?: number
  refit?: boolean
  strategy?: 'random' | 'portfolio' | 'halving' | 'progressive' | 'bayesian'
  minDisagreement?: number
  stacking?: boolean | 'auto'
  stackingPassthrough?: boolean
  metaEstimator?: EstimatorSpec | {
    cls: EstimatorClass
    params?: Record<string, unknown>
  }
  onProgress?: (event: Record<string, unknown>) => void
}

export interface AutoFitResult {
  model: Estimator | null
  /** Preprocessing is owned by fitted Pipeline children. */
  preprocessor: null
  archive: Archive
  leaderboard: CandidateResult[]
  bestParams: {
    model: Record<string, unknown>
    preprocess: PreprocessResolvedConfig | null
  }
  bestCandidate: CandidateTemplate
  bestModelName: string
  bestScore: number
}

export declare function sampleParam(param: SearchParam, rng: RngFn): unknown
export declare function sampleConfig(space: SearchSpace, rng: RngFn): Record<string, unknown>
export declare function randomConfigs(space: SearchSpace, n: number, opts?: { seed?: number }): Record<string, unknown>[]
export declare function gridConfigs(space: SearchSpace, opts?: { steps?: number }): Record<string, unknown>[]
export declare function createCandidate(
  model: {
    name?: string
    displayName?: string
    classId: string
  },
  params: Record<string, unknown>,
  preprocess?: CandidatePreprocess | null
): CandidateTemplate
export declare function candidateCanonicalBytes(candidate: CandidateTemplate): Uint8Array
export declare function candidateHash(candidate: CandidateTemplate): string
export declare function makeCandidateId(candidate: CandidateTemplate): string
export declare function seedFor(candidate: CandidateTemplate, foldIdx: number, baseSeed: number): number
export declare function normalizeModelSpecs(
  models: (ModelSpec | AutoMLEstimatorSpec)[],
  label?: string
): ReadonlyArray<ModelSpec & { classId: string; displayName: string }>
export declare function autoFit(
  models: (ModelSpec | AutoMLEstimatorSpec)[],
  X: Matrix | number[][],
  y: Labels | number[],
  opts?: AutoFitOpts
): Promise<AutoFitResult>
