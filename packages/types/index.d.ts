// wlearn bundle format constants
export declare const BUNDLE_MAGIC: Uint8Array
export declare const BUNDLE_VERSION: 1
export declare const HEADER_SIZE: 16
export declare const DTYPE: {
  readonly FLOAT32: 'float32'
  readonly FLOAT64: 'float64'
  readonly INT32: 'int32'
}

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

export interface Estimator {
  fit(X: Matrix | number[][], y: Labels | number[]): this
  predict(X: Matrix | number[][]): Labels
  score(X: Matrix | number[][], y: Labels | number[]): number
  save(): Uint8Array
  dispose(): void
  getParams(): Record<string, unknown>
  setParams(p: Record<string, unknown>): this
  readonly capabilities: Capabilities
  readonly isFitted: boolean
}

export interface Classifier extends Estimator {
  predictProba(X: Matrix | number[][]): Float64Array
  readonly classes: Int32Array
}

// Search space IR (for AutoML)
export type SearchParam =
  | { type: 'categorical'; values: unknown[] }
  | { type: 'uniform'; low: number; high: number }
  | { type: 'log_uniform'; low: number; high: number }
  | { type: 'int_uniform'; low: number; high: number }
  | { type: 'int_log_uniform'; low: number; high: number }

export type SearchSpace = Record<string, SearchParam & { condition?: Record<string, unknown> }>

// Bundle format v1
export interface BundleManifest {
  typeId: string
  bundleVersion: number
  requires?: string[]
  params?: Record<string, unknown>
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

// Loader
export type LoaderFn = (
  manifest: BundleManifest,
  toc: BundleTOCEntry[],
  blobs: Uint8Array
) => Estimator | Promise<Estimator>

// RNG
export type RngFn = () => number
export declare function makeLCG(seed?: number): RngFn
export declare function shuffle<T extends ArrayLike<number> & { [i: number]: number }>(arr: T, rng: RngFn): T

// Metrics
export type AveragingMethod = 'binary' | 'micro' | 'macro'

export interface ConfusionMatrixResult {
  matrix: Int32Array
  labels: Int32Array
}

export declare function accuracy(yTrue: Labels, yPred: Labels): number
export declare function r2Score(yTrue: Labels, yPred: Labels): number
export declare function meanSquaredError(yTrue: Labels, yPred: Labels): number
export declare function meanAbsoluteError(yTrue: Labels, yPred: Labels): number
export declare function confusionMatrix(yTrue: Labels, yPred: Labels): ConfusionMatrixResult
export declare function precisionScore(yTrue: Labels, yPred: Labels, opts?: { average?: AveragingMethod }): number
export declare function recallScore(yTrue: Labels, yPred: Labels, opts?: { average?: AveragingMethod }): number
export declare function f1Score(yTrue: Labels, yPred: Labels, opts?: { average?: AveragingMethod }): number
export declare function logLoss(yTrue: Labels, yProba: Float64Array, opts?: { nClasses?: number; eps?: number }): number
export declare function rocAuc(yTrue: Labels, yProba: Float64Array): number

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

export interface VotingEnsembleParams {
  estimators?: EstimatorSpec[]
  weights?: number[] | Float64Array
  voting?: VotingMethod
  task?: TaskType
}

export interface StackingEnsembleParams {
  estimators?: EstimatorSpec[]
  finalEstimator?: EstimatorSpec
  cv?: number
  task?: TaskType
  passthrough?: boolean
  seed?: number
}

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
}

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
export interface ModelSpec {
  name: string
  cls: EstimatorClass
  searchSpace?: SearchSpace
  params?: Record<string, unknown>
}

export interface CandidateResult {
  id: number
  modelName: string
  params: Record<string, unknown>
  scores: Float64Array
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
  maxTimeMs?: number
}

export interface HalvingOpts extends SearchOpts {
  factor?: number
  minResources?: number
}

export interface AutoFitOpts extends SearchOpts {
  ensemble?: boolean
  ensembleSize?: number
  refit?: boolean
}

export interface AutoFitResult {
  model: Estimator | null
  leaderboard: CandidateResult[]
  bestParams: Record<string, unknown>
  bestModelName: string
  bestScore: number
}

export declare function sampleParam(param: SearchParam, rng: RngFn): unknown
export declare function sampleConfig(space: SearchSpace, rng: RngFn): Record<string, unknown>
export declare function randomConfigs(space: SearchSpace, n: number, opts?: { seed?: number }): Record<string, unknown>[]
export declare function gridConfigs(space: SearchSpace, opts?: { steps?: number }): Record<string, unknown>[]
export declare function autoFit(
  models: (ModelSpec | EstimatorSpec)[],
  X: Matrix | number[][],
  y: Labels | number[],
  opts?: AutoFitOpts
): Promise<AutoFitResult>
