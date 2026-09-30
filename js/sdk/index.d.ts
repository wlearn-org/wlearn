import type {
  Capabilities,
  Estimator,
  EstimatorClass,
  Labels,
  Matrix,
  SearchSpace
} from '@wlearn/types'

export {
  Pipeline, load, loadSync, register,
  encodeBundle, decodeBundle, validateBundle,
  normalizeX, normalizeY,
  accuracy, r2Score, meanSquaredError, meanAbsoluteError,
  confusionMatrix, precisionScore, recallScore, f1Score, logLoss, rocAuc,
  kFold, stratifiedKFold, trainTestSplit, crossValScore,
  StandardScaler, MinMaxScaler,
  TASK_KINDS, createTask, validateTask, taskRows,
  PREDICTION_FIELDS, createPrediction, validatePrediction,
  MEASURE_DIRECTIONS, MEASURE_RESPONSES, listMeasures, evaluateMeasure,
  evaluateMetricSet, aggregateMeasure,
  RESAMPLING_STRATEGIES, createResamplingPlan, validateResamplingPlan,
  serializeResamplingPlan, deserializeResamplingPlan,
  groupKFold, timeSeriesSplit, slidingWindowSplit, slidingIndexSplit,
  slidingPeriodSplit,
  TRIAL_STATUSES, Archive
} from '@wlearn/core'

export { Preprocessor } from '@wlearn/preprocess'

export {
  autoFit, registerBayesianSearch, BayesianSearch, BayesianStrategy
} from '@wlearn/automl'

export {
  StackingEnsemble, VotingEnsemble, BaggedEstimator
} from '@wlearn/ensemble'

export declare const LinearModel: EstimatorClass
export declare const SVMModel: EstimatorClass
export declare const XGBModel: EstimatorClass
export declare const LGBModel: EstimatorClass
export declare const KNNModel: EstimatorClass
export declare const EBMModel: EstimatorClass
export declare const TsetlinModel: EstimatorClass
export declare const BARTModel: EstimatorClass

export declare const XLearnLR: EstimatorClass
export declare const XLearnFM: EstimatorClass
export declare const XLearnFFM: EstimatorClass
export declare const XLearnLRClassifier: EstimatorClass
export declare const XLearnLRRegressor: EstimatorClass
export declare const XLearnFMClassifier: EstimatorClass
export declare const XLearnFMRegressor: EstimatorClass
export declare const XLearnFFMClassifier: EstimatorClass
export declare const XLearnFFMRegressor: EstimatorClass

export declare const MLPModel: EstimatorClass
export declare const TabMModel: EstimatorClass
export declare const NAMModel: EstimatorClass
export declare const MLPClassifier: EstimatorClass
export declare const MLPRegressor: EstimatorClass
export declare const TabMClassifier: EstimatorClass
export declare const TabMRegressor: EstimatorClass
export declare const NAMClassifier: EstimatorClass
export declare const NAMRegressor: EstimatorClass

export declare const RFModel: EstimatorClass
export declare const GAMModel: EstimatorClass
export declare function loadRF(options?: Record<string, unknown>): Promise<unknown>
export declare function loadGAM(options?: Record<string, unknown>): Promise<unknown>

export interface ClusterEstimator {
  fit(X: Matrix | number[][]): this
  predict(X: Matrix | number[][]): Labels
  save(): Uint8Array
  dispose(): void
  getParams(): Record<string, unknown>
  setParams(params: Record<string, unknown>): this
  readonly capabilities: Capabilities & { clusterer: true }
  readonly isFitted: boolean
}

export interface ClusterModelClass {
  create(params?: Record<string, unknown>): Promise<ClusterEstimator>
  load(bytes: Uint8Array): Promise<ClusterEstimator>
  defaultSearchSpace(): SearchSpace
}

export declare const ClusterModel: ClusterModelClass
export declare function loadCluster(options?: Record<string, unknown>): Promise<unknown>
export declare function silhouette(
  X: Matrix | number[][],
  labels: Labels | number[],
  options?: Record<string, unknown>
): number
export declare function calinskiHarabasz(
  X: Matrix | number[][],
  labels: Labels | number[],
  options?: Record<string, unknown>
): number
export declare function daviesBouldin(
  X: Matrix | number[][],
  labels: Labels | number[],
  options?: Record<string, unknown>
): number
export declare function adjustedRand(
  labelsTrue: Labels | number[],
  labelsPred: Labels | number[]
): number

export interface CompiledSearchSpace {
  readonly paramNames: string[]
  readonly nDims: number
  readonly [key: string]: unknown
}

export declare class BayesianOptimizer {
  static create(
    searchSpace: SearchSpace,
    options?: Record<string, unknown>
  ): Promise<BayesianOptimizer>
  observe(params: Record<string, unknown>, score: number): void
  suggest(): Record<string, unknown>
  suggestLiar(hallucinatedScore: number): Record<string, unknown>
  suggestBatch(count: number): Record<string, unknown>[]
  dispose(): void
  readonly nObs: number
  readonly bestScore: number
  readonly nContexts: number
  readonly compiled: CompiledSearchSpace
}

export declare function compileSpace(searchSpace: SearchSpace): CompiledSearchSpace
export declare function encodeParams(
  compiled: CompiledSearchSpace,
  params: Record<string, unknown>
): Float64Array
export declare function decodeParams(
  compiled: CompiledSearchSpace,
  values: Float64Array
): Record<string, unknown>
export declare function countFreeParams(searchSpace: SearchSpace): number
export declare function loadBO(options?: Record<string, unknown>): Promise<unknown>

export interface MitraParams {
  maxSupport?: number
  seed?: number
}

export interface MitraOnnxSession {
  run(feeds: Record<string, unknown>): Promise<Record<string, unknown>>
  release?(): void | Promise<void>
}

export type MitraOnnxSource = Uint8Array | MitraOnnxSession

export interface MitraLoadOptions {
  ort?: unknown
  trustedOnnxSha256?: string
  allowUnverifiedLegacyModel?: boolean
}

export interface MitraClassifierInstance extends Estimator {
  fit(X: Matrix | number[][], y: Labels | number[]): this
  predict(X: Matrix | number[][]): Promise<Int32Array>
  predictProba(X: Matrix | number[][]): Promise<Float64Array>
  score(X: Matrix | number[][], y: Labels | number[]): Promise<number>
  readonly classes: Int32Array
  readonly nrClass: number
  readonly nrFeature: number
}

export interface MitraRegressorInstance extends Estimator {
  fit(X: Matrix | number[][], y: Labels | number[]): this
  predict(X: Matrix | number[][]): Promise<Float64Array>
  score(X: Matrix | number[][], y: Labels | number[]): Promise<number>
  readonly nrFeature: number
}

export interface MitraClassifierClass {
  create(
    onnxSource: MitraOnnxSource,
    params?: MitraParams,
    options?: MitraLoadOptions
  ): Promise<MitraClassifierInstance>
  load(
    bytes: Uint8Array,
    onnxSource: MitraOnnxSource,
    options?: MitraLoadOptions
  ): Promise<MitraClassifierInstance>
}

export interface MitraRegressorClass {
  create(
    onnxSource: MitraOnnxSource,
    params?: MitraParams,
    options?: MitraLoadOptions
  ): Promise<MitraRegressorInstance>
  load(
    bytes: Uint8Array,
    onnxSource: MitraOnnxSource,
    options?: MitraLoadOptions
  ): Promise<MitraRegressorInstance>
}

export interface RegisterMitraOptions extends MitraLoadOptions {
  classifierTrustedOnnxSha256?: string
  regressorTrustedOnnxSha256?: string
}

export declare const MitraClassifier: MitraClassifierClass | undefined
export declare const MitraRegressor: MitraRegressorClass | undefined
export declare const registerMitraLoaders:
  | ((
    classifierOnnx?: MitraOnnxSource | null,
    regressorOnnx?: MitraOnnxSource | null,
    options?: RegisterMitraOptions
  ) => void)
  | undefined

export { BasisClassifier, BasisRegressor, BasisTransformer, loadBasis } from '@wlearn/basis'

export * from '@wlearn/sym'
export * from '@wlearn/uncertainty'
