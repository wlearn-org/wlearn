import type {
  EstimatorSpec, Prediction, TargetInput, FitOptions,
  BaggedEstimatorParams,
  BaggedEstimatorMutableParams,
  Capabilities,
  CaruanaOpts,
  CaruanaResult,
  Labels,
  DenseMatrix,
  Scoring,
  MaybePromise,
  OofOpts,
  OofResult,
  StackingEnsembleParams,
  StackingEnsembleMutableParams,
  TaskType,
  VotingEnsembleParams,
  VotingEnsembleMutableParams,
  WeightOptimizationOpts
} from '@wlearn/types'

export type {
  BaggedEstimatorLike,
  BaggedEstimatorParams,
  BaggedEstimatorMutableParams,
  CaruanaOpts,
  CaruanaResult,
  EstimatorSpec,
  OofOpts,
  OofResult,
  PrefittedBaggedSpec,
  StackingEnsembleParams,
  StackingEnsembleMutableParams,
  TaskType,
  VotingEnsembleParams,
  VotingEnsembleMutableParams,
  VotingMethod,
  WeightOptimizationOpts
} from '@wlearn/types'

export interface EnsembleLoadOptions {
  loaderOptions?: Record<string, unknown>
}

export declare class VotingEnsemble {
  constructor(params?: VotingEnsembleParams)
  static create(params?: VotingEnsembleParams): Promise<VotingEnsemble>
  static load(bytes: Uint8Array, options?: EnsembleLoadOptions): Promise<VotingEnsemble>
  fit(X: DenseMatrix | number[][], y: Labels | number[]): Promise<this>
  predict(X: DenseMatrix | number[][]): MaybePromise<Labels>
  predictProba(X: DenseMatrix | number[][]): MaybePromise<Float64Array>
  score(X: DenseMatrix | number[][], y: Labels | number[]): MaybePromise<number>
  save(): Uint8Array
  dispose(): void
  getParams(): Record<string, unknown>
  setParams(params: VotingEnsembleMutableParams): this
  readonly capabilities: Capabilities
  readonly isFitted: boolean
  readonly classes: Int32Array | null
}

export declare class BaggedEstimator {
  constructor(params?: BaggedEstimatorParams)
  static create(params?: BaggedEstimatorParams): Promise<BaggedEstimator>
  static load(bytes: Uint8Array, options?: EnsembleLoadOptions): Promise<BaggedEstimator>
  fit(X: DenseMatrix | number[][], y: Labels | number[]): Promise<this>
  predict(X: DenseMatrix | number[][]): MaybePromise<Labels>
  predictProba(X: DenseMatrix | number[][]): MaybePromise<Float64Array>
  score(X: DenseMatrix | number[][], y: Labels | number[]): MaybePromise<number>
  save(): Uint8Array
  dispose(): void
  getParams(): Record<string, unknown>
  setParams(params: BaggedEstimatorMutableParams): this
  readonly capabilities: Capabilities
  readonly isFitted: boolean
  readonly classes: Int32Array | null
  readonly oofPredictions: Float64Array
}

export declare class StackingEnsemble {
  constructor(params?: StackingEnsembleParams)
  static create(params?: StackingEnsembleParams): Promise<StackingEnsemble>
  static load(bytes: Uint8Array, options?: EnsembleLoadOptions): Promise<StackingEnsemble>
  fit(X: DenseMatrix | number[][], y: Labels | number[]): Promise<this>
  predict(X: DenseMatrix | number[][]): MaybePromise<Labels>
  predictProba(X: DenseMatrix | number[][]): MaybePromise<Float64Array>
  score(X: DenseMatrix | number[][], y: Labels | number[]): MaybePromise<number>
  save(): Uint8Array
  dispose(): void
  getParams(): Record<string, unknown>
  setParams(params: StackingEnsembleMutableParams): this
  readonly capabilities: Capabilities
  readonly isFitted: boolean
  readonly classes: Int32Array | null
}

export declare function caruanaSelect(
  oofPredictions: Float64Array[],
  yTrue: Labels | number[],
  opts?: CaruanaOpts
): CaruanaResult

export declare function getOofPredictions(
  estimatorSpecs: import('@wlearn/types').EstimatorSpec[],
  X: DenseMatrix | number[][],
  y: Labels | number[],
  opts?: OofOpts
): Promise<OofResult>

export declare function projectSimplex(values: ArrayLike<number>): Float64Array

export declare function optimizeWeights(
  oofPredictions: Float64Array[],
  yTrue: Labels | number[],
  initWeights: ArrayLike<number>,
  opts?: WeightOptimizationOpts
): Float64Array

export interface MultiTargetParams {
  estimator?: EstimatorSpec
  targetNames?: string[]
  task?: 'multioutput' | 'multilabel'
}

declare class MultiTarget {
  constructor(params?: MultiTargetParams)
  static create(params?: MultiTargetParams): Promise<MultiTarget>
  fit(X: DenseMatrix | number[][], y: TargetInput, opts?: FitOptions): Promise<this>
  predict(X: DenseMatrix | number[][]): MaybePromise<DenseMatrix>
  predictQuantiles(X: DenseMatrix | number[][], levels: ArrayLike<number>): MaybePromise<Prediction>
  score(X: DenseMatrix | number[][], y: TargetInput): MaybePromise<number>
  save(): Uint8Array
  static load(bytes: Uint8Array, opts?: { loaderOptions?: Record<string, unknown> }): Promise<MultiTarget>
  dispose(): void
  getParams(): Record<string, unknown>
  setParams(params: Pick<MultiTargetParams, 'estimator' | 'targetNames'>): this
  readonly capabilities: Capabilities
  readonly isFitted: boolean
  readonly targetNames: string[] | null
  readonly targetCount: number | null
}

export declare class MultiOutputRegressor extends MultiTarget {
  static create(params?: MultiTargetParams): Promise<MultiOutputRegressor>
  static load(bytes: Uint8Array, opts?: { loaderOptions?: Record<string, unknown> }): Promise<MultiOutputRegressor>
}

export declare class MultiLabelClassifier extends MultiTarget {
  static create(params?: MultiTargetParams): Promise<MultiLabelClassifier>
  static load(bytes: Uint8Array, opts?: { loaderOptions?: Record<string, unknown> }): Promise<MultiLabelClassifier>
  predictProba(X: DenseMatrix | number[][]): MaybePromise<DenseMatrix>
}
