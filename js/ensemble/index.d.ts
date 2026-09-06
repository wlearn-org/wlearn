import type {
  BaggedEstimatorParams,
  BaggedEstimatorMutableParams,
  Capabilities,
  CaruanaOpts,
  CaruanaResult,
  Labels,
  Matrix,
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
  fit(X: Matrix | number[][], y: Labels | number[]): Promise<this>
  predict(X: Matrix | number[][]): MaybePromise<Labels>
  predictProba(X: Matrix | number[][]): MaybePromise<Float64Array>
  score(X: Matrix | number[][], y: Labels | number[]): MaybePromise<number>
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
  fit(X: Matrix | number[][], y: Labels | number[]): Promise<this>
  predict(X: Matrix | number[][]): MaybePromise<Labels>
  predictProba(X: Matrix | number[][]): MaybePromise<Float64Array>
  score(X: Matrix | number[][], y: Labels | number[]): MaybePromise<number>
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
  fit(X: Matrix | number[][], y: Labels | number[]): Promise<this>
  predict(X: Matrix | number[][]): MaybePromise<Labels>
  predictProba(X: Matrix | number[][]): MaybePromise<Float64Array>
  score(X: Matrix | number[][], y: Labels | number[]): MaybePromise<number>
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
  X: Matrix | number[][],
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
