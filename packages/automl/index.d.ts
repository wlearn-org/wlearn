import type {
  Archive,
  AutoFitOpts,
  AutoFitResult,
  AutoMLEstimatorSpec,
  CandidateResult,
  CandidateTemplate,
  CVFold,
  Estimator,
  EstimatorClass,
  HalvingOpts,
  Labels,
  Matrix,
  ModelSpec,
  RngFn,
  ScoringFn,
  ScoringName,
  SearchOpts,
  SearchParam,
  SearchSpace,
  TaskType,
} from '@wlearn/types'

export type {
  AutoFitOpts,
  AutoFitResult,
  AutoMLEstimatorSpec,
  CandidateResult,
  CandidateTemplate,
  HalvingOpts,
  ModelSpec,
  SearchOpts,
} from '@wlearn/types'

export interface CandidateTask {
  candidateId: string
  candidate: CandidateTemplate
  cls: EstimatorClass
  params: Readonly<Record<string, unknown>>
  budget?: Readonly<Record<string, unknown>>
}

export interface SearchFitResult {
  leaderboard: Leaderboard
  archive: Archive
  bestResult: CandidateResult
  rounds?: ReadonlyArray<Readonly<Record<string, unknown>>>
}

export interface ExecutorOptions {
  folds: CVFold[]
  scoring: ScoringName | ScoringFn
  X: Matrix
  y: Labels
  timeLimitMs?: number
  seed?: number
  onProgress?: (event: Readonly<Record<string, unknown>>) => void
}

export interface PortfolioSearchOptions extends SearchOpts {
  maxTimeMs?: number
}

export interface ProgressiveSearchOptions extends SearchOpts {
  promoteCount?: number
  probeFraction?: number
}

export interface BayesianSearchOptions extends SearchOpts {
  acquisitionFn?: string
  kappa?: number
  xi?: number
  kernel?: string
}

export declare function sampleParam(param: SearchParam, rng: RngFn): unknown
export declare function sampleConfig(space: SearchSpace, rng: RngFn): Record<string, unknown>
export declare function randomConfigs(space: SearchSpace, n: number, opts?: { seed?: number }): Record<string, unknown>[]
export declare function gridConfigs(space: SearchSpace, opts?: { steps?: number }): Record<string, unknown>[]

export declare function createCandidate(
  model: { name?: string; displayName?: string; classId: string },
  params: Record<string, unknown>,
  preprocess?: CandidateTemplate['preprocess']
): CandidateTemplate
export declare function candidateCanonicalBytes(candidate: CandidateTemplate): Uint8Array
export declare function candidateHash(candidate: CandidateTemplate): string
export declare function makeCandidateId(candidate: CandidateTemplate): string
export declare function seedFor(candidate: CandidateTemplate, foldIdx: number, baseSeed: number): number
export declare function normalizeModelSpecs(
  models: (ModelSpec | AutoMLEstimatorSpec)[],
  label?: string
): ReadonlyArray<ModelSpec & { classId: string; displayName: string }>

export declare function detectTask(y: Labels): TaskType
export declare function partialShuffle<T extends Int32Array | Uint32Array>(
  indices: T,
  k: number,
  rng: RngFn
): T
export declare function scorerGreaterIsBetter(scoring: ScoringName | ScoringFn): boolean

export declare function autoFit(
  models: (ModelSpec | AutoMLEstimatorSpec)[],
  X: Matrix | number[][],
  y: Labels | number[],
  opts?: AutoFitOpts
): Promise<AutoFitResult>

export declare function registerBayesianSearch(
  search: typeof BayesianSearch | null
): void

export class Leaderboard {
  add(entry: {
    candidateId: string
    candidate: CandidateTemplate
    scores: Float64Array
    /** Run-level seed used as the root for executor-owned deterministic choices. */
    baseSeed?: number
    /** Candidate/fold-derived provenance seeds; model params are not overridden. */
    foldSeeds?: Uint32Array
    fitTimeMs: number
  }): CandidateResult
  ranked(): CandidateResult[]
  best(): CandidateResult | null
  top(k: number): CandidateResult[]
  toJSON(): ReadonlyArray<Readonly<Record<string, unknown>>>
  toArchive(opts?: {
    metric?: string
    direction?: 'maximize' | 'minimize'
    metadata?: Record<string, unknown>
  }): Archive
  readonly length: number
  static fromJSON(value: ReadonlyArray<Readonly<Record<string, unknown>>>): Leaderboard
}

export class Executor {
  constructor(options: ExecutorOptions)
  readonly leaderboard: Leaderboard
  readonly archive: Archive
  readonly firstError: unknown | null
  readonly isTimedOut: boolean
  evaluateCandidate(task: CandidateTask): Promise<CandidateResult>
  recordFailure(task: CandidateTask | null, error: unknown, phase?: string): unknown
  runStrategy(strategy: SearchStrategy): Promise<{ leaderboard: Leaderboard; archive: Archive }>
}

export interface SearchStrategy {
  next(): CandidateTask | null
  report(result: CandidateResult): void
  isDone(): boolean
  dispose?(): void
}

export class RandomStrategy implements SearchStrategy {
  constructor(models: (ModelSpec | AutoMLEstimatorSpec)[], opts?: SearchOpts)
  next(): CandidateTask | null
  report(result: CandidateResult): void
  isDone(): boolean
}

export class HalvingStrategy implements SearchStrategy {
  constructor(models: (ModelSpec | AutoMLEstimatorSpec)[], opts?: HalvingOpts & {
    nSamples?: number
    greaterIsBetter?: boolean
  })
  next(): CandidateTask | null
  report(result: CandidateResult): void
  isDone(): boolean
  readonly rounds: ReadonlyArray<Readonly<Record<string, unknown>>>
}

export class ProgressiveStrategy implements SearchStrategy {
  constructor(models: (ModelSpec | AutoMLEstimatorSpec)[], opts?: ProgressiveSearchOptions & {
    greaterIsBetter?: boolean
  })
  next(): CandidateTask | null
  report(result: CandidateResult): void
  isDone(): boolean
  readonly phase: string
}

export class PortfolioStrategy implements SearchStrategy {
  constructor(models: (ModelSpec | AutoMLEstimatorSpec)[], opts?: { task?: TaskType; seed?: number })
  next(): CandidateTask | null
  report(result: CandidateResult): void
  isDone(): boolean
}

export class BayesianStrategy implements SearchStrategy {
  constructor(models: (ModelSpec | AutoMLEstimatorSpec)[], opts?: BayesianSearchOptions)
  init(): Promise<void>
  next(): CandidateTask | null
  report(result: CandidateResult): void
  isDone(): boolean
  dispose(): void
}

declare class SearchBase {
  readonly leaderboard: Leaderboard | null
  readonly bestResult: CandidateResult | null
  readonly archive: Archive | null
  fit(X: Matrix | number[][], y: Labels | number[]): Promise<SearchFitResult>
  refitBest(X: Matrix | number[][], y: Labels | number[]): Promise<Estimator>
}

export class RandomSearch extends SearchBase {
  constructor(models: (ModelSpec | AutoMLEstimatorSpec)[], opts?: SearchOpts)
}

export class SuccessiveHalvingSearch extends SearchBase {
  constructor(models: (ModelSpec | AutoMLEstimatorSpec)[], opts?: HalvingOpts)
  readonly rounds: ReadonlyArray<Readonly<Record<string, unknown>>> | null
}

export class ProgressiveSearch extends SearchBase {
  constructor(models: (ModelSpec | AutoMLEstimatorSpec)[], opts?: ProgressiveSearchOptions)
}

export class PortfolioSearch extends SearchBase {
  constructor(models: (ModelSpec | AutoMLEstimatorSpec)[], opts?: PortfolioSearchOptions)
}

export class BayesianSearch extends SearchBase {
  constructor(models: (ModelSpec | AutoMLEstimatorSpec)[], opts?: BayesianSearchOptions)
}

export type Portfolio = Readonly<Record<
  string,
  ReadonlyArray<Readonly<Record<string, unknown>>>
>>

export declare const PORTFOLIO: Readonly<Record<TaskType, Portfolio>>
export declare function getPortfolio(task?: TaskType): Portfolio
