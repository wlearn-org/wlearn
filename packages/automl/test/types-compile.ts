import {
  BayesianSearch,
  BayesianStrategy,
  Executor,
  HalvingStrategy,
  Leaderboard,
  PORTFOLIO,
  PortfolioSearch,
  PortfolioStrategy,
  ProgressiveSearch,
  ProgressiveStrategy,
  RandomSearch,
  RandomStrategy,
  SuccessiveHalvingSearch,
  autoFit,
  candidateCanonicalBytes,
  candidateHash,
  createCandidate,
  detectTask,
  getPortfolio,
  gridConfigs,
  makeCandidateId,
  normalizeModelSpecs,
  partialShuffle,
  randomConfigs,
  registerBayesianSearch,
  sampleConfig,
  sampleParam,
  scorerGreaterIsBetter,
  seedFor,
} from '@wlearn/automl'
import type {
  AutoMLEstimatorSpec,
  CandidateResult,
  ModelSpec,
} from '@wlearn/automl'
import type {
  Estimator,
  SearchParam,
  SearchSpace,
} from '@wlearn/types'

class TypedModel {
  static readonly classId = 'wlearn.test.typed@1'
  static async create(): Promise<Estimator> {
    throw new Error('type-only fixture')
  }
  static defaultSearchSpace(): SearchSpace { return {} }
}

class MissingClassIdModel {
  static async create(): Promise<Estimator> {
    throw new Error('type-only fixture')
  }
}

const model: ModelSpec = {
  name: 'typed',
  cls: TypedModel,
  portfolioKey: 'linear',
}
const tuple: AutoMLEstimatorSpec = ['typed', TypedModel, { alpha: 1 }]
const inputs = [model, tuple]
const X = { dtype: 'float64' as const, rows: 2, cols: 1, data: new Float64Array([1, 2]) }
const y = new Int32Array([0, 1])
const searchSpace: SearchSpace = {}
const parameter = { type: 'categorical', values: [1, 2] } as SearchParam
const rng = (): number => 0.5

const candidate = createCandidate({
  displayName: 'typed',
  classId: TypedModel.classId,
}, { alpha: 1 })
candidateCanonicalBytes(candidate)
candidateHash(candidate)
makeCandidateId(candidate)
seedFor(candidate, 0, 42)
normalizeModelSpecs(inputs)
sampleParam(parameter, rng)
sampleConfig(searchSpace, rng)
randomConfigs(searchSpace, 2, { seed: 42 })
gridConfigs(searchSpace, { steps: 3 })
detectTask(y)
partialShuffle(new Int32Array([0, 1]), 1, rng)
scorerGreaterIsBetter('accuracy')
getPortfolio('classification')
PORTFOLIO.classification

const random = new RandomSearch(inputs, { nIter: 1 })
const halving = new SuccessiveHalvingSearch(inputs, { nIter: 1 })
const progressive = new ProgressiveSearch(inputs, { nIter: 1 })
const portfolio = new PortfolioSearch(inputs, { task: 'classification' })
const bayesian = new BayesianSearch(inputs, { nIter: 1 })
random.fit(X, y)
halving.refitBest(X, y)
progressive.fit(X, y)
portfolio.fit(X, y)
bayesian.fit(X, y)
registerBayesianSearch(BayesianSearch)
registerBayesianSearch(null)

declare const candidateResult: CandidateResult
const randomStrategy = new RandomStrategy(inputs, { nIter: 1 })
const halvingStrategy = new HalvingStrategy(inputs, { nIter: 1, nSamples: 2 })
const progressiveStrategy = new ProgressiveStrategy(inputs, { nIter: 1 })
const portfolioStrategy = new PortfolioStrategy(inputs, { task: 'classification' })
const bayesianStrategy = new BayesianStrategy(inputs, { nIter: 1 })
for (const strategy of [
  randomStrategy,
  halvingStrategy,
  progressiveStrategy,
  portfolioStrategy,
  bayesianStrategy,
]) {
  strategy.next()
  strategy.report(candidateResult)
  strategy.isDone()
}
bayesianStrategy.init()
bayesianStrategy.dispose()

const leaderboard = new Leaderboard()
leaderboard.add({
  candidateId: makeCandidateId(candidate),
  candidate,
  scores: new Float64Array([1]),
  fitTimeMs: 1,
})
leaderboard.best()
leaderboard.ranked()
leaderboard.top(1)
leaderboard.toJSON()
leaderboard.toArchive()
Leaderboard.fromJSON([])

const executor = new Executor({
  folds: [{ train: new Int32Array([0]), test: new Int32Array([1]) }],
  scoring: 'accuracy',
  X,
  y,
})
const firstExecutorError: unknown | null = executor.firstError
void firstExecutorError
executor.evaluateCandidate({
  candidateId: makeCandidateId(candidate),
  candidate,
  cls: TypedModel,
  params: {},
})
executor.recordFailure(null, new Error('type-only fixture'))
executor.runStrategy(randomStrategy)

autoFit(inputs, X, y, { strategy: 'random', nIter: 1 })

// Tuple inputs have no separate classId field, so their class must expose one.
// @ts-expect-error MissingClassIdModel has no stable classId.
const invalidTuple: AutoMLEstimatorSpec = ['invalid', MissingClassIdModel]
void invalidTuple

// Object inputs must supply classId when the class does not expose it.
// @ts-expect-error Neither object nor class provides a stable classId.
const invalidObject: ModelSpec = { name: 'invalid', cls: MissingClassIdModel }
void invalidObject
