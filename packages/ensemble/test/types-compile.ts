import {
  BaggedEstimator,
  StackingEnsemble,
  VotingEnsemble,
  caruanaSelect,
  optimizeWeights,
  projectSimplex,
  type EstimatorSpec
} from '@wlearn/ensemble'
import type { EstimatorClass } from '@wlearn/types'

const Model = null as unknown as EstimatorClass
const spec: EstimatorSpec = ['base', Model, { seed: 7 }]
const X = [[0], [1], [2], [3]]
const y = new Int32Array([0, 0, 1, 1])

async function compileSurface(): Promise<void> {
  const vote = await VotingEnsemble.create({ estimators: [spec] })
  await vote.fit(X, y)
  vote.setParams({ voting: 'hard' })
  // @ts-expect-error constructor-only fields are not mutable
  vote.setParams({ task: 'regression' })

  const bag = await BaggedEstimator.create({ estimator: spec, kFold: 2 })
  await bag.fit(X, y)
  bag.setParams({ nRepeats: 2 })
  // @ts-expect-error constructor-only fields are not mutable
  bag.setParams({ estimator: spec })

  const stack = await StackingEnsemble.create({
    estimators: [spec, ['prefitted', bag]],
    finalEstimator: ['meta', Model],
    cv: 2
  })
  await stack.fit(X, y)
  stack.setParams({ passthrough: true })
  // @ts-expect-error constructor-only fields are not mutable
  stack.setParams({ finalEstimator: spec })

  const candidates = [bag.oofPredictions, bag.oofPredictions]
  const selected = caruanaSelect(candidates, y, { refineWeights: true })
  optimizeWeights(candidates, y, selected.weights, { classes: new Int32Array([0, 1]) })
  projectSimplex([0.25, 0.75])
}

void compileSurface
