import { createRequire } from 'node:module'
import { test } from 'node:test'
import assert from 'node:assert/strict'
const require = createRequire(import.meta.url)
const { load, Pipeline } = require('@wlearn/core')
const { CalibratedClassifier, ConformalClassifier, ConformalRegressor, CrossVennAbersClassifier } = require('@wlearn/uncertainty')
const { LinearModel } = require('@wlearn/liblinear')
const { RFModel } = require('@wlearn/rf')
const X = Array.from({ length: 32 }, (_, i) => [Math.sin(i), Math.cos(i)])
const y = X.map(row => row[0] > 0 ? 7 : -3)
const Z = X.map(row => [row[0] + .02, row[1] - .01])

for (const [name, Model, params] of [
  ['linear', LinearModel, {}], ['rf', RFModel, { nTrees: 5 }]
]) for (const Wrapper of [CalibratedClassifier, ConformalClassifier]) {
  test(`${Wrapper.name} owns a real ${name} classifier and reloads its class axis`, async () => {
    const estimator = await Model.create({ ...params, task: 'classification' })
    let model, restored
    try {
      model = await Wrapper.create({ estimator })
      await model.fit(X, y)
      await model.calibrate(Z, y)
      const expected = await model.predictProba(Z)
      assert.equal(expected.length, Z.length * 2)
      restored = await load(model.save())
      assert.deepEqual(await restored.predictProba(Z), expected)
      assert.deepEqual(Array.from(restored.classes()), [-3, 7])
    } finally { restored?.dispose(); if (model) model.dispose(); else estimator.dispose() }
  })
}

test('cross Venn-Abers aligns property-based model classes in complementary folds', async () => {
  const model = await CrossVennAbersClassifier.create({ estimator: ['linear', LinearModel, { task: 'classification' }], cv: 3 })
  let restored
  try {
    await model.fit(X, y)
    const expected = await model.predictProba(Z)
    restored = await load(model.save())
    assert.deepEqual(await restored.predictProba(Z), expected)
  } finally { model.dispose(); restored?.dispose() }
})

test('conformal regression owns a fitted preprocessing Pipeline and persists intervals', async () => {
  const { Preprocessor } = require('@wlearn/preprocess')
  const pipeline = new Pipeline([
    ['prepare', await Preprocessor.create({ scale: 'standard' })],
    ['rf', await RFModel.create({ task: 'regression', nTrees: 5 })]
  ])
  const model = await ConformalRegressor.create({ estimator: pipeline })
  let restored
  try {
    await model.fit(X, X.map(row => row[0] * 2 + row[1]))
    await model.calibrate(Z, Z.map(row => row[0] * 2 + row[1] + .1))
    const intervals = await model.predictInterval(Z, [.8])
    restored = await load(model.save())
    assert.deepEqual(await restored.predictInterval(Z, [.8]), intervals)
  } finally { model.dispose(); restored?.dispose() }
})

const modelCases = [
  ['libsvm', 'SVMModel', { probability: 1 }],
  ['xgboost', 'XGBModel', { numRound: 3, max_depth: 2 }],
  ['lightgbm', 'LGBModel', { numRound: 3, min_data_in_leaf: 1, verbosity: -1 }],
  ['nanoflann', 'KNNModel', { k: 3 }],
  ['ebm', 'EBMModel', { maxRounds: 3, maxInteractions: 0, minSamplesLeaf: 1 }],
  ['stochtree', 'BARTModel', { numTrees: 3, numGfr: 1, numBurnin: 1, numSamples: 2, minSamplesLeaf: 1 }],
  ['tsetlin', 'TsetlinModel', { nClauses: 12, epochs: 2 }],
  ['gam', 'GAMModel', { nLambda: 3, nFolds: 0, maxIter: 30 }],
  ['xlearn', 'XLearnFM', { epoch: 2, k: 2 }],
  ['basis', 'BasisClassifier', { nComponents: 4 }],
  ['nn', 'MLPClassifier', { epochs: 1, hidden_sizes: [4] }],
  ['nn', 'NAMClassifier', { epochs: 1, hidden_sizes: [4] }],
  ['nn', 'TabMClassifier', { epochs: 1, hidden_sizes: [4], n_ensemble: 2 }],
  ['sym', 'SymbolicClassifier', { population: 16, eliteCount: 2, generations: 2, operatorSet: 'basic' }]
]
for (const [packageName, name, params] of modelCases) {
  test(`calibration composes with ${name} and its real artifact loader`, async () => {
    const Model = require('@wlearn/' + packageName)[name]
    const estimator = await Model.create({ ...params, task: 'classification' })
    let model, restored
    try {
      model = await CalibratedClassifier.create({ estimator })
      const labels = y.map(value => value === 7 ? 1 : 0)
      await model.fit(X, labels)
      await model.calibrate(Z, labels)
      const probabilities = await model.predictProba(Z)
      assert.equal(probabilities.length, 2 * Z.length)
      assert(probabilities.every(Number.isFinite))
      restored = await load(model.save())
      const actual = await restored.predictProba(Z)
      const error = Math.max(...actual.map((v, i) => Math.abs(v - probabilities[i])))
      // LIBSVM serializes support vectors as decimal text; the ecosystem
      // roundtrip contract permits 1e-5 absolute prediction error.
      assert(error <= 1e-5, `${name} roundtrip error ${error}`)
    } finally { restored?.dispose(); if (model) model.dispose(); else estimator.dispose() }
  })
}

test('cross Venn-Abers waits for objective-dependent fold capabilities', async () => {
  const { XGBModel } = require('@wlearn/xgboost')
  const model = await CrossVennAbersClassifier.create({ estimator: ['xgb', XGBModel, { numRound: 2 }], cv: 3 })
  try {
    await model.fit(X, y.map(v => v === 7 ? 1 : 0))
    assert((await model.predictProba(Z)).every(Number.isFinite))
  } finally { model.dispose() }
})
