import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)
const { autoFit } = require('@wlearn/automl')

const families = [
  ['liblinear', 'LinearModel', { C: 1 }],
  ['libsvm', 'SVMModel', { C: 1, kernel: 'LINEAR' }],
  ['xgboost', 'XGBModel', { numRound: 3, max_depth: 2 }],
  ['lightgbm', 'LGBModel', { numRound: 3, verbosity: -1, min_data_in_leaf: 1 }],
  ['rf', 'RFModel', { nEstimators: 3, maxDepth: 2 }],
  ['gam', 'GAMModel', { nLambda: 3, nFolds: 0, maxIter: 30 }],
]

for (const [name, exportName, params] of families) {
  test(`${name} AutoML default space respects integer-valued regression`, async () => {
    const Model = require(`@wlearn/${name}`)[exportName]
    const X = Array.from({ length: 12 }, (_, i) => [i / 12, (i % 3) / 3])
    const y = Float64Array.from({ length: 12 }, (_, i) => i - 6)
    const result = await autoFit([{ name, classId: `test.${name}`, cls: Model, params }], X, y, {
      task: 'regression', cv: 2, nIter: 2, seed: 42, ensemble: false
    })
    try {
      assert.equal(result.model.capabilities.regressor, true)
      assert.equal(result.model.capabilities.classifier, false)
      assert.equal(result.archive.records().filter(row => row.status === 'failed').length, 0)
    } finally { result.model.dispose() }
  })
}


for (const labels of [[-5, 9], [-5, 3, 9]]) {
  test(`GAM exposes the common ${labels.length}-class prediction contract`, async () => {
    const core = require('@wlearn/core')
    const { GAMModel } = require('@wlearn/gam')
    const X = Array.from({ length: 24 }, (_, i) => [i % labels.length, i / 24])
    const y = Int32Array.from(X, row => labels[row[0]])
    const model = await GAMModel.create({ task: 'classification', nLambda: 3, maxIter: 50 })
    let restored
    try {
      model.fit(X, y)
      const proba = model.predictProba(X)
      assert.equal(proba.length, X.length * labels.length)
      assert.deepEqual(Array.from(model.classes), labels)
      assert(Array.from(model.predict(X)).every(label => labels.includes(label)))
      core.createPrediction({ truth: y, proba, classes: model.classes })
      restored = await core.load(model.save())
      assert.deepEqual(restored.classes, model.classes)
      assert.deepEqual(restored.predict(X), model.predict(X))
    } finally { model.dispose(); restored?.dispose() }
  })
}


test('GAM probability scoring works through AutoML and its fitted ensemble', async () => {
  const { GAMModel } = require('@wlearn/gam')
  const X = Array.from({ length: 24 }, (_, i) => [i % 3, i / 24])
  const y = Int32Array.from(X, row => [-5, 3, 9][row[0]])
  const result = await autoFit([{ name: 'gam', classId: 'test.gam', cls: GAMModel,
    params: { nLambda: 3, maxIter: 50 }, searchSpace: {} }], X, y, {
    task: 'classification', scoring: 'log_loss', cv: 3, nIter: 1
  })
  try {
    assert.equal(result.archive.records().filter(row => row.status === 'failed').length, 0)
    assert.equal(result.model.predictProba(X).length, X.length * 3)
  } finally { result.model.dispose() }
})
