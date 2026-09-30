import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
const require = createRequire(import.meta.url)
const { autoFit } = require('@wlearn/automl')
const { LinearModel } = require('@wlearn/liblinear')
const { load } = require('@wlearn/core')

const X = Array.from({ length: 24 }, (_, i) => [i - 12, i % 3])
const y = Int32Array.from(X, row => +(row[0] >= 0))

for (const seed of [0, 1, 42]) {
  test(`default AutoML ensembles public liblinear candidates (seed ${seed})`, async () => {
    const result = await autoFit([{name: 'linear', classId: 'wlearn.liblinear.classifier@1', cls: LinearModel}], X, y, {
      nIter: 6, cv: 2, seed, task: 'classification'
    })
    let restored
    try {
      assert(result.leaderboard.every(e => e.supportsPredictProba))
      assert(result.model.predictProba(X).every(Number.isFinite))
      restored = await load(result.model.save())
      assert.deepEqual(restored.predict(X), result.model.predict(X))
    } finally { restored?.dispose(); result.model.dispose() }
  })
}

test('explicit liblinear SVM remains searchable with default ensembling', async () => {
  const result = await autoFit([{
    name: 'svm', classId: 'wlearn.liblinear.classifier@1', cls: LinearModel, searchSpace: {},
    params: { solver: 'L2R_L2LOSS_SVC_DUAL' }
  }], X, y, { nIter: 1, cv: 2, task: 'classification', preprocess: true })
  try {
    assert.equal(result.leaderboard[0].supportsPredictProba, false)
    assert.equal(result.model.capabilities.predictProba, false)
    assert.equal(result.model.predict(X).length, y.length)
  } finally { result.model.dispose() }
})
