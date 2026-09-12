const { it } = require('node:test')
const assert = require('node:assert/strict')
const { MultiOutputRegressor, MultiLabelClassifier } = require('../src/index.js')
const { MockModel } = require('./mock-model.js')
const { Pipeline, load, validateBundle, encodeBundle, ValidationError, NotFittedError, DisposedError } = require('@wlearn/core')

const X = { data: Float64Array.of(1, 2, 3, 4), rows: 4, cols: 1 }
const Y = [[1, 10], [2, 20], [3, 30], [4, 40]]

it('multioutput preserves target axes, distinct means and nested artifacts', async () => {
  const m = await MultiOutputRegressor.create({ estimator: ['mean', MockModel], targetNames: ['small', 'large'] })
  await m.fit(X, Y)
  const pred = m.predict(X)
  assert.equal(typeof pred.then, 'undefined')
  assert.deepEqual(Array.from(pred.data), [2.5, 25, 2.5, 25, 2.5, 25, 2.5, 25])
  assert.equal(m.score(X, Y), 0)
  const restored = await load(m.save())
  assert.deepEqual(restored.predict(X), pred)
  assert.deepEqual(validateBundle(restored.save()).manifest, validateBundle(m.save()).manifest)
  assert.throws(() => m.predict({ data: Float64Array.of(1, 2), rows: 1, cols: 2 }), ValidationError)
  await assert.rejects(restored.fit(X, Y), /specification/)
  restored.setParams({ estimator: ['mean', MockModel] })
  assert.throws(() => restored.predict(X), NotFittedError)
  await restored.fit(X, Y)
  const p = new Pipeline([['heads', restored]])
  await p.fit(X, Y)
  const copy = await load(p.save())
  assert.deepEqual(copy.predict(X), pred)
  copy.dispose(); p.dispose(); m.dispose()
  assert.throws(() => m.predict(X), DisposedError)
})

it('multilabel keeps constant heads and remaps descending probability columns', async () => {
  const labels = [[0, 0, 1], [1, 0, 1], [1, 0, 1], [0, 0, 1]]
  const m = new MultiLabelClassifier({ estimator: ['binary', MockModel, { classOrder: 'descending' }] })
  await m.fit(X, labels)
  const p = m.predictProba(X)
  assert.equal(p.cols, 3)
  for (let r = 0; r < p.rows; r++) {
    assert.equal(p.data[r * 3], 0.1)
    assert.equal(p.data[r * 3 + 1], 0)
    assert.equal(p.data[r * 3 + 2], 1)
  }
  const restored = await load(m.save())
  assert.deepEqual(restored.predictProba(X), p)
  restored.dispose(); m.dispose()
})

it('all-constant multilabel artifacts need no synthetic child model', async () => {
  const m = new MultiLabelClassifier({ estimator: ['binary', MockModel] })
  await m.fit(X, [[0, 1], [0, 1], [0, 1], [0, 1]])
  assert.equal(validateBundle(m.save()).toc.length, 0)
  const copy = await load(m.save())
  assert.equal(copy.score(X, [[0, 1], [0, 1], [0, 1], [0, 1]]), 1)
  m.dispose(); copy.dispose()
})

it('failed replacement disposes new heads and preserves the fitted model', async () => {
  let fail = false, created = 0, disposed = 0
  class Fragile extends MockModel {
    static async create(params) { created++; return new Fragile(params) }
    fit(X, y) { if (fail && y[0] > 1) throw new Error('head failed'); return super.fit(X, y) }
    dispose() { disposed++; super.dispose() }
  }
  const m = new MultiOutputRegressor({ estimator: ['m', Fragile] })
  await m.fit(X, Y)
  const before = m.predict(X)
  fail = true
  await assert.rejects(m.fit(X, Y), /head failed/)
  assert.equal(created, 4); assert.equal(disposed, 2)
  assert.deepEqual(m.predict(X), before)
  m.dispose(); assert.equal(disposed, 4)
})

it('multi-target loaders reject missing heads before loading child artifacts', async () => {
  const m = new MultiOutputRegressor({ estimator: ['mean', MockModel] })
  await m.fit(X, Y)
  const { manifest } = validateBundle(m.save())
  await assert.rejects(load(encodeBundle(manifest, [])), ValidationError)
  m.dispose()
})

it('public CV and Measure preserve matrix targets and multilabel probability meaning', async () => {
  const { crossValScore, getScorer } = require('@wlearn/core')
  const scores = await crossValScore(MultiOutputRegressor, X, Y, {
    task: 'multioutput', cv: 2, scoring: 'mse', params: { estimator: ['mean', MockModel] }
  })
  assert.equal(scores.length, 2)
  assert(scores.every(Number.isFinite))
  const labels = [[0, 0], [1, 0], [1, 1], [0, 1]]
  const cls = await crossValScore(MultiLabelClassifier, X, labels, {
    task: 'multilabel', cv: 2, scoring: 'multilabel_log_loss', params: { estimator: ['binary', MockModel] }
  })
  assert(cls.every(Number.isFinite))
  assert.equal(getScorer('hamming_loss')([[0, 1], [1, 1]], [[0, 0], [1, 1]], { taskKind: 'multilabel' }), 0.25)
  assert.equal(getScorer('subset_accuracy')([[0, 1], [1, 1]], [[0, 0], [1, 1]], { taskKind: 'multilabel' }), 0.5)
})

it('does not relabel child quantile levels or return disposed async predictions', async () => {
  const { createPrediction } = require('@wlearn/core')
  let resolve
  class Quantiles extends MockModel {
    static async create(params) { return new Quantiles(params) }
    get capabilities() { return { ...super.capabilities, predictQuantiles: true } }
    predictQuantiles(X) {
      return createPrediction({ rows: X.rows, quantiles: new Float64Array(X.rows * 2), quantileLevels: [0.2, 0.8] })
    }
    predict(X) { const p = super.predict(X); return new Promise(r => { resolve = () => r(p) }) }
  }
  const m = new MultiOutputRegressor({ estimator: ['q', Quantiles] })
  await m.fit(X, [[1], [2], [3], [4]])
  assert.throws(() => m.predictQuantiles(X, [0.1, 0.9]), ValidationError)
  assert.equal(m.predictQuantiles(X, [0.2, 0.8]).quantiles.length, 8)
  const prediction = m.predict(X)
  m.dispose()
  resolve()
  await assert.rejects(prediction, DisposedError)
})
