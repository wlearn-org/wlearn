import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { BaggedEstimator } from '../src/bagging.js'
import { MockModel } from './mock-model.js'
import { ValidationError, NotFittedError, DisposedError } from '@wlearn/core'

const X = {
  data: new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
  rows: 10, cols: 2
}
const yCls = new Int32Array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
const yReg = new Float64Array([1.1, 2.3, 3.7, 4.2, 5.8, 6.1, 7.5, 8.9, 9.4, 10.6])

describe('BaggedEstimator classification', () => {
  it('fit and predict', async () => {
    const bag = await BaggedEstimator.create({
      estimator: ['mock', MockModel, { task: 'classification' }],
      kFold: 2,
      task: 'classification',
    })
    await bag.fit(X, yCls)
    assert(bag.isFitted)
    const preds = bag.predict(X)
    assert.equal(preds.length, 10)
    bag.dispose()
  })

  it('predictProba returns probabilities', async () => {
    const bag = await BaggedEstimator.create({
      estimator: ['mock', MockModel, { task: 'classification' }],
      kFold: 2,
      task: 'classification',
    })
    await bag.fit(X, yCls)
    const proba = bag.predictProba(X)
    // 10 samples * 2 classes = 20
    assert.equal(proba.length, 20)
    // Probabilities should be >= 0
    for (let i = 0; i < proba.length; i++) {
      assert(proba[i] >= 0, `proba[${i}] = ${proba[i]} < 0`)
    }
    bag.dispose()
  })

  it('score returns accuracy', async () => {
    const bag = await BaggedEstimator.create({
      estimator: ['mock', MockModel, { task: 'classification' }],
      kFold: 2,
      task: 'classification',
    })
    await bag.fit(X, yCls)
    const s = bag.score(X, yCls)
    assert(s >= 0 && s <= 1)
    bag.dispose()
  })

  it('oofPredictions has correct shape', async () => {
    const bag = await BaggedEstimator.create({
      estimator: ['mock', MockModel, { task: 'classification' }],
      kFold: 2,
      task: 'classification',
    })
    await bag.fit(X, yCls)
    const oof = bag.oofPredictions
    // 10 samples * 2 classes
    assert.equal(oof.length, 20)
    bag.dispose()
  })

  it('classes are detected', async () => {
    const bag = await BaggedEstimator.create({
      estimator: ['mock', MockModel, { task: 'classification' }],
      kFold: 2,
      task: 'classification',
    })
    await bag.fit(X, yCls)
    assert.deepEqual([...bag.classes], [0, 1])
    bag.dispose()
  })
})

describe('BaggedEstimator regression', () => {
  it('fit and predict', async () => {
    const bag = await BaggedEstimator.create({
      estimator: ['mock', MockModel, { task: 'regression' }],
      kFold: 2,
      task: 'regression',
    })
    await bag.fit(X, yReg)
    assert(bag.isFitted)
    const preds = bag.predict(X)
    assert.equal(preds.length, 10)
    bag.dispose()
  })

  it('score returns r2', async () => {
    const bag = await BaggedEstimator.create({
      estimator: ['mock', MockModel, { task: 'regression' }],
      kFold: 2,
      task: 'regression',
    })
    await bag.fit(X, yReg)
    const s = bag.score(X, yReg)
    assert(isFinite(s))
    bag.dispose()
  })

  it('oofPredictions has correct shape', async () => {
    const bag = await BaggedEstimator.create({
      estimator: ['mock', MockModel, { task: 'regression' }],
      kFold: 2,
      task: 'regression',
    })
    await bag.fit(X, yReg)
    const oof = bag.oofPredictions
    assert.equal(oof.length, 10)
    bag.dispose()
  })

  it('predictProba throws for regression', async () => {
    const bag = await BaggedEstimator.create({
      estimator: ['mock', MockModel, { task: 'regression' }],
      kFold: 2,
      task: 'regression',
    })
    await bag.fit(X, yReg)
    assert.throws(() => bag.predictProba(X), ValidationError)
    bag.dispose()
  })
})

describe('BaggedEstimator nRepeats', () => {
  it('multiple repeats create more fold models', async () => {
    const bag = await BaggedEstimator.create({
      estimator: ['mock', MockModel, { task: 'classification' }],
      kFold: 2,
      nRepeats: 2,
      task: 'classification',
    })
    await bag.fit(X, yCls)
    assert(bag.isFitted)
    const preds = bag.predict(X)
    assert.equal(preds.length, 10)
    bag.dispose()
  })
})

describe('BaggedEstimator save/load', () => {
  it('round-trip preserves predictions', async () => {
    const bag = await BaggedEstimator.create({
      estimator: ['mock', MockModel, { task: 'classification' }],
      kFold: 2,
      task: 'classification',
    })
    await bag.fit(X, yCls)
    const preds1 = bag.predict(X)

    const bytes = bag.save()
    const bag2 = await BaggedEstimator.load(bytes)
    const preds2 = bag2.predict(X)

    assert.equal(preds1.length, preds2.length)
    for (let i = 0; i < preds1.length; i++) {
      assert(Math.abs(preds1[i] - preds2[i]) < 1e-10)
    }
    bag.dispose()
    bag2.dispose()
  })
})

describe('BaggedEstimator lifecycle', () => {
  it('throws NotFittedError before fit', async () => {
    const bag = await BaggedEstimator.create({
      estimator: ['mock', MockModel, { task: 'classification' }],
      kFold: 2,
      task: 'classification',
    })
    assert.throws(() => bag.predict(X), NotFittedError)
    bag.dispose()
  })

  it('throws DisposedError after dispose', async () => {
    const bag = await BaggedEstimator.create({
      estimator: ['mock', MockModel, { task: 'classification' }],
      kFold: 2,
      task: 'classification',
    })
    bag.dispose()
    assert.throws(() => bag.predict(X), DisposedError)
  })

  it('getParams returns config', async () => {
    const bag = await BaggedEstimator.create({
      estimator: ['mock', MockModel, { task: 'classification' }],
      kFold: 3,
      nRepeats: 2,
      seed: 123,
      task: 'classification',
    })
    const p = bag.getParams()
    assert.equal(p.kFold, 3)
    assert.equal(p.nRepeats, 2)
    assert.equal(p.seed, 123)
    assert.equal(p.task, 'classification')
    bag.dispose()
  })

  it('capabilities reflect task', async () => {
    const clsBag = await BaggedEstimator.create({
      estimator: ['mock', MockModel, {}],
      task: 'classification',
    })
    assert(clsBag.capabilities.classifier)
    assert(!clsBag.capabilities.regressor)
    assert(clsBag.capabilities.predictProba)
    clsBag.dispose()

    const regBag = await BaggedEstimator.create({
      estimator: ['mock', MockModel, {}],
      task: 'regression',
    })
    assert(!regBag.capabilities.classifier)
    assert(regBag.capabilities.regressor)
    assert(!regBag.capabilities.predictProba)
    regBag.dispose()
  })
})
