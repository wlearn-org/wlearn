import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { VotingEnsemble } from '../src/voting.js'
import { MockModel } from './mock-model.js'
import { ValidationError, NotFittedError, DisposedError, load } from '@wlearn/core'

const X = { data: new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), rows: 6, cols: 2 }
const yCls = new Int32Array([0, 0, 0, 1, 1, 1])
const yReg = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

describe('VotingEnsemble classification', () => {
  it('creates and fits', async () => {
    const ens = await VotingEnsemble.create({
      estimators: [
        ['m1', MockModel, { task: 'classification' }],
        ['m2', MockModel, { task: 'classification' }],
      ],
      task: 'classification',
    })
    assert.equal(ens.isFitted, false)
    await ens.fit(X, yCls)
    assert.equal(ens.isFitted, true)
    ens.dispose()
  })

  it('predict returns valid labels', async () => {
    const ens = await VotingEnsemble.create({
      estimators: [
        ['m1', MockModel, { task: 'classification' }],
        ['m2', MockModel, { task: 'classification' }],
      ],
      task: 'classification',
    })
    await ens.fit(X, yCls)
    const preds = ens.predict(X)
    assert.equal(preds.length, 6)
    for (const p of preds) {
      assert(p === 0 || p === 1, `unexpected prediction: ${p}`)
    }
    ens.dispose()
  })

  it('predictProba returns correct shape', async () => {
    const ens = await VotingEnsemble.create({
      estimators: [
        ['m1', MockModel, { task: 'classification' }],
        ['m2', MockModel, { task: 'classification' }],
      ],
      task: 'classification',
      voting: 'soft',
    })
    await ens.fit(X, yCls)
    const proba = ens.predictProba(X)
    assert.equal(proba.length, 6 * 2) // 6 samples * 2 classes
    // Probabilities should sum to ~1 per row
    for (let i = 0; i < 6; i++) {
      const sum = proba[i * 2] + proba[i * 2 + 1]
      assert(Math.abs(sum - 1.0) < 1e-9, `row ${i} sums to ${sum}`)
    }
    ens.dispose()
  })

  it('score returns accuracy', async () => {
    const ens = await VotingEnsemble.create({
      estimators: [
        ['m1', MockModel, { task: 'classification' }],
      ],
      task: 'classification',
    })
    await ens.fit(X, yCls)
    const s = ens.score(X, yCls)
    assert(s >= 0 && s <= 1)
    ens.dispose()
  })

  it('custom weights', async () => {
    const ens = await VotingEnsemble.create({
      estimators: [
        ['m1', MockModel, { task: 'classification' }],
        ['m2', MockModel, { task: 'classification', bias: 1 }],
      ],
      weights: [0.9, 0.1],
      task: 'classification',
    })
    await ens.fit(X, yCls)
    const preds = ens.predict(X)
    assert.equal(preds.length, 6)
    ens.dispose()
  })

  it('hard voting', async () => {
    const ens = await VotingEnsemble.create({
      estimators: [
        ['m1', MockModel, { task: 'classification' }],
        ['m2', MockModel, { task: 'classification' }],
      ],
      voting: 'hard',
      task: 'classification',
    })
    await ens.fit(X, yCls)
    const preds = ens.predict(X)
    assert.equal(preds.length, 6)
    // Hard voting should not support predictProba
    assert.throws(() => ens.predictProba(X), ValidationError)
    ens.dispose()
  })

  it('save and load round-trip', async () => {
    const ens = await VotingEnsemble.create({
      estimators: [
        ['m1', MockModel, { task: 'classification' }],
        ['m2', MockModel, { task: 'classification' }],
      ],
      weights: [0.6, 0.4],
      task: 'classification',
    })
    await ens.fit(X, yCls)
    const predsBefore = ens.predict(X)

    const bytes = ens.save()
    assert(bytes instanceof Uint8Array)
    assert(bytes.length > 0)

    const loaded = await VotingEnsemble.load(bytes)
    const predsAfter = loaded.predict(X)
    assert.deepEqual([...predsBefore], [...predsAfter])

    // Also test via registry load
    const fromRegistry = await load(bytes)
    const predsRegistry = fromRegistry.predict(X)
    assert.deepEqual([...predsBefore], [...predsRegistry])

    ens.dispose()
    loaded.dispose()
    fromRegistry.dispose()
  })
})

describe('VotingEnsemble regression', () => {
  it('weighted average predictions', async () => {
    const ens = await VotingEnsemble.create({
      estimators: [
        ['m1', MockModel, { task: 'regression' }],
        ['m2', MockModel, { task: 'regression', bias: 1 }],
      ],
      weights: [0.5, 0.5],
      task: 'regression',
    })
    await ens.fit(X, yReg)
    const preds = ens.predict(X)
    assert.equal(preds.length, 6)
    for (const p of preds) assert(isFinite(p))
    ens.dispose()
  })

  it('score returns r2', async () => {
    const ens = await VotingEnsemble.create({
      estimators: [['m1', MockModel, { task: 'regression' }]],
      task: 'regression',
    })
    await ens.fit(X, yReg)
    const s = ens.score(X, yReg)
    assert(isFinite(s))
    ens.dispose()
  })
})

describe('VotingEnsemble lifecycle', () => {
  it('throws NotFittedError before fit', async () => {
    const ens = await VotingEnsemble.create({
      estimators: [['m1', MockModel, { task: 'classification' }]],
      task: 'classification',
    })
    assert.throws(() => ens.predict(X), NotFittedError)
  })

  it('dispose is idempotent', async () => {
    const ens = await VotingEnsemble.create({
      estimators: [['m1', MockModel, { task: 'classification' }]],
      task: 'classification',
    })
    await ens.fit(X, yCls)
    ens.dispose()
    ens.dispose() // should not throw
  })

  it('throws DisposedError after dispose', async () => {
    const ens = await VotingEnsemble.create({
      estimators: [['m1', MockModel, { task: 'classification' }]],
      task: 'classification',
    })
    await ens.fit(X, yCls)
    ens.dispose()
    assert.throws(() => ens.predict(X), DisposedError)
  })

  it('getParams and setParams', async () => {
    const ens = await VotingEnsemble.create({
      estimators: [['m1', MockModel, { task: 'classification' }]],
      voting: 'soft',
      task: 'classification',
    })
    const p = ens.getParams()
    assert.equal(p.voting, 'soft')
    assert.equal(p.task, 'classification')
    ens.setParams({ voting: 'hard' })
    assert.equal(ens.getParams().voting, 'hard')
  })

  it('capabilities reflect task', async () => {
    const cls = await VotingEnsemble.create({
      estimators: [['m1', MockModel, {}]],
      task: 'classification',
    })
    assert.equal(cls.capabilities.classifier, true)
    assert.equal(cls.capabilities.regressor, false)

    const reg = await VotingEnsemble.create({
      estimators: [['m1', MockModel, {}]],
      task: 'regression',
    })
    assert.equal(reg.capabilities.classifier, false)
    assert.equal(reg.capabilities.regressor, true)
  })
})
