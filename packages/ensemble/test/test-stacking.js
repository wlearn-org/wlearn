const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const { StackingEnsemble } = require('../src/stacking.js')
const { MockModel } = require('./mock-model.js')
const { ValidationError, NotFittedError, DisposedError, load } = require('@wlearn/core')

const X = { data: new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]), rows: 10, cols: 2 }
const yCls = new Int32Array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
const yReg = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

describe('StackingEnsemble classification', () => {
  it('creates, fits, and predicts', async () => {
    const stk = await StackingEnsemble.create({
      estimators: [
        ['m1', MockModel, { task: 'classification' }],
        ['m2', MockModel, { task: 'classification' }],
      ],
      finalEstimator: ['meta', MockModel, { task: 'classification' }],
      cv: 3,
      task: 'classification',
    })
    assert.equal(stk.isFitted, false)
    await stk.fit(X, yCls)
    assert.equal(stk.isFitted, true)

    const preds = stk.predict(X)
    assert.equal(preds.length, 10)
    for (const p of preds) {
      assert(p === 0 || p === 1, `unexpected: ${p}`)
    }
    stk.dispose()
  })

  it('score returns accuracy', async () => {
    const stk = await StackingEnsemble.create({
      estimators: [['m1', MockModel, { task: 'classification' }]],
      finalEstimator: ['meta', MockModel, { task: 'classification' }],
      cv: 2,
      task: 'classification',
    })
    await stk.fit(X, yCls)
    const s = stk.score(X, yCls)
    assert(s >= 0 && s <= 1)
    stk.dispose()
  })

  it('predictProba returns correct shape', async () => {
    const stk = await StackingEnsemble.create({
      estimators: [['m1', MockModel, { task: 'classification' }]],
      finalEstimator: ['meta', MockModel, { task: 'classification' }],
      cv: 2,
      task: 'classification',
    })
    await stk.fit(X, yCls)
    const proba = stk.predictProba(X)
    assert.equal(proba.length, 10 * 2)
    stk.dispose()
  })

  it('save and load round-trip', async () => {
    const stk = await StackingEnsemble.create({
      estimators: [
        ['m1', MockModel, { task: 'classification' }],
        ['m2', MockModel, { task: 'classification' }],
      ],
      finalEstimator: ['meta', MockModel, { task: 'classification' }],
      cv: 2,
      task: 'classification',
    })
    await stk.fit(X, yCls)
    const predsBefore = stk.predict(X)

    const bytes = stk.save()
    assert(bytes instanceof Uint8Array)

    const loaded = await StackingEnsemble.load(bytes)
    const predsAfter = loaded.predict(X)
    assert.deepEqual([...predsBefore], [...predsAfter])

    // Also via registry
    const fromReg = await load(bytes)
    assert.deepEqual([...predsBefore], [...fromReg.predict(X)])

    stk.dispose()
    loaded.dispose()
    fromReg.dispose()
  })

  it('passthrough includes original features', async () => {
    const stk = await StackingEnsemble.create({
      estimators: [['m1', MockModel, { task: 'classification' }]],
      finalEstimator: ['meta', MockModel, { task: 'classification' }],
      cv: 2,
      task: 'classification',
      passthrough: true,
    })
    await stk.fit(X, yCls)
    // nMetaCols = 1 model * 2 classes + 2 original cols = 4
    const params = stk.getParams()
    assert.equal(params.passthrough, true)
    const preds = stk.predict(X)
    assert.equal(preds.length, 10)
    stk.dispose()
  })
})

describe('StackingEnsemble regression', () => {
  it('fits and predicts', async () => {
    const stk = await StackingEnsemble.create({
      estimators: [
        ['m1', MockModel, { task: 'regression' }],
        ['m2', MockModel, { task: 'regression', bias: 1 }],
      ],
      finalEstimator: ['meta', MockModel, { task: 'regression' }],
      cv: 2,
      task: 'regression',
    })
    await stk.fit(X, yReg)
    const preds = stk.predict(X)
    assert.equal(preds.length, 10)
    for (const p of preds) assert(isFinite(p))
    stk.dispose()
  })

  it('score returns r2', async () => {
    const stk = await StackingEnsemble.create({
      estimators: [['m1', MockModel, { task: 'regression' }]],
      finalEstimator: ['meta', MockModel, { task: 'regression' }],
      cv: 2,
      task: 'regression',
    })
    await stk.fit(X, yReg)
    const s = stk.score(X, yReg)
    assert(isFinite(s))
    stk.dispose()
  })
})

describe('StackingEnsemble lifecycle', () => {
  it('throws NotFittedError before fit', async () => {
    const stk = await StackingEnsemble.create({
      estimators: [['m1', MockModel, {}]],
      finalEstimator: ['meta', MockModel, {}],
      task: 'classification',
    })
    assert.throws(() => stk.predict(X), NotFittedError)
  })

  it('throws without finalEstimator', async () => {
    const stk = await StackingEnsemble.create({
      estimators: [['m1', MockModel, { task: 'classification' }]],
      task: 'classification',
    })
    await assert.rejects(() => stk.fit(X, yCls), ValidationError)
  })

  it('dispose is idempotent', async () => {
    const stk = await StackingEnsemble.create({
      estimators: [['m1', MockModel, { task: 'classification' }]],
      finalEstimator: ['meta', MockModel, { task: 'classification' }],
      cv: 2,
      task: 'classification',
    })
    await stk.fit(X, yCls)
    stk.dispose()
    stk.dispose()
  })

  it('throws DisposedError after dispose', async () => {
    const stk = await StackingEnsemble.create({
      estimators: [['m1', MockModel, { task: 'classification' }]],
      finalEstimator: ['meta', MockModel, { task: 'classification' }],
      cv: 2,
      task: 'classification',
    })
    await stk.fit(X, yCls)
    stk.dispose()
    assert.throws(() => stk.predict(X), DisposedError)
  })

  it('getParams and setParams', async () => {
    const stk = await StackingEnsemble.create({
      estimators: [['m1', MockModel, {}]],
      finalEstimator: ['meta', MockModel, {}],
      cv: 3,
      task: 'classification',
    })
    const p = stk.getParams()
    assert.equal(p.cv, 3)
    assert.equal(p.passthrough, false)
    stk.setParams({ cv: 5 })
    assert.equal(stk.getParams().cv, 5)
  })

  for (const failure of ['later-base', 'meta']) {
    it(`transactionally releases full-data children after ${failure} failure`, async () => {
      const events = []
      const live = { count: 0 }
      const fitError = new Error(`${failure} fit failed`)
      function modelClass(label, { failFull = false, cleanupThrows = false } = {}) {
        return class {
          #disposed = false
          #full = false
          static async create() {
            live.count++
            return new this()
          }
          fit(X) {
            this.#full = X.rows === 10
            if (this.#full && failFull) throw fitError
            return this
          }
          predict(X) { return new Int32Array(X.rows) }
          predictProba(X) {
            const out = new Float64Array(X.rows * 2)
            out.fill(0.5)
            return out
          }
          dispose() {
            if (this.#disposed) return
            this.#disposed = true
            live.count--
            events.push(`${label}:dispose:${this.#full ? 'full' : 'oof'}`)
            if (this.#full && cleanupThrows) {
              throw new Error(`${label} cleanup failed`)
            }
          }
        }
      }
      const Base1 = modelClass('base1', { cleanupThrows: true })
      const Base2 = modelClass('base2', {
        failFull: failure === 'later-base',
      })
      const Meta = modelClass('meta', { failFull: failure === 'meta' })
      const ensemble = await StackingEnsemble.create({
        estimators: [['base1', Base1, {}], ['base2', Base2, {}]],
        finalEstimator: ['meta', Meta, {}],
        cv: 2,
        task: 'classification',
      })

      await assert.rejects(() => ensemble.fit(X, yCls), error => error === fitError)
      assert.equal(live.count, 0)
      const fullDisposals = events.filter(event => event.endsWith(':full'))
      assert.deepEqual(fullDisposals, failure === 'later-base'
        ? ['base2:dispose:full', 'base1:dispose:full']
        : ['meta:dispose:full', 'base2:dispose:full', 'base1:dispose:full'])
      ensemble.dispose()
      assert.equal(live.count, 0)
    })
  }
})
