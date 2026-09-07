const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const { StackingEnsemble } = require('../src/stacking.js')
const { MockModel } = require('./mock-model.js')
const { ValidationError, NotFittedError, DisposedError, load } = require('@wlearn/core')

const X = { data: new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]), rows: 10, cols: 2 }
const yCls = new Int32Array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
const yReg = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

it('stacking fits and persists a custom CV plan', async () => {
  const { createResamplingPlan } = require('@wlearn/core')
  const model = await StackingEnsemble.create({
    estimators: [['base', MockModel, {}]], finalEstimator: ['meta', MockModel, {}],
    cv: createResamplingPlan({ n: 10, k: 2 }), task: 'regression'
  })
  await model.fit(X, yReg)
  const restored = await load(model.save())
  assert.deepEqual(restored.predict(X), model.predict(X))
  restored.dispose()
  model.dispose()
})

class HardMetaMock extends MockModel {
  static async create(params = {}) { return new HardMetaMock(params) }
  get capabilities() { return { ...super.capabilities, predictProba: false } }
}

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
    assert(preds instanceof Int32Array)
    assert.equal(preds.length, 10)
    for (const p of preds) {
      assert(p === 0 || p === 1, `unexpected: ${p}`)
    }
    stk.dispose()
  })

  it('derives probability capability from the fitted meta-model', async () => {
    const stk = await StackingEnsemble.create({
      estimators: [['m1', MockModel, { task: 'classification' }]],
      finalEstimator: ['meta', HardMetaMock, { task: 'classification' }],
      cv: 2,
      task: 'classification',
    })
    await stk.fit(X, yCls)
    assert.equal(stk.capabilities.predictProba, false)
    assert(stk.predict(X) instanceof Int32Array)
    assert.throws(
      () => stk.predictProba(X),
      error => error instanceof ValidationError && /does not support/.test(error.message)
    )
    stk.dispose()
  })

  it('rejects a base model without an explicit probability capability', async () => {
    const stk = await StackingEnsemble.create({
      estimators: [['base', HardMetaMock, { task: 'classification' }]],
      finalEstimator: ['meta', MockModel, { task: 'classification' }],
      cv: 2,
      task: 'classification',
    })
    await assert.rejects(
      () => stk.fit(X, yCls),
      error => error instanceof ValidationError && /capability/.test(error.message)
    )
    assert.equal(stk.isFitted, false)
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

  it('aligns base and meta-model probability columns', async () => {
    const labels = new Int32Array([2, 2, 2, 2, 2, 2, 1, 1, 1, 1])
    const params = { task: 'classification', classOrder: 'descending' }
    const stk = await StackingEnsemble.create({
      estimators: [['reversed', MockModel, params]],
      finalEstimator: ['meta', MockModel, params],
      cv: 2,
      task: 'classification',
    })
    await stk.fit(X, labels)
    assert.deepEqual([...stk.classes], [1, 2])
    const proba = stk.predictProba(X)
    assert(Math.abs(proba[0] - 0.1) < 1e-12)
    assert(Math.abs(proba[1] - 0.9) < 1e-12)
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

  it('rejects malformed base OOF predictions', async () => {
    class ShortRegressionMock extends MockModel {
      static async create(params = {}) {
        return new ShortRegressionMock({ task: 'regression', ...params })
      }
      predict(input) {
        return new Float64Array(Math.max(0, input.rows - 1))
      }
    }
    const stk = await StackingEnsemble.create({
      estimators: [['short', ShortRegressionMock, {}]],
      finalEstimator: ['meta', MockModel, { task: 'regression' }],
      cv: 2,
      task: 'regression',
    })
    await assert.rejects(() => stk.fit(X, yReg), /wrong shape/)
    stk.dispose()
  })

  it('validates regression base and meta predictions at inference', async () => {
    class ToggleRegressionMock extends MockModel {
      static instances = []
      static async create(params = {}) {
        const model = new ToggleRegressionMock({ task: 'regression', ...params })
        ToggleRegressionMock.instances.push(model)
        return model
      }
      malformed = false
      predict(input) {
        if (this.malformed) return new Float64Array(input.rows).fill(NaN)
        return super.predict(input)
      }
    }
    const baseMalformed = await StackingEnsemble.create({
      estimators: [['base', ToggleRegressionMock, {}]],
      finalEstimator: ['meta', MockModel, { task: 'regression' }],
      cv: 2,
      task: 'regression',
    })
    await baseMalformed.fit(X, yReg)
    for (const model of ToggleRegressionMock.instances) model.malformed = true
    assert.throws(() => baseMalformed.predict(X), /finite numbers/)
    baseMalformed.dispose()

    ToggleRegressionMock.instances = []
    const metaMalformed = await StackingEnsemble.create({
      estimators: [['base', MockModel, { task: 'regression' }]],
      finalEstimator: ['meta', ToggleRegressionMock, {}],
      cv: 2,
      task: 'regression',
    })
    await metaMalformed.fit(X, yReg)
    for (const model of ToggleRegressionMock.instances) model.malformed = true
    assert.throws(() => metaMalformed.predict(X), /finite numbers/)
    metaMalformed.dispose()
  })
})

describe('StackingEnsemble lifecycle', () => {
  it('rejects lifecycle races while child creation is pending', async () => {
    class DeferredCreateModel extends MockModel {
      static first = true
      static release = null
      static disposed = 0
      static create(params = {}) {
        if (!DeferredCreateModel.first) {
          return Promise.resolve(new DeferredCreateModel(params))
        }
        DeferredCreateModel.first = false
        return new Promise(resolve => {
          DeferredCreateModel.release = () => resolve(
            new DeferredCreateModel(params)
          )
        })
      }
      dispose() {
        super.dispose()
        DeferredCreateModel.disposed++
      }
    }
    const ensemble = await StackingEnsemble.create({
      estimators: [[
        'deferred', DeferredCreateModel, { task: 'classification' },
      ]],
      finalEstimator: ['meta', MockModel, { task: 'classification' }],
      cv: 2,
      task: 'classification',
    })
    const pending = ensemble.fit(X, yCls)
    await assert.rejects(() => ensemble.fit(X, yCls), /already in progress/)
    assert.throws(() => ensemble.setParams({ seed: 9 }), /in progress/)
    assert.throws(() => ensemble.dispose(), /in progress/)
    DeferredCreateModel.release()
    await pending
    assert.equal(ensemble.isFitted, true)
    ensemble.dispose()
    assert.equal(DeferredCreateModel.disposed, 3)
  })

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
    assert.equal(stk.isFitted, false)
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

  it('invalidates fitted state when training parameters change', async () => {
    const stk = await StackingEnsemble.create({
      estimators: [['m1', MockModel, { task: 'classification' }]],
      finalEstimator: ['meta', MockModel, { task: 'classification' }],
      cv: 2,
      task: 'classification',
    })
    await stk.fit(X, yCls)
    assert.equal(stk.isFitted, true)
    stk.setParams({ seed: 123 })
    assert.equal(stk.isFitted, false)
    assert.throws(() => stk.save(), NotFittedError)
    stk.dispose()
  })

  it('rejects invalid training config transactionally', async () => {
    const stk = await StackingEnsemble.create({
      estimators: [['m1', MockModel, { task: 'classification' }]],
      finalEstimator: ['meta', MockModel, { task: 'classification' }],
      cv: 2,
      task: 'classification',
    })
    await stk.fit(X, yCls)
    assert.throws(() => stk.setParams({ cv: 1 }), ValidationError)
    assert.throws(() => stk.setParams({ task: 'regression' }), /Unknown.*task/)
    assert.equal(stk.getParams().cv, 2)
    assert.equal(stk.isFitted, true)
    const invalid = await StackingEnsemble.create({
      estimators: [['m1', MockModel, {}]],
      finalEstimator: ['meta', MockModel, {}],
      cv: 2,
      task: 'unknown',
    })
    await assert.rejects(() => invalid.fit(X, yCls), ValidationError)
    stk.dispose()
    invalid.dispose()
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
          get classes() { return new Int32Array([0, 1]) }
          get capabilities() { return { predictProba: true } }
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

  it('accepts a fitted BaggedEstimator without retraining it', async () => {
    const { BaggedEstimator } = require('../src/bagging.js')
    const bagged = await BaggedEstimator.create({
      estimator: ['bag-base', MockModel, { task: 'classification' }],
      kFold: 2,
      task: 'classification',
    })
    await bagged.fit(X, yCls)
    const stacking = await StackingEnsemble.create({
      estimators: [['bagged', bagged]],
      finalEstimator: ['meta', MockModel, { task: 'classification' }],
      cv: 2,
      task: 'classification',
    })
    await stacking.fit(X, yCls)
    assert.equal(stacking.isFitted, true)
    assert.equal(stacking.predict(X).length, X.rows)
    stacking.dispose()
    assert.equal(bagged.isFitted, false, 'ownership transfers after stacking fit commits')
  })

  it('rejects a pre-fitted BaggedEstimator with incompatible OOF rows', async () => {
    const { BaggedEstimator } = require('../src/bagging.js')
    const bagged = await BaggedEstimator.create({
      estimator: ['bag-base', MockModel, { task: 'classification' }],
      kFold: 2,
      task: 'classification',
    })
    await bagged.fit(X, yCls)
    const shortX = { data: X.data.slice(0, 16), rows: 8, cols: 2 }
    const shortY = yCls.slice(0, 8)
    const stacking = await StackingEnsemble.create({
      estimators: [['bagged', bagged]],
      finalEstimator: ['meta', MockModel, { task: 'classification' }],
      cv: 2,
      task: 'classification',
    })
    await assert.rejects(
      () => stacking.fit(shortX, shortY),
      error => error instanceof ValidationError && /OOF shape/.test(error.message)
    )
    assert.equal(bagged.isFitted, true, 'rejected stacking fit does not take ownership')
    stacking.dispose()
    bagged.dispose()
  })

  it('rejects non-finite pre-fitted OOF data without taking ownership', async () => {
    let disposed = false
    const oof = new Float64Array(X.rows * 2).fill(0.5)
    oof[0] = NaN
    const prefitted = {
      isFitted: true,
      get oofPredictions() { return oof },
      get classes() { return new Int32Array([0, 1]) },
      getParams() { return { task: 'classification' } },
      dispose() { disposed = true },
    }
    const stacking = await StackingEnsemble.create({
      estimators: [['prefitted', prefitted]],
      finalEstimator: ['meta', MockModel, { task: 'classification' }],
      cv: 2,
      task: 'classification',
    })
    await assert.rejects(
      () => stacking.fit(X, yCls),
      error => error instanceof ValidationError && /OOF predictions must be finite/.test(error.message)
    )
    assert.equal(disposed, false)
    assert.equal(stacking.isFitted, false)
    stacking.dispose()
  })

  it('rejects pre-fitted bagging task and class mismatches without ownership', async () => {
    const { BaggedEstimator } = require('../src/bagging.js')
    const classificationBag = await BaggedEstimator.create({
      estimator: ['bag-base', MockModel, { task: 'classification' }],
      kFold: 2,
      task: 'classification',
    })
    await classificationBag.fit(X, yCls)
    const classMismatch = await StackingEnsemble.create({
      estimators: [['bagged', classificationBag]],
      finalEstimator: ['meta', MockModel, { task: 'classification' }],
      cv: 2,
      task: 'classification',
    })
    const shiftedLabels = new Int32Array(yCls.length)
    for (let index = 0; index < yCls.length; index++) shiftedLabels[index] = yCls[index] + 1
    await assert.rejects(
      () => classMismatch.fit(X, shiftedLabels),
      error => error instanceof ValidationError && /classes/.test(error.message)
    )
    assert.equal(classificationBag.isFitted, true)
    classMismatch.dispose()
    classificationBag.dispose()

    const regressionBag = await BaggedEstimator.create({
      estimator: ['bag-base', MockModel, { task: 'regression' }],
      kFold: 2,
      task: 'regression',
    })
    await regressionBag.fit(X, yReg)
    const taskMismatch = await StackingEnsemble.create({
      estimators: [['bagged', regressionBag]],
      finalEstimator: ['meta', MockModel, { task: 'classification' }],
      cv: 2,
      task: 'classification',
    })
    await assert.rejects(
      () => taskMismatch.fit(X, yCls),
      error => error instanceof ValidationError && /task/.test(error.message)
    )
    assert.equal(regressionBag.isFitted, true)
    taskMismatch.dispose()
    regressionBag.dispose()
  })

  it('does not reject a committed refit when old-model cleanup throws', async () => {
    class ReplacementModel {
      static generation = 1
      static live = 0
      #generation = ReplacementModel.generation
      #full = false
      #disposed = false
      static async create() {
        ReplacementModel.live++
        return new ReplacementModel()
      }
      fit(input) { this.#full = input.rows === X.rows; return this }
      get classes() { return new Int32Array([0, 1]) }
      get capabilities() { return { predictProba: true } }
      predict(input) { return new Int32Array(input.rows) }
      predictProba(input) { return new Float64Array(input.rows * 2).fill(0.5) }
      dispose() {
        if (this.#disposed) return
        this.#disposed = true
        ReplacementModel.live--
        if (this.#full && this.#generation === 1) {
          throw new Error('old cleanup failed')
        }
      }
    }
    const ensemble = await StackingEnsemble.create({
      estimators: [['base', ReplacementModel, {}]],
      finalEstimator: ['meta', ReplacementModel, {}],
      cv: 2,
      task: 'classification',
    })
    await ensemble.fit(X, yCls)
    ReplacementModel.generation = 2
    await ensemble.fit(X, yCls)
    assert.equal(ensemble.isFitted, true)
    assert.equal(ReplacementModel.live, 2)
    ensemble.dispose()
    assert.equal(ReplacementModel.live, 0)
  })
})
