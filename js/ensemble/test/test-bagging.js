const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const { BaggedEstimator } = require('../src/bagging.js')
const { getOofPredictions } = require('../src/oof.js')
const { MockModel } = require('./mock-model.js')
const { ValidationError, NotFittedError, DisposedError } = require('@wlearn/core')

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
    assert(preds instanceof Int32Array)
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

  it('aligns probability columns from fold-model class order', async () => {
    const labels = new Int32Array([2, 2, 2, 2, 2, 2, 1, 1, 1, 1])
    const bag = await BaggedEstimator.create({
      estimator: [
        'reversed', MockModel,
        { task: 'classification', classOrder: 'descending' },
      ],
      kFold: 2,
      task: 'classification',
    })
    await bag.fit(X, labels)
    assert.deepEqual([...bag.classes], [1, 2])
    assert.deepEqual([...bag.predict(X)], new Array(10).fill(2))
    const proba = bag.predictProba(X)
    assert(Math.abs(proba[0] - 0.1) < 1e-12)
    assert(Math.abs(proba[1] - 0.9) < 1e-12)
    bag.dispose()
  })

  it('rejects a child without an explicit probability capability', async () => {
    class NoProbabilityCapabilityMock extends MockModel {
      static async create(params = {}) {
        return new NoProbabilityCapabilityMock(params)
      }
      get capabilities() {
        return { ...super.capabilities, predictProba: false }
      }
    }
    const bag = await BaggedEstimator.create({
      estimator: [
        'no-probability', NoProbabilityCapabilityMock,
        { task: 'classification' },
      ],
      kFold: 2,
      task: 'classification',
    })
    await assert.rejects(
      () => bag.fit(X, yCls),
      error => error instanceof ValidationError && /capability/.test(error.message)
    )
    assert.equal(bag.isFitted, false)
    bag.dispose()
  })
})

describe('OOF classification contracts', () => {
  it('waits for asynchronous child fit before probability prediction', async () => {
    class DeferredFitMock extends MockModel {
      static async create(params = {}) { return new DeferredFitMock(params) }
      async fit(input, labels) {
        await Promise.resolve()
        return super.fit(input, labels)
      }
      predictProba(input) {
        assert.equal(this.isFitted, true)
        return super.predictProba(input)
      }
    }

    const result = await getOofPredictions([
      ['deferred', DeferredFitMock, { task: 'classification' }],
    ], X, yCls, { cv: 2, task: 'classification' })
    assert.equal(result.oofPreds[0].length, X.rows * 2)
  })

  it('aligns probability columns and requires the capability descriptor', async () => {
    const labels = new Int32Array([2, 2, 2, 2, 2, 2, 1, 1, 1, 1])
    class ReversedProbabilityMock extends MockModel {
      static async create(params = {}) {
        return new ReversedProbabilityMock(params)
      }
      predictProba(input) {
        const output = new Float64Array(input.rows * 2)
        for (let row = 0; row < input.rows; row++) {
          output[row * 2] = 0.9
          output[row * 2 + 1] = 0.1
        }
        return output
      }
    }
    const aligned = await getOofPredictions([
      ['reversed', ReversedProbabilityMock, {
        task: 'classification', classOrder: 'descending',
      }],
    ], X, labels, { cv: 2, task: 'classification' })
    assert.deepEqual([...aligned.classes], [1, 2])
    assert(Math.abs(aligned.oofPreds[0][0] - 0.1) < 1e-12)
    assert(Math.abs(aligned.oofPreds[0][1] - 0.9) < 1e-12)

    class NoProbabilityCapabilityMock extends MockModel {
      static async create(params = {}) {
        return new NoProbabilityCapabilityMock(params)
      }
      get capabilities() {
        return { ...super.capabilities, predictProba: false }
      }
    }
    await assert.rejects(
      () => getOofPredictions([
        ['no-probability', NoProbabilityCapabilityMock, {
          task: 'classification',
        }],
      ], X, yCls, { cv: 2, task: 'classification' }),
      error => error instanceof ValidationError && /capability/.test(error.message)
    )
  })
})

describe('OOF regression contracts', () => {
  it('rejects malformed regression predictions', async () => {
    class ShortRegressionMock extends MockModel {
      static async create(params = {}) {
        return new ShortRegressionMock({ task: 'regression', ...params })
      }
      predict(input) {
        return new Float64Array(Math.max(0, input.rows - 1))
      }
    }
    await assert.rejects(
      () => getOofPredictions([
        ['short', ShortRegressionMock, { task: 'regression' }],
      ], X, yReg, { cv: 2, task: 'regression' }),
      /wrong shape/
    )
  })
})

describe('BaggedEstimator regression', () => {
  it('rejects malformed fold predictions', async () => {
    class NonfiniteRegressionMock extends MockModel {
      static async create(params = {}) {
        return new NonfiniteRegressionMock({ task: 'regression', ...params })
      }
      predict(input) { return new Float64Array(input.rows).fill(Infinity) }
    }
    const bag = await BaggedEstimator.create({
      estimator: ['nonfinite', NonfiniteRegressionMock, {}],
      kFold: 2,
      task: 'regression',
    })
    await assert.rejects(() => bag.fit(X, yReg), /finite numbers/)
    assert.equal(bag.isFitted, false)
    bag.dispose()
  })

  it('fit and predict', async () => {
    const bag = await BaggedEstimator.create({
      estimator: ['mock', MockModel, { task: 'regression' }],
      kFold: 2,
      task: 'regression',
    })
    await bag.fit(X, yReg)
    assert(bag.isFitted)
    const preds = bag.predict(X)
    assert(preds instanceof Float64Array)
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
    const bag = await BaggedEstimator.create({
      estimator: [
        'deferred', DeferredCreateModel, { task: 'classification' },
      ],
      kFold: 2,
      task: 'classification',
    })
    const pending = bag.fit(X, yCls)
    await assert.rejects(() => bag.fit(X, yCls), /already in progress/)
    assert.throws(() => bag.setParams({ seed: 9 }), /in progress/)
    assert.throws(() => bag.dispose(), /in progress/)
    DeferredCreateModel.release()
    await pending
    assert.equal(bag.isFitted, true)
    bag.dispose()
    assert.equal(DeferredCreateModel.disposed, 2)
  })

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

  it('invalidates fitted state when training parameters change', async () => {
    const bag = await BaggedEstimator.create({
      estimator: ['mock', MockModel, { task: 'classification' }],
      kFold: 2,
      task: 'classification',
    })
    await bag.fit(X, yCls)
    assert.equal(bag.isFitted, true)
    bag.setParams({ nRepeats: 2 })
    assert.equal(bag.isFitted, false)
    assert.throws(() => bag.save(), NotFittedError)
    bag.dispose()
  })

  it('rejects invalid training config transactionally', async () => {
    const bag = await BaggedEstimator.create({
      estimator: ['mock', MockModel, { task: 'classification' }],
      kFold: 2,
      task: 'classification',
    })
    await bag.fit(X, yCls)
    assert.throws(() => bag.setParams({ nRepeats: 0 }), ValidationError)
    assert.throws(() => bag.setParams({ estimator: null }), /Unknown.*estimator/)
    assert.equal(bag.getParams().nRepeats, 1)
    assert.equal(bag.isFitted, true)
    const invalid = await BaggedEstimator.create({
      estimator: ['mock', MockModel, {}],
      kFold: 2,
      nRepeats: -1,
      task: 'classification',
    })
    await assert.rejects(() => invalid.fit(X, yCls), ValidationError)
    bag.dispose()
    invalid.dispose()
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

  it('preserves the previous fitted state when a refit fails', async () => {
    class TrackingModel {
      static fail = false
      static live = 0
      #disposed = false
      static async create() {
        TrackingModel.live++
        return new TrackingModel()
      }
      fit() {
        if (TrackingModel.fail) throw new Error('refit failed')
        return this
      }
      get classes() { return new Int32Array([0, 1]) }
      get capabilities() { return { predictProba: true } }
      predictProba(X) {
        const output = new Float64Array(X.rows * 2)
        for (let row = 0; row < X.rows; row++) {
          output[row * 2] = 0.75
          output[row * 2 + 1] = 0.25
        }
        return output
      }
      dispose() {
        if (this.#disposed) return
        this.#disposed = true
        TrackingModel.live--
      }
    }
    const bag = await BaggedEstimator.create({
      estimator: ['tracking', TrackingModel, {}],
      kFold: 2,
      task: 'classification',
    })
    await bag.fit(X, yCls)
    assert.equal(TrackingModel.live, 2)
    const before = [...bag.predict(X)]
    TrackingModel.fail = true
    await assert.rejects(() => bag.fit(X, yCls), /refit failed/)
    assert.equal(bag.isFitted, true)
    assert.deepEqual([...bag.predict(X)], before)
    assert.equal(TrackingModel.live, 2)
    bag.dispose()
    assert.equal(TrackingModel.live, 0)
  })

  it('does not reject a committed refit when old-model cleanup throws', async () => {
    class CleanupModel {
      static generation = 1
      static live = 0
      #generation = CleanupModel.generation
      #disposed = false
      static async create() {
        CleanupModel.live++
        return new CleanupModel()
      }
      fit() { return this }
      get classes() { return new Int32Array([0, 1]) }
      get capabilities() { return { predictProba: true } }
      predictProba(X) { return new Float64Array(X.rows * 2).fill(0.5) }
      dispose() {
        if (this.#disposed) return
        this.#disposed = true
        CleanupModel.live--
        if (this.#generation === 1) throw new Error('old cleanup failed')
      }
    }
    const bag = await BaggedEstimator.create({
      estimator: ['cleanup', CleanupModel, {}],
      kFold: 2,
      task: 'classification',
    })
    await bag.fit(X, yCls)
    CleanupModel.generation = 2
    await bag.fit(X, yCls)
    assert.equal(bag.isFitted, true)
    assert.equal(CleanupModel.live, 2)
    bag.dispose()
    assert.equal(CleanupModel.live, 0)
  })
})
