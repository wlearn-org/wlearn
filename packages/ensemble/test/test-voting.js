const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const { VotingEnsemble } = require('../src/voting.js')
const { MockModel } = require('./mock-model.js')
const {
  Pipeline, ValidationError, NotFittedError, DisposedError, load
} = require('@wlearn/core')

const X = { data: new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), rows: 6, cols: 2 }
const yCls = new Int32Array([0, 0, 0, 1, 1, 1])
const yReg = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

class HardOnlyMock extends MockModel {
  static async create(params = {}) { return new HardOnlyMock(params) }
  get classes() { return undefined }
  get predictProba() { return undefined }
}

class NoProbabilityCapabilityMock extends MockModel {
  static async create(params = {}) { return new NoProbabilityCapabilityMock(params) }
  get capabilities() { return { ...super.capabilities, predictProba: false } }
}

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
    assert(preds instanceof Int32Array)
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
    assert(preds instanceof Int32Array)
    assert.equal(preds.length, 6)
    ens.dispose()
  })

  it('aligns probability columns from child class order', async () => {
    const labels = new Int32Array([2, 2, 2, 2, 1, 1])
    const ens = await VotingEnsemble.create({
      estimators: [[
        'reversed', MockModel,
        { task: 'classification', classOrder: 'descending' },
      ]],
      task: 'classification',
    })
    await ens.fit(X, labels)
    assert.deepEqual([...ens.classes], [1, 2])
    assert.deepEqual([...ens.predict(X)], [2, 2, 2, 2, 2, 2])
    const proba = ens.predictProba(X)
    assert(Math.abs(proba[0] - 0.1) < 1e-12)
    assert(Math.abs(proba[1] - 0.9) < 1e-12)
    ens.dispose()
  })

  it('rejects a callable probability method with a false capability', async () => {
    const ens = await VotingEnsemble.create({
      estimators: [[
        'no-probability', NoProbabilityCapabilityMock,
        { task: 'classification' },
      ]],
      voting: 'soft',
      task: 'classification',
    })
    await assert.rejects(
      () => ens.fit(X, yCls),
      error => error instanceof ValidationError && /capability/.test(error.message)
    )
    assert.equal(ens.isFitted, false)
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

  it('hard voting does not require probability class metadata', async () => {
    const ens = await VotingEnsemble.create({
      estimators: [['hard-only', HardOnlyMock, { task: 'classification' }]],
      voting: 'hard',
      task: 'classification',
    })
    await ens.fit(X, yCls)
    assert.equal(ens.predict(X).length, X.rows)
    assert.throws(
      () => ens.setParams({ voting: 'soft' }),
      error => error instanceof ValidationError && /predictProba/.test(error.message)
    )
    assert.equal(ens.getParams().voting, 'hard')
    assert.equal(ens.isFitted, true)
    assert.equal(ens.predict(X).length, X.rows)
    ens.dispose()
  })

  it('hard voting rejects malformed child label output', async () => {
    const cases = [
      ['wrong-shape', () => new Int32Array(0), /wrong shape/],
      ['non-finite', () => new Float64Array(6).fill(NaN), /declared int32/],
      ['fractional', () => new Float64Array(6).fill(0.5), /declared int32/],
      ['unknown', () => new Int32Array(6).fill(7), /declared int32/],
    ]
    for (const [name, output, expected] of cases) {
      class MalformedOutputMock extends HardOnlyMock {
        static async create(params = {}) { return new MalformedOutputMock(params) }
        predict() { return output() }
      }
      const ens = await VotingEnsemble.create({
        estimators: [[name, MalformedOutputMock, { task: 'classification' }]],
        voting: 'hard',
        task: 'classification',
      })
      await ens.fit(X, yCls)
      assert.throws(
        () => ens.predict(X),
        error => error instanceof ValidationError && expected.test(error.message)
      )
      ens.dispose()
    }
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

  it('fits as the asynchronous final step of a Pipeline', async () => {
    const ens = await VotingEnsemble.create({
      estimators: [
        ['m1', MockModel, { task: 'classification' }],
        ['m2', MockModel, { task: 'classification' }],
      ],
      voting: 'soft',
      task: 'classification',
    })
    const pipeline = new Pipeline([['vote', ens]])
    const pending = pipeline.fit(X, yCls)
    assert(pending instanceof Promise)
    assert.equal(pipeline.isFitted, false)
    await pending
    assert.equal(pipeline.isFitted, true)
    assert.equal(pipeline.predictProba(X).length, X.rows * 2)
    pipeline.dispose()
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
    assert(preds instanceof Float64Array)
    assert.equal(preds.length, 6)
    for (const p of preds) assert(isFinite(p))
    ens.dispose()
  })

  it('normalizes relative weights', async () => {
    const ens = await VotingEnsemble.create({
      estimators: [
        ['m1', MockModel, { task: 'regression' }],
        ['m2', MockModel, { task: 'regression', bias: 2 }],
      ],
      weights: [1, 1],
      task: 'regression',
    })
    await ens.fit(X, yReg)
    assert.deepEqual(ens.getParams().weights, [0.5, 0.5])
    assert(Math.abs(ens.predict(X)[0] - 4.5) < 1e-12)
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

  it('rejects malformed regression child predictions', async () => {
    class MalformedRegressionMock extends MockModel {
      static mode = 'short'
      static async create(params = {}) {
        return new MalformedRegressionMock({ task: 'regression', ...params })
      }
      predict(input) {
        if (MalformedRegressionMock.mode === 'short') {
          return new Float64Array(Math.max(0, input.rows - 1))
        }
        return new Float64Array(input.rows).fill(NaN)
      }
    }
    const ens = await VotingEnsemble.create({
      estimators: [['malformed', MalformedRegressionMock, {}]],
      task: 'regression',
    })
    await ens.fit(X, yReg)
    assert.throws(() => ens.predict(X), /wrong shape/)
    MalformedRegressionMock.mode = 'nonfinite'
    assert.throws(() => ens.predict(X), /finite numbers/)
    ens.dispose()
  })
})

describe('VotingEnsemble lifecycle', () => {
  it('rejects lifecycle races while child creation is pending', async () => {
    class DeferredCreateModel extends MockModel {
      static release = null
      static disposed = 0
      static create(params = {}) {
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
    const ensemble = await VotingEnsemble.create({
      estimators: [[
        'deferred', DeferredCreateModel, { task: 'classification' },
      ]],
      voting: 'soft',
      task: 'classification',
    })
    const pending = ensemble.fit(X, yCls)
    await assert.rejects(() => ensemble.fit(X, yCls), /already in progress/)
    assert.throws(() => ensemble.setParams({ voting: 'hard' }), /in progress/)
    assert.throws(() => ensemble.dispose(), /in progress/)
    DeferredCreateModel.release()
    await pending
    assert.equal(ensemble.isFitted, true)
    ensemble.dispose()
    assert.equal(DeferredCreateModel.disposed, 1)
  })

  it('rejects invalid fit configuration before training', async () => {
    const wrongWeights = await VotingEnsemble.create({
      estimators: [
        ['m1', MockModel, {}],
        ['m2', MockModel, {}],
      ],
      weights: [1],
      task: 'classification',
    })
    await assert.rejects(() => wrongWeights.fit(X, yCls), ValidationError)
    wrongWeights.dispose()

    const nonfinite = await VotingEnsemble.create({
      estimators: [['m1', MockModel, {}]],
      weights: [NaN],
      task: 'classification',
    })
    await assert.rejects(() => nonfinite.fit(X, yCls), ValidationError)
    nonfinite.dispose()

    for (const weights of [[0], [-1]]) {
      const invalidWeights = await VotingEnsemble.create({
        estimators: [['m1', MockModel, {}]],
        weights,
        task: 'classification',
      })
      await assert.rejects(() => invalidWeights.fit(X, yCls), ValidationError)
      invalidWeights.dispose()
    }

    const duplicateNames = await VotingEnsemble.create({
      estimators: [['m1', MockModel, {}], ['m1', MockModel, {}]],
      task: 'classification',
    })
    await assert.rejects(() => duplicateNames.fit(X, yCls), ValidationError)
    duplicateNames.dispose()
  })

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
    assert.equal(ens.isFitted, false)
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

  it('preserves fitted inference configuration after an invalid setParams', async () => {
    const ens = await VotingEnsemble.create({
      estimators: [['m1', MockModel, {}], ['m2', MockModel, {}]],
      weights: [0.5, 0.5],
      task: 'classification',
    })
    await ens.fit(X, yCls)
    const before = ens.getParams()
    assert.throws(() => ens.setParams({ weights: [1] }), ValidationError)
    assert.throws(() => ens.setParams({ voting: 'invalid' }), ValidationError)
    assert.throws(() => ens.setParams({ task: 'regression' }), /Unknown.*task/)
    assert.equal(ens.isFitted, true)
    assert.deepEqual(ens.getParams(), before)
    assert.equal(ens.predict(X).length, X.rows)
    ens.dispose()
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

  it('releases a later failed child in reverse order and preserves the fit error', async () => {
    const events = []
    const live = { count: 0 }
    const fitError = new Error('second fit failed')
    function modelClass(label, { fail = false, cleanupThrows = false } = {}) {
      return class {
        #disposed = false
        static async create() {
          live.count++
          events.push(`${label}:create`)
          return new this()
        }
        fit() {
          events.push(`${label}:fit`)
          if (fail) throw fitError
          return this
        }
        get classes() { return new Int32Array([0, 1]) }
        dispose() {
          if (this.#disposed) return
          this.#disposed = true
          live.count--
          events.push(`${label}:dispose`)
          if (cleanupThrows) throw new Error(`${label} cleanup failed`)
        }
      }
    }
    const First = modelClass('first', { cleanupThrows: true })
    const Second = modelClass('second', { fail: true })
    const ensemble = await VotingEnsemble.create({
      estimators: [['first', First, {}], ['second', Second, {}]],
      voting: 'hard',
      task: 'classification',
    })

    await assert.rejects(() => ensemble.fit(X, yCls), error => error === fitError)
    assert.equal(live.count, 0)
    assert.deepEqual(events, [
      'first:create', 'first:fit', 'second:create', 'second:fit',
      'second:dispose', 'first:dispose',
    ])
    ensemble.dispose()
    assert.equal(live.count, 0)
  })

  it('does not reject a committed refit when old-model cleanup throws', async () => {
    class ReplacementModel {
      static generation = 1
      static live = 0
      #generation = ReplacementModel.generation
      #disposed = false
      static async create() {
        ReplacementModel.live++
        return new ReplacementModel()
      }
      fit() { return this }
      get classes() { return new Int32Array([0, 1]) }
      get capabilities() { return { predictProba: true } }
      predictProba(X) { return new Float64Array(X.rows * 2).fill(0.5) }
      dispose() {
        if (this.#disposed) return
        this.#disposed = true
        ReplacementModel.live--
        if (this.#generation === 1) throw new Error('old cleanup failed')
      }
    }
    const ensemble = await VotingEnsemble.create({
      estimators: [['replacement', ReplacementModel, {}]],
      task: 'classification',
    })
    await ensemble.fit(X, yCls)
    ReplacementModel.generation = 2
    await ensemble.fit(X, yCls)
    assert.equal(ensemble.isFitted, true)
    assert.equal(ReplacementModel.live, 1)
    ensemble.dispose()
    assert.equal(ReplacementModel.live, 0)
  })
})
