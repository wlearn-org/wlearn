const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const { stratifiedKFold, kFold, ValidationError } = require('@wlearn/core')
const { Executor } = require('../src/executor.js')
const {
  createCandidate, makeCandidateId, seedFor,
} = require('../src/candidate.js')
const { SearchableMock, SearchableMockReg } = require('./mock-model.js')

const X = {
  data: new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
  rows: 10, cols: 2
}
const yCls = new Int32Array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
const yReg = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

function makeTask(name, cls, params, classId = `wlearn.test.${name}@1`) {
  const candidate = createCandidate({ displayName: name, classId }, params)
  return { candidateId: makeCandidateId(candidate), candidate, cls, params }
}

describe('Executor evaluateCandidate', () => {
  it('waits for asynchronous fit before prediction and disposal', async () => {
    let disposals = 0
    class DeferredFitModel {
      #fitted = false

      static async create() { return new DeferredFitModel() }
      async fit() {
        await Promise.resolve()
        this.#fitted = true
        return this
      }
      predict(input) {
        assert.equal(this.#fitted, true)
        return new Int32Array(input.rows)
      }
      dispose() {
        assert.equal(this.#fitted, true)
        disposals++
      }
    }

    const folds = stratifiedKFold(yCls, 2, { shuffle: true, seed: 42 })
    const exec = new Executor({
      folds, scoring: 'accuracy', X, y: yCls, seed: 42,
    })
    const result = await exec.evaluateCandidate(
      makeTask('deferred-fit', DeferredFitModel, {})
    )
    assert.equal(result.foldScores.length, 2)
    assert.equal(disposals, 2)
  })

  it('returns CandidateResult with correct shape', async () => {
    const folds = stratifiedKFold(yCls, 3, { shuffle: true, seed: 42 })
    const exec = new Executor({
      folds, scoring: 'accuracy', X, y: yCls, seed: 42,
    })
    const result = await exec.evaluateCandidate(makeTask(
      'mock', SearchableMock, { bias: 0, task: 'classification' }
    ))
    assert.equal(typeof result.candidateId, 'string')
    assert.equal(typeof result.meanScore, 'number')
    assert.equal(result.foldScores.length, 3)
    assert.equal(typeof result.stdScore, 'number')
    assert.equal(typeof result.fitTimeMs, 'number')
    assert.equal(typeof result.nTrainUsed, 'number')
    assert.equal(typeof result.nTest, 'number')
  })

  it('records result in leaderboard', async () => {
    const folds = stratifiedKFold(yCls, 3, { shuffle: true, seed: 42 })
    const exec = new Executor({
      folds, scoring: 'accuracy', X, y: yCls, seed: 42,
    })
    assert.equal(exec.leaderboard.length, 0)
    assert.equal(exec.archive.size, 0)
    const task = makeTask(
      'mock', SearchableMock, { bias: 0, task: 'classification' }
    )
    await exec.evaluateCandidate(task)
    assert.equal(exec.leaderboard.length, 1)
    assert.equal(exec.archive.size, 1)
    const record = exec.archive.records()[0]
    assert.equal(record.status, 'ok')
    assert.equal(record.scores.accuracy, exec.leaderboard.best().meanScore)
    assert.equal(record.metadata.sourceCandidateId, task.candidateId)
    assert.deepEqual(record.metadata.candidate, task.candidate)
  })

  it('evaluates multiple candidates correctly', async () => {
    const folds = stratifiedKFold(yCls, 2, { shuffle: true, seed: 42 })
    const exec = new Executor({
      folds, scoring: 'accuracy', X, y: yCls, seed: 42,
    })
    await exec.evaluateCandidate(makeTask(
      'mock1', SearchableMock, { bias: 0, task: 'classification' }
    ))
    await exec.evaluateCandidate(makeTask(
      'mock2', SearchableMock, { bias: 1, task: 'classification' }
    ))
    assert.equal(exec.leaderboard.length, 2)
  })

  it('rejects a stale candidate ID before model construction', async () => {
    let creates = 0
    class NeverCreated {
      static async create() { creates++; return SearchableMock.create({}) }
    }
    const exec = new Executor({
      folds: kFold(10, 2), scoring: 'accuracy', X, y: yCls,
    })
    const task = makeTask('stale', NeverCreated, {})
    task.candidateId = `wlc1_${'0'.repeat(64)}`
    await assert.rejects(
      () => exec.evaluateCandidate(task),
      error => error instanceof ValidationError && /candidateId/.test(error.message)
    )
    assert.equal(creates, 0)
  })

  it('records exact non-default fold seeds without subsampling', async () => {
    const folds = kFold(10, 2, { shuffle: true, seed: 3 })
    const exec = new Executor({
      folds, scoring: 'accuracy', X, y: yCls, seed: 7,
    })
    const params = JSON.parse('{"__proto__":{"safe":true}}')
    const task = makeTask('seeded', SearchableMock, params)
    const result = await exec.evaluateCandidate(task)
    assert.equal(result.baseSeed, 7)
    assert.deepEqual([...result.foldSeeds], [
      seedFor(task.candidate, 0, 7), seedFor(task.candidate, 1, 7),
    ])
    const record = exec.archive.records()[0]
    assert.equal(record.seed, 7)
    assert.deepEqual(record.metadata.foldSeeds, [
      { foldId: 0, seed: result.foldSeeds[0] },
      { foldId: 1, seed: result.foldSeeds[1] },
    ])
    assert(Object.hasOwn(record.metadata.candidate.model.params, '__proto__'))
    assert.equal(makeCandidateId(record.metadata.candidate), record.candidateId)
  })

  it('validates the base seed before constructing archive state', () => {
    for (const seed of [-1, 2 ** 32, 1.5, NaN]) {
      assert.throws(() => new Executor({
        folds: kFold(10, 2), scoring: 'accuracy', X, y: yCls, seed,
      }), ValidationError)
    }
  })
})

describe('Executor subsample budget', () => {
  it('subsamples train indices not test', async () => {
    const folds = kFold(10, 2, { shuffle: true, seed: 42 })
    const exec = new Executor({
      folds, scoring: 'accuracy', X, y: yCls, seed: 42,
    })
    const result = await exec.evaluateCandidate({
      ...makeTask('mock', SearchableMock, { bias: 0, task: 'classification' }),
      budget: { type: 'subsample', value: 0.5 },
    })
    // nTrainUsed should be roughly half of the original train size
    assert(result.nTrainUsed < folds[0].train.length)
    // nTest should be full size
    assert.equal(result.nTest, folds[0].test.length)
  })

  it('full subsample (value=1) uses all train data', async () => {
    const folds = kFold(10, 2, { shuffle: true, seed: 42 })
    const exec = new Executor({
      folds, scoring: 'accuracy', X, y: yCls, seed: 42,
    })
    const result = await exec.evaluateCandidate({
      ...makeTask('mock', SearchableMock, { bias: 0, task: 'classification' }),
      budget: { type: 'subsample', value: 1.0 },
    })
    assert.equal(result.nTrainUsed, folds[0].train.length)
  })
})

describe('Executor rounds budget', () => {
  it('sets roundsParam when model has budgetSpec', async () => {
    // Create a model class that records the params passed to create()
    let capturedParams = null
    class BudgetMock {
      static budgetSpec() { return { roundsParam: 'nEstimators' } }
      static async create(params) {
        capturedParams = { ...params }
        return SearchableMock.create(params)
      }
    }
    const folds = stratifiedKFold(yCls, 2, { shuffle: true, seed: 42 })
    const exec = new Executor({
      folds, scoring: 'accuracy', X, y: yCls, seed: 42,
    })
    await exec.evaluateCandidate({
      ...makeTask('budget-mock', BudgetMock, { bias: 0, task: 'classification' }),
      budget: { type: 'rounds', value: 50 },
    })
    assert.equal(capturedParams.nEstimators, 50)
  })

  it('candidate config wins over rounds budget', async () => {
    let capturedParams = null
    class BudgetMock {
      static budgetSpec() { return { roundsParam: 'nEstimators' } }
      static async create(params) {
        capturedParams = { ...params }
        return SearchableMock.create(params)
      }
    }
    const folds = stratifiedKFold(yCls, 2, { shuffle: true, seed: 42 })
    const exec = new Executor({
      folds, scoring: 'accuracy', X, y: yCls, seed: 42,
    })
    await exec.evaluateCandidate({
      ...makeTask('budget-mock', BudgetMock, {
        bias: 0, task: 'classification', nEstimators: 200
      }),
      budget: { type: 'rounds', value: 50 },
    })
    assert.equal(capturedParams.nEstimators, 200)
  })

  it('ignores rounds budget when model lacks budgetSpec', async () => {
    const folds = stratifiedKFold(yCls, 2, { shuffle: true, seed: 42 })
    const exec = new Executor({
      folds, scoring: 'accuracy', X, y: yCls, seed: 42,
    })
    // SearchableMock has no budgetSpec -- should not throw
    const result = await exec.evaluateCandidate({
      ...makeTask('mock', SearchableMock, { bias: 0, task: 'classification' }),
      budget: { type: 'rounds', value: 50 },
    })
    assert(typeof result.meanScore === 'number')
  })
})

describe('Executor time limit', () => {
  it('isTimedOut becomes true after time limit', async () => {
    const folds = stratifiedKFold(yCls, 2, { shuffle: true, seed: 42 })
    const exec = new Executor({
      folds, scoring: 'accuracy', X, y: yCls, seed: 42,
      timeLimitMs: 1, // 1ms limit
    })
    // Wait a tick to ensure time passes
    await new Promise(r => setTimeout(r, 5))
    assert.equal(exec.isTimedOut, true)
  })

  it('isTimedOut is false when no limit', () => {
    const folds = stratifiedKFold(yCls, 2, { shuffle: true, seed: 42 })
    const exec = new Executor({
      folds, scoring: 'accuracy', X, y: yCls, seed: 42,
      timeLimitMs: 0,
    })
    assert.equal(exec.isTimedOut, false)
  })
})

describe('Executor runStrategy', () => {
  it('runs a simple strategy to completion', async () => {
    const folds = stratifiedKFold(yCls, 2, { shuffle: true, seed: 42 })
    const exec = new Executor({
      folds, scoring: 'accuracy', X, y: yCls, seed: 42,
    })

    // Minimal strategy: yields 2 candidates then done
    const candidates = [
      makeTask('m1', SearchableMock, { bias: 0, task: 'classification' }),
      makeTask('m2', SearchableMock, { bias: 1, task: 'classification' }),
    ]
    let idx = 0
    const strategy = {
      next() { return idx < candidates.length ? candidates[idx++] : null },
      report() {},
      isDone() { return idx >= candidates.length },
    }

    const { leaderboard, archive } = await exec.runStrategy(strategy)
    assert.equal(leaderboard.length, 2)
    assert.equal(archive.size, 2)
  })

  it('records failed candidates in archive', async () => {
    class FailingModel {
      static async create() {
        throw new Error('create failed')
      }
    }
    const folds = stratifiedKFold(yCls, 2, { shuffle: true, seed: 42 })
    const exec = new Executor({
      folds, scoring: 'accuracy', X, y: yCls, seed: 42,
    })

    let yielded = false
    const strategy = {
      next() {
        if (yielded) return null
        yielded = true
        return makeTask(
          'bad', FailingModel,
          JSON.parse('{"__proto__":{"failed":true}}')
        )
      },
      report() {},
      isDone() { return yielded },
    }

    const { leaderboard, archive } = await exec.runStrategy(strategy)
    assert.equal(leaderboard.length, 0)
    assert.equal(archive.size, 1)
    const record = archive.records()[0]
    assert.equal(record.seed, 42)
    assert.equal(record.metadata.foldSeeds.length, folds.length)
    assert(Object.hasOwn(record.metadata.candidate.model.params, '__proto__'))
    assert.equal(makeCandidateId(record.metadata.candidate), record.candidateId)
    assert.equal(archive.records()[0].status, 'failed')
    assert.equal(archive.records()[0].error.phase, 'fit')
  })
})
