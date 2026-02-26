import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { stratifiedKFold, kFold } from '@wlearn/core'
import { Executor } from '../src/executor.js'
import { SearchableMock, SearchableMockReg } from './mock-model.js'

const X = {
  data: new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
  rows: 10, cols: 2
}
const yCls = new Int32Array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
const yReg = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

describe('Executor evaluateCandidate', () => {
  it('returns CandidateResult with correct shape', async () => {
    const folds = stratifiedKFold(yCls, 3, { shuffle: true, seed: 42 })
    const exec = new Executor({
      folds, scoring: 'accuracy', X, y: yCls, seed: 42,
    })
    const result = await exec.evaluateCandidate({
      candidateId: 'mock:{}',
      cls: SearchableMock,
      params: { bias: 0, task: 'classification' },
    })
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
    await exec.evaluateCandidate({
      candidateId: 'mock:{}',
      cls: SearchableMock,
      params: { bias: 0, task: 'classification' },
    })
    assert.equal(exec.leaderboard.length, 1)
  })

  it('evaluates multiple candidates correctly', async () => {
    const folds = stratifiedKFold(yCls, 2, { shuffle: true, seed: 42 })
    const exec = new Executor({
      folds, scoring: 'accuracy', X, y: yCls, seed: 42,
    })
    await exec.evaluateCandidate({
      candidateId: 'mock1:{}',
      cls: SearchableMock,
      params: { bias: 0, task: 'classification' },
    })
    await exec.evaluateCandidate({
      candidateId: 'mock2:{}',
      cls: SearchableMock,
      params: { bias: 1, task: 'classification' },
    })
    assert.equal(exec.leaderboard.length, 2)
  })
})

describe('Executor subsample budget', () => {
  it('subsamples train indices not test', async () => {
    const folds = kFold(10, 2, { shuffle: true, seed: 42 })
    const exec = new Executor({
      folds, scoring: 'accuracy', X, y: yCls, seed: 42,
    })
    const result = await exec.evaluateCandidate({
      candidateId: 'mock:{}',
      cls: SearchableMock,
      params: { bias: 0, task: 'classification' },
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
      candidateId: 'mock:{}',
      cls: SearchableMock,
      params: { bias: 0, task: 'classification' },
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
      candidateId: 'budget-mock:{}',
      cls: BudgetMock,
      params: { bias: 0, task: 'classification' },
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
      candidateId: 'budget-mock:{}',
      cls: BudgetMock,
      params: { bias: 0, task: 'classification', nEstimators: 200 },
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
      candidateId: 'mock:{}',
      cls: SearchableMock,
      params: { bias: 0, task: 'classification' },
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
      { candidateId: 'm:1', cls: SearchableMock, params: { bias: 0, task: 'classification' } },
      { candidateId: 'm:2', cls: SearchableMock, params: { bias: 1, task: 'classification' } },
    ]
    let idx = 0
    const strategy = {
      next() { return idx < candidates.length ? candidates[idx++] : null },
      report() {},
      isDone() { return idx >= candidates.length },
    }

    const { leaderboard } = await exec.runStrategy(strategy)
    assert.equal(leaderboard.length, 2)
  })
})
