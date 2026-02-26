import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { autoFit } from '../src/auto-fit.js'
import { SearchableMock, SearchableMockReg, MockModel } from './mock-model.js'
import { ValidationError } from '@wlearn/core'

const X = {
  data: new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
  rows: 10, cols: 2
}
const yCls = new Int32Array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
const yReg = new Float64Array([1.1, 2.3, 3.7, 4.2, 5.8, 6.1, 7.5, 8.9, 9.4, 10.6])

describe('autoFit classification', () => {
  it('returns fitted model with refit=true', async () => {
    const result = await autoFit(
      [{ name: 'mock', cls: SearchableMock }],
      X, yCls,
      { nIter: 3, cv: 2, task: 'classification' }
    )
    assert(result.model !== null)
    assert(result.model.isFitted)
    const preds = result.model.predict(X)
    assert.equal(preds.length, 10)
    result.model.dispose()
  })

  it('returns leaderboard', async () => {
    const result = await autoFit(
      [{ name: 'mock', cls: SearchableMock }],
      X, yCls,
      { nIter: 5, cv: 2, task: 'classification' }
    )
    assert(result.leaderboard.length === 5)
    assert(result.bestScore >= 0)
    assert.equal(result.bestModelName, 'mock')
    if (result.model) result.model.dispose()
  })

  it('accepts EstimatorSpec tuples', async () => {
    const result = await autoFit(
      [['mock', SearchableMock, { task: 'classification' }]],
      X, yCls,
      { nIter: 2, cv: 2, task: 'classification' }
    )
    assert(result.model !== null)
    assert(result.leaderboard.length === 2)
    if (result.model) result.model.dispose()
  })

  it('refit=false and ensemble=false returns null model', async () => {
    const result = await autoFit(
      [{ name: 'mock', cls: SearchableMock }],
      X, yCls,
      { nIter: 2, cv: 2, task: 'classification', refit: false, ensemble: false }
    )
    assert.equal(result.model, null)
    assert(result.bestScore >= 0)
  })
})

describe('autoFit ensemble', () => {
  it('ensemble=true returns VotingEnsemble', async () => {
    const result = await autoFit(
      [
        { name: 'm1', cls: SearchableMock },
        { name: 'm2', cls: SearchableMock },
      ],
      X, yCls,
      { nIter: 3, cv: 2, task: 'classification', ensemble: true, ensembleSize: 5 }
    )
    assert(result.model !== null)
    // VotingEnsemble has capabilities property
    assert(result.model.capabilities.classifier)
    const preds = result.model.predict(X)
    assert.equal(preds.length, 10)
    result.model.dispose()
  })
})

describe('autoFit regression', () => {
  it('works with regression task', async () => {
    const result = await autoFit(
      [{ name: 'mock', cls: SearchableMockReg }],
      X, yReg,
      { nIter: 3, cv: 2, task: 'regression' }
    )
    assert(result.model !== null)
    assert(isFinite(result.bestScore))
    result.model.dispose()
  })
})

describe('autoFit onProgress', () => {
  it('calls onProgress for each candidate during search', async () => {
    const events = []
    const result = await autoFit(
      [{ name: 'mock', cls: SearchableMock }],
      X, yCls,
      {
        nIter: 3, cv: 2, task: 'classification',
        ensemble: false, refit: true,
        onProgress: (e) => events.push(e),
      }
    )
    assert.equal(events.length, 3)
    for (const e of events) {
      assert.equal(e.phase, 'search')
      assert.equal(typeof e.candidatesDone, 'number')
      assert.equal(typeof e.bestScore, 'number')
      assert.equal(typeof e.bestModel, 'string')
      assert.equal(typeof e.lastCandidate.model, 'string')
      assert.equal(typeof e.lastCandidate.score, 'number')
      assert.equal(typeof e.lastCandidate.timeMs, 'number')
      assert.equal(typeof e.elapsedMs, 'number')
    }
    assert.equal(events[0].candidatesDone, 1)
    assert.equal(events[2].candidatesDone, 3)
    if (result.model) result.model.dispose()
  })

  it('emits ensemble phase event when ensemble=true', async () => {
    const events = []
    const result = await autoFit(
      [
        { name: 'm1', cls: SearchableMock },
        { name: 'm2', cls: SearchableMock },
      ],
      X, yCls,
      {
        nIter: 2, cv: 2, task: 'classification',
        ensemble: true, ensembleSize: 3,
        onProgress: (e) => events.push(e),
      }
    )
    const phases = events.map(e => e.phase)
    assert(phases.includes('search'))
    assert(phases.includes('ensemble'))
    if (result.model) result.model.dispose()
  })

  it('works with portfolio strategy', async () => {
    const events = []
    const result = await autoFit(
      [{ name: 'mock', cls: SearchableMock }],
      X, yCls,
      {
        strategy: 'portfolio', cv: 2, task: 'classification',
        ensemble: false, refit: true,
        onProgress: (e) => events.push(e),
      }
    )
    assert(events.length > 0)
    assert.equal(events[0].phase, 'search')
    if (result.model) result.model.dispose()
  })
})

describe('autoFit validation', () => {
  it('throws on empty models', async () => {
    await assert.rejects(
      () => autoFit([], X, yCls, { task: 'classification' }),
      ValidationError
    )
  })

  it('auto-detects classification task from Int32Array', async () => {
    const result = await autoFit(
      [{ name: 'mock', cls: SearchableMock }],
      X, yCls,
      { nIter: 2, cv: 2 }
    )
    assert(result.bestScore >= 0 && result.bestScore <= 1)
    if (result.model) result.model.dispose()
  })

  it('auto-detects regression task from Float64Array', async () => {
    const result = await autoFit(
      [{ name: 'mock', cls: SearchableMockReg }],
      X, yReg,
      { nIter: 2, cv: 2 }
    )
    assert(isFinite(result.bestScore))
    if (result.model) result.model.dispose()
  })
})
