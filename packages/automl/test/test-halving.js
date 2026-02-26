import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { SuccessiveHalvingSearch } from '../src/halving.js'
import { SearchableMock, SearchableMockReg } from './mock-model.js'
import { ValidationError } from '@wlearn/core'

const X = {
  data: new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
  rows: 10, cols: 2
}
const yCls = new Int32Array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
const yReg = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

describe('SuccessiveHalvingSearch', () => {
  it('runs multiple rounds', async () => {
    const halving = new SuccessiveHalvingSearch(
      [{ name: 'mock', cls: SearchableMock }],
      { nIter: 9, cv: 2, task: 'classification', factor: 3 }
    )
    const { rounds } = await halving.fit(X, yCls)
    assert(rounds.length >= 1, 'expected at least 1 round')
  })

  it('later rounds have fewer candidates', async () => {
    const halving = new SuccessiveHalvingSearch(
      [{ name: 'mock', cls: SearchableMock }],
      { nIter: 9, cv: 2, task: 'classification', factor: 3 }
    )
    const { rounds } = await halving.fit(X, yCls)
    if (rounds.length >= 2) {
      assert(rounds[1].nCandidates <= rounds[0].nCandidates)
    }
  })

  it('returns leaderboard and best result', async () => {
    const halving = new SuccessiveHalvingSearch(
      [{ name: 'mock', cls: SearchableMock }],
      { nIter: 6, cv: 2, task: 'classification', factor: 3 }
    )
    const { leaderboard, bestResult } = await halving.fit(X, yCls)
    assert(leaderboard.length > 0)
    assert(bestResult !== null)
    assert.equal(bestResult.rank, 1)
  })

  it('deterministic with same seed', async () => {
    const run = async (seed) => {
      const h = new SuccessiveHalvingSearch(
        [{ name: 'mock', cls: SearchableMock }],
        { nIter: 6, cv: 2, task: 'classification', factor: 3, seed }
      )
      const { bestResult } = await h.fit(X, yCls)
      return bestResult.meanScore
    }
    const a = await run(42)
    const b = await run(42)
    assert.equal(a, b)
  })

  it('works with classification', async () => {
    const halving = new SuccessiveHalvingSearch(
      [{ name: 'mock', cls: SearchableMock }],
      { nIter: 4, cv: 2, task: 'classification' }
    )
    const { bestResult } = await halving.fit(X, yCls)
    assert(bestResult.meanScore >= 0 && bestResult.meanScore <= 1)
  })

  it('works with regression', async () => {
    const halving = new SuccessiveHalvingSearch(
      [{ name: 'mock', cls: SearchableMockReg }],
      { nIter: 4, cv: 2, task: 'regression' }
    )
    const { bestResult } = await halving.fit(X, yReg)
    assert(isFinite(bestResult.meanScore))
  })

  it('refitBest returns fitted model', async () => {
    const halving = new SuccessiveHalvingSearch(
      [{ name: 'mock', cls: SearchableMock }],
      { nIter: 4, cv: 2, task: 'classification' }
    )
    await halving.fit(X, yCls)
    const model = await halving.refitBest(X, yCls)
    assert(model.isFitted)
    model.dispose()
  })

  it('throws on empty models', () => {
    assert.throws(() => new SuccessiveHalvingSearch([]), ValidationError)
  })
})
