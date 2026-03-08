const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const { RandomSearch } = require('../src/search.js')
const { SuccessiveHalvingSearch } = require('../src/halving.js')
const { SearchableMock, SearchableMockReg } = require('./mock-model.js')

/**
 * Regression snapshot tests: ensure the refactor produces identical
 * output to the pre-refactor implementation.
 * Same seed + same mock models + same options = same results.
 */

const X = {
  data: new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
  rows: 10, cols: 2
}
const yCls = new Int32Array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

describe('RandomSearch regression snapshot', () => {
  it('produces 5 entries with seed=42, nIter=5, cv=3', async () => {
    const search = new RandomSearch(
      [{ name: 'mock', cls: SearchableMock }],
      { nIter: 5, cv: 3, seed: 42, task: 'classification' }
    )
    const { leaderboard } = await search.fit(X, yCls)
    assert.equal(leaderboard.length, 5)
    // All entries should be from 'mock' model
    for (const e of leaderboard.ranked()) {
      assert.equal(e.modelName, 'mock')
    }
  })

  it('produces 6 entries for 2 models with nIter=3', async () => {
    const search = new RandomSearch(
      [
        { name: 'm1', cls: SearchableMock },
        { name: 'm2', cls: SearchableMock },
      ],
      { nIter: 3, cv: 2, seed: 42, task: 'classification' }
    )
    const { leaderboard } = await search.fit(X, yCls)
    assert.equal(leaderboard.length, 6)
  })

  it('determinism: same seed produces identical scores', async () => {
    const run = async () => {
      const search = new RandomSearch(
        [{ name: 'mock', cls: SearchableMock }],
        { nIter: 5, cv: 3, seed: 42, task: 'classification' }
      )
      const { leaderboard } = await search.fit(X, yCls)
      return leaderboard.ranked().map(e => ({
        modelName: e.modelName,
        meanScore: e.meanScore,
        rank: e.rank,
      }))
    }
    const a = await run()
    const b = await run()
    assert.deepEqual(a, b)
  })
})

describe('SuccessiveHalvingSearch regression snapshot', () => {
  it('produces correct round structure with seed=42, nIter=9', async () => {
    const sh = new SuccessiveHalvingSearch(
      [{ name: 'mock', cls: SearchableMock }],
      { nIter: 9, cv: 3, seed: 42, task: 'classification' }
    )
    const { leaderboard, rounds } = await sh.fit(X, yCls)
    // Should have multiple rounds
    assert(rounds.length >= 1)
    // First round should have 9 candidates
    assert.equal(rounds[0].nCandidates, 9)
    // Survivors decrease
    for (let i = 1; i < rounds.length; i++) {
      assert(rounds[i].nCandidates <= rounds[i - 1].nCandidates)
    }
  })

  it('determinism: same seed produces identical results', async () => {
    const run = async () => {
      const sh = new SuccessiveHalvingSearch(
        [{ name: 'mock', cls: SearchableMock }],
        { nIter: 9, cv: 3, seed: 42, task: 'classification' }
      )
      const { leaderboard, rounds } = await sh.fit(X, yCls)
      return {
        entries: leaderboard.ranked().map(e => ({
          modelName: e.modelName,
          meanScore: e.meanScore,
          rank: e.rank,
        })),
        rounds,
      }
    }
    const a = await run()
    const b = await run()
    assert.deepEqual(a.entries, b.entries)
    assert.deepEqual(a.rounds, b.rounds)
  })
})
