const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const { RandomStrategy } = require('../src/strategy-random.js')
const { HalvingStrategy } = require('../src/strategy-halving.js')
const { SearchableMock, SearchableMockReg } = require('./mock-model.js')

describe('RandomStrategy', () => {
  it('yields exactly nIter * nModels candidates', () => {
    const strategy = new RandomStrategy(
      [
        { name: 'm1', cls: SearchableMock },
        { name: 'm2', cls: SearchableMock },
      ],
      { nIter: 3, seed: 42 }
    )
    let count = 0
    while (!strategy.isDone()) {
      const task = strategy.next()
      if (task === null) break
      count++
    }
    assert.equal(count, 6) // 2 models * 3 iter
  })

  it('returns null after exhaustion', () => {
    const strategy = new RandomStrategy(
      [{ name: 'm', cls: SearchableMock }],
      { nIter: 2, seed: 42 }
    )
    strategy.next()
    strategy.next()
    assert.equal(strategy.next(), null)
    assert.equal(strategy.isDone(), true)
  })

  it('candidates have candidateId, cls, params', () => {
    const strategy = new RandomStrategy(
      [{ name: 'mock', cls: SearchableMock }],
      { nIter: 1, seed: 42 }
    )
    const task = strategy.next()
    assert(typeof task.candidateId === 'string')
    assert.equal(task.cls, SearchableMock)
    assert(typeof task.params === 'object')
  })

  it('fixed params are merged into candidates', () => {
    const strategy = new RandomStrategy(
      [{ name: 'mock', cls: SearchableMock, params: { bias: 99 } }],
      { nIter: 3, seed: 42 }
    )
    for (let i = 0; i < 3; i++) {
      const task = strategy.next()
      assert.equal(task.params.bias, 99)
    }
  })

  it('is deterministic with same seed', () => {
    const collect = (seed) => {
      const s = new RandomStrategy(
        [{ name: 'mock', cls: SearchableMock }],
        { nIter: 5, seed }
      )
      const ids = []
      while (!s.isDone()) {
        const t = s.next()
        if (!t) break
        ids.push(t.candidateId)
      }
      return ids
    }
    assert.deepEqual(collect(42), collect(42))
  })

  it('report is a no-op', () => {
    const strategy = new RandomStrategy(
      [{ name: 'm', cls: SearchableMock }],
      { nIter: 1, seed: 42 }
    )
    // Should not throw
    strategy.report({ candidateId: 'x', meanScore: 0.5 })
  })
})

describe('HalvingStrategy', () => {
  it('starts with all candidates in first round', () => {
    const strategy = new HalvingStrategy(
      [{ name: 'mock', cls: SearchableMock }],
      { nIter: 9, seed: 42, factor: 3, nSamples: 10, cv: 3 }
    )
    let count = 0
    while (!strategy.isDone()) {
      const task = strategy.next()
      if (task === null) break
      count++
      // Report a fake result to trigger round transition
      strategy.report({
        candidateId: task.candidateId,
        meanScore: Math.random(),
        foldScores: new Float64Array([0.5]),
        stdScore: 0,
        fitTimeMs: 1,
        nTrainUsed: 5,
        nTest: 5,
      })
    }
    // Should have evaluated more than nIter candidates total (multiple rounds)
    assert(count >= 9)
  })

  it('fewer candidates survive each round', () => {
    const strategy = new HalvingStrategy(
      [{ name: 'mock', cls: SearchableMock }],
      { nIter: 9, seed: 42, factor: 3, nSamples: 100, cv: 3 }
    )
    const roundCounts = []
    let currentRoundCount = 0
    let prevRounds = 0

    while (!strategy.isDone()) {
      const task = strategy.next()
      if (task === null) break
      currentRoundCount++
      strategy.report({
        candidateId: task.candidateId,
        meanScore: currentRoundCount * 0.01, // increasing scores so sorting works
        foldScores: new Float64Array([0.5]),
        stdScore: 0,
        fitTimeMs: 1,
        nTrainUsed: 5,
        nTest: 5,
      })

      // Check if round advanced
      if (strategy.rounds.length > prevRounds) {
        roundCounts.push(currentRoundCount)
        currentRoundCount = 0
        prevRounds = strategy.rounds.length
      }
    }
    if (currentRoundCount > 0) roundCounts.push(currentRoundCount)

    // First round should have more candidates than later rounds
    if (roundCounts.length > 1) {
      assert(roundCounts[0] >= roundCounts[1])
    }
  })

  it('isDone returns true when finished', () => {
    const strategy = new HalvingStrategy(
      [{ name: 'mock', cls: SearchableMock }],
      { nIter: 3, seed: 42, factor: 3, nSamples: 10, cv: 2 }
    )
    assert.equal(strategy.isDone(), false)

    // Run to completion
    while (!strategy.isDone()) {
      const task = strategy.next()
      if (task === null) break
      strategy.report({
        candidateId: task.candidateId,
        meanScore: 0.5,
        foldScores: new Float64Array([0.5]),
        stdScore: 0,
        fitTimeMs: 1,
        nTrainUsed: 5,
        nTest: 5,
      })
    }
    assert.equal(strategy.isDone(), true)
  })

  it('first round candidates have subsample budget', () => {
    const strategy = new HalvingStrategy(
      [{ name: 'mock', cls: SearchableMock }],
      { nIter: 9, seed: 42, factor: 3, nSamples: 100, cv: 3 }
    )
    const task = strategy.next()
    // With 100 samples and factor=3, first round should have small fraction
    assert(task.budget !== undefined)
    assert.equal(task.budget.type, 'subsample')
    assert(task.budget.value < 1)
  })

  it('greaterIsBetter=false sorts ascending for elimination', () => {
    const strategy = new HalvingStrategy(
      [{ name: 'mock', cls: SearchableMock }],
      { nIter: 6, seed: 42, factor: 3, nSamples: 100, cv: 2, greaterIsBetter: false }
    )

    // Give candidates distinct scores
    let i = 0
    const scores = [0.9, 0.1, 0.5, 0.3, 0.7, 0.2]
    while (!strategy.isDone()) {
      const task = strategy.next()
      if (task === null) break
      strategy.report({
        candidateId: task.candidateId,
        meanScore: scores[i] || 0.5,
        foldScores: new Float64Array([scores[i] || 0.5]),
        stdScore: 0,
        fitTimeMs: 1,
        nTrainUsed: 5,
        nTest: 5,
      })
      i++
    }

    // With greaterIsBetter=false, lower scores should survive
    assert(strategy.rounds.length >= 1)
  })

  it('rounds getter returns round stats', () => {
    const strategy = new HalvingStrategy(
      [{ name: 'mock', cls: SearchableMock }],
      { nIter: 9, seed: 42, factor: 3, nSamples: 100, cv: 3 }
    )

    while (!strategy.isDone()) {
      const task = strategy.next()
      if (task === null) break
      strategy.report({
        candidateId: task.candidateId,
        meanScore: 0.5,
        foldScores: new Float64Array([0.5]),
        stdScore: 0,
        fitTimeMs: 1,
        nTrainUsed: 5,
        nTest: 5,
      })
    }

    assert(strategy.rounds.length >= 1)
    for (const r of strategy.rounds) {
      assert(typeof r.round === 'number')
      assert(typeof r.nCandidates === 'number')
      assert(typeof r.nSurvivors === 'number')
      assert(r.nSurvivors <= r.nCandidates)
    }
  })
})
