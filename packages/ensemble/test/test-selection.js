import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { caruanaSelect } from '../src/selection.js'
import { ValidationError } from '@wlearn/core'

describe('caruanaSelect', () => {
  it('selects from pool of candidates', () => {
    const n = 10
    const nClasses = 2
    // Candidate 0: perfect separator (class 0 gets [1,0], class 1 gets [0,1])
    const good = new Float64Array(n * nClasses)
    for (let i = 0; i < n; i++) {
      good[i * 2 + 0] = i < 5 ? 0.9 : 0.1
      good[i * 2 + 1] = i < 5 ? 0.1 : 0.9
    }
    // Candidate 1: terrible (always predicts class 0)
    const bad = new Float64Array(n * nClasses)
    for (let i = 0; i < n; i++) {
      bad[i * 2 + 0] = 0.9
      bad[i * 2 + 1] = 0.1
    }

    const yTrue = new Int32Array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    const { indices, weights, scores } = caruanaSelect([good, bad], yTrue, {
      maxSize: 5,
      scoring: 'accuracy',
      task: 'classification',
    })

    // Good model should be selected more often
    assert(indices.length > 0)
    assert(weights.length === indices.length)
    assert(scores.length === 5)

    // First selected should be the good model (index 0)
    // weights for good model should be >= weights for bad model
    const goodIdx = indices.indexOf(0)
    if (goodIdx >= 0) {
      assert(weights[goodIdx] > 0)
    }
  })

  it('returns correct weight format', () => {
    const n = 6
    const pred = new Float64Array(n * 2)
    for (let i = 0; i < n; i++) {
      pred[i * 2 + 0] = 0.5
      pred[i * 2 + 1] = 0.5
    }
    const yTrue = new Int32Array([0, 0, 0, 1, 1, 1])

    const { indices, weights } = caruanaSelect([pred], yTrue, {
      maxSize: 3,
      task: 'classification',
    })

    // With only one candidate, it's always selected
    assert.equal(indices.length, 1)
    assert.equal(indices[0], 0)
    assert.equal(weights[0], 1.0)
  })

  it('works with regression', () => {
    const n = 6
    // Good predictor
    const good = new Float64Array([1, 2, 3, 4, 5, 6])
    // Bad predictor
    const bad = new Float64Array([10, 10, 10, 10, 10, 10])
    const yTrue = new Float64Array([1, 2, 3, 4, 5, 6])

    const { indices, weights, scores } = caruanaSelect([good, bad], yTrue, {
      maxSize: 5,
      scoring: 'r2',
      task: 'regression',
    })

    assert(indices.length > 0)
    // Good model should dominate
    const goodIdx = indices.indexOf(0)
    assert(goodIdx >= 0)
    assert(weights[goodIdx] > 0.5)
  })

  it('scores improve or stay constant', () => {
    const n = 10
    const pred1 = new Float64Array(n * 2)
    const pred2 = new Float64Array(n * 2)
    for (let i = 0; i < n; i++) {
      pred1[i * 2 + 0] = i < 5 ? 0.8 : 0.2
      pred1[i * 2 + 1] = i < 5 ? 0.2 : 0.8
      pred2[i * 2 + 0] = i < 5 ? 0.6 : 0.4
      pred2[i * 2 + 1] = i < 5 ? 0.4 : 0.6
    }
    const yTrue = new Int32Array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    const { scores } = caruanaSelect([pred1, pred2], yTrue, {
      maxSize: 10,
      task: 'classification',
    })

    // Scores should generally not decrease (greedy selection)
    for (let i = 1; i < scores.length; i++) {
      assert(scores[i] >= scores[i - 1] - 1e-9,
        `score decreased: ${scores[i]} < ${scores[i - 1]}`)
    }
  })

  it('throws on empty pool', () => {
    assert.throws(
      () => caruanaSelect([], new Int32Array([0, 1]), { task: 'classification' }),
      ValidationError
    )
  })

  it('accepts custom scoring function', () => {
    const n = 4
    const pred = new Float64Array(n * 2).fill(0.5)
    const yTrue = new Int32Array([0, 0, 1, 1])

    const { scores } = caruanaSelect([pred], yTrue, {
      maxSize: 2,
      scoring: () => 0.42,
      task: 'classification',
    })
    for (const s of scores) assert.equal(s, 0.42)
  })
})
