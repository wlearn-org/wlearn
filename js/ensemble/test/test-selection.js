const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const { caruanaSelect } = require('../src/selection.js')

it('selection rejects invalid probability mass even with hard-label scoring', () => {
  for (const prediction of [[-1, 2, 2, -1], [0.2, 0.2, 0.3, 0.3]]) {
    assert.throws(() => caruanaSelect([prediction], [0, 1], { maxSize: 1 }), /proba/i)
  }
})

it('MAE selection retains its objective during optional refinement', () => {
  const predictions = [[0, 0, 9], [3, 3, -9]].map(x => Float64Array.from(x))
  const result = caruanaSelect(predictions, new Float64Array(3), {
    maxSize: 2, task: 'regression', scoring: 'neg_mae'
  })
  assert.deepEqual(Array.from(result.weights), [0.5, 0.5])
})

it('probability measures select using probabilities and their declared direction', () => {
  const good = Float64Array.from([0.9, 0.1, 0.1, 0.9])
  const weak = Float64Array.from([0.6, 0.4, 0.4, 0.6])
  for (const scoring of ['log_loss', 'roc_auc']) {
    const result = caruanaSelect([good, weak], Int32Array.from([5, 2]), {
      maxSize: 1, scoring, classes: [5, 2]
    })
    assert.deepEqual(Array.from(result.indices), [0])
  }
})
const { ValidationError } = require('@wlearn/core')

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

  it('maps probability columns to noncontiguous class labels', () => {
    const yTrue = new Int32Array([2, 2, 5, 5])
    const bad = new Float64Array([
      0.1, 0.9,
      0.1, 0.9,
      0.9, 0.1,
      0.9, 0.1,
    ])
    const good = new Float64Array([
      0.9, 0.1,
      0.9, 0.1,
      0.1, 0.9,
      0.1, 0.9,
    ])
    const { indices, scores } = caruanaSelect([bad, good], yTrue, {
      maxSize: 1,
      task: 'classification',
      classes: new Int32Array([2, 5]),
      refineWeights: false,
    })
    assert.deepEqual(Array.from(indices), [1])
    assert.deepEqual(Array.from(scores), [1])

    const swapColumns = values => {
      const out = new Float64Array(values.length)
      for (let i = 0; i < values.length; i += 2) {
        out[i] = values[i + 1]
        out[i + 1] = values[i]
      }
      return out
    }
    const reversed = caruanaSelect([
      swapColumns(bad), swapColumns(good)
    ], yTrue, {
      maxSize: 1,
      task: 'classification',
      classes: new Int32Array([5, 2]),
      refineWeights: false,
    })
    assert.deepEqual(Array.from(reversed.indices), [1])
    assert.deepEqual(Array.from(reversed.scores), [1])
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

  it('rejects class metadata inconsistent with probability width', () => {
    assert.throws(
      () => caruanaSelect(
        [new Float64Array([0.5, 0.5, 0.5, 0.5])],
        new Int32Array([2, 5]),
        { task: 'classification', nClasses: 3, classes: [2, 5, 9] }
      ),
      /nClasses must match/
    )
  })

  it('rejects class metadata outside the unique-int32 contract', () => {
    const predictions = [
      new Float64Array([0.8, 0.2, 0.2, 0.8])
    ]
    const yTrue = new Int32Array([2, 5])
    for (const classes of [
      [2.5, 5], [NaN, 5], [2147483648, 5], ['2', '5'], [2n, 5n], [2, 2]
    ]) {
      assert.throws(
        () => caruanaSelect(predictions, yTrue, {
          maxSize: 1,
          task: 'classification',
          classes,
          refineWeights: false,
        }),
        /unique int32/
      )
    }
  })

  it('rejects mismatched or non-finite candidate predictions', () => {
    const good = new Float64Array([0.8, 0.2, 0.2, 0.8])
    const yTrue = new Int32Array([2, 5])
    for (const bad of [
      new Float64Array([0.8, 0.2]),
      new Float64Array([0.8, 0.2, NaN, 0.8]),
    ]) {
      assert.throws(
        () => caruanaSelect([good, bad], yTrue, {
          maxSize: 1,
          task: 'classification',
          classes: [2, 5],
          refineWeights: false,
        }),
        ValidationError
      )
    }
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
