const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const {
  slidingWindowSplit,
  slidingIndexSplit,
  slidingPeriodSplit
} = require('../src/resampling.js')

function assertFoldInvariants(folds, n) {
  assert(folds.length > 0)
  for (const fold of folds) {
    assert(fold.train.length > 0)
    assert(fold.test.length > 0)
    const train = new Set(fold.train)
    assert.equal(train.size, fold.train.length)
    const test = new Set(fold.test)
    assert.equal(test.size, fold.test.length)
    for (const idx of fold.train) {
      assert(idx >= 0 && idx < n)
      assert.equal(test.has(idx), false)
    }
    for (const idx of fold.test) {
      assert(idx >= 0 && idx < n)
      assert.equal(train.has(idx), false)
    }
    assert(Math.max(...fold.train) < Math.min(...fold.test))
  }
}

describe('resampling property probes', () => {
  it('slidingWindowSplit emits valid forward-only folds over parameter sweeps', () => {
    for (let n = 5; n <= 24; n++) {
      for (let lookback = 1; lookback <= Math.min(6, n - 2); lookback++) {
        for (let horizon = 1; horizon <= Math.min(3, n - lookback); horizon++) {
          const folds = slidingWindowSplit(n, {
            lookback,
            assessStart: 1,
            assessStop: horizon,
            step: 1 + (n + lookback + horizon) % 3,
            complete: true
          })
          assertFoldInvariants(folds, n)
          for (const fold of folds) {
            assert(fold.train.length <= lookback)
            assert(fold.test.length === horizon)
          }
        }
      }
    }
  })

  it('slidingIndexSplit emits valid folds for duplicated sorted indices', () => {
    const index = new Float64Array([0, 0, 1, 2, 2, 3, 4, 5, 5, 6])
    const folds = slidingIndexSplit(index, {
      lookback: 2,
      assessStart: 1,
      assessStop: 1,
      complete: true
    })
    assertFoldInvariants(folds, index.length)
    for (const fold of folds) {
      const trainMax = Math.max(...fold.train.map(i => index[i]))
      const testMin = Math.min(...fold.test.map(i => index[i]))
      assert(trainMax < testMin)
    }
  })

  it('slidingPeriodSplit emits valid folds for date periods', () => {
    const index = [
      '2026-01-01',
      '2026-01-02',
      '2026-01-03',
      '2026-01-04',
      '2026-01-05',
      '2026-01-06',
      '2026-01-07'
    ]
    const folds = slidingPeriodSplit(index, {
      period: 'day',
      lookback: 2,
      assessStart: 1,
      assessStop: 2,
      complete: true
    })
    assertFoldInvariants(folds, index.length)
  })
})
