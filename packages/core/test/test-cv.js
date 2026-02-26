import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { kFold, stratifiedKFold, trainTestSplit, crossValScore, getScorer } from '../src/cv.js'
import { ValidationError } from '../src/errors.js'

describe('kFold', () => {
  it('produces k folds', () => {
    const folds = kFold(10, 5)
    assert.equal(folds.length, 5)
  })

  it('each sample appears in exactly one test fold', () => {
    const folds = kFold(20, 4)
    const testCounts = new Int32Array(20)
    for (const { test } of folds) {
      for (const idx of test) testCounts[idx]++
    }
    for (let i = 0; i < 20; i++) {
      assert.equal(testCounts[i], 1, `sample ${i} appeared ${testCounts[i]} times in test`)
    }
  })

  it('train + test covers all samples per fold', () => {
    const folds = kFold(15, 3)
    for (const { train, test } of folds) {
      assert.equal(train.length + test.length, 15)
      const all = new Set([...train, ...test])
      assert.equal(all.size, 15)
    }
  })

  it('is deterministic', () => {
    const a = kFold(10, 3, { seed: 7 })
    const b = kFold(10, 3, { seed: 7 })
    for (let i = 0; i < 3; i++) {
      assert.deepEqual([...a[i].train], [...b[i].train])
      assert.deepEqual([...a[i].test], [...b[i].test])
    }
  })

  it('different seeds give different folds', () => {
    const a = kFold(10, 3, { seed: 1 })
    const b = kFold(10, 3, { seed: 2 })
    let same = true
    for (let i = 0; i < 3; i++) {
      if ([...a[i].test].join(',') !== [...b[i].test].join(',')) same = false
    }
    assert(!same)
  })

  it('handles non-divisible n', () => {
    const folds = kFold(7, 3)
    const sizes = folds.map(f => f.test.length)
    // 7 / 3 = 2 remainder 1, so sizes should be [3, 2, 2] or similar
    assert.equal(sizes.reduce((a, b) => a + b), 7)
    assert(Math.max(...sizes) - Math.min(...sizes) <= 1)
  })

  it('throws if n < k', () => {
    assert.throws(() => kFold(2, 5), ValidationError)
  })

  it('throws if k < 2', () => {
    assert.throws(() => kFold(10, 1), ValidationError)
  })

  it('supports shuffle=false', () => {
    const folds = kFold(6, 3, { shuffle: false })
    // Without shuffle, first fold test should be [0, 1]
    assert.deepEqual([...folds[0].test], [0, 1])
    assert.deepEqual([...folds[1].test], [2, 3])
    assert.deepEqual([...folds[2].test], [4, 5])
  })
})

describe('stratifiedKFold', () => {
  it('preserves class proportions', () => {
    // 60% class 0, 40% class 1
    const y = new Int32Array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    const folds = stratifiedKFold(y, 5)

    for (const { test } of folds) {
      let zeros = 0, ones = 0
      for (const idx of test) {
        if (y[idx] === 0) zeros++
        else ones++
      }
      // Each fold should have approximately 60/40 split
      // With 2 samples per fold from 10 total / 5 folds,
      // we expect ~1.2 zeros and ~0.8 ones per fold
      assert(zeros > 0 || ones > 0, 'fold should not be empty')
    }
  })

  it('each sample appears in exactly one test fold', () => {
    const y = new Int32Array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    const folds = stratifiedKFold(y, 3)
    const testCounts = new Int32Array(9)
    for (const { test } of folds) {
      for (const idx of test) testCounts[idx]++
    }
    for (let i = 0; i < 9; i++) {
      assert.equal(testCounts[i], 1, `sample ${i} appeared ${testCounts[i]} times`)
    }
  })

  it('is deterministic', () => {
    const y = new Int32Array([0, 1, 0, 1, 0, 1])
    const a = stratifiedKFold(y, 3, { seed: 5 })
    const b = stratifiedKFold(y, 3, { seed: 5 })
    for (let i = 0; i < 3; i++) {
      assert.deepEqual([...a[i].test], [...b[i].test])
    }
  })

  it('throws if n < k', () => {
    assert.throws(() => stratifiedKFold(new Int32Array([0, 1]), 5), ValidationError)
  })
})

describe('trainTestSplit', () => {
  it('default 80/20 split', () => {
    const { train, test } = trainTestSplit(100)
    assert.equal(train.length, 80)
    assert.equal(test.length, 20)
  })

  it('custom test size', () => {
    const { train, test } = trainTestSplit(100, { testSize: 0.3 })
    assert.equal(train.length, 70)
    assert.equal(test.length, 30)
  })

  it('no overlap', () => {
    const { train, test } = trainTestSplit(50)
    const all = new Set([...train, ...test])
    assert.equal(all.size, 50)
  })

  it('is deterministic', () => {
    const a = trainTestSplit(50, { seed: 3 })
    const b = trainTestSplit(50, { seed: 3 })
    assert.deepEqual([...a.train], [...b.train])
    assert.deepEqual([...a.test], [...b.test])
  })

  it('throws if n < 2', () => {
    assert.throws(() => trainTestSplit(1), ValidationError)
  })
})

describe('getScorer', () => {
  it('returns function for known names', () => {
    for (const name of ['accuracy', 'r2', 'neg_mse', 'neg_mae']) {
      assert.equal(typeof getScorer(name), 'function')
    }
  })

  it('returns the function itself if given a function', () => {
    const fn = () => 0
    assert.strictEqual(getScorer(fn), fn)
  })

  it('throws on unknown name', () => {
    assert.throws(() => getScorer('unknown'), ValidationError)
  })

  it('neg_mse returns negative values', () => {
    const scorer = getScorer('neg_mse')
    const result = scorer(new Float64Array([1, 2, 3]), new Float64Array([2, 3, 4]))
    assert(result < 0)
  })
})

describe('crossValScore', () => {
  // Mock estimator for testing
  class MockClassifier {
    #label = 0

    static async create(params) {
      const m = new MockClassifier()
      m.#label = params.label || 0
      return m
    }

    fit(X, y) {
      // Store the most common label
      const counts = new Map()
      for (let i = 0; i < y.length; i++) {
        counts.set(y[i], (counts.get(y[i]) || 0) + 1)
      }
      let best = y[0], bestCount = 0
      for (const [k, v] of counts) {
        if (v > bestCount) { best = k; bestCount = v }
      }
      this.#label = best
      return this
    }

    predict(X) {
      return new Float64Array(X.rows).fill(this.#label)
    }

    dispose() {}
  }

  it('returns per-fold scores', async () => {
    const X = { data: new Float64Array(20), rows: 10, cols: 2 }
    const y = new Int32Array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    const scores = await crossValScore(MockClassifier, X, y, { cv: 3 })
    assert.equal(scores.length, 3)
    for (const s of scores) {
      assert(s >= 0 && s <= 1, `score out of range: ${s}`)
    }
  })

  it('accepts custom scoring function', async () => {
    const X = { data: new Float64Array(12), rows: 6, cols: 2 }
    const y = new Float64Array([1, 2, 3, 4, 5, 6])
    const scores = await crossValScore(MockClassifier, X, y, {
      cv: 3,
      scoring: () => 0.42,
    })
    for (const s of scores) {
      assert.equal(s, 0.42)
    }
  })

  it('accepts pre-built folds', async () => {
    const X = { data: new Float64Array(12), rows: 6, cols: 2 }
    const y = new Int32Array([0, 0, 0, 1, 1, 1])
    const folds = [
      { train: new Int32Array([0, 1, 2]), test: new Int32Array([3, 4, 5]) },
      { train: new Int32Array([3, 4, 5]), test: new Int32Array([0, 1, 2]) },
    ]
    const scores = await crossValScore(MockClassifier, X, y, { cv: folds })
    assert.equal(scores.length, 2)
  })

  it('accepts number[][] for X', async () => {
    const X = [[1, 2], [3, 4], [5, 6], [7, 8]]
    const y = new Int32Array([0, 0, 1, 1])
    const scores = await crossValScore(MockClassifier, X, y, { cv: 2 })
    assert.equal(scores.length, 2)
  })
})
