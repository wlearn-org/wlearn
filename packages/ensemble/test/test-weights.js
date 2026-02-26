import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { projectSimplex, optimizeWeights } from '../src/weights.js'
import { caruanaSelect } from '../src/selection.js'
import { ValidationError } from '@wlearn/core'

describe('projectSimplex', () => {
  it('projects a vector onto the simplex', () => {
    const v = new Float64Array([0.5, 0.3, 0.2])
    const w = projectSimplex(v)
    assert.equal(w.length, 3)
    // Sum should be ~1
    let sum = 0
    for (let i = 0; i < w.length; i++) sum += w[i]
    assert(Math.abs(sum - 1) < 1e-10)
    // All non-negative
    for (let i = 0; i < w.length; i++) {
      assert(w[i] >= 0)
    }
  })

  it('already on simplex stays unchanged', () => {
    const v = new Float64Array([0.6, 0.3, 0.1])
    const w = projectSimplex(v)
    for (let i = 0; i < v.length; i++) {
      assert(Math.abs(w[i] - v[i]) < 1e-10)
    }
  })

  it('negative values get projected to zero', () => {
    const v = new Float64Array([2, -1, -1])
    const w = projectSimplex(v)
    let sum = 0
    for (let i = 0; i < w.length; i++) {
      assert(w[i] >= -1e-15)
      sum += w[i]
    }
    assert(Math.abs(sum - 1) < 1e-10)
  })

  it('single element returns [1.0]', () => {
    const w = projectSimplex(new Float64Array([5.0]))
    assert.equal(w.length, 1)
    assert.equal(w[0], 1.0)
  })

  it('empty input returns empty', () => {
    const w = projectSimplex(new Float64Array(0))
    assert.equal(w.length, 0)
  })

  it('uniform case', () => {
    const v = new Float64Array([1, 1, 1, 1])
    const w = projectSimplex(v)
    for (let i = 0; i < 4; i++) {
      assert(Math.abs(w[i] - 0.25) < 1e-10)
    }
  })
})

describe('optimizeWeights', () => {
  it('classification: optimizes weights for two models', () => {
    const n = 10
    const nc = 2
    // Model 0: perfect for class 0
    const m0 = new Float64Array(n * nc)
    // Model 1: perfect for class 1
    const m1 = new Float64Array(n * nc)
    const yTrue = new Int32Array(n)
    for (let i = 0; i < n; i++) {
      if (i < 5) {
        yTrue[i] = 0
        m0[i * nc + 0] = 0.9; m0[i * nc + 1] = 0.1
        m1[i * nc + 0] = 0.5; m1[i * nc + 1] = 0.5
      } else {
        yTrue[i] = 1
        m0[i * nc + 0] = 0.5; m0[i * nc + 1] = 0.5
        m1[i * nc + 0] = 0.1; m1[i * nc + 1] = 0.9
      }
    }

    const init = new Float64Array([0.5, 0.5])
    const w = optimizeWeights([m0, m1], yTrue, init, { task: 'classification' })
    assert.equal(w.length, 2)

    let sum = 0
    for (let i = 0; i < w.length; i++) {
      assert(w[i] >= 0)
      sum += w[i]
    }
    assert(Math.abs(sum - 1) < 1e-10)
  })

  it('regression: optimizes weights for two models', () => {
    const n = 6
    const yTrue = new Float64Array([1, 2, 3, 4, 5, 6])
    // Model 0: perfect
    const m0 = new Float64Array([1, 2, 3, 4, 5, 6])
    // Model 1: bad
    const m1 = new Float64Array([3, 3, 3, 3, 3, 3])

    const init = new Float64Array([0.5, 0.5])
    const w = optimizeWeights([m0, m1], yTrue, init, { task: 'regression' })
    assert.equal(w.length, 2)

    // Good model should get more weight
    assert(w[0] > w[1])

    let sum = 0
    for (let i = 0; i < w.length; i++) {
      assert(w[i] >= 0)
      sum += w[i]
    }
    assert(Math.abs(sum - 1) < 1e-10)
  })

  it('single model returns [1.0]', () => {
    const m = new Float64Array([1, 2, 3])
    const yTrue = new Float64Array([1, 2, 3])
    const w = optimizeWeights([m], yTrue, new Float64Array([1.0]), { task: 'regression' })
    assert.equal(w.length, 1)
    assert.equal(w[0], 1.0)
  })

  it('throws on empty models', () => {
    assert.throws(
      () => optimizeWeights([], new Float64Array([1]), new Float64Array(0), { task: 'regression' }),
      ValidationError
    )
  })

  it('weights sum to 1', () => {
    const n = 8
    const nc = 2
    const models = []
    for (let m = 0; m < 3; m++) {
      const preds = new Float64Array(n * nc)
      for (let i = 0; i < n; i++) {
        preds[i * nc + 0] = 0.4 + m * 0.1
        preds[i * nc + 1] = 0.6 - m * 0.1
      }
      models.push(preds)
    }
    const yTrue = new Int32Array([0, 0, 0, 0, 1, 1, 1, 1])
    const init = new Float64Array([1 / 3, 1 / 3, 1 / 3])
    const w = optimizeWeights(models, yTrue, init, { task: 'classification' })

    let sum = 0
    for (let i = 0; i < w.length; i++) sum += w[i]
    assert(Math.abs(sum - 1) < 1e-10)
  })
})

describe('caruanaSelect with refineWeights', () => {
  it('refineWeights=true produces valid weights', () => {
    const n = 10
    const nc = 2
    const good = new Float64Array(n * nc)
    const mid = new Float64Array(n * nc)
    for (let i = 0; i < n; i++) {
      good[i * 2 + 0] = i < 5 ? 0.9 : 0.1
      good[i * 2 + 1] = i < 5 ? 0.1 : 0.9
      mid[i * 2 + 0] = i < 5 ? 0.7 : 0.3
      mid[i * 2 + 1] = i < 5 ? 0.3 : 0.7
    }
    const yTrue = new Int32Array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    const { indices, weights } = caruanaSelect([good, mid], yTrue, {
      maxSize: 5,
      scoring: 'accuracy',
      task: 'classification',
      refineWeights: true,
    })

    assert(indices.length > 0)
    let sum = 0
    for (let i = 0; i < weights.length; i++) {
      assert(weights[i] >= 0)
      sum += weights[i]
    }
    assert(Math.abs(sum - 1) < 1e-10)
  })
})
