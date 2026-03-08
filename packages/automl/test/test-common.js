const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const {
  detectTask, makeCandidateId, seedFor, partialShuffle,
  scorerGreaterIsBetter
} = require('../src/common.js')

describe('detectTask', () => {
  it('classifies Int32Array as classification', () => {
    assert.equal(detectTask(new Int32Array([0, 1, 0, 1])), 'classification')
  })

  it('classifies non-integer Float64Array as regression', () => {
    assert.equal(detectTask(new Float64Array([1.1, 2.2, 3.3])), 'regression')
  })

  it('classifies few unique integers as classification', () => {
    assert.equal(detectTask(new Float64Array([0, 1, 2, 0, 1, 2])), 'classification')
  })

  it('classifies many unique integers as regression', () => {
    const y = new Float64Array(100)
    for (let i = 0; i < 100; i++) y[i] = i
    assert.equal(detectTask(y), 'regression')
  })
})

describe('makeCandidateId', () => {
  it('produces stable ids', () => {
    const a = makeCandidateId('lr', { C: 1, eps: 0.01 })
    const b = makeCandidateId('lr', { C: 1, eps: 0.01 })
    assert.equal(a, b)
  })

  it('is key-order independent', () => {
    const a = makeCandidateId('lr', { C: 1, eps: 0.01 })
    const b = makeCandidateId('lr', { eps: 0.01, C: 1 })
    assert.equal(a, b)
  })

  it('different params produce different ids', () => {
    const a = makeCandidateId('lr', { C: 1 })
    const b = makeCandidateId('lr', { C: 2 })
    assert.notEqual(a, b)
  })

  it('different model labels produce different ids', () => {
    const a = makeCandidateId('lr', { C: 1 })
    const b = makeCandidateId('svm', { C: 1 })
    assert.notEqual(a, b)
  })

  it('handles nested objects', () => {
    const a = makeCandidateId('m', { a: { b: 1 } })
    const b = makeCandidateId('m', { a: { b: 1 } })
    assert.equal(a, b)
  })

  it('handles arrays in params', () => {
    const a = makeCandidateId('m', { x: [1, 2, 3] })
    const b = makeCandidateId('m', { x: [1, 2, 3] })
    assert.equal(a, b)
  })
})

describe('seedFor', () => {
  it('is deterministic', () => {
    const a = seedFor('lr:{"C":1}', 0, 42)
    const b = seedFor('lr:{"C":1}', 0, 42)
    assert.equal(a, b)
  })

  it('varies with fold index', () => {
    const a = seedFor('lr:{"C":1}', 0, 42)
    const b = seedFor('lr:{"C":1}', 1, 42)
    assert.notEqual(a, b)
  })

  it('varies with candidate id', () => {
    const a = seedFor('lr:{"C":1}', 0, 42)
    const b = seedFor('lr:{"C":2}', 0, 42)
    assert.notEqual(a, b)
  })

  it('varies with base seed', () => {
    const a = seedFor('lr:{"C":1}', 0, 42)
    const b = seedFor('lr:{"C":1}', 0, 99)
    assert.notEqual(a, b)
  })

  it('returns positive integer', () => {
    const s = seedFor('test', 5, 123)
    assert(Number.isInteger(s))
    assert(s >= 0)
  })
})

describe('partialShuffle', () => {
  it('selects k elements from array', () => {
    const rng = () => 0.5
    const arr = new Int32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    const result = partialShuffle(arr, 3, rng)
    assert.equal(result.length, 3)
  })

  it('returns all elements when k >= n', () => {
    const rng = () => 0.5
    const arr = new Int32Array([0, 1, 2])
    const result = partialShuffle(arr, 5, rng)
    assert.equal(result.length, 3)
  })

  it('is deterministic with same rng', () => {
    const makeRng = () => {
      let s = 42
      return () => { s = (s * 1664525 + 1013904223) & 0x7fffffff; return s / 0x7fffffff }
    }
    const a = partialShuffle(new Int32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 4, makeRng())
    const b = partialShuffle(new Int32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 4, makeRng())
    assert.deepEqual([...a], [...b])
  })

  it('selected elements are from original array', () => {
    const rng = () => 0.3
    const arr = new Int32Array([10, 20, 30, 40, 50])
    const result = partialShuffle(arr, 3, rng)
    const original = new Set([10, 20, 30, 40, 50])
    for (const v of result) assert(original.has(v))
  })

  it('selected elements are unique', () => {
    const rng = () => 0.7
    const arr = new Int32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    const result = partialShuffle(arr, 5, rng)
    const unique = new Set(result)
    assert.equal(unique.size, result.length)
  })
})

describe('scorerGreaterIsBetter', () => {
  it('returns true for accuracy', () => {
    assert.equal(scorerGreaterIsBetter('accuracy'), true)
  })

  it('returns true for r2', () => {
    assert.equal(scorerGreaterIsBetter('r2'), true)
  })

  it('returns true for neg_mse', () => {
    assert.equal(scorerGreaterIsBetter('neg_mse'), true)
  })

  it('returns true for custom function', () => {
    assert.equal(scorerGreaterIsBetter(() => 0.5), true)
  })
})
