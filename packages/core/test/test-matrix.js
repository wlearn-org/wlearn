const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const { normalizeX, normalizeY, makeDense, validateMatrix } = require('../src/matrix.js')
const { ValidationError } = require('../src/errors.js')

describe('normalizeX', () => {
  it('passes through Float64Array typed matrix', () => {
    const data = new Float64Array([1, 2, 3, 4])
    const m = { data, rows: 2, cols: 2 }
    const result = normalizeX(m)
    assert.strictEqual(result.data, data) // same reference, no copy
    assert.equal(result.rows, 2)
    assert.equal(result.cols, 2)
  })

  it('coerces non-Float64Array typed matrix', () => {
    const data = new Float32Array([1, 2, 3, 4])
    const m = { data, rows: 2, cols: 2 }
    const result = normalizeX(m)
    assert(result.data instanceof Float64Array)
    assert.deepEqual([...result.data], [1, 2, 3, 4])
  })

  it('converts number[][] to typed matrix', () => {
    const X = [[1, 2], [3, 4]]
    const result = normalizeX(X)
    assert(result.data instanceof Float64Array)
    assert.equal(result.rows, 2)
    assert.equal(result.cols, 2)
    assert.deepEqual([...result.data], [1, 2, 3, 4])
  })

  it('throws in error mode for number[][]', () => {
    assert.throws(() => normalizeX([[1, 2]], 'error'), ValidationError)
  })

  it('throws in error mode for non-Float64Array typed matrix', () => {
    const m = { data: new Float32Array([1]), rows: 1, cols: 1 }
    assert.throws(() => normalizeX(m, 'error'), ValidationError)
  })

  it('warns in warn mode for number[][]', () => {
    const warnings = []
    const orig = console.warn
    console.warn = (...args) => warnings.push(args.join(' '))
    try {
      normalizeX([[1, 2]], 'warn')
      assert.equal(warnings.length, 1)
      assert(warnings[0].includes('@wlearn/core'))
    } finally {
      console.warn = orig
    }
  })

  it('throws for invalid input', () => {
    assert.throws(() => normalizeX(42), ValidationError)
    assert.throws(() => normalizeX('foo'), ValidationError)
    assert.throws(() => normalizeX(null), ValidationError)
  })
})

describe('normalizeY', () => {
  it('passes through Float64Array', () => {
    const y = new Float64Array([1, 2, 3])
    assert.strictEqual(normalizeY(y), y)
  })

  it('passes through Float32Array', () => {
    const y = new Float32Array([1, 2, 3])
    assert.strictEqual(normalizeY(y), y)
  })

  it('passes through Int32Array', () => {
    const y = new Int32Array([0, 1, 2])
    assert.strictEqual(normalizeY(y), y)
  })

  it('converts number[] to Float64Array', () => {
    const result = normalizeY([1, 2, 3])
    assert(result instanceof Float64Array)
    assert.deepEqual([...result], [1, 2, 3])
  })
})

describe('makeDense', () => {
  it('creates dense matrix with validation', () => {
    const data = new Float64Array([1, 2, 3, 4, 5, 6])
    const m = makeDense(data, 2, 3)
    assert.strictEqual(m.data, data)
    assert.equal(m.rows, 2)
    assert.equal(m.cols, 3)
  })

  it('throws on shape mismatch', () => {
    const data = new Float64Array([1, 2, 3])
    assert.throws(() => makeDense(data, 2, 2), ValidationError)
  })

  it('throws on invalid data type', () => {
    assert.throws(() => makeDense([1, 2, 3, 4], 2, 2), ValidationError)
  })

  it('throws on invalid dimensions', () => {
    const data = new Float64Array(4)
    assert.throws(() => makeDense(data, 0, 4), ValidationError)
    assert.throws(() => makeDense(data, -1, 4), ValidationError)
  })

  it('accepts Float32Array', () => {
    const data = new Float32Array([1, 2, 3, 4])
    const m = makeDense(data, 2, 2)
    assert.strictEqual(m.data, data)
  })
})

describe('validateMatrix', () => {
  it('validates correct matrix', () => {
    const m = { data: new Float64Array([1, 2, 3, 4]), rows: 2, cols: 2 }
    assert.strictEqual(validateMatrix(m), m)
  })

  it('throws on non-object', () => {
    assert.throws(() => validateMatrix(null), ValidationError)
    assert.throws(() => validateMatrix(42), ValidationError)
  })

  it('throws on bad dimensions', () => {
    assert.throws(
      () => validateMatrix({ data: new Float64Array(4), rows: 0, cols: 4 }),
      ValidationError
    )
  })

  it('throws on wrong data type', () => {
    assert.throws(
      () => validateMatrix({ data: [1, 2, 3, 4], rows: 2, cols: 2 }),
      ValidationError
    )
  })

  it('throws on shape mismatch', () => {
    assert.throws(
      () => validateMatrix({ data: new Float64Array(3), rows: 2, cols: 2 }),
      ValidationError
    )
  })
})
