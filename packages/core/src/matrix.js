const { ValidationError } = require('./errors.js')

function normalizeX(X, coerce = 'auto') {
  // Fast path: typed matrix { data, rows, cols }
  if (X && typeof X === 'object' && !Array.isArray(X) && X.data) {
    const { data, rows, cols } = X
    if (!(data instanceof Float64Array)) {
      if (coerce === 'error') throw new ValidationError('Expected Float64Array in typed matrix')
      return { data: new Float64Array(data), rows, cols }
    }
    return { data, rows, cols }
  }

  // Slow path: number[][]
  if (Array.isArray(X) && Array.isArray(X[0])) {
    if (coerce === 'error') {
      throw new ValidationError('Input coercion disabled (coerce: "error"). Pass { data: Float64Array, rows, cols } instead of number[][].')
    }
    const rows = X.length
    const cols = X[0].length
    const data = new Float64Array(rows * cols)
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        data[i * cols + j] = X[i][j]
      }
    }
    if (coerce === 'warn') {
      const bytes = data.byteLength
      console.warn(`@wlearn/core: Converted number[][] to Float64Array (copied ${(bytes / 1024).toFixed(1)} KB, shape ${rows}x${cols}). For performance, pass { data, rows, cols }.`)
    }
    return { data, rows, cols }
  }

  throw new ValidationError('X must be number[][] or { data: Float64Array, rows, cols }')
}

function normalizeY(y) {
  if (y instanceof Int32Array) return y
  if (y instanceof Float32Array) return y
  if (y instanceof Float64Array) return y
  return new Float64Array(y)
}

function makeDense(data, rows, cols) {
  if (!rows || !cols || rows < 1 || cols < 1) {
    throw new ValidationError(`Invalid dimensions: rows=${rows}, cols=${cols}`)
  }
  if (!(data instanceof Float32Array) && !(data instanceof Float64Array)) {
    throw new ValidationError('data must be Float32Array or Float64Array')
  }
  if (data.length !== rows * cols) {
    throw new ValidationError(`data.length (${data.length}) !== rows * cols (${rows * cols})`)
  }
  return { data, rows, cols }
}

function validateMatrix(m) {
  if (!m || typeof m !== 'object') {
    throw new ValidationError('Matrix must be an object')
  }
  const { data, rows, cols } = m
  if (typeof rows !== 'number' || typeof cols !== 'number' || rows < 1 || cols < 1) {
    throw new ValidationError(`Invalid dimensions: rows=${rows}, cols=${cols}`)
  }
  if (!(data instanceof Float32Array) && !(data instanceof Float64Array)) {
    throw new ValidationError('data must be Float32Array or Float64Array')
  }
  if (data.length !== rows * cols) {
    throw new ValidationError(`data.length (${data.length}) !== rows * cols (${rows * cols})`)
  }
  return m
}

module.exports = { normalizeX, normalizeY, makeDense, validateMatrix }
