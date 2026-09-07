const { ValidationError } = require('./errors.js')

function matrixSize(rows, cols) {
  if (cols === 0) {
    throw new ValidationError('Matrix has zero columns')
  }
  if (!Number.isSafeInteger(rows) || !Number.isSafeInteger(cols) ||
      rows < 1 || cols < 1) {
    throw new ValidationError(`Invalid dimensions: rows=${rows}, cols=${cols}`)
  }
  const size = rows * cols
  if (!Number.isSafeInteger(size)) {
    throw new ValidationError(`Matrix size is not a safe integer: rows=${rows}, cols=${cols}`)
  }
  return size
}

function normalizeX(X, coerce = 'auto') {
  if (X?.indptr != null || X?.indices != null) {
    throw new ValidationError('normalizeX requires a dense matrix; CSR is not supported')
  }
  // Fast path: typed matrix { data, rows, cols }
  if (X && typeof X === 'object' && !Array.isArray(X) && X.data != null) {
    const { data, rows, cols } = X
    const size = matrixSize(rows, cols)
    if (typeof data.length !== 'number' || !Number.isSafeInteger(data.length)) {
      throw new ValidationError('Typed matrix data must be an array-like numeric buffer')
    }
    if (data.length !== size) {
      throw new ValidationError(`data.length (${data.length}) !== rows * cols (${size})`)
    }
    if (!(data instanceof Float64Array)) {
      if (coerce === 'error') throw new ValidationError('Expected Float64Array in typed matrix')
      try {
        return { data: new Float64Array(data), rows, cols }
      } catch (error) {
        throw new ValidationError(`Typed matrix data cannot be converted to Float64Array: ${error.message}`)
      }
    }
    return { data, rows, cols }
  }

  // Slow path: number[][]
  if (Array.isArray(X)) {
    if (X.length === 0 || !Array.isArray(X[0])) {
      throw new ValidationError('X must be a non-empty rectangular number[][]')
    }
    if (coerce === 'error') {
      throw new ValidationError('Input coercion disabled (coerce: "error"). Pass { data: Float64Array, rows, cols } instead of number[][].')
    }
    const rows = X.length
    const cols = X[0].length
    const data = new Float64Array(matrixSize(rows, cols))
    for (let i = 0; i < rows; i++) {
      if (!Array.isArray(X[i]) || X[i].length !== cols) {
        throw new ValidationError(`X must be rectangular; row ${i} has length ${X[i]?.length}, expected ${cols}`)
      }
      for (let j = 0; j < cols; j++) {
        const value = X[i][j]
        if (typeof value !== 'number') {
          throw new ValidationError(`X[${i}][${j}] must be a number`)
        }
        data[i * cols + j] = value
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
  const size = matrixSize(rows, cols)
  if (!(data instanceof Float32Array) && !(data instanceof Float64Array)) {
    throw new ValidationError('data must be Float32Array or Float64Array')
  }
  if (data.length !== size) {
    throw new ValidationError(`data.length (${data.length}) !== rows * cols (${size})`)
  }
  return { data, rows, cols }
}

function validateMatrix(m) {
  if (m?.indptr != null || m?.indices != null) {
    throw new ValidationError('validateMatrix requires a dense matrix; CSR is not supported')
  }
  if (!m || typeof m !== 'object') {
    throw new ValidationError('Matrix must be an object')
  }
  const { data, rows, cols } = m
  const size = matrixSize(rows, cols)
  if (!(data instanceof Float32Array) && !(data instanceof Float64Array)) {
    throw new ValidationError('data must be Float32Array or Float64Array')
  }
  if (data.length !== size) {
    throw new ValidationError(`data.length (${data.length}) !== rows * cols (${size})`)
  }
  return m
}

// Copies selected rows into host-owned buffers for CV/module boundaries.
function subsetRows(X, indices) {
  const { data, rows, cols } = normalizeX(X)
  validateIndices(indices, rows)
  const out = new Float64Array(indices.length * cols)
  for (let i = 0; i < indices.length; i++) {
    const offset = indices[i] * cols
    out.set(data.subarray(offset, offset + cols), i * cols)
  }
  return { data: out, rows: indices.length, cols }
}

function subsetLabels(y, indices) {
  const labels = normalizeY(y)
  validateIndices(indices, labels.length)
  const out = new labels.constructor(indices.length)
  for (let i = 0; i < indices.length; i++) out[i] = labels[indices[i]]
  return out
}

function validateIndices(indices, rows) {
  if (!(Array.isArray(indices) || indices instanceof Int32Array) || !indices.length) {
    throw new ValidationError('Row indices must be a non-empty array or Int32Array')
  }
  for (const index of indices) {
    if (!Number.isInteger(index) || index < 0 || index >= rows) {
      throw new ValidationError('Row indices must be integers within the input rows')
    }
  }
}

module.exports = { normalizeX, normalizeY, makeDense, validateMatrix, subsetRows, subsetLabels }
