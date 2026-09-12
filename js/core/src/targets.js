const { normalizeX, normalizeY, subsetRows, subsetLabels } = require('./matrix.js')
const { ValidationError } = require('./errors.js')

function isTargetMatrix(y) {
  return !!(y && y.data != null) || (Array.isArray(y) && Array.isArray(y[0]))
}

function normalizeTargets(y, kind) {
  const matrix = isTargetMatrix(y)
  const multiple = kind === 'multioutput' || kind === 'multilabel'
  if (matrix !== multiple) {
    throw new ValidationError(matrix
      ? 'Matrix targets require an explicit multioutput or multilabel task'
      : 'Multioutput and multilabel tasks require a target matrix')
  }
  if (!matrix && !(Array.isArray(y) || y instanceof Int32Array || y instanceof Float32Array || y instanceof Float64Array)) {
    throw new ValidationError('Targets must be a vector or dense matrix')
  }
  const result = matrix ? normalizeX(y) : normalizeY(y)
  const values = matrix ? result.data : result
  if (!values.length) throw new ValidationError('Targets must not be empty')
  for (const value of values) {
    if (!Number.isFinite(value)) throw new ValidationError('Targets must be finite')
    if (kind === 'multilabel' && value !== 0 && value !== 1) throw new ValidationError('Multilabel targets must be 0 or 1')
  }
  return result
}

function targetRows(y) {
  if (isTargetMatrix(y)) return normalizeX(y).rows
  if (!(Array.isArray(y) || ArrayBuffer.isView(y)) || !Number.isSafeInteger(y.length) || y.length < 1) {
    throw new ValidationError('Targets require at least one row')
  }
  return y.length
}

function subsetTargets(y, indices) {
  return isTargetMatrix(y) ? subsetRows(y, indices) : subsetLabels(y, indices)
}

module.exports = { isTargetMatrix, normalizeTargets, targetRows, subsetTargets }

function validateSampleWeight(weights, rows) {
  if (!(Array.isArray(weights) || weights instanceof Float64Array || weights instanceof Float32Array || weights instanceof Int32Array) || weights.length !== rows || !rows) {
    throw new ValidationError('sampleWeight must have one entry per target row')
  }
  let total = 0
  for (const v of weights) {
    if (typeof v !== 'number' || !Number.isFinite(v) || v < 0) throw new ValidationError('sampleWeight must be finite and nonnegative')
    total += v
  }
  if (!(total > 0) || !Number.isFinite(total)) throw new ValidationError('sampleWeight must have a finite positive sum')
  return weights
}

module.exports.validateSampleWeight = validateSampleWeight
