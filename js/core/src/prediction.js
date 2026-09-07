const { ValidationError } = require('./errors.js')
const { normalizeY } = require('./matrix.js')

const PREDICTION_FIELDS = ['response', 'proba', 'score', 'decision', 'interval', 'quantiles']

function _toFloat64(x, name) {
  if (x == null) return undefined
  if (x instanceof Float64Array) return x
  if (x instanceof Float32Array || x instanceof Int32Array || Array.isArray(x)) {
    return new Float64Array(x)
  }
  throw new ValidationError(`Prediction.${name} must be an array or typed array`)
}

function _inferRows(prediction) {
  if (prediction.truth) return prediction.truth.length
  if (prediction.response) return prediction.response.length
  if (prediction.score) return prediction.score.length
  if (prediction.decision) return prediction.decision.length
  if (prediction.rowIds) return prediction.rowIds.length
  if (prediction.probaRows) return prediction.probaRows
  if (prediction.proba && prediction.classes && prediction.classes.length > 0) {
    return prediction.proba.length / prediction.classes.length
  }
  return undefined
}

function createPrediction({
  taskId,
  rowIds,
  truth,
  response,
  proba,
  probaRows,
  score,
  decision,
  interval,
  quantiles,
  classes,
  featureSchemaHash,
  modelArtifactHash,
  warnings = [],
  metadata = {}
} = {}) {
  const prediction = {
    warnings: [...warnings],
    metadata: { ...metadata }
  }
  if (taskId) prediction.taskId = String(taskId)
  if (rowIds) prediction.rowIds = rowIds
  if (truth != null) prediction.truth = normalizeY(truth)
  if (response != null) prediction.response = normalizeY(response)
  if (proba != null) prediction.proba = _toFloat64(proba, 'proba')
  if (probaRows != null) prediction.probaRows = probaRows
  if (score != null) prediction.score = _toFloat64(score, 'score')
  if (decision != null) prediction.decision = _toFloat64(decision, 'decision')
  if (interval != null) prediction.interval = _toFloat64(interval, 'interval')
  if (quantiles != null) prediction.quantiles = _toFloat64(quantiles, 'quantiles')
  if (classes != null) prediction.classes = normalizeY(classes)
  if (featureSchemaHash) prediction.featureSchemaHash = String(featureSchemaHash)
  if (modelArtifactHash) prediction.modelArtifactHash = String(modelArtifactHash)
  return validatePrediction(prediction)
}

function validatePrediction(prediction) {
  if (!prediction || typeof prediction !== 'object') {
    throw new ValidationError('Prediction must be an object')
  }

  let hasField = false
  for (const field of PREDICTION_FIELDS) {
    if (prediction[field] != null) hasField = true
  }
  if (!hasField) {
    throw new ValidationError('Prediction must contain at least one prediction field')
  }

  const rows = _inferRows(prediction)
  if (rows == null || rows < 1) {
    throw new ValidationError('Prediction row count could not be inferred')
  }
  if (!Number.isInteger(rows)) {
    throw new ValidationError(`Prediction row count must be an integer, got ${rows}`)
  }

  for (const field of ['truth', 'response', 'score', 'decision']) {
    if (prediction[field] && prediction[field].length !== rows) {
      throw new ValidationError(`Prediction.${field} length (${prediction[field].length}) must match row count (${rows})`)
    }
  }
  for (const field of ['interval', 'quantiles']) {
    if (prediction[field] && prediction[field].length % rows !== 0) {
      throw new ValidationError(`Prediction.${field} length (${prediction[field].length}) must be divisible by row count (${rows})`)
    }
  }
  if (prediction.rowIds && prediction.rowIds.length !== rows) {
    throw new ValidationError(`Prediction.rowIds length (${prediction.rowIds.length}) must match row count (${rows})`)
  }
  if (prediction.proba) {
    const nClasses = prediction.classes ? prediction.classes.length : undefined
    if (prediction.probaRows != null && (!Number.isInteger(prediction.probaRows) || prediction.probaRows < 1)) {
      throw new ValidationError('Prediction.probaRows must be a positive integer')
    }
    if (prediction.probaRows != null && prediction.probaRows !== rows) {
      throw new ValidationError(`Prediction.probaRows (${prediction.probaRows}) must match row count (${rows})`)
    }
    if (prediction.classes && prediction.classes.length === 0) {
      throw new ValidationError('Prediction.classes must be non-empty')
    }
    if (nClasses && prediction.proba.length !== rows * nClasses) {
      throw new ValidationError(`Prediction.proba length (${prediction.proba.length}) must equal rows * classes (${rows * nClasses})`)
    }
    if (!nClasses && prediction.probaRows && prediction.proba.length % prediction.probaRows !== 0) {
      throw new ValidationError(`Prediction.proba length (${prediction.proba.length}) must be divisible by probaRows (${prediction.probaRows})`)
    }
  }
  if (prediction.classes) {
    const labels = Array.from(prediction.classes)
    if (labels.some(value => !Number.isFinite(value)) || new Set(labels).size !== labels.length) {
      throw new ValidationError('Prediction.classes must contain unique finite labels')
    }
  }
  if (prediction.proba) {
    const cols = prediction.proba.length / rows
    if (!Number.isInteger(cols) || cols < 1) {
      throw new ValidationError('Prediction.proba must contain complete probability rows')
    }
    for (let row = 0; row < rows; row++) {
      let sum = 0
      for (let col = 0; col < cols; col++) {
        const value = prediction.proba[row * cols + col]
        if (!Number.isFinite(value) || value < 0 || value > 1) {
          throw new ValidationError('Prediction.proba values must be finite and in [0, 1]')
        }
        sum += value
      }
      // Allow Float32 backend rounding without accepting unnormalized scores.
      if (Math.abs(sum - 1) > 1e-6) {
        throw new ValidationError('Prediction.proba rows must sum to 1')
      }
    }
  }
  return prediction
}

function predictionRows(prediction) {
  validatePrediction(prediction)
  return _inferRows(prediction)
}

function predictionField(prediction, field) {
  validatePrediction(prediction)
  if (!PREDICTION_FIELDS.includes(field) && field !== 'truth') {
    throw new ValidationError(`Unknown prediction field "${field}"`)
  }
  return prediction[field]
}

module.exports = {
  PREDICTION_FIELDS,
  createPrediction,
  validatePrediction,
  predictionRows,
  predictionField
}
