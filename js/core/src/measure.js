const { ValidationError } = require('./errors.js')
const {
  accuracy,
  r2Score,
  meanSquaredError,
  meanAbsoluteError,
  precisionScore,
  recallScore,
  f1Score,
  logLoss,
  rocAuc
} = require('./metrics.js')
const { createPrediction, validatePrediction } = require('./prediction.js')

// The callable form retains the original two-array scorer contract. Named and
// structured measures additionally carry the response and optimization direction.
function getScorer(scoring) {
  if (typeof scoring === 'function') return scoring
  const measure = typeof scoring === 'string' ? getMeasureDef(scoring) : defineMeasure(scoring)
  const scorer = (truth, values, opts = {}) => {
    const field = measure.response === 'score' && opts.classes &&
      values.length === truth.length * opts.classes.length ? 'proba' : measure.response
    const value = evaluateMeasure(measure, createPrediction({
      truth, [field]: values, classes: opts.classes
    }), opts)
    if (!Number.isFinite(value)) throw new ValidationError(`Scorer "${measure.id}" must return a finite number`)
    return value
  }
  return Object.assign(scorer, { measure, direction: measure.direction, response: measure.response })
}

async function scoreEstimator(model, X, y, scoring) {
  const scorer = getScorer(scoring)
  const response = scorer.response || 'response'
  let method = { response: 'predict', proba: 'predictProba', score: 'decisionFunction', decision: 'decisionFunction' }[response]
  if (response === 'score' && typeof model[method] !== 'function') method = 'predictProba'
  if (!method || typeof model[method] !== 'function') {
    throw new ValidationError(`Scoring response "${response}" requires ${method || 'a supported prediction method'}`)
  }
  const values = await model[method](X)
  const classes = method === 'predictProba'
    ? (typeof model.classes === 'function' ? model.classes() : model.classes)
    : undefined
  if (method === 'predictProba' && !classes) {
    throw new ValidationError('Probability scoring requires the model class order')
  }
  const value = scorer(y, values, { classes })
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new ValidationError('Scorer must return a finite number')
  }
  return value
}

const MEASURE_DIRECTIONS = ['maximize', 'minimize']
const MEASURE_RESPONSES = ['response', 'proba', 'score', 'decision', 'distribution']
const registry = new Map()

function defineMeasure(def) {
  if (!def || typeof def !== 'object') {
    throw new ValidationError('Measure definition must be an object')
  }
  if (typeof def.id !== 'string' || def.id.length === 0) {
    throw new ValidationError('Measure.id must be a non-empty string')
  }
  if (!Array.isArray(def.taskKinds) || def.taskKinds.length === 0) {
    throw new ValidationError(`Measure "${def.id}" must declare taskKinds`)
  }
  if (!MEASURE_DIRECTIONS.includes(def.direction)) {
    throw new ValidationError(`Measure "${def.id}" has invalid direction "${def.direction}"`)
  }
  if (!MEASURE_RESPONSES.includes(def.response)) {
    throw new ValidationError(`Measure "${def.id}" has invalid response "${def.response}"`)
  }
  if (typeof def.fn !== 'function') {
    throw new ValidationError(`Measure "${def.id}" must provide fn`)
  }
  return {
    id: def.id,
    label: def.label || def.id,
    taskKinds: [...def.taskKinds],
    direction: def.direction,
    response: def.response,
    range: def.range ? [def.range[0], def.range[1]] : [-Infinity, Infinity],
    average: def.average || 'macro',
    naValue: def.naValue,
    requiresTruth: def.requiresTruth !== false,
    supportsSampleWeight: !!def.supportsSampleWeight,
    metadata: { ...(def.metadata || {}) },
    aggregator: def.aggregator || meanAggregator,
    fn: def.fn
  }
}

function registerMeasure(def) {
  const measure = defineMeasure(def)
  registry.set(measure.id, measure)
  return measure
}

function getMeasureDef(id) {
  const measure = registry.get(id)
  if (!measure) {
    throw new ValidationError(`Unknown measure "${id}". Available: ${listMeasures().join(', ')}`)
  }
  return measure
}

function listMeasures() {
  return [...registry.keys()].sort()
}

function evaluateMeasure(measureOrId, prediction, opts = {}) {
  const measure = typeof measureOrId === 'string' ? getMeasureDef(measureOrId) : defineMeasure(measureOrId)
  validatePrediction(prediction)

  const truth = opts.truth || prediction.truth
  if (measure.requiresTruth && !truth) {
    throw new ValidationError(`Measure "${measure.id}" requires truth labels`)
  }

  const value = measure.fn({
    truth,
    response: prediction.response,
    proba: prediction.proba,
    score: prediction.score,
    decision: prediction.decision,
    prediction,
    opts
  })
  if (typeof value !== 'number' || (Number.isNaN(value) && !_allowsNaN(opts))) {
    if (measure.naValue != null) return measure.naValue
    throw new ValidationError(`Measure "${measure.id}" returned a non-number`)
  }
  return value
}

function aggregateMeasure(measureOrId, values) {
  const measure = typeof measureOrId === 'string' ? getMeasureDef(measureOrId) : defineMeasure(measureOrId)
  if (!values || values.length === 0) {
    throw new ValidationError(`Measure "${measure.id}" cannot aggregate empty values`)
  }
  return measure.aggregator(values)
}

function evaluateMetricSet(measures, prediction, opts = {}) {
  const result = {}
  for (const measure of measures) {
    const id = typeof measure === 'string' ? measure : measure.id
    result[id] = evaluateMeasure(measure, prediction, opts[id] || opts)
  }
  return result
}

function meanAggregator(values) {
  let sum = 0
  let n = 0
  for (const value of values) {
    if (Number.isNaN(value)) continue
    sum += value
    n++
  }
  return n === 0 ? NaN : sum / n
}

function _requireField(value, measureId, field) {
  if (!value) throw new ValidationError(`Measure "${measureId}" requires prediction.${field}`)
  return value
}

function _labelsFromTruth(truth) {
  return [...new Set(Array.from(truth))].sort((a, b) => a - b)
}

function _allowsNaN(opts = {}) {
  return opts.allowNaN === true ||
    opts.allow_nan === true ||
    opts.undefinedValue === 'nan' ||
    opts.undefined_value === 'nan' ||
    opts.undefinedValue === 'warn' ||
    opts.undefined_value === 'warn' ||
    opts.zeroDivision === 'nan' ||
    opts.zero_division === 'nan'
}

function _classesOpt(prediction, opts = {}) {
  if (opts.classes) return opts.classes
  return prediction.classes ? Array.from(prediction.classes) : undefined
}

function _binaryScoreFromPrediction(score, proba, truth, classes) {
  if (score) return score
  if (!proba) return undefined
  if (proba.length === truth.length) return proba
  if (proba.length === truth.length * 2) {
    const truthLabels = _labelsFromTruth(truth)
    const positiveLabel = truthLabels[truthLabels.length - 1]
    const classList = classes ? Array.from(classes) : truthLabels
    const positiveColumn = classList.indexOf(positiveLabel)
    if (positiveColumn < 0) {
      throw new ValidationError(`roc_auc: positive class "${positiveLabel}" missing from prediction.classes`)
    }
    const out = new Float64Array(truth.length)
    for (let i = 0; i < truth.length; i++) out[i] = proba[i * 2 + positiveColumn]
    return out
  }
  return undefined
}

function _aucScoreFromPrediction(score, proba, truth, prediction, opts = {}) {
  if (score) return score
  if (!proba) return undefined
  if ((opts.multiClass || opts.multi_class) && prediction.classes && proba.length === truth.length * prediction.classes.length) {
    return proba
  }
  if ((opts.multiClass || opts.multi_class) && proba.length % truth.length === 0) {
    return proba
  }
  return _binaryScoreFromPrediction(score, proba, truth, _classesOpt(prediction, opts))
}

function registerBuiltinMeasures() {
  if (registry.size > 0) return

  registerMeasure({
    id: 'accuracy',
    taskKinds: ['classification'],
    direction: 'maximize',
    response: 'response',
    supportsSampleWeight: true,
    fn: ({ truth, response, opts }) => accuracy(truth, _requireField(response, 'accuracy', 'response'), opts)
  })
  registerMeasure({
    id: 'precision',
    taskKinds: ['classification'],
    direction: 'maximize',
    response: 'response',
    supportsSampleWeight: true,
    fn: ({ truth, response, opts }) => precisionScore(truth, _requireField(response, 'precision', 'response'), opts)
  })
  registerMeasure({
    id: 'recall',
    taskKinds: ['classification'],
    direction: 'maximize',
    response: 'response',
    supportsSampleWeight: true,
    fn: ({ truth, response, opts }) => recallScore(truth, _requireField(response, 'recall', 'response'), opts)
  })
  registerMeasure({
    id: 'f1',
    taskKinds: ['classification'],
    direction: 'maximize',
    response: 'response',
    supportsSampleWeight: true,
    fn: ({ truth, response, opts }) => f1Score(truth, _requireField(response, 'f1', 'response'), opts)
  })
  registerMeasure({
    id: 'log_loss',
    taskKinds: ['classification'],
    direction: 'minimize',
    response: 'proba',
    supportsSampleWeight: true,
    fn: ({ truth, proba, prediction, opts }) => logLoss(truth, _requireField(proba, 'log_loss', 'proba'), {
      ...opts,
      classes: _classesOpt(prediction, opts)
    })
  })
  registerMeasure({
    id: 'roc_auc',
    taskKinds: ['classification'],
    direction: 'maximize',
    response: 'score',
    supportsSampleWeight: true,
    fn: ({ truth, score, proba, prediction, opts }) => rocAuc(
      truth,
      _requireField(_aucScoreFromPrediction(score, proba, truth, prediction, opts), 'roc_auc', 'score'),
      { ...opts, classes: _classesOpt(prediction, opts) }
    )
  })
  registerMeasure({
    id: 'roc_auc_ovr',
    taskKinds: ['classification'],
    direction: 'maximize',
    response: 'proba',
    average: 'macro',
    supportsSampleWeight: true,
    fn: ({ truth, score, proba, prediction, opts }) => rocAuc(
      truth,
      _requireField(score || proba, 'roc_auc_ovr', 'proba'),
      { ...opts, classes: _classesOpt(prediction, opts), multiClass: 'ovr' }
    )
  })
  registerMeasure({
    id: 'roc_auc_ovo',
    taskKinds: ['classification'],
    direction: 'maximize',
    response: 'proba',
    average: 'macro',
    supportsSampleWeight: true,
    fn: ({ truth, score, proba, prediction, opts }) => rocAuc(
      truth,
      _requireField(score || proba, 'roc_auc_ovo', 'proba'),
      { ...opts, classes: _classesOpt(prediction, opts), multiClass: 'ovo' }
    )
  })
  registerMeasure({
    id: 'r2',
    taskKinds: ['regression'],
    direction: 'maximize',
    response: 'response',
    supportsSampleWeight: true,
    fn: ({ truth, response, opts }) => r2Score(truth, _requireField(response, 'r2', 'response'), opts)
  })
  registerMeasure({
    id: 'mse',
    taskKinds: ['regression'],
    direction: 'minimize',
    response: 'response',
    supportsSampleWeight: true,
    fn: ({ truth, response, opts }) => meanSquaredError(truth, _requireField(response, 'mse', 'response'), opts)
  })
  registerMeasure({
    id: 'neg_mse',
    taskKinds: ['regression'],
    direction: 'maximize',
    response: 'response',
    supportsSampleWeight: true,
    fn: ({ truth, response, opts }) => -meanSquaredError(truth, _requireField(response, 'neg_mse', 'response'), opts)
  })
  registerMeasure({
    id: 'mae',
    taskKinds: ['regression'],
    direction: 'minimize',
    response: 'response',
    supportsSampleWeight: true,
    fn: ({ truth, response, opts }) => meanAbsoluteError(truth, _requireField(response, 'mae', 'response'), opts)
  })
  registerMeasure({
    id: 'neg_mae',
    taskKinds: ['regression'],
    direction: 'maximize',
    response: 'response',
    supportsSampleWeight: true,
    fn: ({ truth, response, opts }) => -meanAbsoluteError(truth, _requireField(response, 'neg_mae', 'response'), opts)
  })
}

registerBuiltinMeasures()

module.exports = {
  getScorer,
  scoreEstimator,
  MEASURE_DIRECTIONS,
  MEASURE_RESPONSES,
  defineMeasure,
  registerMeasure,
  getMeasureDef,
  listMeasures,
  evaluateMeasure,
  aggregateMeasure,
  evaluateMetricSet,
  meanAggregator,
  registerBuiltinMeasures
}
