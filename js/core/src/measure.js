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
const { normalizeTargets, isTargetMatrix } = require('./targets.js')
const { normalizeX } = require('./matrix.js')

// The callable form retains the original two-array scorer contract. Named and
// structured measures additionally carry the response and optimization direction.
function getScorer(scoring) {
  if (typeof scoring === 'function') return scoring
  const measure = typeof scoring === 'string' ? getMeasureDef(scoring) : defineMeasure(scoring)
  const scorer = (truth, values, opts = {}) => {
    const yn = normalizeTargets(truth, opts.taskKind)
    const multiple = isTargetMatrix(yn)
    const shape = multiple ? { rows: yn.rows, targetCount: yn.cols, taskKind: opts.taskKind } : { rows: yn.length }
    const flatTruth = multiple ? yn.data : yn
    let field = measure.response
    const structured = values && !isTargetMatrix(values) && !ArrayBuffer.isView(values) && !Array.isArray(values) && typeof values === 'object'
    if (isTargetMatrix(values)) {
      const matrix = normalizeX(values)
      if (matrix.rows !== shape.rows || (multiple && matrix.cols !== shape.targetCount)) throw new ValidationError('Score prediction target shape mismatch')
      values = matrix.data
    }
    if (field === 'score' && opts.classes && values.length === shape.rows * opts.classes.length) field = 'proba'
    const data = structured ? values : { [field]: values }
    const value = evaluateMeasure(measure, createPrediction({
      ...data, ...shape, truth: flatTruth, classes: opts.classes ?? data.classes,
      taskKind: opts.taskKind ?? data.taskKind,
      quantileLevels: opts.quantileLevels ?? data.quantileLevels,
      coverageLevels: opts.coverageLevels ?? data.coverageLevels
    }), opts)
    if (!Number.isFinite(value)) throw new ValidationError(`Scorer "${measure.id}" must return a finite number`)
    return value
  }

  return Object.assign(scorer, { measure, direction: measure.direction, response: measure.response })
}

async function scoreEstimator(model, X, y, scoring) {
  const scorer = getScorer(scoring)
  const response = scorer.response || 'response'
  let method = { response: 'predict', proba: 'predictProba', score: 'decisionFunction', decision: 'decisionFunction', quantiles: 'predictQuantiles', interval: 'predictInterval', sets: 'predictSet', region: 'predictRegion', samples: 'predictDistribution', distribution: 'predictDistribution' }[response]
  if (response === 'score' && typeof model[method] !== 'function') method = 'predictProba'
  if (!method || typeof model[method] !== 'function') {
    throw new ValidationError(`Scoring response "${response}" requires ${method || 'a supported prediction method'}`)
  }
  const values = await model[method](X, ...(scorer.measure?.metadata?.predictionArgs || []))
  const taskKind = model.capabilities?.multilabel ? 'multilabel' : model.capabilities?.multioutput ? 'multioutput' : undefined
  const classes = method === 'predictProba' && taskKind !== 'multilabel'
    ? (typeof model.classes === 'function' ? model.classes() : model.classes)
    : undefined
  if (method === 'predictProba' && taskKind !== 'multilabel' && !classes) {
    throw new ValidationError('Probability scoring requires the model class order')
  }
  const value = scorer(y, values, { classes, taskKind })
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new ValidationError('Scorer must return a finite number')
  }
  return value
}

const MEASURE_DIRECTIONS = ['maximize', 'minimize']
const MEASURE_RESPONSES = ['response', 'proba', 'score', 'decision', 'distribution', 'quantiles', 'interval', 'sets', 'region', 'samples']
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

  if (prediction.taskKind && !measure.taskKinds.includes(prediction.taskKind)) throw new ValidationError(`Measure ${measure.id} does not support ${prediction.taskKind}`)
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

function targetMean(metric, { truth, response, prediction, opts }) {
  _requireField(response, 'regression', 'response')
  const t = prediction.targetCount ?? 1
  if (t === 1) return metric(truth, response, opts)
  const rows = truth.length / t
  let sum = 0
  for (let c = 0; c < t; c++) {
    const y = new Float64Array(rows), p = new Float64Array(rows)
    for (let r = 0; r < rows; r++) { y[r] = truth[r * t + c]; p[r] = response[r * t + c] }
    sum += metric(y, p, opts)
  }
  return sum / t
}

function multilabelScore(kind, { truth, response, proba, prediction, opts }) {
  if (prediction.taskKind !== 'multilabel') throw new ValidationError('Multilabel scoring requires explicit task and target axes')
  const t = prediction.targetCount ?? 1, rows = truth.length / t
  const values = kind === 'log_loss' ? proba : response
  _requireField(values, kind, kind === 'log_loss' ? 'proba' : 'response')
  const rowLoss = new Float64Array(rows)
  for (let r = 0; r < rows; r++) {
    let total = 0
    for (let c = 0; c < t; c++) {
      const i = r * t + c
      if (kind === 'log_loss') {
        const p = Math.max(1e-15, Math.min(1 - 1e-15, values[i]))
        total -= truth[i] ? Math.log(p) : Math.log1p(-p)
      } else if (truth[i] !== values[i]) total++
    }
    rowLoss[r] = kind === 'subset_accuracy' ? Number(total === 0) : total / t
  }
  return meanAbsoluteError(new Float64Array(rows), rowLoss, opts)
}

function registerBuiltinMeasures() {
  if (registry.size > 0) return

  for (const [id, kind, response, direction] of [
    ['subset_accuracy', 'subset_accuracy', 'response', 'maximize'],
    ['hamming_loss', 'hamming_loss', 'response', 'minimize'],
    ['multilabel_log_loss', 'log_loss', 'proba', 'minimize']
  ]) registerMeasure({ id, taskKinds: ['multilabel'], direction, response, supportsSampleWeight: true,
    fn: ctx => multilabelScore(kind, ctx) })

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
    taskKinds: ['regression', 'multioutput'],
    direction: 'maximize',
    response: 'response',
    supportsSampleWeight: true,
    fn: ctx => targetMean(r2Score, ctx)
  })
  registerMeasure({
    id: 'mse',
    taskKinds: ['regression', 'multioutput'],
    direction: 'minimize',
    response: 'response',
    supportsSampleWeight: true,
    fn: ctx => targetMean(meanSquaredError, ctx)
  })
  registerMeasure({
    id: 'neg_mse',
    taskKinds: ['regression', 'multioutput'],
    direction: 'maximize',
    response: 'response',
    supportsSampleWeight: true,
    fn: ctx => -targetMean(meanSquaredError, ctx)
  })
  registerMeasure({
    id: 'mae',
    taskKinds: ['regression', 'multioutput'],
    direction: 'minimize',
    response: 'response',
    supportsSampleWeight: true,
    fn: ctx => targetMean(meanAbsoluteError, ctx)
  })
  registerMeasure({
    id: 'neg_mae',
    taskKinds: ['regression', 'multioutput'],
    direction: 'maximize',
    response: 'response',
    supportsSampleWeight: true,
    fn: ctx => -targetMean(meanAbsoluteError, ctx)
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
