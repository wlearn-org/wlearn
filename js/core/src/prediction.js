const { ValidationError } = require('./errors.js')

const PREDICTION_FIELDS = ['response', 'proba', 'score', 'decision', 'interval', 'quantiles', 'sets', 'samples', 'region']
const ARRAY_FIELDS = ['truth', 'response', 'proba', 'score', 'decision', 'interval', 'quantiles', 'classes', 'sets', 'samples', 'quantileLevels', 'coverageLevels']

function numericArray(value, name) {
  if (!(Array.isArray(value) || value instanceof Float64Array || value instanceof Float32Array || value instanceof Int32Array || value instanceof Uint8Array)) {
    throw new ValidationError(`Prediction.${name} must be a flat numeric array`)
  }
  for (const v of value) if (typeof v !== 'number') throw new ValidationError(`Prediction.${name} must be a flat numeric array`)
  return ArrayBuffer.isView(value) ? value : Float64Array.from(value)
}

function positive(value, name) {
  if (!Number.isSafeInteger(value) || value < 1) throw new ValidationError(`Prediction.${name} must be a positive safe integer`)
  return value
}

function size(...dims) {
  const n = dims.reduce((a, b) => a * b, 1)
  return positive(n, 'size')
}

function levels(value, name, endpoints = false) {
  numericArray(value, name)
  if (!value.length) throw new ValidationError(`Prediction.${name} must be non-empty`)
  let previous = -Infinity
  for (const v of value) {
    if (!Number.isFinite(v) || (endpoints ? v < 0 || v > 1 : v <= 0 || v >= 1) || v <= previous) {
      throw new ValidationError(`Prediction.${name} must contain strictly increasing probabilities`)
    }
    previous = v
  }
  return value.length
}

function inferRows(p) {
  const t = p.targetCount ?? 1
  if (p.rows != null) return p.rows
  if (p.truth != null) return p.truth.length / t
  if (p.response != null) return p.response.length / t
  if (p.score != null) return p.score.length / t
  if (p.decision != null) return p.decision.length / t
  if (p.rowIds != null) return p.rowIds.length
  if (p.probaRows != null) return p.probaRows
  if (p.proba != null) return p.proba.length / (p.taskKind === 'multilabel' ? t : p.classes?.length)
  if (p.quantiles != null) return p.quantiles.length / (t * p.quantileLevels?.length)
  if (p.interval != null) return p.interval.length / (t * p.coverageLevels?.length * 2)
  return undefined
}

function createPrediction(opts = {}) {
  const p = { warnings: [...(opts.warnings || [])], metadata: { ...(opts.metadata || {}) } }
  for (const key of ['taskId', 'taskKind', 'rows', 'targetCount', 'targetNames', 'rowIds', 'probaRows', 'sampleCount', 'sampleKind', 'region', 'featureSchemaHash', 'modelArtifactHash']) {
    if (opts[key] != null) p[key] = opts[key]
  }
  for (const key of ARRAY_FIELDS) if (opts[key] != null) p[key] = numericArray(opts[key], key)
  validatePrediction(p)
  p.rows = inferRows(p)
  return p
}

function validatePrediction(p) {
  if (!p || typeof p !== 'object' || !PREDICTION_FIELDS.some(f => p[f] != null)) {
    throw new ValidationError('Prediction must contain at least one prediction field')
  }
  const t = positive(p.targetCount ?? 1, 'targetCount')
  if (p.taskKind != null && !['classification', 'regression', 'multioutput', 'multilabel'].includes(p.taskKind)) {
    throw new ValidationError('Prediction.taskKind is unsupported')
  }
  if (t > 1 && !['multioutput', 'multilabel'].includes(p.taskKind)) {
    throw new ValidationError('Multiple targets require taskKind multioutput or multilabel')
  }
  if (p.targetNames != null && (!Array.isArray(p.targetNames) || p.targetNames.length !== t ||
      p.targetNames.some(v => typeof v !== 'string' || !v.length) || new Set(p.targetNames).size !== t)) {
    throw new ValidationError('Prediction.targetNames must uniquely name every target')
  }
  const rows = positive(inferRows(p), 'rows')
  const nt = size(rows, t)
  if (p.rowIds != null && p.rowIds.length !== rows) throw new ValidationError('Prediction.rowIds length must match rows')
  if (p.probaRows != null && positive(p.probaRows, 'probaRows') !== rows) throw new ValidationError('Prediction.probaRows must match rows')
  function field(name, n, extended = false) {
    if (p[name] == null) return
    const v = numericArray(p[name], name)
    if (v.length !== n) throw new ValidationError(`Prediction.${name} length must equal ${n}`)
    for (const x of v) if (extended ? Number.isNaN(x) : !Number.isFinite(x)) {
      throw new ValidationError(`Prediction.${name} ${extended ? 'must not contain NaN' : 'must be finite'}`)
    }
  }
  for (const name of ['truth', 'response', 'score', 'decision']) field(name, nt)
  if (p.taskKind === 'multilabel') {
    for (const name of ['truth', 'response']) if (p[name] != null) {
      for (const v of p[name]) if (v !== 0 && v !== 1) throw new ValidationError(`Multilabel ${name} values must be 0 or 1`)
    }
    if (p.classes != null) throw new ValidationError('Multilabel predictions use target axes, not a shared class axis')
  }
  if (p.classes != null) {
    const cls = numericArray(p.classes, 'classes')
    if (!cls.length || new Set(cls).size !== cls.length || Array.from(cls).some(v => !Number.isFinite(v))) {
      throw new ValidationError('Prediction.classes must contain unique finite labels')
    }
  }
  if (p.proba != null) {
    const cols = p.taskKind === 'multilabel' ? t : p.classes?.length ?? p.proba.length / rows
    positive(cols, 'probability columns')
    if (p.taskKind === 'multioutput') throw new ValidationError('Multioutput regression has no class probabilities')
    field('proba', size(rows, cols))
    for (let r = 0; r < rows; r++) {
      let sum = 0
      for (let c = 0; c < cols; c++) {
        const v = p.proba[r * cols + c]
        if (v < 0 || v > 1) throw new ValidationError('Prediction.proba values must be in [0, 1]')
        sum += v
      }
      // Float32 outputs need rounding tolerance; independent labels have no row sum constraint.
      if (p.taskKind !== 'multilabel' && Math.abs(sum - 1) > 1e-6) throw new ValidationError('Prediction.proba rows must sum to 1')
    }
  }
  if (p.quantileLevels != null) levels(p.quantileLevels, 'quantileLevels', true)
  if (p.coverageLevels != null) levels(p.coverageLevels, 'coverageLevels')
  if (p.quantiles != null) {
    const q = levels(p.quantileLevels, 'quantileLevels', true)
    field('quantiles', size(nt, q), true)
    // Layout [row, target, level]. A quantile function must be nondecreasing.
    for (let i = 0; i < p.quantiles.length; i++) if (i % q && p.quantiles[i] < p.quantiles[i - 1]) {
      throw new ValidationError('Prediction.quantiles must be nondecreasing within each target')
    }
  }
  if (p.interval != null) {
    const k = levels(p.coverageLevels, 'coverageLevels')
    field('interval', size(nt, k, 2), true)
    for (let i = 0; i < p.interval.length; i += 2) {
      const lo = p.interval[i], hi = p.interval[i + 1]
      // [+Infinity, -Infinity] denotes the empty set; [-Infinity, +Infinity] all real values.
      if (lo > hi && !(lo === Infinity && hi === -Infinity)) throw new ValidationError('Prediction.interval has reversed bounds')
    }
  }
  if (p.sets != null) {
    const k = levels(p.coverageLevels, 'coverageLevels')
    const cols = p.taskKind === 'multilabel' ? size(t, 2) : positive(p.classes?.length, 'classes length')
    field('sets', size(rows, k, cols))
    for (const v of p.sets) if (v !== 0 && v !== 1) throw new ValidationError('Prediction.sets entries must be 0 or 1')
  }
  if (p.samples != null) {
    const n = positive(p.sampleCount, 'sampleCount')
    if (!['outcome', 'mean'].includes(p.sampleKind)) throw new ValidationError('Prediction.sampleKind must be outcome or mean')
    field('samples', size(rows, n, t))
  }
  if (p.region != null) {
    const k = levels(p.coverageLevels, 'coverageLevels')
    if (p.region.kind !== 'ellipsoid') throw new ValidationError('Prediction.region kind must be ellipsoid')
    for (const [key, n] of [['centers', nt], ['precision', size(t, t)], ['radii', size(rows, k)]]) {
      const v = numericArray(p.region[key], `region.${key}`)
      if (v.length !== n || Array.from(v).some(x => key === 'radii' ? Number.isNaN(x) || x < 0 : !Number.isFinite(x))) {
        throw new ValidationError(`Prediction.region.${key} has invalid dimensions or values`)
      }
    }
    // Positive definiteness is checked by the numerical region owner before construction/import.
  }
  return p
}

function predictionRows(p) { validatePrediction(p); return inferRows(p) }
function predictionField(p, field) {
  validatePrediction(p)
  if (!PREDICTION_FIELDS.includes(field) && field !== 'truth') throw new ValidationError(`Unknown prediction field "${field}"`)
  return p[field]
}

module.exports = { PREDICTION_FIELDS, createPrediction, validatePrediction, predictionRows, predictionField }
