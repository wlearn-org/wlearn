const {
  normalizeX, normalizeTargets, validateSampleWeight, taskParams, validateEstimatorTask,
  createPrediction, validatePrediction, r2Score, lift, isPromiseLike, encodeBundle, validateBundle,
  register, load, assertRequiredLoaders, ValidationError, NotFittedError, DisposedError
} = require('@wlearn/core')
const { classColumnMap, requireProbabilityModel, validateProbabilityOutput, validateRegressionOutput } = require('./class-order.js')

const TYPES = {
  multioutput: 'wlearn.ensemble.multioutput.regressor@1',
  multilabel: 'wlearn.ensemble.multilabel.classifier@1'
}

function disposeAll(models, suppress = false) {
  let first
  for (const model of models) if (model) {
    try { model.dispose() } catch (error) { first ??= error }
  }
  if (first && !suppress) throw first
}

function namesFor(names, count) {
  const result = names ?? Array.from({ length: count }, (_, i) => `target_${i}`)
  if (!Array.isArray(result) || result.length !== count || new Set(result).size !== count ||
      result.some(v => typeof v !== 'string' || !v.length)) throw new ValidationError('targetNames must uniquely name every target')
  return [...result]
}

function specFor(spec) {
  if (!Array.isArray(spec) || spec.length < 2 || spec.length > 3 || typeof spec[0] !== 'string' ||
      !spec[0].length || typeof spec[1]?.create !== 'function' ||
      (spec[2] != null && (typeof spec[2] !== 'object' || Array.isArray(spec[2])))) {
    throw new ValidationError('estimator must be [name, EstimatorClass, params?]; loaded composites need a new specification before refitting')
  }
  return [spec[0], spec[1], { ...(spec[2] || {}) }]
}

class MultiTarget {
  #kind
  #spec
  #names
  #models = []
  #constants = []
  #columns = []
  #features = 0
  #fitted = false
  #disposed = false
  #busy = false

  constructor(params, kind) {
    if (params.task != null && params.task !== kind) throw new ValidationError(`task must be ${kind}`)
    this.#kind = kind
    this.#spec = params.estimator ? specFor(params.estimator) : null
    this.#names = params.targetNames ? [...params.targetNames] : null
  }

  static async create(params = {}) { return new this(params) }

  #alive() { if (this.#disposed) throw new DisposedError('Multi-target estimator has been disposed') }
  #ready() {
    this.#alive()
    if (this.#busy) throw new ValidationError('Multi-target fit is in progress')
    if (!this.#fitted) throw new NotFittedError('Multi-target estimator is not fitted')
  }

  async fit(X, y, opts = {}) {
    this.#alive()
    if (this.#busy) throw new ValidationError('Multi-target fit is already in progress')
    if (!opts || Object.keys(opts).some(key => key !== 'sampleWeight')) throw new ValidationError('fit options support only sampleWeight')
    const [name, Class, params] = specFor(this.#spec)
    const xn = normalizeX(X), yn = normalizeTargets(y, this.#kind)
    if (xn.rows !== yn.rows) throw new ValidationError('X and target rows must match')
    const names = namesFor(this.#names, yn.cols)
    const weights = opts.sampleWeight == null ? null : validateSampleWeight(opts.sampleWeight, yn.rows)
    const task = this.#kind === 'multilabel' ? 'classification' : 'regression'
    taskParams(params, task)
    this.#busy = true
    const models = [], constants = [], columns = []
    try {
      for (let t = 0; t < yn.cols; t++) {
        const column = new Float64Array(yn.rows)
        for (let r = 0; r < yn.rows; r++) column[r] = yn.data[r * yn.cols + t]
        // A fold may contain a constant binary label. Preserve both public states
        // without asking a two-class backend to fit an impossible objective.
        const constant = task === 'classification' && column.every(v => v === column[0]) ? column[0] : null
        constants.push(constant)
        if (constant != null) { models.push(null); columns.push(null); continue }
        const model = await Class.create(taskParams(params, task))
        models.push(model)
        validateEstimatorTask(model, task)
        if (task === 'classification') requireProbabilityModel(model, name)
        if (weights && !model.capabilities?.sampleWeight) throw new ValidationError(`${name} does not support sampleWeight`)
        await model.fit(xn, column, weights ? { sampleWeight: weights } : undefined)
        validateEstimatorTask(model, task)
        columns.push(task === 'classification' ? classColumnMap(model, [0, 1], name) : null)
      }
      const previous = this.#models
      this.#models = models
      this.#constants = constants
      this.#columns = columns
      this.#features = xn.cols
      this.#names = names
      this.#fitted = true
      disposeAll(previous, true)
      return this
    } catch (error) {
      disposeAll(models, true)
      throw error
    } finally { this.#busy = false }
  }

  #matrix(X) {
    this.#ready()
    const xn = normalizeX(X)
    if (xn.cols !== this.#features) throw new ValidationError('Prediction feature count differs from training')
    return xn
  }

  #collect(X, probability) {
    const xn = this.#matrix(X), count = this.#models.length
    const models = this.#models
    const raw = this.#models.map((m, t) => m ? m[probability ? 'predictProba' : 'predict'](xn) : this.#constants[t])
    const combine = values => {
      this.#ready()
      if (models !== this.#models) throw new ValidationError('Multi-target model changed during prediction')
      const data = new Float64Array(xn.rows * count)
      values.forEach((v, t) => {
        if (!this.#models[t]) {
          for (let r = 0; r < xn.rows; r++) data[r * count + t] = v
        } else if (probability) {
          const p = validateProbabilityOutput(v, xn.rows, 2, this.#names[t])
          const positive = this.#columns[t][1]
          for (let r = 0; r < xn.rows; r++) data[r * count + t] = p[r * 2 + positive]
        } else {
          const p = validateRegressionOutput(v, xn.rows, this.#names[t])
          for (let r = 0; r < xn.rows; r++) data[r * count + t] = p[r]
        }
      })
      return { data, rows: xn.rows, cols: count }
    }
    return raw.some(isPromiseLike) ? Promise.all(raw).then(combine) : combine(raw)
  }

  predict(X) {
    if (this.#kind === 'multioutput') return this.#collect(X, false)
    return lift(this.predictProba(X), p => ({ ...p, data: Float64Array.from(p.data, v => v >= 0.5 ? 1 : 0) }))
  }

  predictProba(X) {
    if (this.#kind !== 'multilabel') throw new ValidationError('Multioutput regression does not support probabilities')
    return this.#collect(X, true)
  }

  predictQuantiles(X, levels) {
    const xn = this.#matrix(X)
    if (!this.capabilities.predictQuantiles) throw new ValidationError('Every target estimator must support quantiles')
    const models = this.#models
    levels = Array.from(createPrediction({ rows: 1, quantileLevels: levels, quantiles: new Float64Array(levels?.length || 0) }).quantileLevels)
    const raw = this.#models.map(m => m.predictQuantiles(xn, levels))
    const combine = values => {
      this.#ready()
      if (models !== this.#models) throw new ValidationError('Multi-target model changed during prediction')
      const q = levels.length, t = values.length, data = new Float64Array(xn.rows * t * q)
      values.forEach((p, c) => {
        const checked = validatePrediction(p)
        if (checked.rows !== xn.rows || (checked.targetCount ?? 1) !== 1 || !checked.quantiles ||
            checked.quantileLevels.length !== q || levels.some((v, i) => v !== checked.quantileLevels[i])) {
          throw new ValidationError('Target quantile dimensions and levels must match the request')
        }
        for (let r = 0; r < xn.rows; r++) data.set(checked.quantiles.subarray(r * q, (r + 1) * q), (r * t + c) * q)
      })
      return createPrediction({ rows: xn.rows, taskKind: 'multioutput', targetCount: t, targetNames: this.targetNames, quantiles: data, quantileLevels: levels })
    }
    return raw.some(isPromiseLike) ? Promise.all(raw).then(combine) : combine(raw)
  }

  score(X, y) {
    const yn = normalizeTargets(y, this.#kind)
    return lift(this.predict(X), p => {
      if (p.rows !== yn.rows || p.cols !== yn.cols) throw new ValidationError('Score target shape mismatch')
      if (this.#kind === 'multilabel') {
        let correct = 0
        for (let r = 0; r < p.rows; r++) {
          let same = true
          for (let t = 0; t < p.cols; t++) same &&= p.data[r * p.cols + t] === yn.data[r * p.cols + t]
          if (same) correct++
        }
        return correct / p.rows
      }
      let sum = 0
      for (let t = 0; t < p.cols; t++) {
        const truth = new Float64Array(p.rows), response = new Float64Array(p.rows)
        for (let r = 0; r < p.rows; r++) { truth[r] = yn.data[r * p.cols + t]; response[r] = p.data[r * p.cols + t] }
        sum += r2Score(truth, response)
      }
      return sum / p.cols
    })
  }

  get targetNames() { return this.#names ? [...this.#names] : null }
  get targetCount() { return this.#models.length || this.#names?.length || null }
  get isFitted() { return this.#fitted && !this.#disposed }
  get capabilities() {
    return { classifier: this.#kind === 'multilabel', regressor: this.#kind === 'multioutput',
      multioutput: this.#kind === 'multioutput', multilabel: this.#kind === 'multilabel',
      predictProba: this.#kind === 'multilabel', predictQuantiles: this.#kind === 'multioutput' && this.#models.length > 0 && this.#models.every(m => m?.capabilities?.predictQuantiles),
      decisionFunction: false, sampleWeight: true, csr: false, earlyStopping: false }
  }
  getParams() { return { task: this.#kind, targetNames: this.targetNames, estimatorName: this.#spec?.[0] ?? null, estimatorParams: { ...(this.#spec?.[2] || {}) } } }
  setParams(params) {
    this.#alive()
    if (this.#busy) throw new ValidationError('Cannot change parameters during fit')
    if (!params || Object.keys(params).some(k => !['estimator', 'targetNames'].includes(k))) throw new ValidationError('Supported params are estimator and targetNames')
    const spec = params.estimator ? specFor(params.estimator) : this.#spec
    const names = params.targetNames === undefined ? this.#names : params.targetNames
    if (names != null) namesFor(names, names.length)
    const previous = this.#models
    this.#fitted = false
    this.#models = []
    this.#spec = spec
    this.#names = names == null ? null : [...names]
    disposeAll(previous)
    return this
  }
  dispose() {
    if (this.#disposed) return
    if (this.#busy) throw new ValidationError('Cannot dispose during fit')
    this.#disposed = true
    this.#fitted = false
    const previous = this.#models
    this.#models = []
    disposeAll(previous)
  }
  save() {
    this.#ready()
    const artifacts = [], requires = new Set()
    this.#models.forEach((m, t) => {
      if (!m) return
      const data = m.save(), manifest = validateBundle(data).manifest
      requires.add(manifest.typeId)
      for (const id of manifest.requires || []) requires.add(id)
      artifacts.push({ id: `target_${t}`, data, mediaType: 'application/x-wlearn-bundle' })
    })
    return encodeBundle({ typeId: TYPES[this.#kind], requires: [...requires].sort(), params: {
      ...this.getParams(), nFeatures: this.#features, constants: this.#constants
    } }, artifacts)
  }
  static async load(bytes, options = {}) {
    const type = this === MultiLabelClassifier ? TYPES.multilabel : TYPES.multioutput
    if (validateBundle(bytes).manifest.typeId !== type) throw new ValidationError('Multi-target bundle type mismatch')
    return load(bytes, options)
  }
  static async _restore(manifest, toc, blobs, context) {
    const kind = manifest.typeId === TYPES.multilabel ? 'multilabel' : 'multioutput'
    const p = manifest.params
    if (!p || p.task !== kind || !Array.isArray(p.constants) || !p.constants.length ||
        !Number.isSafeInteger(p.nFeatures) || p.nFeatures < 1) throw new ValidationError('Invalid multi-target manifest')
    if (!Array.isArray(p.targetNames)) throw new ValidationError('Missing target names')
    const names = namesFor(p.targetNames, p.constants.length)
    if (p.constants.some(v => v !== null && !(kind === 'multilabel' && (v === 0 || v === 1)))) throw new ValidationError('Invalid constant target')
    const ids = p.constants.flatMap((v, i) => v === null ? [`target_${i}`] : [])
    if (toc.length !== ids.length || toc.some(e => !ids.includes(e.id) || e.mediaType !== 'application/x-wlearn-bundle')) throw new ValidationError('Invalid multi-target artifacts')
    if (p.estimatorName != null && (typeof p.estimatorName !== 'string' || !p.estimatorParams || typeof p.estimatorParams !== 'object' || Array.isArray(p.estimatorParams))) throw new ValidationError('Invalid estimator description')
    assertRequiredLoaders(manifest)
    const result = kind === 'multilabel' ? new MultiLabelClassifier() : new MultiOutputRegressor()
    result.#spec = p.estimatorName ? [p.estimatorName, null, p.estimatorParams] : null
    result.#names = names
    result.#constants = p.constants
    result.#features = p.nFeatures
    try {
      for (let t = 0; t < names.length; t++) {
        if (p.constants[t] !== null) { result.#models.push(null); result.#columns.push(null); continue }
        const entry = toc.find(e => e.id === `target_${t}`)
        const model = await load(blobs.subarray(entry.offset, entry.offset + entry.length), context)
        result.#models.push(model)
        validateEstimatorTask(model, kind === 'multilabel' ? 'classification' : 'regression')
        if (kind === 'multilabel') requireProbabilityModel(model, names[t])
        result.#columns.push(kind === 'multilabel' ? classColumnMap(model, [0, 1], names[t]) : null)
      }
      result.#fitted = true
      return result
    } catch (error) { disposeAll(result.#models, true); throw error }
  }
}

class MultiOutputRegressor extends MultiTarget { constructor(params = {}) { super(params, 'multioutput') } }
class MultiLabelClassifier extends MultiTarget { constructor(params = {}) { super(params, 'multilabel') } }
for (const type of Object.values(TYPES)) register(type, MultiTarget._restore, { acceptsContext: true, sync: false })
module.exports = { MultiOutputRegressor, MultiLabelClassifier }
