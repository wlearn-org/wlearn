const { Step } = require('./step.js')
const { DisposedError, NotFittedError, ValidationError } = require('./errors.js')
const { encodeBundle, validateBundle } = require('./bundle.js')
const {
  register, load: registryLoad, assertRequiredLoaders
} = require('./registry.js')
const { isPromiseLike, lift } = require('./lift.js')
const { targetRows, validateSampleWeight } = require('./targets.js')

const PIPELINE_TYPE_ID = 'wlearn.pipeline@1'
let registered = false

/**
 * A pipeline of named estimator steps executed sequentially.
 *
 * Intermediate steps must implement `transform()` (or `fitTransform()`).
 * The last step is the final estimator (predict/score).
 *
 * @example
 * const pipe = new Pipeline([['scaler', scaler], ['clf', model]])
 * pipe.fit(X, y)
 * pipe.predict(X)
 * const bytes = pipe.save()   // WLRN bundle
 * const restored = await load(bytes)
 */
class Pipeline {
  #steps
  #provenance
  #fitted = false
  #disposed = false
  #fitInProgress = false

  /**
   * @param {Array<[string, Object]>} steps - Array of `[name, estimator]` tuples.
   *   Each estimator must implement the wlearn estimator contract (`fit`, `predict`, `save`, `dispose`).
   * @throws {ValidationError} If steps is empty.
   */
  constructor(steps, { provenance = null } = {}) {
    this.#steps = steps.map(([name, estimator]) => new Step(name, estimator))
    if (this.#steps.length === 0) {
      throw new ValidationError('Pipeline requires at least one step')
    }
    this.#provenance = provenance === null
      ? null
      : _freezeJSON(_cloneJSON(provenance))
  }

  #ensureAlive() {
    if (this.#disposed) throw new DisposedError('Pipeline has been disposed.')
  }

  #ensureFitted() {
    this.#ensureAlive()
    if (!this.#fitted) throw new NotFittedError('Pipeline is not fitted. Call fit() first.')
  }

  #transformThrough(X) {
    let current = X
    for (let i = 0; i < this.#steps.length - 1; i++) {
      const est = this.#steps[i].estimator
      current = lift(current, value => est.transform(value))
    }
    return current
  }

  #fitIntermediate(estimator, X, y, opts) {
    const args = opts && estimator.capabilities?.sampleWeight ? [X, y, opts] : [X, y]
    if (typeof estimator.fitTransform === 'function') {
      return estimator.fitTransform(...args)
    }
    const fitted = estimator.fit(...args)
    return isPromiseLike(fitted)
      ? Promise.resolve(fitted).then(() => estimator.transform(X))
      : estimator.transform(X)
  }

  #commitFit() {
    this.#fitInProgress = false
    this.#ensureAlive()
    this.#fitted = true
    return this
  }

  #failFit(error) {
    this.#fitInProgress = false
    throw error
  }

  /**
   * Fit all steps. Intermediate steps are fit-transformed; the last step is fit only.
   * @param {Object} X - Feature matrix (`{ data, rows, cols }` or `number[][]`).
   * @param {Float64Array|Int32Array|number[]} y - Target labels/values.
   * Returns synchronously for synchronous children and lifts to a Promise when a
   * composite child has asynchronous fit semantics.
   * @returns {this|Promise<this>}
   */
  fit(X, y, opts = {}) {
    this.#ensureAlive()
    if (this.#fitInProgress) {
      throw new ValidationError('Pipeline fit is already in progress')
    }
    if (!opts || typeof opts !== 'object' || Array.isArray(opts) ||
        Object.keys(opts).some(key => key !== 'sampleWeight')) {
      throw new ValidationError('Pipeline fit options support only sampleWeight')
    }
    let weighted = null
    if (opts.sampleWeight != null) {
      const weights = validateSampleWeight(opts.sampleWeight, targetRows(y))
      const last = this.#steps[this.#steps.length - 1]
      if (!last.estimator.capabilities?.sampleWeight) {
        throw new ValidationError(`Pipeline step "${last.name}" does not support sampleWeight`)
      }
      weighted = { sampleWeight: weights }
    }
    this.#fitInProgress = true
    this.#fitted = false
    try {
      let current = X
      for (let i = 0; i < this.#steps.length - 1; i++) {
        const estimator = this.#steps[i].estimator
        current = isPromiseLike(current)
          ? Promise.resolve(current).then(
            value => this.#fitIntermediate(estimator, value, y, weighted)
          )
          : this.#fitIntermediate(estimator, current, y, weighted)
      }

      const last = this.#steps[this.#steps.length - 1].estimator
      const finish = value => {
        const fitted = weighted ? last.fit(value, y, weighted) : last.fit(value, y)
        return isPromiseLike(fitted)
          ? Promise.resolve(fitted).then(() => this.#commitFit())
          : this.#commitFit()
      }
      const result = isPromiseLike(current)
        ? Promise.resolve(current).then(finish)
        : finish(current)
      return isPromiseLike(result)
        ? Promise.resolve(result).catch(error => this.#failFit(error))
        : result
    } catch (error) {
      return this.#failFit(error)
    }
  }

  /**
   * Transform through intermediate steps, then predict with the last step.
   * @param {Object} X - Feature matrix.
   * @returns {Float64Array|Int32Array|Promise<Float64Array|Int32Array>}
   */
  predict(X, opts) { return this.#predictMethod('predict', X, opts) }

  /**
   * Transform through intermediate steps, then call `predictProba` on the last step.
   * @param {Object} X - Feature matrix.
   * @returns {Float64Array|Promise<Float64Array>} Class probability estimates.
   * @throws {ValidationError} If the last step does not support `predictProba`.
   */
  predictProba(X, opts) { return this.#predictMethod('predictProba', X, opts) }
  predictQuantiles(X, levels, opts) { return this.#predictMethod('predictQuantiles', X, levels, opts) }
  predictInterval(X, coverage, opts) { return this.#predictMethod('predictInterval', X, coverage, opts) }
  predictSet(X, coverage, opts) { return this.#predictMethod('predictSet', X, coverage, opts) }
  predictRegion(X, coverage, opts) { return this.#predictMethod('predictRegion', X, coverage, opts) }
  predictDistribution(X, opts) { return this.#predictMethod('predictDistribution', X, opts) }

  #predictMethod(method, X, ...args) {
    this.#ensureFitted()
    const last = this.#steps[this.#steps.length - 1].estimator
    if (typeof last[method] !== 'function') throw new ValidationError(`Last step does not support ${method}`)
    return lift(this.#transformThrough(X), value => last[method](value, ...args))
  }

  /**
   * Transform through intermediate steps, then score with the last step.
   * @param {Object} X - Feature matrix.
   * @param {Float64Array|Int32Array|number[]} y - True labels/values.
   * @returns {number|Promise<number>} Score (accuracy for classifiers, R2 for regressors).
   */
  score(X, y) {
    this.#ensureFitted()
    const transformed = this.#transformThrough(X)
    return lift(transformed, value => this.#steps[this.#steps.length - 1].estimator.score(value, y))
  }

  /**
   * Serialize the fitted pipeline as a WLRN bundle.
   * Each step's model is saved as a nested artifact.
   * @returns {Uint8Array} Bundle bytes (loadable via `load()` from `@wlearn/core`).
   */
  save() {
    this.#ensureFitted()
    // Validate every child before invoking any serializer.
    for (const step of this.#steps) {
      for (const method of ['getParams', 'save']) {
        if (typeof step.estimator[method] !== 'function') {
          throw new ValidationError(`Pipeline step "${step.name}" does not support ${method}()`)
        }
      }
    }
    const manifest = {
      typeId: PIPELINE_TYPE_ID,
      steps: this.#steps.map(s => ({
        name: s.name,
        params: s.estimator.getParams()
      }))
    }
    if (this.#provenance !== null) {
      manifest.metadata = { provenance: _cloneJSON(this.#provenance) }
    }
    const artifacts = this.#steps.map(s => ({
      id: s.name,
      data: s.estimator.save(),
      mediaType: 'application/x-wlearn-bundle'
    }))
    return encodeBundle(manifest, artifacts)
  }

  /**
   * Load a pipeline from WLRN bundle bytes.
   * Dispatches each step's blob through the global registry to reconstruct estimators.
   * @param {Uint8Array} bytes - Bundle bytes produced by `pipeline.save()`.
   * @returns {Promise<Pipeline>} A fitted pipeline ready for predict/score.
   */
  static async load(bytes, options = {}) {
    const { manifest } = validateBundle(bytes)
    if (manifest.typeId !== PIPELINE_TYPE_ID) {
      throw new ValidationError(
        `Pipeline.load expected typeId "${PIPELINE_TYPE_ID}", got "${manifest.typeId}"`
      )
    }
    return registryLoad(bytes, options)
  }

  /** Dispose all step estimators and mark the pipeline as disposed. */
  dispose() {
    if (this.#disposed) return
    if (this.#fitInProgress) {
      throw new ValidationError('Cannot dispose Pipeline while fit is in progress')
    }
    this.#disposed = true
    this.#fitInProgress = false
    let firstError = null
    for (let i = this.#steps.length - 1; i >= 0; i--) {
      try {
        this.#steps[i].estimator.dispose()
      } catch (error) {
        if (firstError === null) firstError = error
      }
    }
    if (firstError !== null) throw firstError
  }

  getParams() {
    const params = {}
    for (const step of this.#steps) {
      params[step.name] = step.estimator.getParams()
    }
    return params
  }

  setParams(p) {
    this.#ensureAlive()
    if (this.#fitInProgress) {
      throw new ValidationError('Cannot set Pipeline params while fit is in progress')
    }
    if (!p || typeof p !== 'object' || Array.isArray(p)) {
      throw new ValidationError('Pipeline params must be an object keyed by step name')
    }
    const stepNames = new Set(this.#steps.map(step => step.name))
    for (const name of Object.keys(p)) {
      if (!stepNames.has(name)) {
        throw new ValidationError(`Unknown Pipeline step parameter "${name}"`)
      }
    }
    const selected = this.#steps.filter(step =>
      Object.prototype.hasOwnProperty.call(p, step.name)
    )
    for (const step of selected) {
      if (typeof step.estimator.setParams !== 'function') {
        throw new ValidationError(
          `Pipeline step "${step.name}" does not support setParams`
        )
      }
    }
    if (selected.length > 0) this.#fitted = false
    for (const step of selected) {
      step.estimator.setParams(p[step.name])
    }
    return this
  }

  get capabilities() {
    const capabilities = this.#steps[this.#steps.length - 1].estimator.capabilities
    return capabilities == null ? capabilities : { ...capabilities }
  }

  get classes() {
    this.#ensureFitted()
    const estimator = this.#steps[this.#steps.length - 1].estimator
    const classes = estimator.classes
    const resolved = typeof classes === 'function' ? classes.call(estimator) : classes
    return resolved ?? null
  }

  get isFitted() { return this.#fitted && !this.#disposed }
  get provenance() { return _cloneJSON(this.#provenance) }

  static registerLoader() {
    if (registered) return
    registered = true
    register(PIPELINE_TYPE_ID, (manifest, toc, blobs, context) => {
      return Pipeline._loadFromParts(manifest, toc, blobs, context)
    }, { acceptsContext: true, sync: false })
  }

  static async _loadFromParts(manifest, toc, blobs, context) {
    if (manifest.typeId !== PIPELINE_TYPE_ID) {
      throw new ValidationError(
        `Pipeline.load expected typeId "${PIPELINE_TYPE_ID}", got "${manifest.typeId}"`
      )
    }
    if (!Array.isArray(manifest.steps) || manifest.steps.length === 0) {
      throw new ValidationError('Pipeline manifest must contain at least one step')
    }
    assertRequiredLoaders(manifest)
    const steps = []
    try {
      for (const stepInfo of manifest.steps) {
        const tocEntry = toc.find(t => t.id === stepInfo.name)
        if (!tocEntry) {
          throw new ValidationError(`No artifact found for pipeline step "${stepInfo.name}"`)
        }
        const blob = blobs.subarray(tocEntry.offset, tocEntry.offset + tocEntry.length)
        const estimator = await registryLoad(blob, context)
        steps.push([stepInfo.name, estimator])
      }
      const provenance = manifest.metadata?.provenance ?? null
      const pipeline = new Pipeline(steps, { provenance })
      pipeline.#fitted = true
      return pipeline
    } catch (error) {
      _disposeLoaded(steps.map(([, estimator]) => estimator))
      throw error
    }
  }
}

function _cloneJSON(value) {
  if (value === null || typeof value !== 'object') return value
  if (Array.isArray(value)) return value.map(_cloneJSON)
  const result = {}
  for (const [key, child] of Object.entries(value)) {
    Object.defineProperty(result, key, {
      value: _cloneJSON(child),
      enumerable: true,
      configurable: true,
      writable: true,
    })
  }
  return result
}

function _freezeJSON(value) {
  if (!value || typeof value !== 'object' || Object.isFrozen(value)) return value
  for (const child of Object.values(value)) _freezeJSON(child)
  return Object.freeze(value)
}

function _disposeLoaded(estimators) {
  for (let i = estimators.length - 1; i >= 0; i--) {
    try {
      if (typeof estimators[i]?.dispose === 'function') estimators[i].dispose()
    } catch {
      // Preserve the load error; cleanup is best effort for partially loaded state.
    }
  }
}

// Auto-register pipeline loader
Pipeline.registerLoader()

module.exports = { Pipeline }
