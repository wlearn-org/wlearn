const { Step } = require('./step.js')
const { DisposedError, NotFittedError, ValidationError } = require('./errors.js')
const { encodeBundle, validateBundle } = require('./bundle.js')
const {
  register, load: registryLoad, assertRequiredLoaders
} = require('./registry.js')

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
      current = est.transform(current)
    }
    return current
  }

  /**
   * Fit all steps. Intermediate steps are fit-transformed; the last step is fit only.
   * @param {Object} X - Feature matrix (`{ data, rows, cols }` or `number[][]`).
   * @param {Float64Array|Int32Array|number[]} y - Target labels/values.
   * @returns {this}
   */
  fit(X, y) {
    this.#ensureAlive()
    let current = X
    for (let i = 0; i < this.#steps.length - 1; i++) {
      const est = this.#steps[i].estimator
      if (typeof est.fitTransform === 'function') {
        current = est.fitTransform(current, y)
      } else {
        est.fit(current, y)
        current = est.transform(current)
      }
    }
    // Last step: fit only
    const last = this.#steps[this.#steps.length - 1].estimator
    last.fit(current, y)
    this.#fitted = true
    return this
  }

  /**
   * Transform through intermediate steps, then predict with the last step.
   * @param {Object} X - Feature matrix.
   * @returns {Float64Array|Int32Array|Promise<Float64Array|Int32Array>}
   */
  predict(X) {
    this.#ensureFitted()
    const transformed = this.#transformThrough(X)
    return this.#steps[this.#steps.length - 1].estimator.predict(transformed)
  }

  /**
   * Transform through intermediate steps, then call `predictProba` on the last step.
   * @param {Object} X - Feature matrix.
   * @returns {Float64Array|Promise<Float64Array>} Class probability estimates.
   * @throws {ValidationError} If the last step does not support `predictProba`.
   */
  predictProba(X) {
    this.#ensureFitted()
    const last = this.#steps[this.#steps.length - 1].estimator
    if (typeof last.predictProba !== 'function') {
      throw new ValidationError('Last step does not support predictProba')
    }
    const transformed = this.#transformThrough(X)
    return last.predictProba(transformed)
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
    return this.#steps[this.#steps.length - 1].estimator.score(transformed, y)
  }

  /**
   * Serialize the fitted pipeline as a WLRN bundle.
   * Each step's model is saved as a nested artifact.
   * @returns {Uint8Array} Bundle bytes (loadable via `load()` from `@wlearn/core`).
   */
  save() {
    this.#ensureFitted()
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
    this.#disposed = true
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
    for (const step of this.#steps) {
      if (p[step.name]) {
        step.estimator.setParams(p[step.name])
      }
    }
    return this
  }

  get capabilities() {
    return this.#steps[this.#steps.length - 1].estimator.capabilities
  }

  get isFitted() { return this.#fitted }
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
