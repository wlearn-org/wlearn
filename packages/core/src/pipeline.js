import { Step } from './step.js'
import { DisposedError, NotFittedError, ValidationError } from './errors.js'
import { encodeBundle, decodeBundle } from './bundle.js'
import { register, load as registryLoad } from './registry.js'

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
export class Pipeline {
  #steps
  #fitted = false
  #disposed = false

  /**
   * @param {Array<[string, Object]>} steps - Array of `[name, estimator]` tuples.
   *   Each estimator must implement the wlearn estimator contract (`fit`, `predict`, `save`, `dispose`).
   * @throws {ValidationError} If steps is empty.
   */
  constructor(steps) {
    this.#steps = steps.map(([name, estimator]) => new Step(name, estimator))
    if (this.#steps.length === 0) {
      throw new ValidationError('Pipeline requires at least one step')
    }
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
  static async load(bytes) {
    const { manifest, toc, blobs } = decodeBundle(bytes)
    const steps = []
    for (const stepInfo of manifest.steps) {
      const tocEntry = toc.find(t => t.id === stepInfo.name)
      if (!tocEntry) {
        throw new ValidationError(`No artifact found for pipeline step "${stepInfo.name}"`)
      }
      const blob = blobs.subarray(tocEntry.offset, tocEntry.offset + tocEntry.length)
      const estimator = await registryLoad(blob)
      steps.push([stepInfo.name, estimator])
    }
    const pipeline = new Pipeline(steps)
    pipeline.#fitted = true
    return pipeline
  }

  /** Dispose all step estimators and mark the pipeline as disposed. */
  dispose() {
    if (this.#disposed) return
    this.#disposed = true
    for (const step of this.#steps) {
      step.estimator.dispose()
    }
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

  static registerLoader() {
    if (registered) return
    registered = true
    register(PIPELINE_TYPE_ID, (manifest, toc, blobs) => {
      // Reconstruct the full bundle bytes so Pipeline.load can decode them
      // This is called from registry.load, which already decoded once.
      // We need to pass the original bytes to Pipeline.load, but we only
      // have the decoded parts. Re-encode is wasteful. Instead, handle
      // the decoded parts directly.
      // Actually, Pipeline.load will be called externally with the original
      // bytes. The registry loader receives (manifest, toc, blobs) and
      // needs to reconstruct the pipeline from those parts.
      return Pipeline._loadFromParts(manifest, toc, blobs)
    })
  }

  static async _loadFromParts(manifest, toc, blobs) {
    const steps = []
    for (const stepInfo of manifest.steps) {
      const tocEntry = toc.find(t => t.id === stepInfo.name)
      if (!tocEntry) {
        throw new ValidationError(`No artifact found for pipeline step "${stepInfo.name}"`)
      }
      const blob = blobs.subarray(tocEntry.offset, tocEntry.offset + tocEntry.length)
      const estimator = await registryLoad(blob)
      steps.push([stepInfo.name, estimator])
    }
    const pipeline = new Pipeline(steps)
    pipeline.#fitted = true
    return pipeline
  }
}

// Auto-register pipeline loader
Pipeline.registerLoader()
