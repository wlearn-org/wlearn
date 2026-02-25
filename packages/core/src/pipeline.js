import { Step } from './step.js'
import { DisposedError, NotFittedError, ValidationError } from './errors.js'
import { encodeBundle, decodeBundle } from './bundle.js'
import { register, load as registryLoad } from './registry.js'

const PIPELINE_TYPE_ID = 'wlearn.pipeline@1'
let registered = false

export class Pipeline {
  #steps
  #fitted = false
  #disposed = false

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

  predict(X) {
    this.#ensureFitted()
    const transformed = this.#transformThrough(X)
    return this.#steps[this.#steps.length - 1].estimator.predict(transformed)
  }

  predictProba(X) {
    this.#ensureFitted()
    const last = this.#steps[this.#steps.length - 1].estimator
    if (typeof last.predictProba !== 'function') {
      throw new ValidationError('Last step does not support predictProba')
    }
    const transformed = this.#transformThrough(X)
    return last.predictProba(transformed)
  }

  score(X, y) {
    this.#ensureFitted()
    const transformed = this.#transformThrough(X)
    return this.#steps[this.#steps.length - 1].estimator.score(transformed, y)
  }

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
