const { effectiveSearchSpace } = require('./conditions.js')
const { makeLCG, ValidationError } = require('@wlearn/core')
const { BayesianOptimizer, countFreeParams } = require('@wlearn/bo')
const { sampleConfig } = require('./sampler.js')
const {
  candidateHash, createCandidate, createCandidateTask, normalizeModelSpecs,
  preprocessChoices
} = require('./candidate.js')

const { ceil, min, max } = Math

class BayesianStrategy {
  static optimizerClass = BayesianOptimizer

  #models
  #nIter
  #seed
  #opts
  #optimizers = new Map()
  #warmupQueues = new Map()
  #warmupCounts = new Map()
  #reported = new Map()
  #yielded = 0
  #total = 0
  #modelCycle = []
  #cycleIdx = 0
  #initialized = false
  #disposed = false
  #seen = new Map()

  constructor(models, opts = {}) {
    this.#models = normalizeModelSpecs(models).flatMap(model => (
      preprocessChoices(model).map(preprocess => ({
        ...model,
        activePreprocess: preprocess,
        variantId: variantId(model.classId, preprocess)
      }))
    ))
    this.#nIter = opts.nIter ?? 30
    this.#seed = opts.seed ?? 42
    this.#opts = opts
    this.#total = this.#models.length * this.#nIter
    this.#modelCycle = this.#models.map(m => m.variantId)
  }

  async init() {
    if (this.#disposed) {
      throw new ValidationError('BayesianStrategy has been disposed')
    }
    if (this.#initialized) return
    this.#initialized = true

    const rng = makeLCG(this.#seed)

    try {
      for (const model of this.#models) {
        const effectiveSpace = this.#getEffectiveSpace(model)
        const nFree = countFreeParams(effectiveSpace)
        const autoWarmup = nFree === 0
          ? this.#nIter
          : min(ceil(this.#nIter * 0.4), max(3, nFree + 1))
        const nInitial = nFree === 0
          ? this.#nIter
          : min(this.#nIter, this.#opts.nInitial ?? autoWarmup)

        this.#warmupCounts.set(model.variantId, nInitial)
        this.#reported.set(model.variantId, 0)

        const configRng = makeLCG((rng() * 0x7fffffff) | 0)
        const queue = []
        for (let i = 0; i < nInitial; i++) {
          const config = sampleConfig(effectiveSpace, configRng)
          const params = { ...config, ...(model.params || {}) }
          queue.push(createCandidateTask(
            model, params, model.activePreprocess, this.#seen
          ))
        }
        this.#warmupQueues.set(model.variantId, queue)

        if (nFree === 0) continue

        const optSeed = (rng() * 0x7fffffff) | 0
        const optimizer = await BayesianStrategy.optimizerClass.create(effectiveSpace, {
          kernel: this.#opts.kernel || 'matern52',
          acquisitionFn: this.#opts.acquisitionFn || 'ei',
          kappa: this.#opts.kappa ?? 2.0,
          xi: this.#opts.xi ?? 0.01,
          seed: optSeed,
        })
        this.#optimizers.set(model.variantId, optimizer)
      }
    } catch (error) {
      try { this.dispose() } catch {}
      throw error
    }
  }

  next() {
    this.#assertReady()
    if (this.#yielded >= this.#total) return null

    for (let attempts = 0; attempts < this.#modelCycle.length; attempts++) {
      const currentVariantId = this.#modelCycle[this.#cycleIdx]
      this.#cycleIdx = (this.#cycleIdx + 1) % this.#modelCycle.length

      const model = this.#models.find(m => m.variantId === currentVariantId)
      const reported = this.#reported.get(currentVariantId)
      const warmupCount = this.#warmupCounts.get(currentVariantId)
      const queue = this.#warmupQueues.get(currentVariantId)

      if (queue.length > 0) {
        this.#yielded++
        return queue.shift()
      }

      if (reported < warmupCount) continue

      const optimizer = this.#optimizers.get(currentVariantId)
      if (!optimizer) continue
      const suggested = optimizer.suggest()
      const params = { ...suggested, ...(model.params || {}) }
      this.#yielded++
      return createCandidateTask(
        model, params, model.activePreprocess, this.#seen
      )
    }

    return null
  }

  report(result) {
    this.#assertReady()
    const { candidate, meanScore } = result
    if (!candidate) return
    const currentVariantId = variantId(
      candidate.model.classId, candidate.preprocess
    )
    const optimizer = this.#optimizers.get(currentVariantId)
    const model = this.#models.find(m => m.variantId === currentVariantId)
    if (!optimizer || !model) return

    const params = candidate.model.params
    const observeParams = { ...params }
    if (model.params) {
      for (const key of Object.keys(model.params)) delete observeParams[key]
    }

    if (typeof meanScore === 'number' && isFinite(meanScore)) {
      try {
        optimizer.observe(observeParams, meanScore)
      } catch (_) {}
    }

    this.#reported.set(
      currentVariantId, (this.#reported.get(currentVariantId) || 0) + 1
    )
  }

  isDone() {
    return this.#yielded >= this.#total
  }

  dispose() {
    if (this.#disposed) return
    this.#disposed = true
    let firstError = null
    const optimizers = Array.from(this.#optimizers.values()).reverse()
    for (const opt of optimizers) {
      try {
        opt.dispose()
      } catch (error) {
        if (firstError === null) firstError = error
      }
    }
    this.#optimizers.clear()
    if (firstError !== null) throw firstError
  }

  #getEffectiveSpace(model) {
    const task = this.#opts.task || model.task || null
    const space = model.searchSpace || model.cls.defaultSearchSpace?.(task) || {}
    return effectiveSearchSpace(space, model.params || {})
  }

  #assertReady() {
    if (this.#disposed) {
      throw new ValidationError('BayesianStrategy has been disposed')
    }
    if (!this.#initialized) {
      throw new ValidationError('BayesianStrategy.init() must be awaited before use')
    }
  }
}

function variantId(classId, preprocess) {
  const candidate = createCandidate(
    { displayName: 'Bayesian variant', classId }, {}, preprocess
  )
  return `wlcv1_${candidateHash(candidate)}`
}

module.exports = { BayesianStrategy }
