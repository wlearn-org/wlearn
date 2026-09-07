const { effectiveSearchSpace } = require('./conditions.js')
const { makeLCG } = require('@wlearn/core')
const { sampleConfig } = require('./sampler.js')
const {
  createCandidateTask, normalizeModelSpecs, preprocessChoices
} = require('./candidate.js')

/**
 * Random search strategy: generates nIter random configs per model,
 * yields them one at a time. No adaptive behavior.
 */
class RandomStrategy {
  #queue = []
  #index = 0
  #total = 0

  /**
   * @param {Array<{ name: string, cls: object, searchSpace?: object, params?: object }>} models
   * @param {object} opts
   * @param {number} opts.nIter - candidates per model
   * @param {number} opts.seed
   */
  constructor(models, { nIter = 20, seed = 42 } = {}) {
    const rng = makeLCG(seed)
    const seen = new Map()

    for (const model of normalizeModelSpecs(models)) {
      const space = model.searchSpace || model.cls.defaultSearchSpace?.(model.params?.task) || {}
      const effectiveSpace = effectiveSearchSpace(space, model.params || {})

      const configRng = makeLCG((rng() * 0x7fffffff) | 0)
      for (let i = 0; i < nIter; i++) {
        const config = sampleConfig(effectiveSpace, configRng)
        const params = { ...config, ...(model.params || {}) }
        for (const preprocess of preprocessChoices(model)) {
          this.#queue.push(createCandidateTask(
            model, params, preprocess, seen
          ))
        }
      }
    }
    this.#total = this.#queue.length
  }

  /**
   * Return next candidate to evaluate, or null when exhausted.
   */
  next() {
    if (this.#index >= this.#total) return null
    return this.#queue[this.#index++]
  }

  /**
   * Report result. No-op for random search.
   */
  report(_result) {}

  /**
   * True when all candidates have been yielded.
   */
  isDone() {
    return this.#index >= this.#total
  }
}

module.exports = { RandomStrategy }
