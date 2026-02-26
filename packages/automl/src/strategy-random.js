import { makeLCG } from '@wlearn/core'
import { sampleConfig } from './sampler.js'
import { makeCandidateId } from './common.js'

/**
 * Random search strategy: generates nIter random configs per model,
 * yields them one at a time. No adaptive behavior.
 */
export class RandomStrategy {
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

    for (const model of models) {
      const space = model.searchSpace || model.cls.defaultSearchSpace?.() || {}
      // Remove fixed params from search space
      const effectiveSpace = { ...space }
      if (model.params) {
        for (const key of Object.keys(model.params)) {
          delete effectiveSpace[key]
        }
      }

      const configRng = makeLCG((rng() * 0x7fffffff) | 0)
      for (let i = 0; i < nIter; i++) {
        const config = sampleConfig(effectiveSpace, configRng)
        const params = { ...config, ...(model.params || {}) }
        const candidateId = makeCandidateId(model.name, params)
        this.#queue.push({
          candidateId,
          cls: model.cls,
          params,
        })
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
