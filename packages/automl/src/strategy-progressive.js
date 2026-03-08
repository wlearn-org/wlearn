const { makeLCG } = require('@wlearn/core')
const { sampleConfig } = require('./sampler.js')
const { makeCandidateId } = require('./common.js')

const { max, ceil } = Math

/**
 * Progressive evaluation strategy: probe all candidates cheaply (1 fold),
 * then promote top N to full evaluation.
 *
 * Phase 1 (probe): yield all candidates for cheap 1-fold evaluation
 * Phase 2 (promote): yield top N candidates for full K-fold evaluation
 *
 * This pairs with ProgressiveSearch which manages two Executors
 * (probe executor with 1 fold, full executor with K folds).
 */
class ProgressiveStrategy {
  #allCandidates = []
  #promotedCandidates = []
  #phase = 'probe'
  #probeIndex = 0
  #promoteIndex = 0
  #probeResults = []
  #promoteCount
  #greaterIsBetter
  #done = false
  #probeFraction

  /**
   * @param {Array<{ name: string, cls: object, searchSpace?: object, params?: object }>} models
   * @param {object} opts
   * @param {number} opts.nIter - candidates per model
   * @param {number} opts.seed
   * @param {number} opts.promoteCount - how many to promote from probe to full eval
   * @param {boolean} opts.greaterIsBetter - sort direction
   * @param {number} opts.probeFraction - subsample fraction for probe phase (0-1)
   */
  constructor(models, { nIter = 20, seed = 42, promoteCount = 10,
    greaterIsBetter = true, probeFraction = 0.5 } = {}) {
    this.#promoteCount = promoteCount
    this.#greaterIsBetter = greaterIsBetter
    this.#probeFraction = probeFraction

    const rng = makeLCG(seed)
    for (const model of models) {
      const space = model.searchSpace || model.cls.defaultSearchSpace?.() || {}
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
        this.#allCandidates.push({ candidateId, cls: model.cls, params })
      }
    }
  }

  get phase() { return this.#phase }

  next() {
    if (this.#done) return null

    if (this.#phase === 'probe') {
      if (this.#probeIndex >= this.#allCandidates.length) return null
      const cand = this.#allCandidates[this.#probeIndex++]
      // Attach subsample budget for cheaper probe
      if (this.#probeFraction < 1) {
        return { ...cand, budget: { type: 'subsample', value: this.#probeFraction } }
      }
      return cand
    }

    // Promote phase
    if (this.#promoteIndex >= this.#promotedCandidates.length) return null
    return this.#promotedCandidates[this.#promoteIndex++]
  }

  report(result) {
    if (this.#phase === 'probe') {
      this.#probeResults.push(result)
      if (this.#probeResults.length >= this.#allCandidates.length) {
        this.#transitionToPromote()
      }
      return
    }
    // Promote phase: nothing to do per-result
  }

  #transitionToPromote() {
    // Sort probe results
    const sorted = [...this.#probeResults]
    if (this.#greaterIsBetter) {
      sorted.sort((a, b) => b.meanScore - a.meanScore)
    } else {
      sorted.sort((a, b) => a.meanScore - b.meanScore)
    }

    // Select top N
    const topN = sorted.slice(0, max(1, this.#promoteCount))
    const topIds = new Set(topN.map(r => r.candidateId))

    // Build promoted list from original candidates (to preserve cls reference)
    this.#promotedCandidates = this.#allCandidates.filter(
      c => topIds.has(c.candidateId)
    )

    this.#phase = 'promote'
    this.#promoteIndex = 0
  }

  isDone() {
    if (this.#done) return true
    if (this.#phase === 'promote' &&
        this.#promoteIndex >= this.#promotedCandidates.length &&
        this.#promotedCandidates.length > 0) {
      this.#done = true
      return true
    }
    return false
  }
}

module.exports = { ProgressiveStrategy }
