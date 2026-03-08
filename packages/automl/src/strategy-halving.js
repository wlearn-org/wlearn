const { makeLCG } = require('@wlearn/core')
const { sampleConfig } = require('./sampler.js')
const { makeCandidateId } = require('./common.js')

const { ceil, log, max, min, floor } = Math

/**
 * Successive halving strategy: multi-round elimination tournament.
 *
 * Evaluates many candidates on small subsamples, progressively
 * eliminates the worst and increases resource allocation.
 *
 * Round transitions happen inside report() when all candidates
 * in the current round have been evaluated. next() returns null
 * only when fully done.
 */
class HalvingStrategy {
  #candidates       // all candidates for current round
  #roundIndex = 0   // index within current round's candidates
  #round = 0        // current round number
  #nRounds
  #factor
  #nSamples         // total sample count (for computing fractions)
  #minResources
  #greaterIsBetter
  #roundResults = [] // results collected for current round
  #rounds = []       // completed round stats
  #done = false
  #finalRound = false

  /**
   * @param {Array<{ name: string, cls: object, searchSpace?: object, params?: object }>} models
   * @param {object} opts
   * @param {number} opts.nIter - candidates per model
   * @param {number} opts.seed
   * @param {number} opts.factor - elimination factor (keep top 1/factor)
   * @param {number} opts.nSamples - total sample count for fraction computation
   * @param {boolean} opts.greaterIsBetter - sort direction for elimination
   * @param {number} opts.cv - fold count (for minResources default)
   */
  constructor(models, { nIter = 20, seed = 42, factor = 3, nSamples, greaterIsBetter = true, cv = 5 } = {}) {
    this.#factor = factor
    this.#nSamples = nSamples || 0
    this.#greaterIsBetter = greaterIsBetter

    // Generate all candidate configs
    const rng = makeLCG(seed)
    const allCandidates = []
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
        allCandidates.push({
          candidateId,
          cls: model.cls,
          params,
        })
      }
    }

    this.#candidates = allCandidates
    this.#nRounds = max(1, ceil(log(allCandidates.length) / log(factor)))
    this.#minResources = max(cv * 2, floor(this.#nSamples / (factor ** this.#nRounds)))
  }

  /**
   * Return next candidate for current round (with subsample budget),
   * or null when fully done.
   */
  next() {
    if (this.#done) return null
    if (this.#roundIndex >= this.#candidates.length) return null

    const cand = this.#candidates[this.#roundIndex++]

    // For non-final rounds, attach subsample budget
    if (!this.#finalRound) {
      const nResources = min(this.#nSamples, floor(this.#minResources * (this.#factor ** this.#round)))
      const fraction = min(1, nResources / this.#nSamples)
      if (fraction < 1) {
        return { ...cand, budget: { type: 'subsample', value: fraction } }
      }
    }

    return cand
  }

  /**
   * Report a candidate result.
   * When all candidates in the current round have reported,
   * performs elimination and advances to the next round.
   */
  report(result) {
    this.#roundResults.push(result)

    // Check if all candidates in current round have been evaluated
    if (this.#roundResults.length < this.#candidates.length) return

    // Round complete: sort and eliminate
    if (this.#finalRound) {
      // Final round done
      this.#done = true
      return
    }

    const nResources = min(this.#nSamples, floor(this.#minResources * (this.#factor ** this.#round)))
    const fraction = min(1, nResources / this.#nSamples)

    // Sort results
    const sorted = [...this.#roundResults]
    if (this.#greaterIsBetter) {
      sorted.sort((a, b) => b.meanScore - a.meanScore)
    } else {
      sorted.sort((a, b) => a.meanScore - b.meanScore)
    }

    const nSurvivors = max(1, ceil(sorted.length / this.#factor))

    this.#rounds.push({
      round: this.#round,
      nResources,
      fraction,
      nCandidates: this.#roundResults.length,
      nSurvivors,
    })

    // Build survivor candidate list
    const survivorIds = new Set(sorted.slice(0, nSurvivors).map(r => r.candidateId))
    this.#candidates = this.#candidates.filter(c => survivorIds.has(c.candidateId))

    this.#round++
    this.#roundIndex = 0
    this.#roundResults = []

    // Check if we should enter final round (1 or fewer survivors, or max rounds)
    if (this.#candidates.length <= 1 || this.#round >= this.#nRounds) {
      this.#finalRound = true
    }
  }

  isDone() {
    return this.#done
  }

  get rounds() {
    return this.#rounds
  }
}

module.exports = { HalvingStrategy }
