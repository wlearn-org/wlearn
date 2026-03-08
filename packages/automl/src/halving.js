const { stratifiedKFold, kFold, normalizeX, normalizeY,
  ValidationError } = require('@wlearn/core')
const { Executor } = require('./executor.js')
const { HalvingStrategy } = require('./strategy-halving.js')
const { detectTask, scorerGreaterIsBetter } = require('./common.js')

/**
 * Successive halving search: multi-round elimination tournament.
 * Evaluates many candidates on small subsamples, progressively
 * eliminates the worst and increases resource allocation.
 */
class SuccessiveHalvingSearch {
  #models
  #opts
  #leaderboard = null
  #bestResult = null
  #rounds = null

  constructor(models, opts = {}) {
    if (!models || models.length === 0) {
      throw new ValidationError('SuccessiveHalvingSearch: at least one model is required')
    }
    this.#models = models
    this.#opts = {
      scoring: null,
      cv: 5,
      seed: 42,
      task: null,
      nIter: 20,
      maxTimeMs: 0,
      factor: 3,
      minResources: 0,
      onProgress: null,
      ...opts,
    }
  }

  async fit(X, y) {
    const Xn = normalizeX(X)
    const yn = normalizeY(y)
    const n = Xn.rows
    const task = this.#opts.task || detectTask(yn)
    const scoring = this.#opts.scoring || (task === 'classification' ? 'accuracy' : 'r2')
    const greaterIsBetter = scorerGreaterIsBetter(scoring)
    const { cv, seed, nIter, maxTimeMs, factor, onProgress } = this.#opts

    // Generate base folds on full data
    const folds = task === 'classification'
      ? stratifiedKFold(yn, cv, { shuffle: true, seed })
      : kFold(n, cv, { shuffle: true, seed })

    const executor = new Executor({
      folds,
      scoring,
      X: Xn,
      y: yn,
      timeLimitMs: maxTimeMs,
      seed,
      onProgress,
    })

    const strategy = new HalvingStrategy(this.#models, {
      nIter,
      seed,
      factor,
      nSamples: n,
      greaterIsBetter,
      cv,
    })

    const { leaderboard } = await executor.runStrategy(strategy)

    this.#leaderboard = leaderboard
    this.#bestResult = leaderboard.best()
    this.#rounds = strategy.rounds
    return { leaderboard, bestResult: this.#bestResult, rounds: this.#rounds }
  }

  async refitBest(X, y) {
    if (!this.#bestResult) {
      throw new ValidationError('SuccessiveHalvingSearch: must call fit() first')
    }
    const best = this.#bestResult
    const model = this.#models.find(m => m.name === best.modelName)
    const instance = await model.cls.create(best.params)
    const Xn = normalizeX(X)
    const yn = normalizeY(y)
    instance.fit(Xn, yn)
    return instance
  }

  get leaderboard() { return this.#leaderboard }
  get bestResult() { return this.#bestResult }
  get rounds() { return this.#rounds }
}

module.exports = { SuccessiveHalvingSearch }
