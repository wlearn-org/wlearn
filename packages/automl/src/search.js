const { stratifiedKFold, kFold, normalizeX, normalizeY,
  ValidationError } = require('@wlearn/core')
const { Executor } = require('./executor.js')
const { RandomStrategy } = require('./strategy-random.js')
const { detectTask, scorerGreaterIsBetter } = require('./common.js')
const { normalizeModelSpecs } = require('./candidate.js')
const { fitCandidate } = require('./candidate-pipeline.js')
const { cloneLeaderboardEntry } = require('./leaderboard.js')

/**
 * Random hyperparameter search with cross-validation.
 */
class RandomSearch {
  #models
  #opts
  #leaderboard = null
  #bestResult = null
  #archive = null

  /**
   * @param {Array<{ name: string, cls: object, searchSpace?: object, params?: object }>} models
   * @param {object} opts
   */
  constructor(models, opts = {}) {
    if (!models || models.length === 0) {
      throw new ValidationError('RandomSearch: at least one model is required')
    }
    this.#models = normalizeModelSpecs(models, 'RandomSearch models')
    this.#opts = {
      scoring: null, // auto-detect
      cv: 5,
      seed: 42,
      task: null, // auto-detect
      nIter: 20,
      maxTimeMs: 0,
      onProgress: null,
      ...opts,
    }
  }

  /**
   * Run the search.
   */
  async fit(X, y) {
    const Xn = normalizeX(X)
    const yn = normalizeY(y)
    const task = this.#opts.task || detectTask(yn)
    const scoring = this.#opts.scoring || (task === 'classification' ? 'accuracy' : 'r2')
    const { cv, seed, nIter, maxTimeMs, onProgress } = this.#opts

    // Generate folds once, shared across all candidates
    const folds = task === 'classification'
      ? stratifiedKFold(yn, cv, { shuffle: true, seed })
      : kFold(yn.length, cv, { shuffle: true, seed })

    const executor = new Executor({
      folds,
      scoring,
      X: Xn,
      y: yn,
      timeLimitMs: maxTimeMs,
      seed,
      onProgress,
    })

    const strategy = new RandomStrategy(this.#models, { nIter, seed })

    const { leaderboard, archive } = await executor.runStrategy(strategy)

    if (leaderboard.length === 0) {
      if (executor.firstError !== null) throw executor.firstError
      throw new ValidationError('RandomSearch: no candidates were evaluated')
    }

    this.#leaderboard = leaderboard
    this.#archive = archive
    this.#bestResult = leaderboard.best()
    return { leaderboard, archive, bestResult: leaderboard.best() }
  }

  /**
   * Refit the best candidate on full data.
   */
  async refitBest(X, y) {
    if (!this.#bestResult) {
      throw new ValidationError('RandomSearch: must call fit() first')
    }
    const best = this.#bestResult
    const model = this.#models.find(m => m.classId === best.candidate.model.classId)
    return fitCandidate(model, best.candidate, X, y, best.candidateId)
  }

  get leaderboard() { return this.#leaderboard }
  get bestResult() { return cloneLeaderboardEntry(this.#bestResult) }
  get archive() { return this.#archive }
}

module.exports = { RandomSearch }
