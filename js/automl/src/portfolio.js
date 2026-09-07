const { taskParams, resolveCv } = require('@wlearn/core')
/**
 * Zeroshot portfolio: pre-tuned hyperparameter configs per model family.
 *
 * Instead of random search, the portfolio provides a curated set of configs
 * known to work well across diverse datasets. Inspired by AutoGluon's
 * zeroshot portfolio approach (TabRepo).
 */

const { normalizeX, normalizeY,
  ValidationError } = require('@wlearn/core')
const { Executor } = require('./executor.js')
const { detectTask } = require('./common.js')
const {
  createCandidateTask, normalizeModelSpecs,
  preprocessChoices
} = require('./candidate.js')
const { fitCandidate } = require('./candidate-pipeline.js')
const { cloneLeaderboardEntry } = require('./leaderboard.js')

// ---------------------------------------------------------------------------
// Portfolio configs: task -> model_name -> list of param dicts
// ---------------------------------------------------------------------------

const PORTFOLIO = require('./portfolio.json')

/**
 * Return portfolio configs for the given task.
 * @param {string} task - 'classification' or 'regression'
 * @returns {Object} model name -> config list
 */
function getPortfolio(task = 'classification') {
  return PORTFOLIO[task] || PORTFOLIO.classification
}

// ---------------------------------------------------------------------------
// PortfolioStrategy
// ---------------------------------------------------------------------------

/**
 * Yields pre-tuned configs from the zeroshot portfolio.
 * Same interface as RandomStrategy / HalvingStrategy.
 */
class PortfolioStrategy {
  #queue = []
  #index = 0
  #total = 0

  /**
   * @param {Array<{ name: string, cls: object, params?: object }>} models
   * @param {object} opts
   */
  constructor(models, { task = 'classification', seed = 42 } = {}) {
    const portfolio = getPortfolio(task)
    const seen = new Map()

    for (const model of normalizeModelSpecs(models)) {
      const portfolioKey = model.portfolioKey
        ?? model.cls.portfolioKey
        ?? model.classId
      const fixed = model.params || {}

      // New families own their warm starts; the built-in table retains existing
      // portfolioKey behavior. Explicit caller data takes precedence.
      const configs = model.portfolio ?? model.cls.defaultPortfolio?.(task) ?? portfolio[portfolioKey] ?? [{}]
      if (!Array.isArray(configs) || configs.length === 0 || configs.some(config => !config || typeof config !== 'object' || Array.isArray(config))) {
        throw new ValidationError(`Portfolio for ${model.name} must be a nonempty array of parameter objects`)
      }

      for (const config of configs) {
        const params = { ...config, ...fixed }
        for (const preprocess of preprocessChoices(model)) {
          this.#queue.push(createCandidateTask(
            model, params, preprocess, seen
          ))
        }
      }
    }

    this.#total = this.#queue.length
  }

  next() {
    if (this.#index >= this.#total) return null
    return this.#queue[this.#index++]
  }

  report(_result) {}

  isDone() {
    return this.#index >= this.#total
  }
}

// ---------------------------------------------------------------------------
// PortfolioSearch
// ---------------------------------------------------------------------------

/**
 * Evaluate pre-tuned portfolio configs with cross-validation.
 */
class PortfolioSearch {
  #models
  #opts
  #leaderboard = null
  #bestResult = null
  #archive = null

  constructor(models, opts = {}) {
    if (!models || models.length === 0) {
      throw new ValidationError('PortfolioSearch: at least one model is required')
    }
    this.#models = normalizeModelSpecs(models, 'PortfolioSearch models')
    this.#opts = {
      scoring: null,
      cv: 5,
      seed: 42,
      task: null,
      maxTimeMs: 0,
      onProgress: null,
      ...opts,
    }
  }

  async fit(X, y) {
    const Xn = normalizeX(X)
    const yn = normalizeY(y)
    const task = this.#opts.task || detectTask(yn)
    const models = this.#models.map(spec => ({ ...spec, params: taskParams(spec.params, task) }))
    const scoring = this.#opts.scoring || (task === 'classification' ? 'accuracy' : 'r2')
    const { cv, seed, maxTimeMs, onProgress } = this.#opts

    const folds = resolveCv(cv, yn, { task, seed })

    const executor = new Executor({
      folds,
      scoring,
      X: Xn,
      y: yn,
      timeLimitMs: maxTimeMs,
      seed,
      onProgress,
    })

    const strategy = new PortfolioStrategy(models, { task, seed })

    const { leaderboard, archive } = await executor.runStrategy(strategy)

    if (leaderboard.length === 0) {
      if (executor.firstError !== null) throw executor.firstError
      throw new ValidationError('PortfolioSearch: no candidates were evaluated')
    }

    this.#leaderboard = leaderboard
    this.#archive = archive
    this.#bestResult = leaderboard.best()
    return { leaderboard, archive, bestResult: leaderboard.best() }
  }

  async refitBest(X, y) {
    if (!this.#bestResult) {
      throw new ValidationError('PortfolioSearch: must call fit() first')
    }
    const best = this.#bestResult
    const model = this.#models.find(m => m.classId === best.candidate.model.classId)
    return fitCandidate(model, best.candidate, X, y, best.candidateId)
  }

  get leaderboard() { return this.#leaderboard }
  get bestResult() { return cloneLeaderboardEntry(this.#bestResult) }
  get archive() { return this.#archive }
}

module.exports = { PORTFOLIO, getPortfolio, PortfolioStrategy, PortfolioSearch }
