const { taskParams, resolveCv } = require('@wlearn/core')
const {
  normalizeX, normalizeY, ValidationError
} = require('@wlearn/core')
const { Executor } = require('./executor.js')
const { detectTask } = require('./common.js')
const { BayesianStrategy } = require('./strategy-bayesian.js')
const { normalizeModelSpecs } = require('./candidate.js')
const { fitCandidate } = require('./candidate-pipeline.js')
const { cloneLeaderboardEntry } = require('./leaderboard.js')

class BayesianSearch {
  #models
  #opts
  #leaderboard = null
  #bestResult = null
  #archive = null

  constructor(models, opts = {}) {
    if (!models || models.length === 0) {
      throw new ValidationError('BayesianSearch: at least one model is required')
    }
    this.#models = normalizeModelSpecs(models, 'BayesianSearch models')
    this.#opts = {
      scoring: null,
      cv: 5,
      seed: 42,
      task: null,
      nIter: 30,
      maxTimeMs: 0,
      onProgress: null,
      acquisitionFn: 'ei',
      kappa: 2.0,
      xi: 0.01,
      kernel: 'matern52',
      nInitial: null,
      ...opts,
    }
  }

  async fit(X, y) {
    const Xn = normalizeX(X)
    const yn = normalizeY(y)
    const task = this.#opts.task || detectTask(yn)
    const models = this.#models.map(spec => ({ ...spec, params: taskParams(spec.params, task) }))
    const scoring = this.#opts.scoring || (task === 'classification' ? 'accuracy' : 'r2')
    const {
      cv, seed, nIter, maxTimeMs, onProgress,
      acquisitionFn, kappa, xi, kernel, nInitial,
    } = this.#opts

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

    const strategy = new BayesianStrategy(models, {
      nIter,
      seed,
      acquisitionFn,
      kappa,
      xi,
      kernel,
      nInitial,
      task,
    })

    let result
    let operationError = null
    try {
      await strategy.init()
      result = await executor.runStrategy(strategy)
    } catch (error) {
      operationError = error
      throw error
    } finally {
      try {
        strategy.dispose()
      } catch (disposeError) {
        if (operationError === null) throw disposeError
      }
    }

    const { leaderboard, archive } = result
    if (leaderboard.length === 0) {
      if (executor.firstError !== null) throw executor.firstError
      throw new ValidationError('BayesianSearch: no candidates were evaluated')
    }

    this.#leaderboard = leaderboard
    this.#archive = archive
    this.#bestResult = leaderboard.best()
    return { leaderboard, archive, bestResult: leaderboard.best() }
  }

  async refitBest(X, y) {
    if (!this.#bestResult) {
      throw new ValidationError('BayesianSearch: must call fit() first')
    }
    const best = this.#bestResult
    const model = this.#models.find(m => m.classId === best.candidate.model.classId)
    return fitCandidate(model, best.candidate, X, y, best.candidateId)
  }

  get leaderboard() { return this.#leaderboard }
  get bestResult() { return cloneLeaderboardEntry(this.#bestResult) }
  get archive() { return this.#archive }
}

module.exports = { BayesianSearch }
