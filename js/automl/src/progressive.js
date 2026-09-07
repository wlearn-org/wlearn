const { taskParams, resolveCv } = require('@wlearn/core')
const { normalizeX, normalizeY,
  ValidationError } = require('@wlearn/core')
const { Executor } = require('./executor.js')
const { ProgressiveStrategy } = require('./strategy-progressive.js')
const { detectTask, scorerGreaterIsBetter } = require('./common.js')
const { normalizeModelSpecs } = require('./candidate.js')
const { fitCandidate } = require('./candidate-pipeline.js')
const { cloneLeaderboardEntry } = require('./leaderboard.js')

/**
 * Progressive search: probe all candidates cheaply (1 fold + subsample),
 * then promote top N to full K-fold evaluation.
 *
 * Faster than full random search when many candidates are weak.
 * The probe phase filters out bad configs quickly, saving time
 * for thorough evaluation of promising candidates.
 */
class ProgressiveSearch {
  #models
  #opts
  #leaderboard = null
  #bestResult = null
  #archive = null

  constructor(models, opts = {}) {
    if (!models || models.length === 0) {
      throw new ValidationError('ProgressiveSearch: at least one model is required')
    }
    this.#models = normalizeModelSpecs(models, 'ProgressiveSearch models')
    this.#opts = {
      scoring: null,
      cv: 5,
      seed: 42,
      task: null,
      nIter: 20,
      maxTimeMs: 0,
      promoteCount: 10,
      probeFraction: 0.5,
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
    const { cv, seed, nIter, maxTimeMs, promoteCount, probeFraction, onProgress } = this.#opts
    const greaterIsBetter = scorerGreaterIsBetter(scoring)

    // Probe the caller's first fold; regenerating a split could leak groups.
    const fullFolds = resolveCv(cv, yn, { task, seed })
    const singleFold = [fullFolds[0]]

    // Create strategy
    const strategy = new ProgressiveStrategy(models, {
      nIter, seed, promoteCount, greaterIsBetter, probeFraction,
    })

    // Phase 1: probe with 1-fold executor
    const probeExecutor = new Executor({
      folds: singleFold,
      scoring,
      X: Xn,
      y: yn,
      timeLimitMs: maxTimeMs > 0 ? Math.floor(maxTimeMs * 0.3) : 0,
      seed,
      onProgress,
    })

    while (strategy.phase === 'probe' && !strategy.isDone()) {
      if (probeExecutor.isTimedOut) break
      const cand = strategy.next()
      if (cand === null) break
      try {
        const result = await probeExecutor.evaluateCandidate(cand)
        strategy.report(result)
      } catch (error) {
        probeExecutor.recordFailure(cand, error)
        strategy.report({ candidateId: cand.candidateId, candidate: cand.candidate,
          status: 'failed', meanScore: null })
      }
    }

    // Phase 2: full evaluation of promoted candidates
    const fullExecutor = new Executor({
      folds: fullFolds,
      scoring,
      X: Xn,
      y: yn,
      timeLimitMs: maxTimeMs > 0 ? Math.floor(maxTimeMs * 0.7) : 0,
      seed,
      onProgress,
    })

    while (!strategy.isDone()) {
      if (fullExecutor.isTimedOut) break
      const cand = strategy.next()
      if (cand === null) break
      try {
        await fullExecutor.evaluateCandidate(cand)
      } catch (error) {
        fullExecutor.recordFailure(cand, error)
        // Skip failed candidates
      }
    }

    const leaderboard = fullExecutor.leaderboard
    if (leaderboard.length === 0) {
      // Fall back to probe results if no full evals completed
      const probeLeaderboard = probeExecutor.leaderboard
      if (probeLeaderboard.length === 0) {
        const firstError = probeExecutor.firstError ?? fullExecutor.firstError
        if (firstError !== null) throw firstError
        throw new ValidationError('ProgressiveSearch: no candidates were evaluated')
      }
      this.#leaderboard = probeLeaderboard
      this.#archive = probeExecutor.archive
    } else {
      this.#leaderboard = leaderboard
      this.#archive = fullExecutor.archive
    }

    this.#bestResult = this.#leaderboard.best()
    return { leaderboard: this.#leaderboard, archive: this.#archive, bestResult: this.#leaderboard.best() }
  }

  async refitBest(X, y) {
    if (!this.#bestResult) {
      throw new ValidationError('ProgressiveSearch: must call fit() first')
    }
    const best = this.#bestResult
    const model = this.#models.find(m => m.classId === best.candidate.model.classId)
    return fitCandidate(model, best.candidate, X, y, best.candidateId)
  }

  get leaderboard() { return this.#leaderboard }
  get bestResult() { return cloneLeaderboardEntry(this.#bestResult) }
  get archive() { return this.#archive }
}

module.exports = { ProgressiveSearch }
