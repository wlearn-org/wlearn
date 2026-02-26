import { stratifiedKFold, kFold, normalizeX, normalizeY,
  ValidationError } from '@wlearn/core'
import { Executor } from './executor.js'
import { ProgressiveStrategy } from './strategy-progressive.js'
import { detectTask, scorerGreaterIsBetter } from './common.js'

/**
 * Progressive search: probe all candidates cheaply (1 fold + subsample),
 * then promote top N to full K-fold evaluation.
 *
 * Faster than full random search when many candidates are weak.
 * The probe phase filters out bad configs quickly, saving time
 * for thorough evaluation of promising candidates.
 */
export class ProgressiveSearch {
  #models
  #opts
  #leaderboard = null
  #bestResult = null

  constructor(models, opts = {}) {
    if (!models || models.length === 0) {
      throw new ValidationError('ProgressiveSearch: at least one model is required')
    }
    this.#models = models
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
    const scoring = this.#opts.scoring || (task === 'classification' ? 'accuracy' : 'r2')
    const { cv, seed, nIter, maxTimeMs, promoteCount, probeFraction, onProgress } = this.#opts
    const greaterIsBetter = scorerGreaterIsBetter(scoring)

    // Probe folds: use only 1 fold for cheap screening
    const probeFolds = task === 'classification'
      ? stratifiedKFold(yn, 2, { shuffle: true, seed })
      : kFold(yn.length, 2, { shuffle: true, seed })
    // Use only the first fold for probing
    const singleFold = [probeFolds[0]]

    // Full folds for promoted candidates
    const fullFolds = task === 'classification'
      ? stratifiedKFold(yn, cv, { shuffle: true, seed: seed + 1 })
      : kFold(yn.length, cv, { shuffle: true, seed: seed + 1 })

    // Create strategy
    const strategy = new ProgressiveStrategy(this.#models, {
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
      } catch {
        // Report a failing result so the strategy can count it
        strategy.report({
          candidateId: cand.candidateId,
          meanScore: -Infinity,
          foldScores: new Float64Array(1),
          stdScore: 0,
          fitTimeMs: 0,
          nTrainUsed: 0,
          nTest: 0,
        })
      }
    }

    // If probe timed out and strategy hasn't transitioned, force it
    if (strategy.phase === 'probe') {
      // Transition didn't happen (not all probes completed) - use what we have
      // The strategy's report method handles this
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
      } catch {
        // Skip failed candidates
      }
    }

    const leaderboard = fullExecutor.leaderboard
    if (leaderboard.length === 0) {
      // Fall back to probe results if no full evals completed
      const probeLeaderboard = probeExecutor.leaderboard
      if (probeLeaderboard.length === 0) {
        throw new ValidationError('ProgressiveSearch: no candidates were evaluated')
      }
      this.#leaderboard = probeLeaderboard
    } else {
      this.#leaderboard = leaderboard
    }

    this.#bestResult = this.#leaderboard.best()
    return { leaderboard: this.#leaderboard, bestResult: this.#bestResult }
  }

  async refitBest(X, y) {
    if (!this.#bestResult) {
      throw new ValidationError('ProgressiveSearch: must call fit() first')
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
}
