import { crossValScore, stratifiedKFold, kFold, normalizeX, normalizeY,
  makeLCG, ValidationError } from '@wlearn/core'
import { randomConfigs } from './sampler.js'
import { Leaderboard } from './leaderboard.js'

const { ceil, log, max, min, floor, round } = Math

function _now() {
  if (typeof performance !== 'undefined') return performance.now()
  return Date.now()
}

function _detectTask(y) {
  if (y instanceof Int32Array) return 'classification'
  const unique = new Set()
  for (let i = 0; i < y.length; i++) {
    if (y[i] !== round(y[i])) return 'regression'
    unique.add(y[i])
  }
  return unique.size <= 20 ? 'classification' : 'regression'
}

function _subsetFolds(baseFolds, fraction) {
  return baseFolds.map(({ train, test }) => {
    const nTrain = max(1, round(train.length * fraction))
    const nTest = max(1, round(test.length * fraction))
    return {
      train: train.slice(0, nTrain),
      test: test.slice(0, nTest),
    }
  })
}

/**
 * Successive halving search: multi-round elimination tournament.
 * Evaluates many candidates on small subsamples, progressively
 * eliminates the worst and increases resource allocation.
 */
export class SuccessiveHalvingSearch {
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
      ...opts,
    }
  }

  async fit(X, y) {
    const Xn = normalizeX(X)
    const yn = normalizeY(y)
    const n = Xn.rows
    const task = this.#opts.task || _detectTask(yn)
    const scoring = this.#opts.scoring || (task === 'classification' ? 'accuracy' : 'r2')
    const { cv, seed, nIter, maxTimeMs, factor } = this.#opts

    // Generate base folds on full data
    const baseFolds = task === 'classification'
      ? stratifiedKFold(yn, cv, { shuffle: true, seed })
      : kFold(n, cv, { shuffle: true, seed })

    // Generate all candidate configs
    const rng = makeLCG(seed)
    let candidates = []
    for (const model of this.#models) {
      const space = model.searchSpace || model.cls.defaultSearchSpace?.() || {}
      const effectiveSpace = { ...space }
      if (model.params) {
        for (const key of Object.keys(model.params)) {
          delete effectiveSpace[key]
        }
      }
      const configs = randomConfigs(effectiveSpace, nIter, {
        seed: (rng() * 0x7fffffff) | 0,
      })
      for (const config of configs) {
        candidates.push({
          modelName: model.name,
          cls: model.cls,
          params: { ...config, ...(model.params || {}) },
        })
      }
    }

    if (candidates.length === 0) {
      throw new ValidationError('SuccessiveHalvingSearch: no candidates generated')
    }

    const nRounds = max(1, ceil(log(candidates.length) / log(factor)))
    const minRes = this.#opts.minResources || max(cv * 2, floor(n / (factor ** nRounds)))

    const leaderboard = new Leaderboard()
    const rounds = []
    const startTime = _now()

    for (let r = 0; r < nRounds && candidates.length > 1; r++) {
      const nResources = min(n, floor(minRes * (factor ** r)))
      const fraction = min(1, nResources / n)
      const folds = fraction >= 1 ? baseFolds : _subsetFolds(baseFolds, fraction)

      const roundResults = []

      for (const cand of candidates) {
        if (maxTimeMs > 0 && (_now() - startTime) > maxTimeMs) break

        const t0 = _now()
        const scores = await crossValScore(cand.cls, Xn, yn, {
          cv: folds,
          scoring,
          seed,
          params: cand.params,
        })
        const fitTimeMs = _now() - t0

        const entry = leaderboard.add({
          modelName: cand.modelName,
          params: cand.params,
          scores,
          fitTimeMs,
        })

        roundResults.push({ ...cand, meanScore: entry.meanScore })
      }

      // Sort descending and keep top 1/factor
      roundResults.sort((a, b) => b.meanScore - a.meanScore)
      const nSurvivors = max(1, ceil(roundResults.length / factor))
      candidates = roundResults.slice(0, nSurvivors)

      rounds.push({
        round: r,
        nResources,
        fraction,
        nCandidates: roundResults.length,
        nSurvivors: candidates.length,
      })

      if (maxTimeMs > 0 && (_now() - startTime) > maxTimeMs) break
    }

    // Final round with full data if we haven't already
    if (candidates.length >= 1) {
      for (const cand of candidates) {
        if (maxTimeMs > 0 && (_now() - startTime) > maxTimeMs) break
        const t0 = _now()
        const scores = await crossValScore(cand.cls, Xn, yn, {
          cv: baseFolds,
          scoring,
          seed,
          params: cand.params,
        })
        const fitTimeMs = _now() - t0
        leaderboard.add({
          modelName: cand.modelName,
          params: cand.params,
          scores,
          fitTimeMs,
        })
      }
    }

    this.#leaderboard = leaderboard
    this.#bestResult = leaderboard.best()
    this.#rounds = rounds
    return { leaderboard, bestResult: this.#bestResult, rounds }
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
