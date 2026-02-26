import { crossValScore, stratifiedKFold, kFold, normalizeX, normalizeY,
  makeLCG, ValidationError } from '@wlearn/core'
import { randomConfigs } from './sampler.js'
import { Leaderboard } from './leaderboard.js'

function _now() {
  if (typeof performance !== 'undefined') return performance.now()
  return Date.now()
}

function _detectTask(y) {
  if (y instanceof Int32Array) return 'classification'
  const unique = new Set()
  for (let i = 0; i < y.length; i++) {
    if (y[i] !== Math.round(y[i])) return 'regression'
    unique.add(y[i])
  }
  return unique.size <= 20 ? 'classification' : 'regression'
}

/**
 * Random hyperparameter search with cross-validation.
 */
export class RandomSearch {
  #models
  #opts
  #leaderboard = null
  #bestResult = null

  /**
   * @param {Array<{ name: string, cls: object, searchSpace?: object, params?: object }>} models
   * @param {object} opts
   */
  constructor(models, opts = {}) {
    if (!models || models.length === 0) {
      throw new ValidationError('RandomSearch: at least one model is required')
    }
    this.#models = models
    this.#opts = {
      scoring: null, // auto-detect
      cv: 5,
      seed: 42,
      task: null, // auto-detect
      nIter: 20,
      maxTimeMs: 0,
      ...opts,
    }
  }

  /**
   * Run the search.
   */
  async fit(X, y) {
    const Xn = normalizeX(X)
    const yn = normalizeY(y)
    const task = this.#opts.task || _detectTask(yn)
    const scoring = this.#opts.scoring || (task === 'classification' ? 'accuracy' : 'r2')
    const { cv, seed, nIter, maxTimeMs } = this.#opts

    // Generate folds once, shared across all candidates
    const folds = task === 'classification'
      ? stratifiedKFold(yn, cv, { shuffle: true, seed })
      : kFold(yn.length, cv, { shuffle: true, seed })

    const leaderboard = new Leaderboard()
    const rng = makeLCG(seed)
    const startTime = _now()

    for (const model of this.#models) {
      const space = model.searchSpace || model.cls.defaultSearchSpace?.() || {}
      // Remove fixed params from search space
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
        if (maxTimeMs > 0 && (_now() - startTime) > maxTimeMs) break

        const mergedParams = { ...config, ...(model.params || {}) }
        const t0 = _now()

        const scores = await crossValScore(model.cls, Xn, yn, {
          cv: folds,
          scoring,
          seed,
          params: mergedParams,
        })

        const fitTimeMs = _now() - t0
        leaderboard.add({
          modelName: model.name,
          params: mergedParams,
          scores,
          fitTimeMs,
        })
      }

      if (maxTimeMs > 0 && (_now() - startTime) > maxTimeMs) break
    }

    if (leaderboard.length === 0) {
      throw new ValidationError('RandomSearch: no candidates were evaluated')
    }

    this.#leaderboard = leaderboard
    this.#bestResult = leaderboard.best()
    return { leaderboard, bestResult: this.#bestResult }
  }

  /**
   * Refit the best candidate on full data.
   */
  async refitBest(X, y) {
    if (!this.#bestResult) {
      throw new ValidationError('RandomSearch: must call fit() first')
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
