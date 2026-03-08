const { normalizeX, normalizeY, makeLCG, getScorer } = require('@wlearn/core')
const { Leaderboard } = require('./leaderboard.js')
const { now, seedFor, partialShuffle } = require('./common.js')

const { ceil, min } = Math

/**
 * Subset rows of X by index array.
 */
function subsetX(X, indices) {
  const { data, cols } = X
  const rows = indices.length
  const out = new Float64Array(rows * cols)
  for (let i = 0; i < rows; i++) {
    const srcOff = indices[i] * cols
    out.set(data.subarray(srcOff, srcOff + cols), i * cols)
  }
  return { data: out, rows, cols }
}

/**
 * Subset labels by index array.
 */
function subsetY(y, indices) {
  const out = new (y.constructor)(indices.length)
  for (let i = 0; i < indices.length; i++) {
    out[i] = y[indices[i]]
  }
  return out
}

/**
 * Executor: evaluation engine and canonical leaderboard owner.
 *
 * Evaluates candidates across all CV folds, applies budgets,
 * records results in a single Leaderboard instance.
 */
class Executor {
  #folds
  #scorerFn
  #X
  #y
  #timeLimitMs
  #seed
  #startTime
  #leaderboard
  #onProgress

  /**
   * @param {object} opts
   * @param {Array<{train: Int32Array, test: Int32Array}>} opts.folds - CV folds
   * @param {string|Function} opts.scoring - scorer name or function
   * @param {object} opts.X - normalized feature matrix
   * @param {TypedArray} opts.y - normalized labels
   * @param {number} opts.timeLimitMs - global time limit (0 = no limit)
   * @param {number} opts.seed - base seed for reproducibility
   * @param {Function} opts.onProgress - optional progress callback
   */
  constructor({ folds, scoring, X, y, timeLimitMs = 0, seed = 42, onProgress }) {
    this.#folds = folds
    this.#scorerFn = getScorer(scoring)
    this.#X = X
    this.#y = y
    this.#timeLimitMs = timeLimitMs
    this.#seed = seed
    this.#startTime = now()
    this.#leaderboard = new Leaderboard()
    this.#onProgress = onProgress || null
  }

  get leaderboard() {
    return this.#leaderboard
  }

  get isTimedOut() {
    if (this.#timeLimitMs <= 0) return false
    return (now() - this.#startTime) > this.#timeLimitMs
  }

  /**
   * Evaluate one candidate across all CV folds.
   *
   * @param {object} candidateEval
   * @param {string} candidateEval.candidateId - stable identifier
   * @param {object} candidateEval.cls - estimator class with create/fit/predict/dispose
   * @param {object} candidateEval.params - hyperparameters
   * @param {object} [candidateEval.budget] - optional budget constraint
   * @returns {Promise<object>} CandidateResult
   */
  async evaluateCandidate({ candidateId, cls, params, budget }) {
    const folds = this.#folds
    const scores = new Float64Array(folds.length)
    const t0 = now()
    let totalTrainUsed = 0

    // Resolve effective params (apply rounds budget if applicable)
    const effectiveParams = this.#applyRoundsBudget(cls, params, budget)

    for (let f = 0; f < folds.length; f++) {
      let { train, test } = folds[f]

      // Apply subsample budget to train only
      if (budget && budget.type === 'subsample') {
        train = this.#subsampleTrain(train, budget.value, candidateId, f)
      }

      totalTrainUsed += train.length

      const Xtrain = subsetX(this.#X, train)
      const ytrain = subsetY(this.#y, train)
      const Xtest = subsetX(this.#X, test)
      const ytest = subsetY(this.#y, test)

      const model = await cls.create(effectiveParams)
      try {
        model.fit(Xtrain, ytrain)
        const preds = await model.predict(Xtest)
        scores[f] = this.#scorerFn(ytest, preds)
      } finally {
        model.dispose()
      }
    }

    const fitTimeMs = now() - t0

    // Record in leaderboard
    const entry = this.#leaderboard.add({
      modelName: candidateId.split(':')[0],
      params,
      scores,
      fitTimeMs,
    })

    return {
      candidateId,
      meanScore: entry.meanScore,
      foldScores: scores,
      stdScore: entry.stdScore,
      fitTimeMs,
      nTrainUsed: Math.round(totalTrainUsed / folds.length),
      nTest: folds[0].test.length,
    }
  }

  /**
   * Apply rounds budget by setting the model's rounds param if:
   * 1. Budget type is 'rounds'
   * 2. Model exposes budgetSpec().roundsParam
   * 3. Candidate params don't already set that param (candidate config wins)
   */
  #applyRoundsBudget(cls, params, budget) {
    if (!budget || budget.type !== 'rounds') return params
    const spec = cls.budgetSpec?.()
    if (!spec || !spec.roundsParam) return params
    if (params[spec.roundsParam] !== undefined) return params
    return { ...params, [spec.roundsParam]: budget.value }
  }

  /**
   * Subsample train indices using partial Fisher-Yates with deterministic seed.
   * Returns a new array of selected indices. Test indices are never subsampled.
   */
  #subsampleTrain(train, fraction, candidateId, foldIdx) {
    const k = Math.max(1, ceil(train.length * fraction))
    if (k >= train.length) return train
    // Copy to avoid mutating the original fold indices
    const copy = new Int32Array(train)
    const seed = seedFor(candidateId, foldIdx, this.#seed)
    const rng = makeLCG(seed)
    return partialShuffle(copy, k, rng)
  }

  /**
   * Run a strategy to completion.
   * Returns { leaderboard } only. Callers decide "best".
   */
  async runStrategy(strategy) {
    let done = 0
    while (!strategy.isDone()) {
      if (this.isTimedOut) break
      const task = strategy.next()
      if (task === null) break
      try {
        const result = await this.evaluateCandidate(task)
        strategy.report(result)
        done++
        if (this.#onProgress) {
          const best = this.#leaderboard.best()
          this.#onProgress({
            phase: 'search',
            candidatesDone: done,
            bestScore: best ? best.meanScore : null,
            bestModel: best ? best.modelName : null,
            lastCandidate: {
              model: result.candidateId.split(':')[0],
              score: result.meanScore,
              timeMs: result.fitTimeMs,
            },
            elapsedMs: now() - this.#startTime,
          })
        }
      } catch {
        done++
        // Skip failed candidates (invalid params, create errors, etc.)
      }
    }
    return { leaderboard: this.#leaderboard }
  }
}

module.exports = { Executor }
