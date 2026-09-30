const { validateEstimatorTask, subsetRows, subsetLabels } = require('@wlearn/core')
const {
  normalizeX, normalizeY, makeLCG, getScorer, scoreEstimator, Archive, ValidationError
} = require('@wlearn/core')
const { Leaderboard } = require('./leaderboard.js')
const { now, makeCandidateId, seedFor, partialShuffle } = require('./common.js')

const { ceil, min } = Math

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
  #archive
  #metric
  #failedSeq = 0
  #firstError = null
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
    if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
      throw new ValidationError('Executor seed must be an unsigned 32-bit integer')
    }
    this.#folds = folds
    this.#scorerFn = getScorer(scoring)
    this.#X = X
    this.#y = y
    this.#timeLimitMs = timeLimitMs
    this.#seed = seed
    this.#startTime = now()
    this.#leaderboard = new Leaderboard({ direction: this.#scorerFn.direction || 'maximize' })
    this.#metric = typeof scoring === 'string' ? scoring : 'score'
    this.#archive = new Archive({
      id: 'automl',
      measures: [this.#metric],
      primaryMeasure: this.#metric,
      direction: this.#scorerFn.direction || 'maximize',
      metadata: {
        source: '@wlearn/automl',
        seed,
        folds: folds.length
      }
    })
    this.#onProgress = onProgress || null
  }

  get leaderboard() {
    return this.#leaderboard
  }

  get archive() {
    return this.#archive
  }

  get firstError() {
    return this.#firstError
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
  async evaluateCandidate({ candidateId, candidate, cls, params, budget }) {
    if (makeCandidateId(candidate) !== candidateId) {
      throw new ValidationError('candidateId does not match the structured candidate')
    }
    const folds = this.#folds
    const scores = new Float64Array(folds.length)
    const foldSeeds = new Uint32Array(folds.length)
    const t0 = now()
    let totalTrainUsed = 0
    let supportsPredictProba = true

    // Resolve effective params (apply rounds budget if applicable)
    const effectiveParams = this.#applyRoundsBudget(
      cls, candidate.model.params, budget
    )

    for (let f = 0; f < folds.length; f++) {
      foldSeeds[f] = seedFor(candidate, f, this.#seed)
      let { train, test } = folds[f]

      // Apply subsample budget to train only
      if (budget && budget.type === 'subsample') {
        train = this.#subsampleTrain(train, budget.value, candidate, f)
      }

      totalTrainUsed += train.length

      const Xtrain = subsetRows(this.#X, train)
      const ytrain = subsetLabels(this.#y, train)
      const Xtest = subsetRows(this.#X, test)
      const ytest = subsetLabels(this.#y, test)

      const model = await cls.create(effectiveParams)
      let operationError = null
      try {
        await model.fit(Xtrain, ytrain)
        validateEstimatorTask(model, candidate.model.params.task)
        // Capabilities may resolve only after fitting (objective/class count).
        // Every fold must support probabilities before admitting this candidate
        // to a classification soft ensemble; search scores remain independent.
        supportsPredictProba &&= model.capabilities?.predictProba === true &&
          typeof model.predictProba === 'function'
        scores[f] = await scoreEstimator(model, Xtest, ytest, this.#scorerFn)
      } catch (error) {
        operationError = error
        throw error
      } finally {
        try {
          model.dispose()
        } catch (disposeError) {
          if (operationError === null) throw disposeError
        }
      }
    }

    const fitTimeMs = now() - t0

    // Record in leaderboard
    const entry = this.#leaderboard.add({
      candidateId,
      candidate,
      scores,
      baseSeed: this.#seed,
      foldSeeds,
      fitTimeMs,
      supportsPredictProba,
    })

    this.#archive.add({
      trialId: `automl-${entry.id}`,
      candidateId,
      seed: this.#seed,
      params: candidate.model.params,
      budget,
      status: 'ok',
      scores: { [this.#metric]: entry.meanScore },
      primaryScore: entry.meanScore,
      timings: { fitTimeMs },
      metadata: {
        sourceCandidateId: candidateId,
        candidate,
        leaderboardId: entry.id,
        modelName: entry.modelName,
        foldScores: Array.from(scores),
        foldSeeds: Array.from(foldSeeds, (value, foldId) => ({
          foldId,
          seed: value,
        })),
        stdScore: entry.stdScore,
        supportsPredictProba,
        nTrainUsed: Math.round(totalTrainUsed / folds.length),
        nTest: folds[0].test.length
      }
    })

    return {
      candidateId,
      candidate,
      params: candidate.model.params,
      meanScore: entry.meanScore,
      foldScores: scores,
      baseSeed: this.#seed,
      foldSeeds,
      stdScore: entry.stdScore,
      supportsPredictProba,
      fitTimeMs,
      nTrainUsed: Math.round(totalTrainUsed / folds.length),
      nTest: folds[0].test.length,
    }
  }

  recordFailure(task, error, phase = 'fit') {
    if (this.#firstError === null) this.#firstError = error
    const candidateId = task && task.candidateId ? task.candidateId : 'candidate'
    const seq = this.#failedSeq++
    return this.#archive.fail({
      trialId: `automl-failed-${seq}`,
      candidateId,
      seed: this.#seed,
      params: task && task.candidate ? task.candidate.model.params : {},
      budget: task ? task.budget : undefined,
      metadata: {
        sourceCandidateId: candidateId,
        candidate: task ? task.candidate : null,
        foldSeeds: task && task.candidate
          ? this.#foldSeeds(task.candidate)
          : [],
        modelName: task && task.candidate
          ? task.candidate.model.displayName
          : 'candidate'
      }
    }, error, phase)
  }

  #foldSeeds(candidate) {
    return this.#folds.map((_fold, foldId) => ({
      foldId,
      seed: seedFor(candidate, foldId, this.#seed),
    }))
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
  #subsampleTrain(train, fraction, candidate, foldIdx) {
    const k = Math.max(1, ceil(train.length * fraction))
    if (k >= train.length) return train
    // Copy to avoid mutating the original fold indices
    const copy = new Int32Array(train)
    const seed = seedFor(candidate, foldIdx, this.#seed)
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
              model: result.candidate.model.displayName,
              score: result.meanScore,
              timeMs: result.fitTimeMs,
            },
            elapsedMs: now() - this.#startTime,
          })
        }
      } catch (error) {
        done++
        this.recordFailure(task, error)
        // Strategies must count failed evaluations to complete their round.
        strategy.report({ candidateId: task.candidateId, candidate: task.candidate,
          status: 'failed', meanScore: null })
      }
    }
    return { leaderboard: this.#leaderboard, archive: this.#archive }
  }
}

module.exports = { Executor }
