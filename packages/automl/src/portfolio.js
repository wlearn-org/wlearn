/**
 * Zeroshot portfolio: pre-tuned hyperparameter configs per model family.
 *
 * Instead of random search, the portfolio provides a curated set of configs
 * known to work well across diverse datasets. Inspired by AutoGluon's
 * zeroshot portfolio approach (TabRepo).
 */

const { stratifiedKFold, kFold, normalizeX, normalizeY,
  ValidationError } = require('@wlearn/core')
const { Executor } = require('./executor.js')
const { detectTask, makeCandidateId } = require('./common.js')

// ---------------------------------------------------------------------------
// Portfolio configs: task -> model_name -> list of param dicts
// ---------------------------------------------------------------------------

const PORTFOLIO = {
  classification: {
    xgb: [
      { objective: 'multi:softprob', eta: 0.05, max_depth: 6, numRound: 200,
        subsample: 0.8, colsample_bytree: 0.8, min_child_weight: 1.0,
        lambda: 1.0, alpha: 0.0 },
      { objective: 'multi:softprob', eta: 0.01, max_depth: 10, numRound: 500,
        subsample: 0.7, colsample_bytree: 0.65, min_child_weight: 0.6,
        lambda: 0.1, alpha: 0.0 },
      { objective: 'multi:softprob', eta: 0.1, max_depth: 3, numRound: 100,
        subsample: 0.9, colsample_bytree: 0.9, min_child_weight: 1.0,
        lambda: 1.0, alpha: 0.0 },
      { objective: 'multi:softprob', eta: 0.03, max_depth: 7, numRound: 300,
        subsample: 0.8, colsample_bytree: 0.7, min_child_weight: 1.0,
        lambda: 5.0, alpha: 1.0 },
      { objective: 'multi:softprob', eta: 0.02, max_depth: 8, numRound: 400,
        subsample: 0.8, colsample_bytree: 0.8, min_child_weight: 0.8,
        lambda: 0.1, alpha: 0.0 },
      { objective: 'multi:softprob', eta: 0.08, max_depth: 4, numRound: 150,
        subsample: 0.85, colsample_bytree: 0.55, min_child_weight: 1.0,
        lambda: 1.0, alpha: 0.1 },
      { objective: 'multi:softprob', eta: 0.015, max_depth: 9, numRound: 350,
        subsample: 0.75, colsample_bytree: 0.55, min_child_weight: 0.9,
        lambda: 0.5, alpha: 0.0 },
      { objective: 'multi:softprob', eta: 0.3, max_depth: 3, numRound: 50,
        subsample: 0.9, colsample_bytree: 0.9, min_child_weight: 1.0,
        lambda: 1.0, alpha: 0.0 },
      // RF-mode configs (low correlation with boosting for ensemble diversity)
      { objective: 'multi:softprob', num_parallel_tree: 100, numRound: 1,
        subsample: 0.8, colsample_bynode: 0.8, learning_rate: 1.0 },
      { objective: 'multi:softprob', num_parallel_tree: 200, numRound: 1,
        subsample: 0.7, colsample_bynode: 0.6, learning_rate: 1.0 },
    ],
    lgb: [
      { objective: 'multiclass', learning_rate: 0.05, max_depth: 6,
        numRound: 200, num_leaves: 63, subsample: 0.8,
        colsample_bytree: 0.8, min_child_weight: 1.0,
        reg_lambda: 1.0, reg_alpha: 0.0, verbosity: -1 },
      { objective: 'multiclass', learning_rate: 0.01, max_depth: -1,
        numRound: 500, num_leaves: 127, subsample: 0.7,
        colsample_bytree: 0.65, reg_lambda: 0.1, reg_alpha: 0.0, verbosity: -1 },
      { objective: 'multiclass', learning_rate: 0.1, max_depth: 4,
        numRound: 100, num_leaves: 15, subsample: 0.9,
        colsample_bytree: 0.9, reg_lambda: 1.0, reg_alpha: 0.0, verbosity: -1 },
      { objective: 'multiclass', learning_rate: 0.05,
        numRound: 200, num_leaves: 63, subsample: 0.8,
        colsample_bytree: 0.8, extra_trees: true,
        reg_lambda: 1.0, reg_alpha: 0.0, verbosity: -1 },
      { objective: 'multiclass', learning_rate: 0.03, max_depth: 7,
        numRound: 300, num_leaves: 63, subsample: 0.8,
        colsample_bytree: 0.7, reg_lambda: 5.0, reg_alpha: 1.0, verbosity: -1 },
      { objective: 'multiclass', learning_rate: 0.01, max_depth: 8,
        numRound: 500, num_leaves: 95, subsample: 0.75,
        colsample_bytree: 0.55, reg_lambda: 0.5, reg_alpha: 0.0, verbosity: -1 },
    ],
    ebm: [
      { objective: 'classification', learningRate: 0.01, maxRounds: 500,
        maxLeaves: 3, maxBins: 256 },
      { objective: 'classification', learningRate: 0.01, maxRounds: 500,
        maxLeaves: 4, maxInteractions: 15, maxBins: 256 },
      { objective: 'classification', learningRate: 0.05, maxRounds: 300,
        maxLeaves: 3, maxBins: 128 },
      { objective: 'classification', learningRate: 0.005, maxRounds: 800,
        maxLeaves: 5, maxBins: 512 },
    ],
    linear: [
      { solver: 0, C: 1.0 },
      { solver: 0, C: 10.0 },
      { solver: 7, C: 1.0 },
      { solver: 6, C: 0.1 },
    ],
    svm: [
      { svmType: 0, kernel: 2, C: 1.0, gamma: 0, probability: 1 },
      { svmType: 0, kernel: 2, C: 10.0, gamma: 0.01, probability: 1 },
      { svmType: 0, kernel: 1, C: 1.0, degree: 3, gamma: 0, probability: 1 },
      { svmType: 0, kernel: 0, C: 1.0, probability: 1 },
    ],
    knn: [
      { k: 5, metric: 'l2', task: 'classification' },
      { k: 15, metric: 'l2', task: 'classification' },
      { k: 3, metric: 'l1', task: 'classification' },
    ],
    tsetlin: [
      { task: 'classification', nClauses: 100, threshold: 50, s: 3.0, nEpochs: 100 },
      { task: 'classification', nClauses: 500, threshold: 100, s: 5.0, nEpochs: 100 },
      { task: 'classification', nClauses: 50, threshold: 25, s: 2.0, nEpochs: 60 },
    ],
    rf: [
      { nEstimators: 100, maxDepth: 10, maxFeatures: 'sqrt', criterion: 'gini',
        minSamplesSplit: 2, minSamplesLeaf: 1 },
      { nEstimators: 300, maxDepth: 0, maxFeatures: 'log2', criterion: 'gini',
        minSamplesSplit: 5, minSamplesLeaf: 2 },
      { nEstimators: 200, maxDepth: 15, maxFeatures: 'sqrt', criterion: 'entropy',
        extraTrees: 1, minSamplesSplit: 2 },
      { nEstimators: 200, maxDepth: 12, maxFeatures: 'sqrt', criterion: 'hellinger',
        minSamplesSplit: 2, minSamplesLeaf: 1 },
      { nEstimators: 150, maxDepth: 10, maxFeatures: 'sqrt', criterion: 'gini',
        heterogeneous: 1, oobWeighting: 1, minSamplesLeaf: 2 },
      { nEstimators: 200, maxDepth: 10, maxFeatures: 'sqrt', criterion: 'gini',
        histogramBinning: 1, minSamplesSplit: 2, minSamplesLeaf: 1 },
      { nEstimators: 150, maxDepth: 12, maxFeatures: 'sqrt', criterion: 'gini',
        jarf: 1, minSamplesSplit: 2, minSamplesLeaf: 1 },
    ],
    mlp: [
      { hidden_sizes: [64], activation: 'relu', lr: 0.01, epochs: 50,
        optimizer: 'adam', batch_size: 32 },
      { hidden_sizes: [128, 64], activation: 'gelu', lr: 0.001, epochs: 100,
        optimizer: 'adam', batch_size: 32 },
      { hidden_sizes: [256, 128], activation: 'silu', lr: 0.001, epochs: 150,
        optimizer: 'adam', batch_size: 16 },
    ],
    tabm: [
      { hidden_sizes: [128], activation: 'relu', n_ensemble: 32, lr: 0.005,
        epochs: 100, optimizer: 'adam' },
      { hidden_sizes: [64], activation: 'gelu', n_ensemble: 8, lr: 0.01,
        epochs: 50, optimizer: 'adam' },
      { hidden_sizes: [128, 64], activation: 'silu', n_ensemble: 16, lr: 0.001,
        epochs: 150, optimizer: 'adam' },
    ],
    nam: [
      { hidden_sizes: [64], activation: 'exu', lr: 0.01, epochs: 100,
        optimizer: 'adam' },
      { hidden_sizes: [128], activation: 'relu', lr: 0.001, epochs: 100,
        optimizer: 'adam' },
      { hidden_sizes: [64, 32], activation: 'exu', lr: 0.005, epochs: 150,
        optimizer: 'adam' },
    ],
    gam: [
      { family: 'binomial', penalty: 'elasticnet', alpha: 0.5, nLambda: 100, nFolds: 5 },
      { family: 'binomial', penalty: 'lasso', alpha: 1.0, nLambda: 100, nFolds: 5 },
      { family: 'binomial', penalty: 'ridge', alpha: 0.0, nLambda: 100, nFolds: 5 },
      { family: 'binomial', penalty: 'scad', nLambda: 100, nFolds: 5 },
    ],
    bart: [
      { numTrees: 200, numBurnin: 200, numSamples: 100, alpha: 0.95, beta: 2.0 },
      { numTrees: 500, numBurnin: 300, numSamples: 100, alpha: 0.95, beta: 2.0 },
      { numTrees: 100, maxDepth: 5, numBurnin: 100, numSamples: 100,
        alpha: 0.8, beta: 1.0 },
    ],
    fm: [
      { k: 4, lr: 0.01, epoch: 10, opt: 'adagrad' },
      { k: 8, lr: 0.005, epoch: 20, opt: 'adagrad', lambda: 0.001 },
    ],
    xlr: [
      { lr: 0.01, opt: 'ftrl', epoch: 10 },
      { lr: 0.01, lambda: 0.1, opt: 'adagrad', epoch: 20 },
    ],
  },
  regression: {
    xgb: [
      { objective: 'reg:squarederror', eta: 0.05, max_depth: 6, numRound: 200,
        subsample: 0.8, colsample_bytree: 0.8, min_child_weight: 1.0,
        lambda: 1.0, alpha: 0.0 },
      { objective: 'reg:squarederror', eta: 0.01, max_depth: 10, numRound: 500,
        subsample: 0.7, colsample_bytree: 0.65, min_child_weight: 0.6,
        lambda: 0.1, alpha: 0.0 },
      { objective: 'reg:squarederror', eta: 0.1, max_depth: 3, numRound: 100,
        subsample: 0.9, colsample_bytree: 0.9, min_child_weight: 1.0,
        lambda: 1.0, alpha: 0.0 },
      { objective: 'reg:squarederror', eta: 0.03, max_depth: 7, numRound: 300,
        subsample: 0.8, colsample_bytree: 0.7, min_child_weight: 1.0,
        lambda: 5.0, alpha: 1.0 },
      { objective: 'reg:squarederror', eta: 0.02, max_depth: 8, numRound: 400,
        subsample: 0.8, colsample_bytree: 0.8, min_child_weight: 0.8,
        lambda: 0.1, alpha: 0.0 },
      { objective: 'reg:squarederror', eta: 0.08, max_depth: 4, numRound: 150,
        subsample: 0.85, colsample_bytree: 0.55, min_child_weight: 1.0,
        lambda: 1.0, alpha: 0.1 },
      { objective: 'reg:squarederror', eta: 0.015, max_depth: 9, numRound: 350,
        subsample: 0.75, colsample_bytree: 0.55, min_child_weight: 0.9,
        lambda: 0.5, alpha: 0.0 },
      { objective: 'reg:squarederror', eta: 0.3, max_depth: 3, numRound: 50,
        subsample: 0.9, colsample_bytree: 0.9, min_child_weight: 1.0,
        lambda: 1.0, alpha: 0.0 },
      // RF-mode configs (low correlation with boosting for ensemble diversity)
      { objective: 'reg:squarederror', num_parallel_tree: 100, numRound: 1,
        subsample: 0.8, colsample_bynode: 0.8, learning_rate: 1.0 },
      { objective: 'reg:squarederror', num_parallel_tree: 200, numRound: 1,
        subsample: 0.7, colsample_bynode: 0.6, learning_rate: 1.0 },
    ],
    lgb: [
      { objective: 'regression', learning_rate: 0.05, max_depth: 6,
        numRound: 200, num_leaves: 63, subsample: 0.8,
        colsample_bytree: 0.8, min_child_weight: 1.0,
        reg_lambda: 1.0, reg_alpha: 0.0, verbosity: -1 },
      { objective: 'regression', learning_rate: 0.01, max_depth: -1,
        numRound: 500, num_leaves: 127, subsample: 0.7,
        colsample_bytree: 0.65, reg_lambda: 0.1, reg_alpha: 0.0, verbosity: -1 },
      { objective: 'regression', learning_rate: 0.1, max_depth: 4,
        numRound: 100, num_leaves: 15, subsample: 0.9,
        colsample_bytree: 0.9, reg_lambda: 1.0, reg_alpha: 0.0, verbosity: -1 },
      { objective: 'regression', learning_rate: 0.05,
        numRound: 200, num_leaves: 63, subsample: 0.8,
        colsample_bytree: 0.8, extra_trees: true,
        reg_lambda: 1.0, reg_alpha: 0.0, verbosity: -1 },
      { objective: 'regression', learning_rate: 0.03, max_depth: 7,
        numRound: 300, num_leaves: 63, subsample: 0.8,
        colsample_bytree: 0.7, reg_lambda: 5.0, reg_alpha: 1.0, verbosity: -1 },
      { objective: 'regression', learning_rate: 0.01, max_depth: 8,
        numRound: 500, num_leaves: 95, subsample: 0.75,
        colsample_bytree: 0.55, reg_lambda: 0.5, reg_alpha: 0.0, verbosity: -1 },
    ],
    ebm: [
      { objective: 'regression', learningRate: 0.01, maxRounds: 500,
        maxLeaves: 3, maxBins: 256 },
      { objective: 'regression', learningRate: 0.01, maxRounds: 500,
        maxLeaves: 4, maxInteractions: 15, maxBins: 256 },
      { objective: 'regression', learningRate: 0.05, maxRounds: 300,
        maxLeaves: 3, maxBins: 128 },
      { objective: 'regression', learningRate: 0.005, maxRounds: 800,
        maxLeaves: 5, maxBins: 512 },
    ],
    linear: [
      { solver: 11, C: 1.0 },
      { solver: 11, C: 10.0 },
      { solver: 12, C: 1.0 },
      { solver: 13, C: 0.1 },
    ],
    svm: [
      { svmType: 3, kernel: 2, C: 1.0, gamma: 0 },
      { svmType: 3, kernel: 2, C: 10.0, gamma: 0.01 },
      { svmType: 3, kernel: 1, C: 1.0, degree: 3, gamma: 0 },
      { svmType: 3, kernel: 0, C: 1.0 },
    ],
    knn: [
      { k: 5, metric: 'l2', task: 'regression' },
      { k: 15, metric: 'l2', task: 'regression' },
      { k: 3, metric: 'l1', task: 'regression' },
    ],
    tsetlin: [
      { task: 'regression', nClauses: 100, threshold: 50, s: 3.0, nEpochs: 100 },
      { task: 'regression', nClauses: 500, threshold: 100, s: 5.0, nEpochs: 100 },
      { task: 'regression', nClauses: 50, threshold: 25, s: 2.0, nEpochs: 60 },
    ],
    rf: [
      { nEstimators: 100, maxDepth: 10, maxFeatures: 'sqrt', criterion: 'mse',
        minSamplesSplit: 2, minSamplesLeaf: 1 },
      { nEstimators: 300, maxDepth: 0, maxFeatures: 'log2', criterion: 'mse',
        minSamplesSplit: 5, minSamplesLeaf: 2 },
      { nEstimators: 200, maxDepth: 15, maxFeatures: 'sqrt', criterion: 'mae',
        extraTrees: 1, minSamplesSplit: 2 },
      { nEstimators: 100, maxDepth: 10, maxFeatures: 'sqrt', criterion: 'mse',
        leafModel: 1, minSamplesLeaf: 5 },
      { nEstimators: 150, maxDepth: 10, maxFeatures: 'sqrt', criterion: 'mse',
        heterogeneous: 1, oobWeighting: 1, minSamplesLeaf: 2 },
      { nEstimators: 200, maxDepth: 10, maxFeatures: 'sqrt', criterion: 'mse',
        histogramBinning: 1, minSamplesSplit: 2, minSamplesLeaf: 1 },
      { nEstimators: 150, maxDepth: 12, maxFeatures: 'sqrt', criterion: 'mse',
        jarf: 1, minSamplesSplit: 2, minSamplesLeaf: 1 },
    ],
    mlp: [
      { hidden_sizes: [64], activation: 'relu', lr: 0.01, epochs: 50,
        optimizer: 'adam', batch_size: 32 },
      { hidden_sizes: [128, 64], activation: 'gelu', lr: 0.001, epochs: 100,
        optimizer: 'adam', batch_size: 32 },
      { hidden_sizes: [256, 128], activation: 'silu', lr: 0.001, epochs: 150,
        optimizer: 'adam', batch_size: 16 },
    ],
    tabm: [
      { hidden_sizes: [128], activation: 'relu', n_ensemble: 32, lr: 0.005,
        epochs: 100, optimizer: 'adam' },
      { hidden_sizes: [64], activation: 'gelu', n_ensemble: 8, lr: 0.01,
        epochs: 50, optimizer: 'adam' },
      { hidden_sizes: [128, 64], activation: 'silu', n_ensemble: 16, lr: 0.001,
        epochs: 150, optimizer: 'adam' },
    ],
    nam: [
      { hidden_sizes: [64], activation: 'exu', lr: 0.01, epochs: 100,
        optimizer: 'adam' },
      { hidden_sizes: [128], activation: 'relu', lr: 0.001, epochs: 100,
        optimizer: 'adam' },
      { hidden_sizes: [64, 32], activation: 'exu', lr: 0.005, epochs: 150,
        optimizer: 'adam' },
    ],
    gam: [
      { family: 'gaussian', penalty: 'elasticnet', alpha: 0.5, nLambda: 100, nFolds: 5 },
      { family: 'gaussian', penalty: 'lasso', alpha: 1.0, nLambda: 100, nFolds: 5 },
      { family: 'gaussian', penalty: 'ridge', alpha: 0.0, nLambda: 100, nFolds: 5 },
      { family: 'gaussian', penalty: 'scad', nLambda: 100, nFolds: 5 },
    ],
    bart: [
      { numTrees: 200, numBurnin: 200, numSamples: 100, alpha: 0.95, beta: 2.0 },
      { numTrees: 500, numBurnin: 300, numSamples: 100, alpha: 0.95, beta: 2.0 },
      { numTrees: 100, maxDepth: 5, numBurnin: 100, numSamples: 100,
        alpha: 0.8, beta: 1.0 },
    ],
    fm: [
      { k: 4, lr: 0.01, epoch: 10, opt: 'adagrad' },
      { k: 8, lr: 0.005, epoch: 20, opt: 'adagrad', lambda: 0.001 },
    ],
    xlr: [
      { lr: 0.01, opt: 'ftrl', epoch: 10 },
      { lr: 0.01, lambda: 0.1, opt: 'adagrad', epoch: 20 },
    ],
  },
}

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

    for (const model of models) {
      const name = model.name
      const cls = model.cls
      const fixed = model.params || {}

      const configs = portfolio[name] || [{}]

      for (const config of configs) {
        const params = { ...config, ...fixed }
        const candidateId = makeCandidateId(name, params)
        this.#queue.push({ candidateId, cls, params })
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

  constructor(models, opts = {}) {
    if (!models || models.length === 0) {
      throw new ValidationError('PortfolioSearch: at least one model is required')
    }
    this.#models = models
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
    const scoring = this.#opts.scoring || (task === 'classification' ? 'accuracy' : 'r2')
    const { cv, seed, maxTimeMs, onProgress } = this.#opts

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

    const strategy = new PortfolioStrategy(this.#models, { task, seed })

    const { leaderboard } = await executor.runStrategy(strategy)

    if (leaderboard.length === 0) {
      throw new ValidationError('PortfolioSearch: no candidates were evaluated')
    }

    this.#leaderboard = leaderboard
    this.#bestResult = leaderboard.best()
    return { leaderboard, bestResult: this.#bestResult }
  }

  async refitBest(X, y) {
    if (!this.#bestResult) {
      throw new ValidationError('PortfolioSearch: must call fit() first')
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

module.exports = { PORTFOLIO, getPortfolio, PortfolioStrategy, PortfolioSearch }
