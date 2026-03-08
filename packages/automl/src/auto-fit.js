const { normalizeX, normalizeY, ValidationError, Preprocessor } = require('@wlearn/core')
const { getOofPredictions, caruanaSelect, VotingEnsemble, StackingEnsemble } = require('@wlearn/ensemble')
const { RandomSearch } = require('./search.js')
const { SuccessiveHalvingSearch } = require('./halving.js')
const { PortfolioSearch } = require('./portfolio.js')
const { ProgressiveSearch } = require('./progressive.js')
const { detectTask } = require('./common.js')

/**
 * Compute pairwise disagreement rate between two prediction vectors.
 * For classification: fraction of samples where argmax differs.
 * For regression: 1 - correlation (capped at [0,1]).
 */
function _disagreementRate(a, b, n, task) {
  if (task === 'classification') {
    const nClasses = a.length / n
    let disagree = 0
    for (let i = 0; i < n; i++) {
      let bestA = 0, bestB = 0, bestVA = -Infinity, bestVB = -Infinity
      for (let c = 0; c < nClasses; c++) {
        const idx = i * nClasses + c
        if (a[idx] > bestVA) { bestVA = a[idx]; bestA = c }
        if (b[idx] > bestVB) { bestVB = b[idx]; bestB = c }
      }
      if (bestA !== bestB) disagree++
    }
    return disagree / n
  }
  // Regression: 1 - abs(correlation)
  let sumA = 0, sumB = 0, sumAA = 0, sumBB = 0, sumAB = 0
  for (let i = 0; i < n; i++) {
    sumA += a[i]; sumB += b[i]
    sumAA += a[i] * a[i]; sumBB += b[i] * b[i]
    sumAB += a[i] * b[i]
  }
  const denom = Math.sqrt((sumAA - sumA * sumA / n) * (sumBB - sumB * sumB / n))
  if (denom < 1e-12) return 1
  const corr = (sumAB - sumA * sumB / n) / denom
  return 1 - Math.abs(corr)
}

/**
 * Filter pool by minimum pairwise disagreement.
 * Always keeps index 0 (best model). Greedily adds candidates that
 * have at least minDisagreement with all already-selected members.
 * Returns array of retained indices.
 */
function _filterByDisagreement(oofPreds, yn, task, minDisagreement) {
  const n = yn.length
  if (oofPreds.length <= 2 || minDisagreement <= 0) {
    return oofPreds.map((_, i) => i)
  }
  const kept = [0]
  for (let i = 1; i < oofPreds.length; i++) {
    let diverse = true
    for (const j of kept) {
      if (_disagreementRate(oofPreds[i], oofPreds[j], n, task) < minDisagreement) {
        diverse = false
        break
      }
    }
    if (diverse) kept.push(i)
  }
  // Always keep at least 2 for ensemble
  if (kept.length < 2 && oofPreds.length >= 2) {
    if (!kept.includes(1)) kept.push(1)
  }
  return kept
}

/**
 * Normalize model specs: accept both ModelSpec objects and [name, cls, params?] tuples.
 */
function _normalizeSpecs(models) {
  return models.map(m => {
    if (Array.isArray(m)) {
      return { name: m[0], cls: m[1], params: m[2] || {} }
    }
    return m
  })
}

/**
 * High-level AutoML: random search + optional Caruana ensemble + refit.
 *
 * @param {Array} models - ModelSpec[] or EstimatorSpec tuples [name, cls, params?]
 * @param {object|number[][]} X - feature matrix
 * @param {TypedArray|number[]} y - labels
 * @param {object} opts
 * @returns {Promise<{ model: object, leaderboard: object[], bestParams: object, bestModelName: string, bestScore: number }>}
 */
async function autoFit(models, X, y, opts = {}) {
  const {
    ensemble = true,
    ensembleSize = 20,
    refit = true,
    strategy = 'random',
    minDisagreement = 0.05,
    stacking = 'auto',
    metaEstimator = null,
    preprocess = false,
    onProgress = null,
    ...searchOpts
  } = opts

  const specs = _normalizeSpecs(models)
  if (specs.length === 0) {
    throw new ValidationError('autoFit: at least one model is required')
  }

  // Optional preprocessing
  let preprocessor = null
  if (preprocess) {
    const ppConfig = typeof preprocess === 'object' ? preprocess : {}
    preprocessor = new Preprocessor(ppConfig)
    const Xpre = normalizeX(X)
    const ypre = normalizeY(y)
    const Xt = preprocessor.fitTransform(Xpre, ypre)
    X = Xt
  }

  // Run search
  const searchOptsWithProgress = { ...searchOpts, onProgress }
  let search
  if (strategy === 'portfolio') {
    search = new PortfolioSearch(specs, searchOptsWithProgress)
  } else if (strategy === 'halving') {
    search = new SuccessiveHalvingSearch(specs, searchOptsWithProgress)
  } else if (strategy === 'progressive') {
    search = new ProgressiveSearch(specs, searchOptsWithProgress)
  } else {
    search = new RandomSearch(specs, searchOptsWithProgress)
  }
  const { leaderboard, bestResult } = await search.fit(X, y)
  const ranked = leaderboard.ranked()

  const Xn = normalizeX(X)
  const yn = normalizeY(y)
  const task = searchOpts.task || detectTask(yn)
  const scoring = searchOpts.scoring || (task === 'classification' ? 'accuracy' : 'r2')
  const cv = searchOpts.cv || 5
  const seed = searchOpts.seed || 42

  let model = null

  if (ensemble) {
    if (onProgress) {
      onProgress({ phase: 'ensemble', message: 'building ensemble' })
    }
    // Diversity-aware pool: best per family + top overall with disagreement filter
    const familyBest = new Map()
    const familySecond = new Map()
    for (const entry of ranked) {
      if (!familyBest.has(entry.modelName)) {
        familyBest.set(entry.modelName, entry)
      } else if (!familySecond.has(entry.modelName)) {
        familySecond.set(entry.modelName, entry)
      }
    }

    // Seed pool: best per family (guaranteed diversity)
    const pool = [...familyBest.values()]
    const poolIds = new Set(pool.map(e => e.id))

    // Add second-best per family if available (for intra-family diversity)
    for (const entry of familySecond.values()) {
      if (pool.length >= ensembleSize * 2) break
      if (!poolIds.has(entry.id)) {
        pool.push(entry)
        poolIds.add(entry.id)
      }
    }

    // Fill remaining slots from top overall
    for (const entry of ranked) {
      if (pool.length >= ensembleSize * 2) break
      if (!poolIds.has(entry.id)) {
        pool.push(entry)
        poolIds.add(entry.id)
      }
    }

    // Map model names to classes
    const clsMap = new Map()
    for (const spec of specs) {
      clsMap.set(spec.name, spec.cls)
    }

    // Build estimator specs for OOF
    const estSpecs = pool.map((entry, i) => {
      const cls = clsMap.get(entry.modelName)
      return [`${entry.modelName}_${i}`, cls, entry.params]
    })

    // Generate OOF predictions
    const { oofPreds } = await getOofPredictions(estSpecs, Xn, yn, {
      cv, seed, task,
    })

    // Disagreement filter: remove near-duplicate predictions
    const filteredIdx = _filterByDisagreement(oofPreds, yn, task, minDisagreement)
    const filteredOofs = filteredIdx.map(i => oofPreds[i])
    const filteredSpecs = filteredIdx.map(i => estSpecs[i])

    // Caruana selection on filtered pool
    const { indices: selIndices, weights } = caruanaSelect(filteredOofs, yn, {
      maxSize: ensembleSize,
      scoring,
      task,
    })

    // Build ensemble from selected
    const indices = selIndices
    const selectedSpecs = Array.from(indices, i => filteredSpecs[i])
    const selectedWeights = weights

    // Determine if two-layer stacking should be used
    const selectedFamilies = new Set(selectedSpecs.map(s => s[0].split('_')[0]))
    const useStacking = stacking === true ||
      (stacking === 'auto' && selectedFamilies.size >= 3 && metaEstimator)

    if (useStacking && metaEstimator) {
      // Two-layer stacking: L0 = selected base models, L1 = meta-model
      const metaSpec = Array.isArray(metaEstimator)
        ? metaEstimator
        : ['meta', metaEstimator.cls || metaEstimator, metaEstimator.params || {}]
      const ens = await StackingEnsemble.create({
        estimators: selectedSpecs,
        finalEstimator: metaSpec,
        passthrough: true,
        task,
        cv,
        seed,
      })
      await ens.fit(Xn, yn)
      model = ens
    } else {
      // Default: VotingEnsemble
      const ens = await VotingEnsemble.create({
        estimators: selectedSpecs,
        weights: selectedWeights,
        voting: task === 'classification' ? 'soft' : undefined,
        task,
      })
      await ens.fit(Xn, yn)
      model = ens
    }
  } else if (refit) {
    model = await search.refitBest(X, y)
  }

  return {
    model,
    preprocessor,
    leaderboard: ranked,
    bestParams: bestResult.params,
    bestModelName: bestResult.modelName,
    bestScore: bestResult.meanScore,
  }
}

module.exports = { autoFit }

