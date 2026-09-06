const { normalizeX, normalizeY, ValidationError } = require('@wlearn/core')
const {
  resolvePreprocessConfig, TYPE_ID: PREPROCESS_TYPE_ID
} = require('@wlearn/preprocess')
const { getOofPredictions, caruanaSelect, VotingEnsemble, StackingEnsemble } = require('@wlearn/ensemble')
const { RandomSearch } = require('./search.js')
const { SuccessiveHalvingSearch } = require('./halving.js')
const { PortfolioSearch } = require('./portfolio.js')
const { ProgressiveSearch } = require('./progressive.js')
const { BayesianSearch: DefaultBayesianSearch } = require('./bayesian.js')
const { detectTask } = require('./common.js')
const {
  classForCandidate, normalizeModelSpecs
} = require('./candidate.js')
const { createCandidatePipelineClass } = require('./candidate-pipeline.js')

let registeredBayesianSearch = null

function registerBayesianSearch(BayesianSearch) {
  if (BayesianSearch == null) {
    registeredBayesianSearch = null
    return
  }
  if (typeof BayesianSearch !== 'function') {
    throw new ValidationError('registerBayesianSearch: expected a BayesianSearch constructor or null')
  }
  registeredBayesianSearch = BayesianSearch
}

function resolveBayesianSearch() {
  return registeredBayesianSearch || DefaultBayesianSearch
}

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
function _resolvePreprocessChoices(value) {
  if (value === false || value === null || value === undefined) return [null]
  if (value === true) {
    return [_resolvedTemplate('wlearn.preprocess.default.v1', {})]
  }
  if (Array.isArray(value)) {
    if (value.length === 0) {
      throw new ValidationError('autoFit preprocess template list must be nonempty')
    }
    const seen = new Set()
    return value.map((template, index) => {
      if (!template || typeof template !== 'object' || Array.isArray(template)) {
        throw new ValidationError(`preprocess[${index}] must be a template object`)
      }
      if (typeof template.templateId !== 'string' || template.templateId.length === 0) {
        throw new ValidationError(`preprocess[${index}].templateId must be a nonempty string`)
      }
      if (seen.has(template.templateId)) {
        throw new ValidationError(`duplicate preprocessing templateId "${template.templateId}"`)
      }
      seen.add(template.templateId)
      if (template.typeId !== PREPROCESS_TYPE_ID) {
        throw new ValidationError(
          `preprocess[${index}].typeId must be "${PREPROCESS_TYPE_ID}"`
        )
      }
      if (Object.prototype.hasOwnProperty.call(template, 'searchSpace')) {
        if (!_isPlainObject(template.searchSpace)) {
          throw new ValidationError(`preprocess[${index}].searchSpace must be an object`)
        }
        if (Object.keys(template.searchSpace).length > 0) {
          throw new ValidationError(
            'preprocessing searchSpace is not supported in the fixed-template V1; enumerate explicit templates'
          )
        }
      }
      const params = Object.prototype.hasOwnProperty.call(template, 'params')
        ? template.params
        : {}
      if (!_isPlainObject(params)) {
        throw new ValidationError(`preprocess[${index}].params must be an object`)
      }
      return _resolvedTemplate(template.templateId, params)
    })
  }
  if (typeof value !== 'object') {
    throw new ValidationError(
      'autoFit preprocess must be false, true, a config object, or a template list'
    )
  }
  return [_resolvedTemplate('wlearn.preprocess.inline.v1', value)]
}

function _isPlainObject(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const prototype = Object.getPrototypeOf(value)
  return prototype === Object.prototype || prototype === null
}

function _resolvedTemplate(templateId, config) {
  return Object.freeze({
    templateId,
    typeId: PREPROCESS_TYPE_ID,
    resolvedParams: resolvePreprocessConfig(config),
  })
}

/**
 * High-level AutoML: random search + optional Caruana ensemble + refit.
 *
 * @param {Array} models - ModelSpec[] or EstimatorSpec tuples [name, cls, params?]
 * @param {object|number[][]} X - feature matrix
 * @param {TypedArray|number[]} y - labels
 * @param {object} opts
 * @returns {Promise<{ model: object, leaderboard: object[], archive: object, bestParams: object, bestModelName: string, bestScore: number }>}
 */
async function autoFit(models, X, y, opts = {}) {
  const {
    ensemble = true,
    ensembleSize = 20,
    refit = true,
    strategy = 'random',
    minDisagreement = 0.05,
    stacking = 'auto',
    stackingPassthrough = undefined,
    metaEstimator = null,
    preprocess = false,
    onProgress = null,
    ...searchOpts
  } = opts

  const preprocessChoices = _resolvePreprocessChoices(preprocess)
  const cv = searchOpts.cv ?? 5
  const seed = searchOpts.seed ?? 42
  const specs = normalizeModelSpecs(models, 'autoFit models').map(spec => ({
    ...spec,
    preprocessChoices,
    createCandidateClass: candidate => (
      createCandidatePipelineClass(spec, candidate, {
        baseSeed: seed,
        foldCount: cv,
      })
    ),
  }))

  // Run search
  const searchOptsWithProgress = { ...searchOpts, onProgress }
  let search
  if (strategy === 'portfolio') {
    search = new PortfolioSearch(specs, searchOptsWithProgress)
  } else if (strategy === 'halving') {
    search = new SuccessiveHalvingSearch(specs, searchOptsWithProgress)
  } else if (strategy === 'progressive') {
    search = new ProgressiveSearch(specs, searchOptsWithProgress)
  } else if (strategy === 'bayesian') {
    const BayesianSearch = resolveBayesianSearch()
    search = new BayesianSearch(specs, searchOptsWithProgress)
  } else {
    search = new RandomSearch(specs, searchOptsWithProgress)
  }
  const { leaderboard, archive, bestResult } = await search.fit(X, y)
  const ranked = leaderboard.ranked()

  const Xn = normalizeX(X)
  const yn = normalizeY(y)
  const task = searchOpts.task || detectTask(yn)
  const scoring = searchOpts.scoring || (task === 'classification' ? 'accuracy' : 'r2')
  let model = null

  if (ensemble) {
    if (onProgress) {
      onProgress({ phase: 'ensemble', message: 'building ensemble' })
    }
    // Diversity-aware pool: best per family + top overall with disagreement filter
    const familyBest = new Map()
    const familySecond = new Map()
    for (const entry of ranked) {
      const classId = entry.candidate.model.classId
      if (!familyBest.has(classId)) {
        familyBest.set(classId, entry)
      } else if (!familySecond.has(classId)) {
        familySecond.set(classId, entry)
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

    const specMap = new Map(specs.map(spec => [spec.classId, spec]))

    // Build estimator specs for OOF
    const estSpecs = pool.map((entry, i) => {
      const spec = specMap.get(entry.candidate.model.classId)
      const cls = classForCandidate(spec, entry.candidate)
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
    const filteredEntries = filteredIdx.map(i => pool[i])

    // Caruana selection on filtered pool
    const { indices: selIndices, weights } = caruanaSelect(filteredOofs, yn, {
      maxSize: ensembleSize,
      scoring,
      task,
    })

    // Build ensemble from selected
    const indices = selIndices
    const selectedSpecs = Array.from(indices, i => filteredSpecs[i])
    const selectedEntries = Array.from(indices, i => filteredEntries[i])
    const selectedWeights = weights

    // Determine if two-layer stacking should be used
    const selectedFamilies = new Set(
      selectedEntries.map(entry => entry.candidate.model.classId)
    )
    const useStacking = stacking === true ||
      (stacking === 'auto' && selectedFamilies.size >= 3 && metaEstimator)

    if (useStacking && metaEstimator) {
      // Two-layer stacking: L0 = selected base models, L1 = meta-model
      const metaSpec = Array.isArray(metaEstimator)
        ? metaEstimator
        : ['meta', metaEstimator.cls || metaEstimator, metaEstimator.params || {}]
      const selectedPreprocessing = selectedEntries.some(
        entry => entry.candidate.preprocess !== null
      )
      if (selectedPreprocessing && stackingPassthrough === true) {
        throw new ValidationError(
          'stackingPassthrough=true is invalid when preprocessing is active'
        )
      }
      const ens = await StackingEnsemble.create({
        estimators: selectedSpecs,
        finalEstimator: metaSpec,
        passthrough: selectedPreprocessing
          ? false
          : (stackingPassthrough ?? true),
        task,
        cv,
        seed,
      })
      model = await fitOwnedEnsemble(ens, Xn, yn)
    } else {
      // Default: VotingEnsemble
      const ens = await VotingEnsemble.create({
        estimators: selectedSpecs,
        weights: selectedWeights,
        voting: task === 'classification' ? 'soft' : undefined,
        task,
      })
      model = await fitOwnedEnsemble(ens, Xn, yn)
    }
  } else if (refit) {
    model = await search.refitBest(X, y)
  }

  return {
    model,
    // Kept for source compatibility. Preprocessing is now owned by the fitted
    // Pipeline(s), so there is no separately managed fitted preprocessor.
    preprocessor: null,
    leaderboard: ranked,
    archive,
    bestParams: {
      model: bestResult.candidate.model.params,
      preprocess: bestResult.candidate.preprocess?.resolvedParams ?? null,
    },
    bestCandidate: bestResult.candidate,
    bestModelName: bestResult.modelName,
    bestScore: bestResult.meanScore,
  }
}

async function fitOwnedEnsemble(ensemble, X, y) {
  try {
    await ensemble.fit(X, y)
    return ensemble
  } catch (error) {
    try { ensemble.dispose() } catch {}
    throw error
  }
}

module.exports = { autoFit, registerBayesianSearch }
