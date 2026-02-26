import { normalizeX, normalizeY, ValidationError } from '@wlearn/core'
import { getOofPredictions, caruanaSelect, VotingEnsemble } from '@wlearn/ensemble'
import { RandomSearch } from './search.js'

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
export async function autoFit(models, X, y, opts = {}) {
  const {
    ensemble = false,
    ensembleSize = 20,
    refit = true,
    ...searchOpts
  } = opts

  const specs = _normalizeSpecs(models)
  if (specs.length === 0) {
    throw new ValidationError('autoFit: at least one model is required')
  }

  // Run search
  const search = new RandomSearch(specs, searchOpts)
  const { leaderboard, bestResult } = await search.fit(X, y)
  const ranked = leaderboard.ranked()

  const Xn = normalizeX(X)
  const yn = normalizeY(y)
  const task = searchOpts.task || _detectTask(yn)
  const scoring = searchOpts.scoring || (task === 'classification' ? 'accuracy' : 'r2')
  const cv = searchOpts.cv || 5
  const seed = searchOpts.seed || 42

  let model = null

  if (ensemble) {
    // Select best config per model family, then fill with more
    const familyBest = new Map()
    for (const entry of ranked) {
      if (!familyBest.has(entry.modelName)) {
        familyBest.set(entry.modelName, entry)
      }
    }

    // Build pool: best per family + top remaining up to ensembleSize * 2
    const pool = [...familyBest.values()]
    const poolIds = new Set(pool.map(e => e.id))
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

    // Caruana selection
    const { indices, weights } = caruanaSelect(oofPreds, yn, {
      maxSize: ensembleSize,
      scoring,
      task,
    })

    // Build VotingEnsemble from selected
    const selectedSpecs = Array.from(indices, i => estSpecs[i])
    const selectedWeights = weights

    const ens = await VotingEnsemble.create({
      estimators: selectedSpecs,
      weights: selectedWeights,
      voting: task === 'classification' ? 'soft' : undefined,
      task,
    })
    await ens.fit(Xn, yn)
    model = ens
  } else if (refit) {
    model = await search.refitBest(X, y)
  }

  return {
    model,
    leaderboard: ranked,
    bestParams: bestResult.params,
    bestModelName: bestResult.modelName,
    bestScore: bestResult.meanScore,
  }
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
