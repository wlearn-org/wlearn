import { getScorer, normalizeY, ValidationError } from '@wlearn/core'
import { optimizeWeights } from './weights.js'

/**
 * Caruana greedy ensemble selection (Caruana et al., 2004).
 *
 * Selects a weighted subset from a pool of OOF predictions by greedily
 * adding the candidate that most improves the ensemble score at each step.
 * Candidates can be selected multiple times (with replacement).
 *
 * @param {Float64Array[]} oofPredictions - per-candidate OOF predictions
 *   Classification: each is n_samples * n_classes (flat row-major proba)
 *   Regression: each is n_samples
 * @param {TypedArray|number[]} yTrue - true labels
 * @param {Object} opts
 * @param {boolean} opts.refineWeights - if true, optimize weights after selection
 * @returns {{ indices: Int32Array, weights: Float64Array, scores: Float64Array }}
 */
export function caruanaSelect(oofPredictions, yTrue, {
  maxSize = 20,
  scoring = 'accuracy',
  task = 'classification',
  nClasses = 0,
  refineWeights = true,
} = {}) {
  const yn = normalizeY(yTrue)
  const n = yn.length
  const nCandidates = oofPredictions.length

  if (nCandidates === 0) {
    throw new ValidationError('caruanaSelect: need at least 1 candidate')
  }

  const scorerFn = getScorer(scoring)

  // Determine prediction size per sample
  const predSize = oofPredictions[0].length / n
  if (predSize !== Math.floor(predSize)) {
    throw new ValidationError('caruanaSelect: oofPredictions[0].length must be divisible by n')
  }

  if (task === 'classification' && nClasses === 0) {
    nClasses = predSize
  }

  // Current ensemble prediction (running weighted average)
  const current = new Float64Array(oofPredictions[0].length)
  const selected = []
  const scores = []

  for (let t = 0; t < maxSize; t++) {
    let bestIdx = -1
    let bestScore = -Infinity

    for (let i = 0; i < nCandidates; i++) {
      // Trial: ((t) * current + P[i]) / (t + 1)
      const trial = _trialPredictions(current, oofPredictions[i], t, t + 1)
      const trialScore = _score(trial, yn, scorerFn, task, nClasses, n)
      if (trialScore > bestScore) {
        bestScore = trialScore
        bestIdx = i
      }
    }

    selected.push(bestIdx)
    scores.push(bestScore)

    // Update running ensemble: current = (t * current + P[bestIdx]) / (t + 1)
    const P = oofPredictions[bestIdx]
    for (let j = 0; j < current.length; j++) {
      current[j] = (t * current[j] + P[j]) / (t + 1)
    }
  }

  // Compute weights from selection counts
  const counts = new Map()
  for (const idx of selected) {
    counts.set(idx, (counts.get(idx) || 0) + 1)
  }
  const uniqueIndices = new Int32Array([...counts.keys()].sort((a, b) => a - b))
  const weights = new Float64Array(uniqueIndices.length)
  for (let i = 0; i < uniqueIndices.length; i++) {
    weights[i] = counts.get(uniqueIndices[i]) / maxSize
  }

  const result = {
    indices: uniqueIndices,
    weights,
    scores: new Float64Array(scores),
  }

  if (refineWeights && uniqueIndices.length > 1) {
    const selectedOofs = Array.from(uniqueIndices, idx => oofPredictions[idx])
    result.weights = optimizeWeights(selectedOofs, yn, weights, { task })
  }

  return result
}

// --- Internal helpers ---

function _trialPredictions(current, candidate, tCount, tTotal) {
  const trial = new Float64Array(current.length)
  for (let j = 0; j < current.length; j++) {
    trial[j] = (tCount * current[j] + candidate[j]) / tTotal
  }
  return trial
}

function _score(preds, yTrue, scorerFn, task, nClasses, n) {
  if (task === 'regression') {
    return scorerFn(yTrue, preds)
  }
  // Classification: convert proba to hard predictions via argmax
  const hardPreds = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let bestC = 0, bestV = -Infinity
    for (let c = 0; c < nClasses; c++) {
      if (preds[i * nClasses + c] > bestV) {
        bestV = preds[i * nClasses + c]
        bestC = c
      }
    }
    hardPreds[i] = bestC
  }
  return scorerFn(yTrue, hardPreds)
}
