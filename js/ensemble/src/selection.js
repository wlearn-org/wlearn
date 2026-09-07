const { getScorer, normalizeY, ValidationError } = require('@wlearn/core')
const { optimizeWeights } = require('./weights.js')
const {
  normalizeClassOrder,
  validateProbabilityOutput,
  validateRegressionOutput,
} = require('./class-order.js')

/**
 * Caruana greedy ensemble selection (Caruana et al., 2004).
 */
function caruanaSelect(oofPredictions, yTrue, {
  maxSize = 20,
  scoring = 'accuracy',
  task = 'classification',
  nClasses = 0,
  refineWeights = true,
  classes,
} = {}) {
  const yn = normalizeY(yTrue)
  const n = yn.length
  const nCandidates = oofPredictions.length

  if (nCandidates === 0) {
    throw new ValidationError('caruanaSelect: need at least 1 candidate')
  }

  if (!Number.isSafeInteger(maxSize) || maxSize < 1 || n < 1) {
    throw new ValidationError('caruanaSelect: maxSize and row count must be positive')
  }
  const scorerFn = getScorer(scoring)
  const maximize = scorerFn.direction !== 'minimize'
  const objective = scorerFn.measure?.id

  // Determine prediction size per sample
  const predSize = oofPredictions[0].length / n
  if (predSize !== Math.floor(predSize)) {
    throw new ValidationError('caruanaSelect: oofPredictions[0].length must be divisible by n')
  }

  if (task === 'classification' && nClasses === 0) {
    nClasses = predSize
  }
  if (task === 'classification' && nClasses !== predSize) {
    throw new ValidationError(
      'caruanaSelect: nClasses must match probability columns per row'
    )
  }
  const classLabels = task === 'classification'
    ? _resolveClasses(yn, nClasses, classes)
    : null
  for (let index = 0; index < oofPredictions.length; index++) {
    const label = `caruanaSelect: oofPredictions[${index}]`
    if (task === 'classification') {
      validateProbabilityOutput(oofPredictions[index], n, nClasses, label)
    } else {
      validateRegressionOutput(oofPredictions[index], n, label)
    }
  }

  // Current ensemble prediction (running weighted average)
  const current = new Float64Array(oofPredictions[0].length)
  const selected = []
  const scores = []

  for (let t = 0; t < maxSize; t++) {
    let bestIdx = -1
    let bestScore = maximize ? -Infinity : Infinity

    for (let i = 0; i < nCandidates; i++) {
      // Trial: ((t) * current + P[i]) / (t + 1)
      const trial = _trialPredictions(current, oofPredictions[i], t, t + 1)
      const trialScore = _score(
        trial, yn, scorerFn, task, nClasses, n, classLabels
      )
      if (maximize ? trialScore > bestScore : trialScore < bestScore) {
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

  // Refinement only optimizes the objective it actually implements. Other
  // metrics keep the greedy weights instead of silently switching losses.
  const canRefine = task === 'regression'
    ? ['mse', 'neg_mse', 'r2'].includes(objective)
    : objective === 'log_loss'
  if (refineWeights && canRefine && uniqueIndices.length > 1) {
    const selectedOofs = Array.from(uniqueIndices, idx => oofPredictions[idx])
    result.weights = optimizeWeights(selectedOofs, yn, weights, {
      task, classes: classLabels
    })
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

function _score(preds, yTrue, scorerFn, task, nClasses, n, classes) {
  if (task === 'regression' || (scorerFn.response && scorerFn.response !== 'response')) {
    const score = scorerFn(yTrue, preds, { classes: classes ?? undefined })
    if (!Number.isFinite(score)) throw new ValidationError('Scorer must return a finite number')
    return score
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
    hardPreds[i] = classes[bestC]
  }
  const score = scorerFn(yTrue, hardPreds)
  if (!Number.isFinite(score)) throw new ValidationError('Scorer must return a finite number')
  return score
}

function _resolveClasses(yTrue, nClasses, classes) {
  const source = classes == null
    ? [...new Set(Array.from(yTrue))].sort((a, b) => a - b)
    : classes
  const labels = normalizeClassOrder(source, nClasses, 'caruanaSelect')
  const known = new Set(labels)
  for (const label of yTrue) {
    if (!known.has(label)) {
      throw new ValidationError(`caruanaSelect: class "${label}" is missing from classes`)
    }
  }
  return labels
}

module.exports = { caruanaSelect }
