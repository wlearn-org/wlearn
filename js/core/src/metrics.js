const { ValidationError } = require('./errors.js')

// --- Internal helpers ---

function _validatePair(yTrue, yPred, name) {
  if (!yTrue || !yPred || yTrue.length === 0 || yPred.length === 0) {
    throw new ValidationError(`${name}: inputs must be non-empty`)
  }
  if (yTrue.length !== yPred.length) {
    throw new ValidationError(`${name}: length mismatch (${yTrue.length} vs ${yPred.length})`)
  }
}

function _validateSampleWeight(sampleWeight, n, name) {
  if (sampleWeight == null) return undefined
  if (sampleWeight.length !== n) {
    throw new ValidationError(`${name}: sampleWeight length mismatch (${sampleWeight.length} vs ${n})`)
  }
  const out = sampleWeight instanceof Float64Array ? sampleWeight : new Float64Array(sampleWeight)
  let total = 0
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(out[i]) || out[i] < 0) {
      throw new ValidationError(`${name}: sampleWeight must contain finite non-negative values`)
    }
    total += out[i]
  }
  if (total <= 0) throw new ValidationError(`${name}: sampleWeight sum must be positive`)
  return out
}

function _weightAt(weights, i) {
  return weights ? weights[i] : 1
}

function _weightSum(weights, n) {
  if (!weights) return n
  let total = 0
  for (let i = 0; i < n; i++) total += weights[i]
  return total
}

function _classInfo(yTrue, yPred, classes) {
  const labelSet = new Set()
  if (classes) {
    for (const label of classes) labelSet.add(label)
  } else {
    for (let i = 0; i < yTrue.length; i++) labelSet.add(yTrue[i])
    if (yPred) {
      for (let i = 0; i < yPred.length; i++) labelSet.add(yPred[i])
    }
  }
  const labels = [...labelSet].sort((a, b) => a - b)
  const labelMap = new Map()
  for (let i = 0; i < labels.length; i++) labelMap.set(labels[i], i)
  return { labels: new Int32Array(labels), labelMap, nClasses: labels.length }
}

function _buildCM(yTrue, yPred, labelMap, nClasses, weights) {
  const cm = weights ? new Float64Array(nClasses * nClasses) : new Int32Array(nClasses * nClasses)
  for (let i = 0; i < yTrue.length; i++) {
    const t = labelMap.get(yTrue[i])
    const p = labelMap.get(yPred[i])
    if (t == null || p == null) throw new ValidationError('confusionMatrix: label missing from class map')
    cm[t * nClasses + p] += _weightAt(weights, i)
  }
  return cm
}

function _classCounts(cm, nClasses) {
  const tp = new Float64Array(nClasses)
  const fp = new Float64Array(nClasses)
  const fn = new Float64Array(nClasses)
  const support = new Float64Array(nClasses)
  for (let c = 0; c < nClasses; c++) {
    tp[c] = cm[c * nClasses + c]
    for (let j = 0; j < nClasses; j++) {
      support[c] += cm[c * nClasses + j]
      if (j !== c) {
        fp[c] += cm[j * nClasses + c]
        fn[c] += cm[c * nClasses + j]
      }
    }
  }
  return { tp, fp, fn, support }
}

function _emitWarning(opts, metric, message) {
  if (opts && Array.isArray(opts.warnings)) {
    opts.warnings.push({ type: 'undefined_metric', metric, message })
  }
}

function _undefinedMetric(metric, message, opts = {}, defaultValue = NaN) {
  const policy = opts.undefinedValue ?? opts.undefined_value ?? opts.naValue ?? opts.na_value
  if (policy === 'error' || policy == null) {
    throw new ValidationError(`${metric}: ${message}`)
  }
  if (policy === 'warn') {
    _emitWarning(opts, metric, message)
    return defaultValue
  }
  if (policy === 'nan') return NaN
  if (typeof policy === 'number') return policy
  throw new ValidationError(`${metric}: unsupported undefinedValue policy "${policy}"`)
}

function _zeroDivision(metric, message, opts = {}) {
  const policy = opts.zeroDivision ?? opts.zero_division ?? opts.undefinedValue ?? opts.undefined_value
  if (policy == null) return 0
  if (policy === 'error') throw new ValidationError(`${metric}: ${message}`)
  if (policy === 'warn') {
    _emitWarning(opts, metric, message)
    return 0
  }
  if (policy === 'nan') return NaN
  if (policy === 0 || policy === 1) return policy
  if (typeof policy === 'number') return policy
  throw new ValidationError(`${metric}: unsupported zeroDivision policy "${policy}"`)
}

function _resolveAverage(average, nClasses) {
  const avg = average || 'binary'
  if (avg === 'binary') {
    if (nClasses > 2) {
      throw new ValidationError('average="binary" requires exactly 2 classes')
    }
    return avg
  }
  if (avg === 'micro' || avg === 'macro' || avg === 'weighted' || avg === 'macro_weighted') return avg
  throw new ValidationError(`Unknown averaging method "${avg}"`)
}

function _weightedMean(values, weights) {
  let total = 0
  let weightSum = 0
  for (let i = 0; i < values.length; i++) {
    if (weights[i] <= 0 || Number.isNaN(values[i])) continue
    total += values[i] * weights[i]
    weightSum += weights[i]
  }
  return weightSum > 0 ? total / weightSum : NaN
}

function _classesFromOpts(yTrue, opts = {}) {
  if (opts.classes) return Array.from(opts.classes)
  const labels = Array.from(new Set(Array.from(yTrue))).sort((a, b) => a - b)
  if (opts.nClasses || opts.n_classes) {
    const nClasses = opts.nClasses || opts.n_classes
    if (labels.length === nClasses) return labels
    const zeroBased = Array.from({ length: nClasses }, (_, i) => i)
    if (labels.every(label => label >= 0 && label < nClasses && Number.isInteger(label))) return zeroBased
  }
  return labels
}

function _positiveLabel(labels, opts = {}) {
  if (opts.positiveLabel != null) return opts.positiveLabel
  if (opts.positive_label != null) return opts.positive_label
  return labels[labels.length - 1]
}

// --- Exports ---

function accuracy(yTrue, yPred, { sampleWeight, sample_weight: sampleWeightSnake } = {}) {
  _validatePair(yTrue, yPred, 'accuracy')
  const weights = _validateSampleWeight(sampleWeight || sampleWeightSnake, yTrue.length, 'accuracy')
  let correct = 0
  for (let i = 0; i < yTrue.length; i++) {
    if (yTrue[i] === yPred[i]) correct += _weightAt(weights, i)
  }
  return correct / _weightSum(weights, yTrue.length)
}

function r2Score(yTrue, yPred, { sampleWeight, sample_weight: sampleWeightSnake } = {}) {
  _validatePair(yTrue, yPred, 'r2Score')
  const weights = _validateSampleWeight(sampleWeight || sampleWeightSnake, yTrue.length, 'r2Score')
  const n = yTrue.length
  const wSum = _weightSum(weights, n)
  let mean = 0
  for (let i = 0; i < n; i++) mean += yTrue[i] * _weightAt(weights, i)
  mean /= wSum
  let ssTot = 0, ssRes = 0
  for (let i = 0; i < n; i++) {
    const w = _weightAt(weights, i)
    const d = yTrue[i] - mean
    ssTot += w * d * d
    const r = yTrue[i] - yPred[i]
    ssRes += w * r * r
  }
  if (ssTot === 0) return 0
  return 1 - ssRes / ssTot
}

function meanSquaredError(yTrue, yPred, { sampleWeight, sample_weight: sampleWeightSnake } = {}) {
  _validatePair(yTrue, yPred, 'meanSquaredError')
  const weights = _validateSampleWeight(sampleWeight || sampleWeightSnake, yTrue.length, 'meanSquaredError')
  let sum = 0
  for (let i = 0; i < yTrue.length; i++) {
    const d = yTrue[i] - yPred[i]
    sum += _weightAt(weights, i) * d * d
  }
  return sum / _weightSum(weights, yTrue.length)
}

function meanAbsoluteError(yTrue, yPred, { sampleWeight, sample_weight: sampleWeightSnake } = {}) {
  _validatePair(yTrue, yPred, 'meanAbsoluteError')
  const weights = _validateSampleWeight(sampleWeight || sampleWeightSnake, yTrue.length, 'meanAbsoluteError')
  let sum = 0
  for (let i = 0; i < yTrue.length; i++) {
    sum += _weightAt(weights, i) * Math.abs(yTrue[i] - yPred[i])
  }
  return sum / _weightSum(weights, yTrue.length)
}

function confusionMatrix(yTrue, yPred, { sampleWeight, sample_weight: sampleWeightSnake, classes } = {}) {
  _validatePair(yTrue, yPred, 'confusionMatrix')
  const weights = _validateSampleWeight(sampleWeight || sampleWeightSnake, yTrue.length, 'confusionMatrix')
  const { labels, labelMap, nClasses } = _classInfo(yTrue, yPred, classes)
  const matrix = _buildCM(yTrue, yPred, labelMap, nClasses, weights)
  return { matrix, labels }
}

function precisionScore(yTrue, yPred, opts = {}) {
  _validatePair(yTrue, yPred, 'precisionScore')
  const weights = _validateSampleWeight(opts.sampleWeight || opts.sample_weight, yTrue.length, 'precisionScore')
  const { labels, labelMap, nClasses } = _classInfo(yTrue, yPred, opts.classes)
  const cm = _buildCM(yTrue, yPred, labelMap, nClasses, weights)
  const { tp, fp, support } = _classCounts(cm, nClasses)
  const avg = _resolveAverage(opts.average, nClasses)

  const values = new Float64Array(nClasses)
  for (let c = 0; c < nClasses; c++) {
    const denom = tp[c] + fp[c]
    values[c] = denom === 0 ? _zeroDivision('precisionScore', `precision is undefined for class ${labels[c]}`, opts) : tp[c] / denom
  }
  if (avg === 'binary') return values[nClasses - 1]
  if (avg === 'micro') {
    let tpSum = 0, fpSum = 0
    for (let c = 0; c < nClasses; c++) { tpSum += tp[c]; fpSum += fp[c] }
    return tpSum + fpSum === 0 ? _zeroDivision('precisionScore', 'micro precision is undefined', opts) : tpSum / (tpSum + fpSum)
  }
  if (avg === 'weighted' || avg === 'macro_weighted') return _weightedMean(values, support)
  let sum = 0
  for (let c = 0; c < nClasses; c++) sum += values[c]
  return sum / nClasses
}

function recallScore(yTrue, yPred, opts = {}) {
  _validatePair(yTrue, yPred, 'recallScore')
  const weights = _validateSampleWeight(opts.sampleWeight || opts.sample_weight, yTrue.length, 'recallScore')
  const { labels, labelMap, nClasses } = _classInfo(yTrue, yPred, opts.classes)
  const cm = _buildCM(yTrue, yPred, labelMap, nClasses, weights)
  const { tp, fn, support } = _classCounts(cm, nClasses)
  const avg = _resolveAverage(opts.average, nClasses)

  const values = new Float64Array(nClasses)
  for (let c = 0; c < nClasses; c++) {
    const denom = tp[c] + fn[c]
    values[c] = denom === 0 ? _zeroDivision('recallScore', `recall is undefined for class ${labels[c]}`, opts) : tp[c] / denom
  }
  if (avg === 'binary') return values[nClasses - 1]
  if (avg === 'micro') {
    let tpSum = 0, fnSum = 0
    for (let c = 0; c < nClasses; c++) { tpSum += tp[c]; fnSum += fn[c] }
    return tpSum + fnSum === 0 ? _zeroDivision('recallScore', 'micro recall is undefined', opts) : tpSum / (tpSum + fnSum)
  }
  if (avg === 'weighted' || avg === 'macro_weighted') return _weightedMean(values, support)
  let sum = 0
  for (let c = 0; c < nClasses; c++) sum += values[c]
  return sum / nClasses
}

function f1Score(yTrue, yPred, opts = {}) {
  _validatePair(yTrue, yPred, 'f1Score')
  const weights = _validateSampleWeight(opts.sampleWeight || opts.sample_weight, yTrue.length, 'f1Score')
  const { labels, labelMap, nClasses } = _classInfo(yTrue, yPred, opts.classes)
  const cm = _buildCM(yTrue, yPred, labelMap, nClasses, weights)
  const { tp, fp, fn, support } = _classCounts(cm, nClasses)
  const avg = _resolveAverage(opts.average, nClasses)

  const values = new Float64Array(nClasses)
  for (let c = 0; c < nClasses; c++) {
    const pDenom = tp[c] + fp[c]
    const rDenom = tp[c] + fn[c]
    const p = pDenom === 0 ? _zeroDivision('f1Score', `precision is undefined for class ${labels[c]}`, opts) : tp[c] / pDenom
    const r = rDenom === 0 ? _zeroDivision('f1Score', `recall is undefined for class ${labels[c]}`, opts) : tp[c] / rDenom
    values[c] = p + r === 0 ? 0 : 2 * p * r / (p + r)
  }
  if (avg === 'binary') return values[nClasses - 1]
  if (avg === 'micro') {
    let tpSum = 0, fpSum = 0, fnSum = 0
    for (let c = 0; c < nClasses; c++) { tpSum += tp[c]; fpSum += fp[c]; fnSum += fn[c] }
    const p = tpSum + fpSum === 0 ? _zeroDivision('f1Score', 'micro precision is undefined', opts) : tpSum / (tpSum + fpSum)
    const r = tpSum + fnSum === 0 ? _zeroDivision('f1Score', 'micro recall is undefined', opts) : tpSum / (tpSum + fnSum)
    return p + r === 0 ? 0 : 2 * p * r / (p + r)
  }
  if (avg === 'weighted' || avg === 'macro_weighted') return _weightedMean(values, support)
  let sum = 0
  for (let c = 0; c < nClasses; c++) sum += values[c]
  return sum / nClasses
}

function logLoss(yTrue, yProba, opts = {}) {
  if (!yTrue || yTrue.length === 0) {
    throw new ValidationError('logLoss: yTrue must be non-empty')
  }
  const n = yTrue.length
  const weights = _validateSampleWeight(opts.sampleWeight || opts.sample_weight, n, 'logLoss')
  const classes = _classesFromOpts(yTrue, opts)
  const nClasses = opts.nClasses || opts.n_classes || classes.length
  if (classes.length !== nClasses) {
    throw new ValidationError('logLoss: classes length must match nClasses')
  }
  if (yProba.length !== n * nClasses) {
    throw new ValidationError(`logLoss: yProba length (${yProba.length}) must be n * nClasses (${n * nClasses})`)
  }
  for (let i = 0; i < yProba.length; i++) {
    if (!Number.isFinite(yProba[i])) {
      throw new ValidationError('logLoss: probabilities must be finite')
    }
  }
  const classMap = new Map()
  for (let i = 0; i < classes.length; i++) classMap.set(classes[i], i)
  const eps = opts.eps == null ? 1e-15 : opts.eps
  let sum = 0
  for (let i = 0; i < n; i++) {
    const classIdx = classMap.get(yTrue[i])
    if (classIdx == null) throw new ValidationError(`logLoss: class "${yTrue[i]}" missing from classes`)
    let p = yProba[i * nClasses + classIdx]
    p = Math.max(eps, Math.min(1 - eps, p))
    sum -= _weightAt(weights, i) * Math.log(p)
  }
  return sum / _weightSum(weights, n)
}

function _binaryRocAuc(yTrue, yScore, opts = {}) {
  if (!yTrue || yTrue.length === 0) throw new ValidationError('rocAuc: yTrue must be non-empty')
  if (yTrue.length !== yScore.length) throw new ValidationError('rocAuc: length mismatch')
  const weights = _validateSampleWeight(opts.sampleWeight || opts.sample_weight, yTrue.length, 'rocAuc')
  const { labels } = _classInfo(yTrue, undefined, opts.classes)
  const present = Array.from(new Set(Array.from(yTrue))).sort((a, b) => a - b)
  if (present.length !== 2) {
    return _undefinedMetric('rocAuc', 'requires exactly 2 classes with positive support', opts)
  }
  const posLabel = _positiveLabel(labels.length ? Array.from(labels) : present, opts)
  const n = yTrue.length

  let wPos = 0, wNeg = 0
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(yScore[i])) throw new ValidationError('rocAuc: scores must be finite')
    if (yTrue[i] === posLabel) wPos += _weightAt(weights, i)
    else wNeg += _weightAt(weights, i)
  }
  if (wPos <= 0 || wNeg <= 0) {
    return _undefinedMetric('rocAuc', 'requires positive total weight for both classes', opts)
  }

  const order = Array.from({ length: n }, (_, i) => i)
  order.sort((a, b) => yScore[a] - yScore[b])

  let negBefore = 0
  let u = 0
  for (let start = 0; start < n;) {
    let end = start + 1
    const score = yScore[order[start]]
    while (end < n && yScore[order[end]] === score) end++

    let groupNeg = 0
    for (let j = start; j < end; j++) {
      const idx = order[j]
      if (yTrue[idx] !== posLabel) groupNeg += _weightAt(weights, idx)
    }
    for (let j = start; j < end; j++) {
      const idx = order[j]
      if (yTrue[idx] === posLabel) u += _weightAt(weights, idx) * (negBefore + groupNeg / 2)
    }
    negBefore += groupNeg
    start = end
  }

  return u / (wPos * wNeg)
}

function _scoreColumn(yScore, n, nClasses, col) {
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) out[i] = yScore[i * nClasses + col]
  return out
}

function _multiclassOvrAuc(yTrue, yScore, classes, opts) {
  const n = yTrue.length
  const nClasses = classes.length
  const values = []
  const supports = []
  for (let c = 0; c < nClasses; c++) {
    const label = classes[c]
    const binaryTruth = new Int32Array(n)
    let support = 0
    for (let i = 0; i < n; i++) {
      const w = _weightAt(opts._weights, i)
      if (yTrue[i] === label) {
        binaryTruth[i] = 1
        support += w
      }
    }
    const auc = _binaryRocAuc(binaryTruth, _scoreColumn(yScore, n, nClasses, c), {
      ...opts,
      classes: [0, 1],
      positiveLabel: 1
    })
    values.push(auc)
    supports.push(support)
  }
  if ((opts.average || 'macro') === 'weighted' || opts.average === 'macro_weighted') {
    return _weightedMean(values, supports)
  }
  return values.reduce((a, b) => a + b, 0) / values.length
}

function _multiclassOvoAuc(yTrue, yScore, classes, opts) {
  const n = yTrue.length
  const nClasses = classes.length
  const values = []
  const supports = []
  for (let a = 0; a < nClasses; a++) {
    for (let b = a + 1; b < nClasses; b++) {
      const labelA = classes[a]
      const labelB = classes[b]
      const truth = []
      const scoreA = []
      const scoreB = []
      const pairWeights = []
      let support = 0
      for (let i = 0; i < n; i++) {
        if (yTrue[i] !== labelA && yTrue[i] !== labelB) continue
        truth.push(yTrue[i])
        scoreA.push(yScore[i * nClasses + a])
        scoreB.push(yScore[i * nClasses + b])
        const w = _weightAt(opts._weights, i)
        pairWeights.push(w)
        support += w
      }
      const aucA = _binaryRocAuc(new Int32Array(truth), new Float64Array(scoreA), {
        ...opts,
        classes: [labelA, labelB],
        positiveLabel: labelA,
        sampleWeight: new Float64Array(pairWeights)
      })
      const aucB = _binaryRocAuc(new Int32Array(truth), new Float64Array(scoreB), {
        ...opts,
        classes: [labelA, labelB],
        positiveLabel: labelB,
        sampleWeight: new Float64Array(pairWeights)
      })
      values.push((aucA + aucB) / 2)
      supports.push(support)
    }
  }
  if ((opts.average || 'macro') === 'weighted' || opts.average === 'macro_weighted') {
    return _weightedMean(values, supports)
  }
  return values.reduce((a, b) => a + b, 0) / values.length
}

function rocAuc(yTrue, yScore, opts = {}) {
  if (!yTrue || yTrue.length === 0) throw new ValidationError('rocAuc: yTrue must be non-empty')
  const n = yTrue.length
  const weights = _validateSampleWeight(opts.sampleWeight || opts.sample_weight, n, 'rocAuc')
  const classes = _classesFromOpts(yTrue, opts)
  const nClasses = classes.length
  if (nClasses < 2) {
    return _undefinedMetric('rocAuc', 'requires at least 2 classes', opts)
  }
  if (yScore.length === n) {
    return _binaryRocAuc(yTrue, yScore, opts)
  }
  if (yScore.length !== n * nClasses) {
    throw new ValidationError(`rocAuc: score length (${yScore.length}) must equal n or n * classes (${n * nClasses})`)
  }
  for (let i = 0; i < yScore.length; i++) {
    if (!Number.isFinite(yScore[i])) throw new ValidationError('rocAuc: scores must be finite')
  }
  if (nClasses === 2 && (opts.multiClass || opts.multi_class || 'raise') === 'raise') {
    const posLabel = _positiveLabel(classes, opts)
    const posCol = classes.indexOf(posLabel)
    return _binaryRocAuc(yTrue, _scoreColumn(yScore, n, nClasses, posCol), { ...opts, classes })
  }
  const multiClass = opts.multiClass || opts.multi_class || 'raise'
  if (multiClass === 'raise') {
    throw new ValidationError('rocAuc: multiclass scores require multiClass="ovr" or "ovo"')
  }
  const average = opts.average || 'macro'
  if (average !== 'macro' && average !== 'weighted' && average !== 'macro_weighted') {
    throw new ValidationError('rocAuc: multiclass average must be "macro" or "weighted"')
  }
  const mcOpts = { ...opts, average, _weights: weights }
  if (multiClass === 'ovr') return _multiclassOvrAuc(yTrue, yScore, classes, mcOpts)
  if (multiClass === 'ovo') return _multiclassOvoAuc(yTrue, yScore, classes, mcOpts)
  throw new ValidationError(`rocAuc: unsupported multiClass "${multiClass}"`)
}

module.exports = {
  accuracy, r2Score, meanSquaredError, meanAbsoluteError,
  confusionMatrix, precisionScore, recallScore, f1Score,
  logLoss, rocAuc
}
