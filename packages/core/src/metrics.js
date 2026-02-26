import { ValidationError } from './errors.js'

// --- Internal helpers ---

function _validatePair(yTrue, yPred, name) {
  if (!yTrue || !yPred || yTrue.length === 0 || yPred.length === 0) {
    throw new ValidationError(`${name}: inputs must be non-empty`)
  }
  if (yTrue.length !== yPred.length) {
    throw new ValidationError(`${name}: length mismatch (${yTrue.length} vs ${yPred.length})`)
  }
}

function _classInfo(yTrue, yPred) {
  const labelSet = new Set()
  for (let i = 0; i < yTrue.length; i++) labelSet.add(yTrue[i])
  if (yPred) {
    for (let i = 0; i < yPred.length; i++) labelSet.add(yPred[i])
  }
  const labels = new Int32Array([...labelSet].sort((a, b) => a - b))
  const labelMap = new Map()
  for (let i = 0; i < labels.length; i++) labelMap.set(labels[i], i)
  return { labels, labelMap, nClasses: labels.length }
}

function _buildCM(yTrue, yPred, labelMap, nClasses) {
  const cm = new Int32Array(nClasses * nClasses)
  for (let i = 0; i < yTrue.length; i++) {
    const t = labelMap.get(yTrue[i])
    const p = labelMap.get(yPred[i])
    cm[t * nClasses + p]++
  }
  return cm
}

function _classCounts(cm, nClasses) {
  const tp = new Float64Array(nClasses)
  const fp = new Float64Array(nClasses)
  const fn = new Float64Array(nClasses)
  for (let c = 0; c < nClasses; c++) {
    tp[c] = cm[c * nClasses + c]
    for (let j = 0; j < nClasses; j++) {
      if (j !== c) {
        fp[c] += cm[j * nClasses + c]
        fn[c] += cm[c * nClasses + j]
      }
    }
  }
  return { tp, fp, fn }
}

// --- Exports ---

export function accuracy(yTrue, yPred) {
  _validatePair(yTrue, yPred, 'accuracy')
  let correct = 0
  for (let i = 0; i < yTrue.length; i++) {
    if (yTrue[i] === yPred[i]) correct++
  }
  return correct / yTrue.length
}

export function r2Score(yTrue, yPred) {
  _validatePair(yTrue, yPred, 'r2Score')
  const n = yTrue.length
  let mean = 0
  for (let i = 0; i < n; i++) mean += yTrue[i]
  mean /= n
  let ssTot = 0, ssRes = 0
  for (let i = 0; i < n; i++) {
    const d = yTrue[i] - mean
    ssTot += d * d
    const r = yTrue[i] - yPred[i]
    ssRes += r * r
  }
  if (ssTot === 0) return 0
  return 1 - ssRes / ssTot
}

export function meanSquaredError(yTrue, yPred) {
  _validatePair(yTrue, yPred, 'meanSquaredError')
  let sum = 0
  for (let i = 0; i < yTrue.length; i++) {
    const d = yTrue[i] - yPred[i]
    sum += d * d
  }
  return sum / yTrue.length
}

export function meanAbsoluteError(yTrue, yPred) {
  _validatePair(yTrue, yPred, 'meanAbsoluteError')
  let sum = 0
  for (let i = 0; i < yTrue.length; i++) {
    sum += Math.abs(yTrue[i] - yPred[i])
  }
  return sum / yTrue.length
}

export function confusionMatrix(yTrue, yPred) {
  _validatePair(yTrue, yPred, 'confusionMatrix')
  const { labels, labelMap, nClasses } = _classInfo(yTrue, yPred)
  const matrix = _buildCM(yTrue, yPred, labelMap, nClasses)
  return { matrix, labels }
}

function _resolveAverage(average, nClasses, labels) {
  if (average === 'binary') {
    if (nClasses > 2) {
      throw new ValidationError('average="binary" requires exactly 2 classes')
    }
    return 'binary'
  }
  return average || 'binary'
}

export function precisionScore(yTrue, yPred, { average = 'binary' } = {}) {
  _validatePair(yTrue, yPred, 'precisionScore')
  const { labels, labelMap, nClasses } = _classInfo(yTrue, yPred)
  const cm = _buildCM(yTrue, yPred, labelMap, nClasses)
  const { tp, fp } = _classCounts(cm, nClasses)
  const avg = _resolveAverage(average, nClasses, labels)

  if (avg === 'binary') {
    const posIdx = nClasses - 1
    const denom = tp[posIdx] + fp[posIdx]
    return denom === 0 ? 0 : tp[posIdx] / denom
  }
  if (avg === 'micro') {
    let tpSum = 0, fpSum = 0
    for (let c = 0; c < nClasses; c++) { tpSum += tp[c]; fpSum += fp[c] }
    return tpSum + fpSum === 0 ? 0 : tpSum / (tpSum + fpSum)
  }
  // macro
  let sum = 0
  for (let c = 0; c < nClasses; c++) {
    const denom = tp[c] + fp[c]
    sum += denom === 0 ? 0 : tp[c] / denom
  }
  return sum / nClasses
}

export function recallScore(yTrue, yPred, { average = 'binary' } = {}) {
  _validatePair(yTrue, yPred, 'recallScore')
  const { labels, labelMap, nClasses } = _classInfo(yTrue, yPred)
  const cm = _buildCM(yTrue, yPred, labelMap, nClasses)
  const { tp, fn } = _classCounts(cm, nClasses)
  const avg = _resolveAverage(average, nClasses, labels)

  if (avg === 'binary') {
    const posIdx = nClasses - 1
    const denom = tp[posIdx] + fn[posIdx]
    return denom === 0 ? 0 : tp[posIdx] / denom
  }
  if (avg === 'micro') {
    let tpSum = 0, fnSum = 0
    for (let c = 0; c < nClasses; c++) { tpSum += tp[c]; fnSum += fn[c] }
    return tpSum + fnSum === 0 ? 0 : tpSum / (tpSum + fnSum)
  }
  // macro
  let sum = 0
  for (let c = 0; c < nClasses; c++) {
    const denom = tp[c] + fn[c]
    sum += denom === 0 ? 0 : tp[c] / denom
  }
  return sum / nClasses
}

export function f1Score(yTrue, yPred, { average = 'binary' } = {}) {
  _validatePair(yTrue, yPred, 'f1Score')
  const { labels, labelMap, nClasses } = _classInfo(yTrue, yPred)
  const cm = _buildCM(yTrue, yPred, labelMap, nClasses)
  const { tp, fp, fn } = _classCounts(cm, nClasses)
  const avg = _resolveAverage(average, nClasses, labels)

  function _f1(tpC, fpC, fnC) {
    const p = tpC + fpC === 0 ? 0 : tpC / (tpC + fpC)
    const r = tpC + fnC === 0 ? 0 : tpC / (tpC + fnC)
    return p + r === 0 ? 0 : 2 * p * r / (p + r)
  }

  if (avg === 'binary') {
    const posIdx = nClasses - 1
    return _f1(tp[posIdx], fp[posIdx], fn[posIdx])
  }
  if (avg === 'micro') {
    let tpSum = 0, fpSum = 0, fnSum = 0
    for (let c = 0; c < nClasses; c++) { tpSum += tp[c]; fpSum += fp[c]; fnSum += fn[c] }
    return _f1(tpSum, fpSum, fnSum)
  }
  // macro
  let sum = 0
  for (let c = 0; c < nClasses; c++) sum += _f1(tp[c], fp[c], fn[c])
  return sum / nClasses
}

export function logLoss(yTrue, yProba, { nClasses, eps = 1e-15 } = {}) {
  if (!yTrue || yTrue.length === 0) {
    throw new ValidationError('logLoss: yTrue must be non-empty')
  }
  const n = yTrue.length
  if (!nClasses) {
    const s = new Set()
    for (let i = 0; i < n; i++) s.add(yTrue[i])
    nClasses = s.size
  }
  if (yProba.length !== n * nClasses) {
    throw new ValidationError(`logLoss: yProba length (${yProba.length}) must be n * nClasses (${n * nClasses})`)
  }
  const { labelMap } = _classInfo(yTrue)
  let sum = 0
  for (let i = 0; i < n; i++) {
    const classIdx = labelMap.get(yTrue[i])
    let p = yProba[i * nClasses + classIdx]
    p = Math.max(eps, Math.min(1 - eps, p))
    sum -= Math.log(p)
  }
  return sum / n
}

export function rocAuc(yTrue, yProba) {
  if (!yTrue || yTrue.length === 0) {
    throw new ValidationError('rocAuc: yTrue must be non-empty')
  }
  if (yTrue.length !== yProba.length) {
    throw new ValidationError('rocAuc: length mismatch')
  }
  const { labels } = _classInfo(yTrue)
  if (labels.length !== 2) {
    throw new ValidationError('rocAuc: requires exactly 2 classes')
  }
  const posLabel = labels[labels.length - 1]
  const n = yTrue.length

  // Sort by descending score
  const indices = Array.from({ length: n }, (_, i) => i)
  indices.sort((a, b) => yProba[b] - yProba[a])

  let nPos = 0, nNeg = 0
  for (let i = 0; i < n; i++) {
    if (yTrue[i] === posLabel) nPos++
    else nNeg++
  }
  if (nPos === 0 || nNeg === 0) return 0

  // Trapezoidal rule on ROC curve (FPR = x-axis, TPR = y-axis)
  let tp = 0, fp = 0, prevTpr = 0, prevFpr = 0, auc = 0

  for (let i = 0; i < n; i++) {
    const idx = indices[i]
    if (yTrue[idx] === posLabel) tp++
    else fp++
    const tpr = tp / nPos
    const fpr = fp / nNeg
    auc += (fpr - prevFpr) * (tpr + prevTpr) / 2
    prevTpr = tpr
    prevFpr = fpr
  }
  return auc
}
