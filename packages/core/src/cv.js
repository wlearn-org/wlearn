import { ValidationError } from './errors.js'
import { makeLCG, shuffle } from './rng.js'
import { normalizeX, normalizeY } from './matrix.js'
import { accuracy, r2Score, meanSquaredError, meanAbsoluteError } from './metrics.js'

// --- Scorer registry ---

const SCORERS = {
  accuracy: (yTrue, yPred) => accuracy(yTrue, yPred),
  r2: (yTrue, yPred) => r2Score(yTrue, yPred),
  neg_mse: (yTrue, yPred) => -meanSquaredError(yTrue, yPred),
  neg_mae: (yTrue, yPred) => -meanAbsoluteError(yTrue, yPred),
}

export function getScorer(scoring) {
  if (typeof scoring === 'function') return scoring
  const fn = SCORERS[scoring]
  if (!fn) {
    throw new ValidationError(`Unknown scoring: "${scoring}". Available: ${Object.keys(SCORERS).join(', ')}`)
  }
  return fn
}

// --- Fold generators ---

export function kFold(n, k = 5, { shuffle: doShuffle = true, seed = 42 } = {}) {
  if (n < k) throw new ValidationError(`kFold: n (${n}) must be >= k (${k})`)
  if (k < 2) throw new ValidationError('kFold: k must be >= 2')

  const indices = Int32Array.from({ length: n }, (_, i) => i)
  if (doShuffle) {
    const rng = makeLCG(seed)
    shuffle(indices, rng)
  }

  const foldSize = Math.floor(n / k)
  const remainder = n % k
  const folds = []
  let offset = 0

  for (let f = 0; f < k; f++) {
    const size = foldSize + (f < remainder ? 1 : 0)
    const testIdx = indices.slice(offset, offset + size)
    const trainParts = []
    if (offset > 0) trainParts.push(indices.slice(0, offset))
    if (offset + size < n) trainParts.push(indices.slice(offset + size))
    const trainIdx = _concat(trainParts)
    folds.push({ train: trainIdx, test: testIdx })
    offset += size
  }
  return folds
}

export function stratifiedKFold(y, k = 5, { shuffle: doShuffle = true, seed = 42 } = {}) {
  const n = y.length
  if (n < k) throw new ValidationError(`stratifiedKFold: n (${n}) must be >= k (${k})`)
  if (k < 2) throw new ValidationError('stratifiedKFold: k must be >= 2')

  // Group indices by class
  const classMap = new Map()
  for (let i = 0; i < n; i++) {
    const label = y[i]
    if (!classMap.has(label)) classMap.set(label, [])
    classMap.get(label).push(i)
  }

  if (doShuffle) {
    const rng = makeLCG(seed)
    for (const indices of classMap.values()) {
      shuffle(indices, rng)
    }
  }

  // Assign each class's samples round-robin to folds
  const foldTests = Array.from({ length: k }, () => [])
  for (const indices of classMap.values()) {
    for (let i = 0; i < indices.length; i++) {
      foldTests[i % k].push(indices[i])
    }
  }

  const allIndices = Int32Array.from({ length: n }, (_, i) => i)
  const folds = []
  for (let f = 0; f < k; f++) {
    const testSet = new Set(foldTests[f])
    const test = new Int32Array(foldTests[f])
    const train = allIndices.filter(i => !testSet.has(i))
    folds.push({ train, test })
  }
  return folds
}

export function trainTestSplit(n, { testSize = 0.2, shuffle: doShuffle = true, seed = 42 } = {}) {
  if (n < 2) throw new ValidationError('trainTestSplit: n must be >= 2')
  const nTest = Math.max(1, Math.round(n * testSize))
  const nTrain = n - nTest
  if (nTrain < 1) throw new ValidationError('trainTestSplit: testSize too large')

  const indices = Int32Array.from({ length: n }, (_, i) => i)
  if (doShuffle) {
    const rng = makeLCG(seed)
    shuffle(indices, rng)
  }
  return {
    train: indices.slice(0, nTrain),
    test: indices.slice(nTrain),
  }
}

// --- CV runner ---

export async function crossValScore(EstimatorClass, X, y, {
  cv = 5,
  scoring = 'accuracy',
  seed = 42,
  params = {},
} = {}) {
  const Xn = normalizeX(X)
  const yn = normalizeY(y)
  const scorerFn = getScorer(scoring)

  // Generate folds
  let folds
  if (Array.isArray(cv)) {
    folds = cv
  } else {
    folds = stratifiedKFold(yn, cv, { shuffle: true, seed })
  }

  const scores = new Float64Array(folds.length)

  for (let f = 0; f < folds.length; f++) {
    const { train, test } = folds[f]
    const Xtrain = _subsetX(Xn, train)
    const ytrain = _subsetY(yn, train)
    const Xtest = _subsetX(Xn, test)
    const ytest = _subsetY(yn, test)

    const model = await EstimatorClass.create(params)
    try {
      model.fit(Xtrain, ytrain)
      const preds = model.predict(Xtest)
      scores[f] = scorerFn(ytest, preds)
    } finally {
      model.dispose()
    }
  }
  return scores
}

// --- Internal helpers ---

function _concat(parts) {
  let total = 0
  for (const p of parts) total += p.length
  const out = new Int32Array(total)
  let off = 0
  for (const p of parts) {
    out.set(p, off)
    off += p.length
  }
  return out
}

function _subsetX(X, indices) {
  const { data, cols } = X
  const rows = indices.length
  const out = new Float64Array(rows * cols)
  for (let i = 0; i < rows; i++) {
    const srcOff = indices[i] * cols
    out.set(data.subarray(srcOff, srcOff + cols), i * cols)
  }
  return { data: out, rows, cols }
}

function _subsetY(y, indices) {
  const out = new (y.constructor)(indices.length)
  for (let i = 0; i < indices.length; i++) {
    out[i] = y[indices[i]]
  }
  return out
}
