const { subsetRows, subsetLabels } = require('./matrix.js')
const { ValidationError } = require('./errors.js')
const { makeLCG, shuffle } = require('./rng.js')
const { normalizeX, normalizeY } = require('./matrix.js')
const { getScorer, scoreEstimator } = require('./measure.js')
const { inferTaskKind, taskParams, validateEstimatorTask } = require('./task.js')

// --- Fold generators ---

function kFold(n, k = 5, { shuffle: doShuffle = true, seed = 42 } = {}) {
  if (!Number.isInteger(n) || n < 2) throw new ValidationError('kFold: n must be an integer >= 2')
  if (!Number.isInteger(k) || k < 2) throw new ValidationError('kFold: k must be an integer >= 2')
  if (n < k) throw new ValidationError(`kFold: n (${n}) must be >= k (${k})`)

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

function stratifiedKFold(y, k = 5, { shuffle: doShuffle = true, seed = 42 } = {}) {
  const n = y.length
  if (!Number.isInteger(k) || k < 2) throw new ValidationError('stratifiedKFold: k must be an integer >= 2')
  if (n < k) throw new ValidationError(`stratifiedKFold: n (${n}) must be >= k (${k})`)

  // Group indices by class
  const classMap = new Map()
  for (let i = 0; i < n; i++) {
    const label = y[i]
    if (!classMap.has(label)) classMap.set(label, [])
    classMap.get(label).push(i)
  }
  for (const [label, indices] of classMap.entries()) {
    if (indices.length < k) {
      throw new ValidationError(`stratifiedKFold: class "${label}" has only ${indices.length} samples, less than k (${k})`)
    }
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

function trainTestSplit(n, { testSize = 0.2, shuffle: doShuffle = true, seed = 42 } = {}) {
  if (!Number.isInteger(n) || n < 2) throw new ValidationError('trainTestSplit: n must be an integer >= 2')
  if (typeof testSize !== 'number' || testSize <= 0 || testSize >= 1) {
    throw new ValidationError('trainTestSplit: testSize must be in (0, 1)')
  }
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

async function crossValScore(EstimatorClass, X, y, {
  cv = 5,
  scoring = 'accuracy',
  seed = 42,
  params = {},
  task,
} = {}) {
  const Xn = normalizeX(X)
  const yn = normalizeY(y)
  const scorerFn = getScorer(scoring)

  const resolvedTask = task || params.task || inferTaskKind(yn)
  const { resolveCv } = require('./resampling.js')
  const folds = resolveCv(cv, yn, { task: resolvedTask, seed })

  const scores = new Float64Array(folds.length)

  for (let f = 0; f < folds.length; f++) {
    const { train, test } = folds[f]
    const Xtrain = subsetRows(Xn, train)
    const ytrain = subsetLabels(yn, train)
    const Xtest = subsetRows(Xn, test)
    const ytest = subsetLabels(yn, test)

    const model = await EstimatorClass.create(taskParams(params, resolvedTask))
    let operationError = null
    try {
      await model.fit(Xtrain, ytrain)
      validateEstimatorTask(model, resolvedTask)
      scores[f] = await scoreEstimator(model, Xtest, ytest, scorerFn)
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



module.exports = { kFold, stratifiedKFold, trainTestSplit, crossValScore, getScorer }
