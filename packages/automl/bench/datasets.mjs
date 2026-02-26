/**
 * Synthetic dataset generators for benchmarking.
 * Friedman 1-3 (regression), moons + hastie (classification).
 * No external dependencies -- uses inline LCG PRNG + Box-Muller.
 */

// --- Seeded PRNG (LCG matching wlearn core) ---

function makeLCG(seed) {
  let s = seed >>> 0
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return s / 0x100000000
  }
}

function gaussianPair(rng) {
  const u1 = rng() || 1e-15
  const u2 = rng()
  const r = Math.sqrt(-2 * Math.log(u1))
  const theta = 2 * Math.PI * u2
  return [r * Math.cos(theta), r * Math.sin(theta)]
}

function gaussian(rng) {
  return gaussianPair(rng)[0]
}

// --- Regression datasets ---

/**
 * Friedman #1: y = 10*sin(pi*x0*x1) + 20*(x2-0.5)^2 + 10*x3 + 5*x4 + noise
 * Features 5-9 are independent noise.
 */
export function makeFriedman1(nSamples, { nFeatures = 10, noise = 1.0, seed = 42 } = {}) {
  if (nFeatures < 5) nFeatures = 5
  const rng = makeLCG(seed)
  const X = new Float64Array(nSamples * nFeatures)
  const y = new Float64Array(nSamples)

  for (let i = 0; i < nSamples; i++) {
    for (let j = 0; j < nFeatures; j++) {
      X[i * nFeatures + j] = rng()
    }
    const x0 = X[i * nFeatures + 0]
    const x1 = X[i * nFeatures + 1]
    const x2 = X[i * nFeatures + 2]
    const x3 = X[i * nFeatures + 3]
    const x4 = X[i * nFeatures + 4]

    y[i] = 10 * Math.sin(Math.PI * x0 * x1)
         + 20 * (x2 - 0.5) ** 2
         + 10 * x3
         + 5 * x4
         + noise * gaussian(rng)
  }

  return { X: { data: X, rows: nSamples, cols: nFeatures }, y }
}

/**
 * Friedman #2: y = sqrt(x0^2 + (x1*x2 - 1/(x1*x3))^2) + noise
 * x0 in [0,100], x1 in [40pi, 560pi], x2 in [0,1], x3 in [1,11]
 */
export function makeFriedman2(nSamples, { noise = 1.0, seed = 42 } = {}) {
  const rng = makeLCG(seed)
  const nFeatures = 4
  const X = new Float64Array(nSamples * nFeatures)
  const y = new Float64Array(nSamples)

  for (let i = 0; i < nSamples; i++) {
    const x0 = rng() * 100
    const x1 = rng() * (560 - 40) * Math.PI + 40 * Math.PI
    const x2 = rng()
    const x3 = rng() * 10 + 1

    X[i * 4 + 0] = x0
    X[i * 4 + 1] = x1
    X[i * 4 + 2] = x2
    X[i * 4 + 3] = x3

    const inner = x1 * x2 - 1 / (x1 * x3)
    y[i] = Math.sqrt(x0 ** 2 + inner ** 2) + noise * gaussian(rng)
  }

  return { X: { data: X, rows: nSamples, cols: nFeatures }, y }
}

/**
 * Friedman #3: y = atan((x1*x2 - 1/(x1*x3)) / x0) + noise
 * Same feature ranges as Friedman #2.
 */
export function makeFriedman3(nSamples, { noise = 1.0, seed = 42 } = {}) {
  const rng = makeLCG(seed)
  const nFeatures = 4
  const X = new Float64Array(nSamples * nFeatures)
  const y = new Float64Array(nSamples)

  for (let i = 0; i < nSamples; i++) {
    const x0 = rng() * 100
    const x1 = rng() * (560 - 40) * Math.PI + 40 * Math.PI
    const x2 = rng()
    const x3 = rng() * 10 + 1

    X[i * 4 + 0] = x0
    X[i * 4 + 1] = x1
    X[i * 4 + 2] = x2
    X[i * 4 + 3] = x3

    const inner = x1 * x2 - 1 / (x1 * x3)
    y[i] = Math.atan(inner / (x0 || 1e-15)) + noise * gaussian(rng)
  }

  return { X: { data: X, rows: nSamples, cols: nFeatures }, y }
}

// --- Classification datasets ---

/**
 * Two interleaving half circles (moons).
 * Binary classification, 2 features.
 */
export function makeMoons(nSamples, { noise = 0.3, seed = 42 } = {}) {
  const rng = makeLCG(seed)
  const nFeatures = 2
  const nHalf = Math.floor(nSamples / 2)
  const nOther = nSamples - nHalf
  const X = new Float64Array(nSamples * nFeatures)
  const y = new Int32Array(nSamples)

  // Upper moon
  for (let i = 0; i < nHalf; i++) {
    const angle = Math.PI * i / nHalf
    X[i * 2 + 0] = Math.cos(angle) + noise * gaussian(rng)
    X[i * 2 + 1] = Math.sin(angle) + noise * gaussian(rng)
    y[i] = 0
  }

  // Lower moon
  for (let i = 0; i < nOther; i++) {
    const idx = nHalf + i
    const angle = Math.PI * i / nOther
    X[idx * 2 + 0] = 1 - Math.cos(angle) + noise * gaussian(rng)
    X[idx * 2 + 1] = 1 - Math.sin(angle) - 0.5 + noise * gaussian(rng)
    y[idx] = 1
  }

  return { X: { data: X, rows: nSamples, cols: nFeatures }, y }
}

/**
 * Hastie et al. "Elements of Statistical Learning" 10.2.
 * 10 standard normal features, y = 1 if sum(x_i^2) > chi2_median(10), else 0.
 * chi2 median for df=10 is approximately 9.3418.
 */
export function makeHastie(nSamples, { seed = 42 } = {}) {
  const rng = makeLCG(seed)
  const nFeatures = 10
  const X = new Float64Array(nSamples * nFeatures)
  const y = new Int32Array(nSamples)
  const chi2Median = 9.3418

  for (let i = 0; i < nSamples; i++) {
    let sumSq = 0
    for (let j = 0; j < nFeatures; j++) {
      const val = gaussian(rng)
      X[i * nFeatures + j] = val
      sumSq += val * val
    }
    y[i] = sumSq > chi2Median ? 1 : 0
  }

  return { X: { data: X, rows: nSamples, cols: nFeatures }, y }
}

/**
 * Train/test split.
 */
export function trainTestSplit(X, y, { testSize = 0.2, seed = 42 } = {}) {
  const n = X.rows
  const nTest = Math.round(n * testSize)
  const nTrain = n - nTest

  // Shuffle indices
  const indices = Array.from({ length: n }, (_, i) => i)
  const rng = makeLCG(seed + 999)
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    const tmp = indices[i]
    indices[i] = indices[j]
    indices[j] = tmp
  }

  const trainIdx = indices.slice(0, nTrain)
  const testIdx = indices.slice(nTrain)

  const subX = (idx) => {
    const rows = idx.length
    const data = new Float64Array(rows * X.cols)
    for (let i = 0; i < rows; i++) {
      const srcOff = idx[i] * X.cols
      data.set(X.data.subarray(srcOff, srcOff + X.cols), i * X.cols)
    }
    return { data, rows, cols: X.cols }
  }

  const subY = (idx) => {
    const out = new (y.constructor)(idx.length)
    for (let i = 0; i < idx.length; i++) out[i] = y[idx[i]]
    return out
  }

  return {
    Xtrain: subX(trainIdx), ytrain: subY(trainIdx),
    Xtest: subX(testIdx), ytest: subY(testIdx),
  }
}
