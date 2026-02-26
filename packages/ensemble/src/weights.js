import { ValidationError } from '@wlearn/core'

/**
 * Project vector onto the probability simplex {w: w >= 0, sum(w) = 1}.
 * O(n log n) algorithm from Duchi et al. 2008.
 *
 * @param {Float64Array} v - input vector
 * @returns {Float64Array} - projected vector
 */
export function projectSimplex(v) {
  const n = v.length
  if (n === 0) return new Float64Array(0)
  if (n === 1) return new Float64Array([1.0])

  // Sort descending
  const u = new Float64Array(v)
  u.sort()
  u.reverse()

  const cssv = new Float64Array(n)
  cssv[0] = u[0]
  for (let i = 1; i < n; i++) {
    cssv[i] = cssv[i - 1] + u[i]
  }

  let rho = -1
  for (let i = 0; i < n; i++) {
    if (u[i] * (i + 1) > cssv[i] - 1) {
      rho = i
    }
  }

  if (rho < 0) {
    // Fallback: uniform
    const out = new Float64Array(n)
    out.fill(1.0 / n)
    return out
  }

  const theta = (cssv[rho] - 1.0) / (rho + 1.0)
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    out[i] = Math.max(v[i] - theta, 0.0)
  }
  return out
}

/**
 * Optimize ensemble weights via projected gradient descent on the simplex.
 *
 * Classification: minimizes negative log-loss.
 * Regression: minimizes MSE.
 *
 * @param {Float64Array[]} oofPredictions - per-model OOF predictions
 * @param {TypedArray} yTrue - true labels
 * @param {Float64Array} initWeights - initial weights (e.g. from Caruana)
 * @param {object} opts
 * @returns {Float64Array} - optimized weights (>= 0, sum = 1)
 */
export function optimizeWeights(oofPredictions, yTrue, initWeights, {
  task = 'classification',
  lr = 0.05,
  nIter = 100,
} = {}) {
  const nModels = oofPredictions.length
  const n = yTrue.length

  if (nModels === 0) {
    throw new ValidationError('optimizeWeights: need at least 1 model')
  }
  if (nModels === 1) {
    return new Float64Array([1.0])
  }

  const w = new Float64Array(initWeights)
  const eps = 1e-15

  if (task === 'classification') {
    const predLen = oofPredictions[0].length
    const nc = predLen / n
    if (nc !== Math.floor(nc)) {
      throw new ValidationError('optimizeWeights: prediction length must be divisible by n')
    }

    for (let iter = 0; iter < nIter; iter++) {
      const grad = new Float64Array(nModels)

      for (let i = 0; i < n; i++) {
        const c = yTrue[i] | 0
        // Ensemble probability for true class
        let pTrue = 0
        for (let m = 0; m < nModels; m++) {
          pTrue += w[m] * oofPredictions[m][i * nc + c]
        }
        pTrue = Math.max(pTrue, eps)

        for (let m = 0; m < nModels; m++) {
          grad[m] -= oofPredictions[m][i * nc + c] / pTrue
        }
      }

      // Normalize gradient
      for (let m = 0; m < nModels; m++) {
        grad[m] /= n
      }

      // Gradient step + project
      for (let m = 0; m < nModels; m++) {
        w[m] -= lr * grad[m]
      }
      const proj = projectSimplex(w)
      for (let m = 0; m < nModels; m++) w[m] = proj[m]
    }
  } else {
    // Regression: minimize MSE
    for (let iter = 0; iter < nIter; iter++) {
      const grad = new Float64Array(nModels)

      for (let i = 0; i < n; i++) {
        let pred = 0
        for (let m = 0; m < nModels; m++) {
          pred += w[m] * oofPredictions[m][i]
        }
        const residual = yTrue[i] - pred
        for (let m = 0; m < nModels; m++) {
          grad[m] -= 2 * residual * oofPredictions[m][i]
        }
      }

      for (let m = 0; m < nModels; m++) {
        grad[m] /= n
      }

      for (let m = 0; m < nModels; m++) {
        w[m] -= lr * grad[m]
      }
      const proj = projectSimplex(w)
      for (let m = 0; m < nModels; m++) w[m] = proj[m]
    }
  }

  return w
}
