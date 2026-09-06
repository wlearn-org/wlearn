const { ValidationError } = require('@wlearn/core')
const {
  normalizeClassOrder,
  validateProbabilityOutput,
  validateRegressionOutput,
} = require('./class-order.js')

/**
 * Project vector onto the probability simplex {w: w >= 0, sum(w) = 1}.
 * O(n log n) algorithm from Duchi et al. 2008.
 */
function projectSimplex(v) {
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

// Projection alone does not make a gradient step safe: regression units and
// tiny class probabilities can make a fixed step overshoot. Armijo backtracking
// keeps the previous feasible weights if no finite descent step is found.
function projectedStep(w, grad, lr, loss) {
  const initial = loss(w)
  for (let attempt = 0, step = lr; attempt < 40; attempt++, step *= 0.5) {
    const proposal = projectSimplex(Float64Array.from(w, (value, i) => value - step * grad[i]))
    let direction = 0
    for (let i = 0; i < w.length; i++) direction += grad[i] * (proposal[i] - w[i])
    const value = loss(proposal)
    // Conservative sufficient decrease avoids nearly undamped oscillation at
    // the stability boundary of a scaled quadratic objective.
    if (Number.isFinite(value) && value <= initial + 0.5 * Math.min(direction, 0)) return proposal
  }
  return w
}

/**
 * Optimize ensemble weights via projected gradient descent on the simplex.
 */
function optimizeWeights(oofPredictions, yTrue, initWeights, {
  task = 'classification',
  lr = 0.05,
  nIter = 100,
  classes,
} = {}) {
  const nModels = oofPredictions.length
  const n = yTrue.length

  if (nModels === 0) {
    throw new ValidationError('optimizeWeights: need at least 1 model')
  }

  const w = projectSimplex(new Float64Array(initWeights))
  const eps = 1e-15

  if (task === 'classification') {
    const predLen = oofPredictions[0].length
    const nc = predLen / n
    if (nc !== Math.floor(nc)) {
      throw new ValidationError('optimizeWeights: prediction length must be divisible by n')
    }
    const source = classes == null
      ? [...new Set(Array.from(yTrue))].sort((a, b) => a - b)
      : classes
    const labels = normalizeClassOrder(source, nc, 'optimizeWeights')
    const classColumns = new Map(
      Array.from(labels, (label, index) => [label, index])
    )
    const yColumns = new Int32Array(n)
    for (let index = 0; index < n; index++) {
      const column = classColumns.get(yTrue[index])
      if (column == null) {
        throw new ValidationError(
          `optimizeWeights: class "${yTrue[index]}" is missing from classes`
        )
      }
      yColumns[index] = column
    }
    for (let index = 0; index < oofPredictions.length; index++) {
      validateProbabilityOutput(
        oofPredictions[index], n, nc,
        `optimizeWeights: oofPredictions[${index}]`
      )
    }
    if (nModels === 1) return new Float64Array([1.0])

    const loss = weights => {
      let value = 0
      for (let i = 0; i < n; i++) {
        let p = 0
        for (let m = 0; m < nModels; m++) p += weights[m] * oofPredictions[m][i * nc + yColumns[i]]
        value -= Math.log(Math.max(p, eps))
      }
      return value / n
    }

    for (let iter = 0; iter < nIter; iter++) {
      const grad = new Float64Array(nModels)

      for (let i = 0; i < n; i++) {
        const c = yColumns[i]
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

      const proj = projectedStep(w, grad, lr, loss)
      for (let m = 0; m < nModels; m++) w[m] = proj[m]
    }
  } else {
    // Regression: minimize MSE
    for (let index = 0; index < oofPredictions.length; index++) {
      validateRegressionOutput(
        oofPredictions[index], n,
        `optimizeWeights: oofPredictions[${index}]`
      )
    }
    if (nModels === 1) return new Float64Array([1.0])
    const loss = weights => {
      let value = 0
      for (let i = 0; i < n; i++) {
        let p = 0
        for (let m = 0; m < nModels; m++) p += weights[m] * oofPredictions[m][i]
        const residual = yTrue[i] - p
        value += residual * residual
      }
      return value / n
    }
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

      const proj = projectedStep(w, grad, lr, loss)
      for (let m = 0; m < nModels; m++) w[m] = proj[m]
    }
  }

  return w
}

module.exports = { optimizeWeights, projectSimplex }
