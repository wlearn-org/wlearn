const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const {
  accuracy,
  meanAbsoluteError,
  meanSquaredError,
  precisionScore,
  recallScore,
  f1Score,
  rocAuc
} = require('../src/metrics.js')
const { makeLCG } = require('../src/rng.js')

function approx(actual, expected, tol = 1e-10) {
  assert(Math.abs(actual - expected) <= tol,
    `expected ~${expected}, got ${actual} (diff=${Math.abs(actual - expected)})`)
}

function randint(rng, hi) {
  return Math.floor(rng() * hi)
}

function repeatInt32(values, weights) {
  const out = []
  for (let i = 0; i < values.length; i++) {
    for (let j = 0; j < weights[i]; j++) out.push(values[i])
  }
  return new Int32Array(out)
}

function repeatFloat(values, weights) {
  const out = []
  for (let i = 0; i < values.length; i++) {
    for (let j = 0; j < weights[i]; j++) out.push(values[i])
  }
  return new Float64Array(out)
}

describe('metric property probes', () => {
  it('sample weights match duplicated rows for deterministic classification sweeps', () => {
    for (let seed = 1; seed <= 80; seed++) {
      const rng = makeLCG(seed)
      const n = 6 + randint(rng, 20)
      const yTrue = new Int32Array(n)
      const yPred = new Int32Array(n)
      const weights = new Float64Array(n)
      const intWeights = new Int32Array(n)
      for (let i = 0; i < n; i++) {
        yTrue[i] = randint(rng, 3)
        yPred[i] = randint(rng, 3)
        intWeights[i] = 1 + randint(rng, 3)
        weights[i] = intWeights[i]
      }
      const yTrueDup = repeatInt32(yTrue, intWeights)
      const yPredDup = repeatInt32(yPred, intWeights)

      approx(accuracy(yTrue, yPred, { sampleWeight: weights }), accuracy(yTrueDup, yPredDup))
      for (const average of ['micro', 'macro', 'weighted']) {
        approx(
          precisionScore(yTrue, yPred, { average, sampleWeight: weights }),
          precisionScore(yTrueDup, yPredDup, { average })
        )
        approx(
          recallScore(yTrue, yPred, { average, sampleWeight: weights }),
          recallScore(yTrueDup, yPredDup, { average })
        )
        approx(
          f1Score(yTrue, yPred, { average, sampleWeight: weights }),
          f1Score(yTrueDup, yPredDup, { average })
        )
      }
    }
  })

  it('sample weights match duplicated rows for deterministic regression sweeps', () => {
    for (let seed = 101; seed <= 180; seed++) {
      const rng = makeLCG(seed)
      const n = 5 + randint(rng, 20)
      const yTrue = new Float64Array(n)
      const yPred = new Float64Array(n)
      const weights = new Float64Array(n)
      const intWeights = new Int32Array(n)
      for (let i = 0; i < n; i++) {
        yTrue[i] = randint(rng, 11) - 5 + rng()
        yPred[i] = randint(rng, 11) - 5 + rng()
        intWeights[i] = 1 + randint(rng, 4)
        weights[i] = intWeights[i]
      }
      const yTrueDup = repeatFloat(yTrue, intWeights)
      const yPredDup = repeatFloat(yPred, intWeights)

      approx(meanSquaredError(yTrue, yPred, { sampleWeight: weights }), meanSquaredError(yTrueDup, yPredDup))
      approx(meanAbsoluteError(yTrue, yPred, { sampleWeight: weights }), meanAbsoluteError(yTrueDup, yPredDup))
    }
  })

  it('AUC is invariant to positive affine score transforms and reverses under negation', () => {
    for (let seed = 201; seed <= 260; seed++) {
      const rng = makeLCG(seed)
      const n = 12 + randint(rng, 20)
      const y = new Int32Array(n)
      const score = new Float64Array(n)
      const transformed = new Float64Array(n)
      const reversed = new Float64Array(n)
      for (let i = 0; i < n; i++) {
        y[i] = i % 2
        score[i] = rng() + i * 1e-8
        transformed[i] = 3.5 + 7 * score[i]
        reversed[i] = -score[i]
      }

      const auc = rocAuc(y, score)
      approx(rocAuc(y, transformed), auc)
      approx(auc + rocAuc(y, reversed), 1)
    }
  })
})
