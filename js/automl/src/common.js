const { makeLCG } = require('@wlearn/core')
const { makeCandidateId, seedFor } = require('./candidate.js')

const { round } = Math

/**
 * Detect task type from labels.
 */
function detectTask(y) {
  if (y instanceof Int32Array) return 'classification'
  const unique = new Set()
  for (let i = 0; i < y.length; i++) {
    if (y[i] !== round(y[i])) return 'regression'
    unique.add(y[i])
  }
  return unique.size <= 20 ? 'classification' : 'regression'
}

/**
 * High-resolution timer.
 */
function now() {
  if (typeof performance !== 'undefined') return performance.now()
  return Date.now()
}

/**
 * Partial Fisher-Yates: shuffle only first k positions of indices array.
 * O(k) time, mutates indices in-place. Returns indices subarray [0..k-1].
 */
function partialShuffle(indices, k, rng) {
  const n = indices.length
  const m = Math.min(k, n)
  for (let i = 0; i < m; i++) {
    const j = i + ((rng() * (n - i)) | 0)
    const tmp = indices[i]
    indices[i] = indices[j]
    indices[j] = tmp
  }
  return indices.subarray ? indices.subarray(0, m) : indices.slice(0, m)
}

/**
 * Map scoring name to greaterIsBetter.
 * All built-in scorers are greater-is-better (neg_mse, neg_mae are negated).
 * Custom functions default to true.
 */
function scorerGreaterIsBetter(scoring) {
  if (typeof scoring === 'function') return true
  switch (scoring) {
    case 'accuracy':
    case 'r2':
    case 'neg_mse':
    case 'neg_mae':
      return true
    default:
      return true
  }
}

module.exports = { detectTask, now, makeCandidateId, seedFor, partialShuffle, scorerGreaterIsBetter }
