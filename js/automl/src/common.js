const { inferTaskKind: detectTask, getScorer } = require('@wlearn/core')
const { makeCandidateId, seedFor } = require('./candidate.js')

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

function scorerGreaterIsBetter(scoring) {
  return getScorer(scoring).direction !== 'minimize'
}

module.exports = { detectTask, now, makeCandidateId, seedFor, partialShuffle, scorerGreaterIsBetter }
