const { makeLCG } = require('@wlearn/core')

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
 * Stable JSON stringify with sorted keys.
 * Numbers use toString() to avoid precision drift.
 * Params must be JSON-serializable primitives only (enforced by SearchSpace IR).
 */
function stableStringify(obj) {
  if (obj === null || obj === undefined) return String(obj)
  if (typeof obj === 'number') return obj.toString()
  if (typeof obj === 'string') return JSON.stringify(obj)
  if (typeof obj === 'boolean') return String(obj)
  if (Array.isArray(obj)) {
    return '[' + obj.map(stableStringify).join(',') + ']'
  }
  if (typeof obj === 'object') {
    const keys = Object.keys(obj).sort()
    return '{' + keys.map(k => JSON.stringify(k) + ':' + stableStringify(obj[k])).join(',') + '}'
  }
  return String(obj)
}

/**
 * Stable candidate ID from model label and params.
 */
function makeCandidateId(modelLabel, params) {
  return modelLabel + ':' + stableStringify(params)
}

/**
 * Simple integer hash for strings (FNV-1a inspired).
 */
function hashString(str) {
  let h = 0x811c9dc5
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i)
    h = (h * 0x01000193) & 0x7fffffff
  }
  return h
}

/**
 * Derive a deterministic seed from base seed, candidate ID, and fold index.
 */
function seedFor(candidateId, foldIdx, baseSeed) {
  const h = hashString(candidateId)
  // Mix: multiply-xor-shift
  let s = (baseSeed * 2654435761 + h * 40503 + foldIdx * 65537) & 0x7fffffff
  s = ((s >>> 16) ^ s) * 0x45d9f3b & 0x7fffffff
  return s
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
