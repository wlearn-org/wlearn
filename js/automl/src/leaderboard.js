const { Archive, ValidationError } = require('@wlearn/core')
const { createCandidate, makeCandidateId } = require('./candidate.js')

function clone(value) {
  if (value == null || typeof value !== 'object') return value
  if (Array.isArray(value)) return value.map(clone)
  if (ArrayBuffer.isView(value)) return new value.constructor(value)
  const out = {}
  for (const [key, val] of Object.entries(value)) {
    Object.defineProperty(out, key, {
      value: clone(val), enumerable: true, configurable: true, writable: true
    })
  }
  return out
}

function cloneEntry(entry) {
  const result = clone(entry)
  result.candidate = createCandidate(
    entry.candidate.model,
    entry.candidate.model.params,
    entry.candidate.preprocess
  )
  return result
}

function cloneLeaderboardEntry(entry) {
  return entry === null ? null : cloneEntry(entry)
}

/**
 * Tracks and ranks candidate evaluation results.
 */
class Leaderboard {
  #entries = []
  #nextId = 0
  #dirty = true

  /**
   * Add a candidate result.
   * @param {{ candidateId: string, candidate: object, scores: Float64Array, fitTimeMs: number }} entry
   * @returns {object} the entry with id assigned
   */
  add({ candidateId, candidate, scores, baseSeed = 42, foldSeeds, fitTimeMs }) {
    const normalizedCandidate = createCandidate(
      candidate.model, candidate.model.params, candidate.preprocess
    )
    if (makeCandidateId(normalizedCandidate) !== candidateId) {
      throw new ValidationError('candidateId does not match the structured candidate')
    }
    let sum = 0
    for (let i = 0; i < scores.length; i++) sum += scores[i]
    const meanScore = sum / scores.length

    let sumSq = 0
    for (let i = 0; i < scores.length; i++) {
      const d = scores[i] - meanScore
      sumSq += d * d
    }
    const stdScore = Math.sqrt(sumSq / scores.length)

    const entry = {
      id: this.#nextId++,
      candidateId,
      candidate: normalizedCandidate,
      modelName: normalizedCandidate.model.displayName,
      params: clone(normalizedCandidate.model.params),
      scores: new Float64Array(scores),
      baseSeed,
      foldSeeds: foldSeeds ? new Uint32Array(foldSeeds) : null,
      meanScore,
      stdScore,
      fitTimeMs,
      rank: 0,
    }
    this.#entries.push(entry)
    this.#dirty = true
    return entry
  }

  /**
   * Return all entries sorted by meanScore descending with ranks assigned.
   */
  ranked() {
    if (this.#dirty) {
      this.#entries.sort((a, b) => b.meanScore - a.meanScore)
      for (let i = 0; i < this.#entries.length; i++) {
        this.#entries[i].rank = i + 1
      }
      this.#dirty = false
    }
    return this.#entries.map(cloneEntry)
  }

  /**
   * Return the best entry (highest meanScore) or null.
   */
  best() {
    if (this.#entries.length === 0) return null
    this.ranked() // ensure sorted
    return cloneEntry(this.#entries[0])
  }

  /**
   * Return top k entries.
   */
  top(k) {
    return this.ranked().slice(0, k)
  }

  /**
   * Serialize to JSON-friendly array.
   */
  toJSON() {
    return this.ranked().map(e => ({
      id: e.id,
      candidateId: e.candidateId,
      candidate: clone(e.candidate),
      modelName: e.modelName,
      params: clone(e.params),
      scores: [...e.scores],
      baseSeed: e.baseSeed,
      foldSeeds: e.foldSeeds ? [...e.foldSeeds] : null,
      meanScore: e.meanScore,
      stdScore: e.stdScore,
      fitTimeMs: e.fitTimeMs,
      rank: e.rank,
    }))
  }

  /**
   * Deserialize from JSON array.
   */
  static fromJSON(arr) {
    const lb = new Leaderboard()
    for (const e of arr) {
      lb.#entries.push({
        ...e,
        candidate: createCandidate(
          e.candidate.model, e.candidate.model.params, e.candidate.preprocess
        ),
        params: clone(e.params),
        scores: new Float64Array(e.scores),
        foldSeeds: e.foldSeeds ? new Uint32Array(e.foldSeeds) : null,
      })
      if (e.id >= lb.#nextId) lb.#nextId = e.id + 1
    }
    lb.#dirty = true
    return lb
  }

  /**
   * Convert ranked entries to an Archive.
   */
  toArchive({ metric = 'score', direction = 'maximize', metadata = {} } = {}) {
    const archive = new Archive({
      id: 'automl',
      measures: [metric],
      primaryMeasure: metric,
      direction,
      metadata
    })
    for (const entry of this.ranked()) {
      archive.add({
        trialId: `automl-${entry.id}`,
        candidateId: entry.candidateId,
        seed: entry.baseSeed,
        params: clone(entry.candidate.model.params),
        status: 'ok',
        scores: { [metric]: entry.meanScore },
        primaryScore: entry.meanScore,
        timings: { fitTimeMs: entry.fitTimeMs },
        metadata: {
          sourceCandidateId: entry.candidateId,
          candidate: clone(entry.candidate),
          leaderboardId: entry.id,
          modelName: entry.modelName,
          foldScores: Array.from(entry.scores),
          foldSeeds: entry.foldSeeds
            ? Array.from(entry.foldSeeds, (value, foldId) => ({ foldId, seed: value }))
            : [],
          stdScore: entry.stdScore
        }
      })
    }
    return archive
  }

  get length() {
    return this.#entries.length
  }
}

module.exports = { Leaderboard, cloneLeaderboardEntry }
