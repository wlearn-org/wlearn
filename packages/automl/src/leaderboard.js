/**
 * Tracks and ranks candidate evaluation results.
 */
export class Leaderboard {
  #entries = []
  #nextId = 0
  #dirty = true

  /**
   * Add a candidate result.
   * @param {{ modelName: string, params: Record<string, unknown>, scores: Float64Array, fitTimeMs: number }} entry
   * @returns {object} the entry with id assigned
   */
  add({ modelName, params, scores, fitTimeMs }) {
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
      modelName,
      params,
      scores,
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
    return this.#entries.slice()
  }

  /**
   * Return the best entry (highest meanScore) or null.
   */
  best() {
    if (this.#entries.length === 0) return null
    this.ranked() // ensure sorted
    return this.#entries[0]
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
      modelName: e.modelName,
      params: e.params,
      scores: [...e.scores],
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
        scores: new Float64Array(e.scores),
      })
      if (e.id >= lb.#nextId) lb.#nextId = e.id + 1
    }
    lb.#dirty = true
    return lb
  }

  get length() {
    return this.#entries.length
  }
}
