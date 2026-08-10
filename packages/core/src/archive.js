const { ValidationError } = require('./errors.js')

const TRIAL_STATUSES = ['pending', 'running', 'ok', 'failed', 'pruned', 'timeout']

class Archive {
  constructor({
    id = 'archive',
    taskId,
    measures = [],
    primaryMeasure,
    direction = 'maximize',
    metadata = {},
    records = []
  } = {}) {
    if (direction !== 'maximize' && direction !== 'minimize') {
      throw new ValidationError('Archive.direction must be "maximize" or "minimize"')
    }
    this.id = id
    this.taskId = taskId
    this.measures = [...measures]
    this.primaryMeasure = primaryMeasure || measures[0]
    this.direction = direction
    this.metadata = { ...metadata }
    this._records = []
    for (const record of records) this.add(record)
  }

  add(record) {
    const normalized = createTrialRecord(record)
    if (this._records.some(r => r.trialId === normalized.trialId)) {
      throw new ValidationError(`Archive already contains trialId "${normalized.trialId}"`)
    }
    this._records.push(normalized)
    return _clone(normalized)
  }

  start(record = {}) {
    return this.add({
      ...record,
      status: record.status || 'running'
    })
  }

  finish(trialId, patch = {}) {
    return this.update(trialId, {
      ...patch,
      status: patch.status || 'ok'
    })
  }

  fail(record, error, phase = 'fit') {
    return this.add({
      ...record,
      status: 'failed',
      error: normalizeTrialError(error, phase)
    })
  }

  update(trialId, patch) {
    const record = this._records.find(r => r.trialId === trialId)
    if (!record) throw new ValidationError(`Archive record "${trialId}" not found`)
    if (patch.trialId && patch.trialId !== trialId) {
      throw new ValidationError('Archive.update cannot change trialId')
    }
    const next = createTrialRecord({ ...record, ...patch })
    Object.assign(record, next)
    return _clone(record)
  }

  records(filter = {}) {
    const entries = !filter || Object.keys(filter).length === 0
      ? this._records
      : this._records.filter(record => {
        for (const [key, value] of Object.entries(filter)) {
          if (record[key] !== value) return false
        }
        return true
      })
    return entries.map(_clone)
  }

  leaderboard({
    metric = this.primaryMeasure,
    direction = this.direction
  } = {}) {
    const groups = new Map()
    for (const record of this._records) {
      if (record.status !== 'ok') continue
      const value = _scoreValue(record, metric)
      if (value == null || Number.isNaN(value)) continue
      const key = record.candidateId
      if (!groups.has(key)) {
        groups.set(key, {
          candidateId: key,
          learnerSpec: record.learnerSpec,
          pipelineSpec: record.pipelineSpec,
          params: record.params || {},
          budget: record.budget,
          values: [],
          trialIds: []
        })
      }
      const group = groups.get(key)
      group.values.push(value)
      group.trialIds.push(record.trialId)
    }

    const rows = [...groups.values()].map(group => {
      const meanScore = _mean(group.values)
      return {
        candidateId: group.candidateId,
        learnerSpec: group.learnerSpec,
        pipelineSpec: group.pipelineSpec,
        params: group.params,
        budget: group.budget,
        metric,
        meanScore,
        stdScore: _std(group.values, meanScore),
        n: group.values.length,
        trialIds: group.trialIds
      }
    })
    rows.sort((a, b) => direction === 'minimize' ? a.meanScore - b.meanScore : b.meanScore - a.meanScore)
    for (let i = 0; i < rows.length; i++) rows[i].rank = i + 1
    return rows.map(_clone)
  }

  toJSON() {
    return {
      id: this.id,
      taskId: this.taskId,
      measures: this.measures,
      primaryMeasure: this.primaryMeasure,
      direction: this.direction,
      metadata: this.metadata,
      records: this._records.map(_clone)
    }
  }

  get size() {
    return this._records.length
  }

  static fromJSON(json) {
    return new Archive(json)
  }
}

function createTrialRecord(record = {}) {
  const trial = {
    trialId: record.trialId || _defaultTrialId(record),
    candidateId: record.candidateId || 'candidate',
    seed: record.seed == null ? 42 : record.seed,
    status: record.status || 'pending',
    params: _clone(record.params || {}),
    timings: _clone(record.timings || {})
  }
  if (record.learnerSpec) trial.learnerSpec = _clone(record.learnerSpec)
  if (record.pipelineSpec) trial.pipelineSpec = _clone(record.pipelineSpec)
  if (record.budget) trial.budget = _clone(record.budget)
  if (record.batch != null) trial.batch = record.batch
  if (record.uhash) trial.uhash = record.uhash
  if (record.foldId) trial.foldId = record.foldId
  if (record.scores) trial.scores = _clone(record.scores)
  if (record.primaryScore != null) trial.primaryScore = record.primaryScore
  if (record.error) trial.error = normalizeTrialError(record.error, record.error.phase)
  if (record.memory) trial.memory = _clone(record.memory)
  if (record.artifactHash) trial.artifactHash = record.artifactHash
  if (record.predictionHash) trial.predictionHash = record.predictionHash
  if (record.resampleResultHash) trial.resampleResultHash = record.resampleResultHash
  if (record.warnings) trial.warnings = _clone(record.warnings)
  if (record.metadata) trial.metadata = _clone(record.metadata)
  return validateTrialRecord(trial)
}

function validateTrialRecord(record) {
  if (!record || typeof record !== 'object') {
    throw new ValidationError('TrialRecord must be an object')
  }
  if (typeof record.trialId !== 'string' || record.trialId.length === 0) {
    throw new ValidationError('TrialRecord.trialId must be a non-empty string')
  }
  if (typeof record.candidateId !== 'string' || record.candidateId.length === 0) {
    throw new ValidationError('TrialRecord.candidateId must be a non-empty string')
  }
  if (!TRIAL_STATUSES.includes(record.status)) {
    throw new ValidationError(`TrialRecord.status "${record.status}" is invalid`)
  }
  if (record.scores) {
    for (const [key, value] of Object.entries(record.scores)) {
      if (typeof value !== 'number' || Number.isNaN(value)) {
        throw new ValidationError(`TrialRecord.scores.${key} must be a number`)
      }
    }
  }
  if (record.primaryScore != null && (typeof record.primaryScore !== 'number' || Number.isNaN(record.primaryScore))) {
    throw new ValidationError('TrialRecord.primaryScore must be a number')
  }
  if (record.batch != null && (!Number.isInteger(record.batch) || record.batch < 1)) {
    throw new ValidationError('TrialRecord.batch must be a positive integer')
  }
  if (record.error) {
    normalizeTrialError(record.error, record.error.phase)
  }
  return record
}

function normalizeTrialError(error, phase = 'fit') {
  if (!error) return undefined
  return {
    name: error.name || 'Error',
    message: error.message || String(error),
    stackHash: error.stackHash,
    phase: phase || error.phase || 'fit'
  }
}

function _scoreValue(record, metric) {
  if (metric && record.scores && record.scores[metric] != null) return record.scores[metric]
  if (record.primaryScore != null) return record.primaryScore
  return undefined
}

function _mean(values) {
  let sum = 0
  for (const value of values) sum += value
  return values.length === 0 ? NaN : sum / values.length
}

function _std(values, mean) {
  if (values.length <= 1) return 0
  let sum = 0
  for (const value of values) {
    const d = value - mean
    sum += d * d
  }
  return Math.sqrt(sum / (values.length - 1))
}

function _defaultTrialId(record) {
  const candidate = record.candidateId || 'candidate'
  const fold = record.foldId || 'fold'
  const seed = record.seed == null ? 42 : record.seed
  return `${candidate}-${fold}-${seed}`
}

function _clone(value) {
  if (value == null || typeof value !== 'object') return value
  if (Array.isArray(value)) return value.map(_clone)
  if (ArrayBuffer.isView(value)) return new value.constructor(value)
  const out = {}
  for (const [key, val] of Object.entries(value)) {
    Object.defineProperty(out, key, {
      value: _clone(val),
      enumerable: true,
      configurable: true,
      writable: true,
    })
  }
  return out
}

module.exports = {
  TRIAL_STATUSES,
  Archive,
  createTrialRecord,
  validateTrialRecord,
  normalizeTrialError
}
