const { ValidationError } = require('./errors.js')
const { makeLCG, shuffle } = require('./rng.js')
const { normalizeY } = require('./matrix.js')
const { targetRows } = require('./targets.js')
const { kFold, stratifiedKFold, trainTestSplit } = require('./cv.js')
const { inferTaskKind } = require('./task.js')

function serializeCv(cv) {
  if (typeof cv === 'number') return cv
  const folds = Array.isArray(cv) ? cv : cv.folds
  return folds.map(fold => ({
    ...fold,
    train: Array.from(fold.train),
    test: Array.from(fold.test),
    ...(fold.validate ? { validate: Array.from(fold.validate) } : {})
  }))
}

function resolveCv(cv, y, { task = typeof y === 'number' ? 'regression' : inferTaskKind(y), seed = 42, requireComplete = false } = {}) {
  const n = typeof y === 'number' ? y : targetRows(y)
  if (!Number.isSafeInteger(n) || n < 1) throw new ValidationError('CV requires a positive row count')
  if (typeof y === 'number' && task === 'classification') throw new ValidationError('Stratified CV requires class labels')
  let source
  if (typeof cv === 'number') {
    source = task === 'classification'
      ? stratifiedKFold(y, cv, { seed }) : kFold(n, cv, { seed })
  } else if (Array.isArray(cv)) {
    source = cv
  } else {
    validateResamplingPlan(cv)
    if (cv.n !== n) throw new ValidationError('CV plan row count must match y')
    source = cv.folds
  }
  if (!source.length) throw new ValidationError('CV folds must be non-empty')
  const folds = source.map((fold, index) => {
    if (!fold || typeof fold !== 'object') throw new ValidationError('CV fold must be an object')
    const out = { ...fold, foldId: fold.foldId || `fold-${index}` }
    for (const field of ['train', 'test', 'validate']) {
      if (field === 'validate' && fold[field] == null) continue
      const values = fold[field]
      if (!(Array.isArray(values) || values instanceof Int32Array) ||
          Array.from(values).some(value => !Number.isInteger(value) || value < 0 || value >= n)) {
        throw new ValidationError(`CV fold ${index}.${field} must contain valid integer row indices`)
      }
      out[field] = new Int32Array(values)
    }
    _validateFold(out, n)
    return out
  })
  if (requireComplete) {
    const counts = new Uint32Array(n)
    for (const fold of folds) for (const row of fold.test) counts[row]++
    // A materialized OOF feature matrix needs one independent prediction per
    // training row. Partial/repeated folds require a different aggregation API.
    if (counts.some(count => count !== 1)) {
      throw new ValidationError('OOF requires every row in test folds exactly once')
    }
  }
  return folds
}

const RESAMPLING_STRATEGIES = [
  'holdout',
  'kfold',
  'stratified_kfold',
  'repeated_kfold',
  'group_kfold',
  'time_series',
  'sliding_window',
  'sliding_index',
  'sliding_period'
]

function createResamplingPlan({
  id,
  strategy = 'kfold',
  n,
  y,
  groups,
  k = 5,
  repeats = 1,
  testSize = 0.2,
  initialWindow,
  horizon = 1,
  lookback,
  assessStart,
  assessStop,
  complete = true,
  index,
  period = 'day',
  skip = 0,
  step = 1,
  shuffle: doShuffle = true,
  seed = 42,
  folds,
  taskId,
  metadata = {}
} = {}) {
  if (!RESAMPLING_STRATEGIES.includes(strategy)) {
    throw new ValidationError(`Unsupported resampling strategy "${strategy}"`)
  }
  _validateK(k)
  if (!Number.isInteger(repeats) || repeats < 1) {
    throw new ValidationError('ResamplingPlan repeats must be >= 1')
  }
  if (typeof testSize !== 'number' || testSize <= 0 || testSize >= 1) {
    throw new ValidationError('ResamplingPlan testSize must be in (0, 1)')
  }
  if (folds) {
    return validateResamplingPlan(_makePlan({ id, strategy, n, folds, taskId, seed, metadata }))
  }
  if (n == null && index) n = index.length
  if (!Number.isInteger(n) || n < 2) {
    throw new ValidationError('ResamplingPlan requires n >= 2')
  }

  if (strategy === 'holdout') {
    return validateResamplingPlan(_makePlan({
      id,
      strategy,
      n,
      folds: [_withId(trainTestSplit(n, { testSize, shuffle: doShuffle, seed }), 'holdout-0')],
      taskId,
      seed,
      metadata: { testSize, ...metadata }
    }))
  }

  if (strategy === 'kfold') {
    return validateResamplingPlan(_makePlan({
      id,
      strategy,
      n,
      folds: kFold(n, k, { shuffle: doShuffle, seed }).map((fold, i) => _withId(fold, `fold-${i}`)),
      taskId,
      seed,
      metadata: { k, shuffle: doShuffle, ...metadata }
    }))
  }

  if (strategy === 'stratified_kfold') {
    if (!y) throw new ValidationError('stratified_kfold requires y')
    return validateResamplingPlan(_makePlan({
      id,
      strategy,
      n,
      folds: stratifiedKFold(normalizeY(y), k, { shuffle: doShuffle, seed }).map((fold, i) => _withId(fold, `fold-${i}`)),
      taskId,
      seed,
      metadata: { k, shuffle: doShuffle, ...metadata },
      constraints: { stratified: true }
    }))
  }

  if (strategy === 'repeated_kfold') {
    const planFolds = []
    for (let r = 0; r < repeats; r++) {
      const repeatSeed = seed + r
      const repeatFolds = kFold(n, k, { shuffle: doShuffle, seed: repeatSeed })
      for (let f = 0; f < repeatFolds.length; f++) {
        planFolds.push(_withId(repeatFolds[f], `repeat-${r}-fold-${f}`, { repeat: r }))
      }
    }
    return validateResamplingPlan(_makePlan({
      id,
      strategy,
      n,
      folds: planFolds,
      taskId,
      seed,
      metadata: { k, repeats, shuffle: doShuffle, ...metadata }
    }))
  }

  if (strategy === 'group_kfold') {
    if (!groups) throw new ValidationError('group_kfold requires groups')
    return validateResamplingPlan(_makePlan({
      id,
      strategy,
      n,
      folds: groupKFold(normalizeY(groups), k, { shuffle: doShuffle, seed }),
      taskId,
      seed,
      metadata: { k, shuffle: doShuffle, ...metadata },
      constraints: { grouped: true }
    }))
  }

  if (strategy === 'time_series') {
    return validateResamplingPlan(_makePlan({
      id,
      strategy,
      n,
      folds: timeSeriesSplit(n, { initialWindow, horizon, step }),
      taskId,
      seed,
      metadata: { initialWindow, horizon, step, ...metadata },
      constraints: { timeOrdered: true }
    }))
  }

  if (strategy === 'sliding_window') {
    const slidingOpts = { lookback, initialWindow, assessStart, assessStop, horizon, step, skip, complete }
    return validateResamplingPlan(_makePlan({
      id,
      strategy,
      n,
      folds: slidingWindowSplit(n, slidingOpts),
      taskId,
      seed,
      metadata: { ..._slidingMetadata(slidingOpts, n), ...metadata },
      constraints: { timeOrdered: true }
    }))
  }

  if (strategy === 'sliding_index') {
    if (!index) throw new ValidationError('sliding_index requires index')
    const slidingOpts = { lookback, assessStart, assessStop, horizon, step, skip, complete }
    return validateResamplingPlan(_makePlan({
      id,
      strategy,
      n,
      folds: slidingIndexSplit(index, slidingOpts),
      taskId,
      seed,
      metadata: { ..._slidingMetadata(slidingOpts), ...metadata },
      constraints: { timeOrdered: true }
    }))
  }

  if (strategy === 'sliding_period') {
    if (!index) throw new ValidationError('sliding_period requires index')
    const slidingOpts = { lookback, assessStart, assessStop, horizon, step, skip, complete, period }
    return validateResamplingPlan(_makePlan({
      id,
      strategy,
      n,
      folds: slidingPeriodSplit(index, slidingOpts),
      taskId,
      seed,
      metadata: { ..._slidingMetadata(slidingOpts), period, ...metadata },
      constraints: { timeOrdered: true }
    }))
  }

  throw new ValidationError(`Unsupported resampling strategy "${strategy}"`)
}

function groupKFold(groups, k = 5, { shuffle: doShuffle = true, seed = 42 } = {}) {
  _validateK(k)
  if (groups.length < k) throw new ValidationError(`groupKFold: n (${groups.length}) must be >= k (${k})`)
  const groupMap = new Map()
  for (let i = 0; i < groups.length; i++) {
    const key = groups[i]
    if (!groupMap.has(key)) groupMap.set(key, [])
    groupMap.get(key).push(i)
  }
  const groupKeys = [...groupMap.keys()]
  if (groupKeys.length < k) {
    throw new ValidationError(`groupKFold: number of groups (${groupKeys.length}) must be >= k (${k})`)
  }
  if (doShuffle) shuffle(groupKeys, makeLCG(seed))

  const foldTests = Array.from({ length: k }, () => [])
  const foldSizes = new Int32Array(k)
  for (const key of groupKeys) {
    let best = 0
    for (let f = 1; f < k; f++) {
      if (foldSizes[f] < foldSizes[best]) best = f
    }
    const rows = groupMap.get(key)
    foldTests[best].push(...rows)
    foldSizes[best] += rows.length
  }

  return foldTests.map((testRows, foldId) => {
    const testSet = new Set(testRows)
    const train = []
    for (let i = 0; i < groups.length; i++) {
      if (!testSet.has(i)) train.push(i)
    }
    return _withId({ train: new Int32Array(train), test: new Int32Array(testRows) }, `fold-${foldId}`)
  })
}

function timeSeriesSplit(n, { initialWindow, horizon = 1, step = 1 } = {}) {
  if (!Number.isInteger(n) || n < 2) throw new ValidationError('timeSeriesSplit: n must be >= 2')
  if (!Number.isInteger(horizon) || horizon < 1) throw new ValidationError('timeSeriesSplit: horizon must be >= 1')
  if (!Number.isInteger(step) || step < 1) throw new ValidationError('timeSeriesSplit: step must be >= 1')
  const startWindow = initialWindow == null ? Math.max(1, Math.floor(n / 2)) : initialWindow
  if (!Number.isInteger(startWindow) || startWindow < 1 || startWindow >= n) {
    throw new ValidationError('timeSeriesSplit: initialWindow must be between 1 and n - 1')
  }

  const folds = []
  let foldId = 0
  for (let trainEnd = startWindow; trainEnd + horizon <= n; trainEnd += step) {
    const train = Int32Array.from({ length: trainEnd }, (_, i) => i)
    const test = Int32Array.from({ length: horizon }, (_, i) => trainEnd + i)
    folds.push(_withId({ train, test }, `fold-${foldId++}`))
  }
  if (folds.length === 0) {
    throw new ValidationError('timeSeriesSplit: no folds could be generated')
  }
  return folds
}

function slidingWindowSplit(n, opts = {}) {
  if (!Number.isInteger(n) || n < 2) throw new ValidationError('slidingWindowSplit: n must be >= 2')
  const cfg = _normalizeSlidingOpts(opts, n)
  const folds = []
  let foldId = 0
  for (let anchor = cfg.complete ? cfg.lookback - 1 : 0; anchor < n; anchor += cfg.stride) {
    const testStart = anchor + cfg.assessStart
    const fullTestEnd = anchor + cfg.assessStop
    if (testStart >= n) break
    if (cfg.complete && fullTestEnd >= n) break
    const trainStart = Math.max(0, anchor - cfg.lookback + 1)
    if (cfg.complete && anchor - cfg.lookback + 1 < 0) continue
    const train = Int32Array.from({ length: anchor - trainStart + 1 }, (_, i) => trainStart + i)
    const testEnd = Math.min(n - 1, fullTestEnd)
    const test = Int32Array.from({ length: testEnd - testStart + 1 }, (_, i) => testStart + i)
    folds.push(_withId({ train, test }, `fold-${foldId++}`, {
      anchor,
      trainStart,
      trainEnd: anchor,
      testStart,
      testEnd
    }))
  }
  if (folds.length === 0) throw new ValidationError('slidingWindowSplit: no folds could be generated')
  return folds
}

function slidingIndexSplit(index, opts = {}) {
  const values = _coerceIndex(index, 'slidingIndexSplit')
  const cfg = _normalizeSlidingOpts(opts, values.length, { valueWindow: true })
  return _slidingValueSplit(values, cfg, 'slidingIndexSplit')
}

function slidingPeriodSplit(index, opts = {}) {
  const values = _periodOrdinals(index, opts.period || 'day')
  const cfg = _normalizeSlidingOpts(opts, values.length, { valueWindow: true })
  return _slidingValueSplit(values, cfg, 'slidingPeriodSplit')
}

function _slidingValueSplit(values, cfg, name) {
  if (values.length < 2) throw new ValidationError(`${name}: index must have length >= 2`)
  for (let i = 1; i < values.length; i++) {
    if (values[i] < values[i - 1]) throw new ValidationError(`${name}: index must be sorted ascending`)
  }
  const folds = []
  const seenAnchors = new Set()
  let foldId = 0
  for (let anchorIdx = 0; anchorIdx < values.length; anchorIdx += cfg.stride) {
    const anchorValue = values[anchorIdx]
    if (seenAnchors.has(anchorValue)) continue
    seenAnchors.add(anchorValue)

    const trainMin = anchorValue - cfg.lookback
    const testMin = anchorValue + cfg.assessStart
    const testMax = anchorValue + cfg.assessStop
    if (cfg.complete && trainMin < values[0]) continue
    if (testMin > values[values.length - 1]) break
    if (cfg.complete && testMax > values[values.length - 1]) break

    const train = []
    const test = []
    for (let i = 0; i < values.length; i++) {
      if (values[i] >= Math.max(values[0], trainMin) && values[i] <= anchorValue) train.push(i)
      if (values[i] >= testMin && values[i] <= Math.min(values[values.length - 1], testMax)) test.push(i)
    }
    if (train.length === 0 || test.length === 0) continue
    folds.push(_withId({
      train: new Int32Array(train),
      test: new Int32Array(test)
    }, `fold-${foldId++}`, {
      anchorIndex: anchorIdx,
      anchorValue,
      trainStart: Math.max(values[0], trainMin),
      trainEnd: anchorValue,
      testStart: testMin,
      testEnd: Math.min(values[values.length - 1], testMax)
    }))
  }
  if (folds.length === 0) throw new ValidationError(`${name}: no folds could be generated`)
  return folds
}

function _normalizeSlidingOpts(opts, n, { valueWindow = false } = {}) {
  const lookback = opts.lookback ?? opts.initialWindow ?? (valueWindow ? 1 : Math.max(1, Math.floor(n / 2)))
  const assessStart = opts.assessStart ?? opts.assess_start ?? 1
  const assessStop = opts.assessStop ?? opts.assess_stop ?? opts.horizon ?? assessStart
  const step = opts.step ?? 1
  const skip = opts.skip ?? 0
  const complete = opts.complete !== false
  if (typeof lookback !== 'number' || !Number.isFinite(lookback) || lookback <= 0) {
    throw new ValidationError('sliding split lookback must be positive')
  }
  if (typeof assessStart !== 'number' || !Number.isFinite(assessStart) || assessStart <= 0) {
    throw new ValidationError('sliding split assessStart must be positive')
  }
  if (typeof assessStop !== 'number' || !Number.isFinite(assessStop) || assessStop < assessStart) {
    throw new ValidationError('sliding split assessStop must be >= assessStart')
  }
  if (!Number.isInteger(step) || step < 1) throw new ValidationError('sliding split step must be >= 1')
  if (!Number.isInteger(skip) || skip < 0) throw new ValidationError('sliding split skip must be >= 0')
  if (!valueWindow && (!Number.isInteger(lookback) || !Number.isInteger(assessStart) || !Number.isInteger(assessStop))) {
    throw new ValidationError('slidingWindowSplit lookback/assessStart/assessStop must be integers')
  }
  return { lookback, assessStart, assessStop, step, skip, stride: step + skip, complete }
}

function _slidingMetadata(opts, n) {
  const cfg = _normalizeSlidingOpts(opts, n || 2, { valueWindow: n == null })
  return {
    lookback: cfg.lookback,
    assessStart: cfg.assessStart,
    assessStop: cfg.assessStop,
    step: cfg.step,
    skip: cfg.skip,
    complete: cfg.complete
  }
}

function _coerceIndex(index, name) {
  if (!index || index.length < 2) throw new ValidationError(`${name}: index must have length >= 2`)
  const values = new Float64Array(index.length)
  for (let i = 0; i < index.length; i++) {
    values[i] = _coerceIndexValue(index[i])
    if (!Number.isFinite(values[i])) throw new ValidationError(`${name}: index values must be finite`)
  }
  return values
}

function _coerceIndexValue(value) {
  if (value instanceof Date) return value.getTime()
  if (typeof value === 'number') return value
  const parsed = Date.parse(value)
  if (Number.isFinite(parsed)) return parsed
  const numeric = Number(value)
  return numeric
}

function _periodOrdinals(index, period) {
  if (typeof period === 'number') {
    if (!Number.isFinite(period) || period <= 0) throw new ValidationError('slidingPeriodSplit: numeric period must be positive')
    const raw = _coerceIndex(index, 'slidingPeriodSplit')
    return Float64Array.from(raw, value => Math.floor(value / period))
  }
  const values = new Float64Array(index.length)
  for (let i = 0; i < index.length; i++) {
    values[i] = _periodOrdinal(index[i], period)
    if (!Number.isFinite(values[i])) throw new ValidationError('slidingPeriodSplit: index values must be finite dates')
  }
  return values
}

function _periodOrdinal(value, period) {
  const date = value instanceof Date ? value : new Date(value)
  const t = date.getTime()
  if (!Number.isFinite(t)) return NaN
  const year = date.getUTCFullYear()
  const month = date.getUTCMonth()
  if (period === 'day') return Math.floor(t / 86400000)
  if (period === 'week') return Math.floor(t / (7 * 86400000))
  if (period === 'month') return year * 12 + month
  if (period === 'quarter') return year * 4 + Math.floor(month / 3)
  if (period === 'year') return year
  throw new ValidationError(`slidingPeriodSplit: unsupported period "${period}"`)
}

function validateResamplingPlan(plan) {
  if (!plan || typeof plan !== 'object') {
    throw new ValidationError('ResamplingPlan must be an object')
  }
  if (typeof plan.id !== 'string' || plan.id.length === 0) {
    throw new ValidationError('ResamplingPlan.id must be a non-empty string')
  }
  if (!RESAMPLING_STRATEGIES.includes(plan.strategy)) {
    throw new ValidationError(`Unsupported resampling strategy "${plan.strategy}"`)
  }
  if (!Number.isInteger(plan.n) || plan.n < 2) {
    throw new ValidationError('ResamplingPlan.n must be >= 2')
  }
  if (!Array.isArray(plan.folds) || plan.folds.length === 0) {
    throw new ValidationError('ResamplingPlan.folds must be non-empty')
  }
  for (const fold of plan.folds) {
    _validateFold(fold, plan.n)
  }
  return plan
}

function serializeResamplingPlan(plan) {
  validateResamplingPlan(plan)
  return {
    ...plan,
    folds: plan.folds.map(fold => ({
      ...fold,
      train: [...fold.train],
      test: [...fold.test],
      validate: fold.validate ? [...fold.validate] : undefined
    }))
  }
}

function deserializeResamplingPlan(plan) {
  if (!plan || typeof plan !== 'object') {
    throw new ValidationError('Serialized ResamplingPlan must be an object')
  }
  return validateResamplingPlan({
    ...plan,
    folds: plan.folds.map(fold => ({
      ...fold,
      train: new Int32Array(fold.train),
      test: new Int32Array(fold.test),
      validate: fold.validate ? new Int32Array(fold.validate) : undefined
    }))
  })
}

function _makePlan({ id, strategy, n, folds, taskId, seed, metadata = {}, constraints = {} }) {
  const plan = {
    id: id || `${strategy}-${n}`,
    strategy,
    n,
    folds,
    seed,
    metadata: { ...metadata }
  }
  if (taskId) plan.taskId = taskId
  if (Object.keys(constraints).length > 0) plan.constraints = { ...constraints }
  return plan
}

function _withId(fold, foldId, metadata) {
  const out = {
    foldId,
    train: fold.train,
    test: fold.test
  }
  if (fold.validate) out.validate = fold.validate
  if (metadata) out.metadata = { ...metadata }
  return out
}

function _validateFold(fold, n) {
  if (!fold || typeof fold !== 'object') throw new ValidationError('CV fold must be an object')
  if (typeof fold.foldId !== 'string' || fold.foldId.length === 0) {
    throw new ValidationError('CV fold must have a non-empty foldId')
  }
  for (const key of ['train', 'test']) {
    if (!(fold[key] instanceof Int32Array)) {
      throw new ValidationError(`CV fold ${fold.foldId}.${key} must be Int32Array`)
    }
    if (fold[key].length === 0) {
      throw new ValidationError(`CV fold ${fold.foldId}.${key} must be non-empty`)
    }
    for (let i = 0; i < fold[key].length; i++) {
      if (fold[key][i] < 0 || fold[key][i] >= n) {
        throw new ValidationError(`CV fold ${fold.foldId}.${key} contains out-of-range index ${fold[key][i]}`)
      }
    }
  }
  const trainSet = new Set(fold.train)
  if (trainSet.size !== fold.train.length) {
    throw new ValidationError(`CV fold ${fold.foldId}.train contains duplicate indices`)
  }
  const testSet = new Set(fold.test)
  if (testSet.size !== fold.test.length) {
    throw new ValidationError(`CV fold ${fold.foldId}.test contains duplicate indices`)
  }
  for (const idx of fold.test) {
    if (trainSet.has(idx)) {
      throw new ValidationError(`CV fold ${fold.foldId} has overlapping train/test index ${idx}`)
    }
  }
  if (fold.validate != null) {
    if (!(fold.validate instanceof Int32Array)) {
      throw new ValidationError(`CV fold ${fold.foldId}.validate must be Int32Array`)
    }
    if (fold.validate.length === 0) {
      throw new ValidationError(`CV fold ${fold.foldId}.validate must be non-empty`)
    }
    const validateSet = new Set()
    for (let i = 0; i < fold.validate.length; i++) {
      const idx = fold.validate[i]
      if (idx < 0 || idx >= n) {
        throw new ValidationError(`CV fold ${fold.foldId}.validate contains out-of-range index ${idx}`)
      }
      if (validateSet.has(idx)) {
        throw new ValidationError(`CV fold ${fold.foldId}.validate contains duplicate indices`)
      }
      if (trainSet.has(idx) || testSet.has(idx)) {
        throw new ValidationError(`CV fold ${fold.foldId} has overlapping validate index ${idx}`)
      }
      validateSet.add(idx)
    }
  }
}

function _validateK(k) {
  if (!Number.isInteger(k) || k < 2) {
    throw new ValidationError('ResamplingPlan k must be >= 2')
  }
}

module.exports = {
  serializeCv,
  resolveCv,
  RESAMPLING_STRATEGIES,
  createResamplingPlan,
  validateResamplingPlan,
  serializeResamplingPlan,
  deserializeResamplingPlan,
  groupKFold,
  timeSeriesSplit,
  slidingWindowSplit,
  slidingIndexSplit,
  slidingPeriodSplit
}
