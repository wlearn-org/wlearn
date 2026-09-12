const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const {
  createTask,
  createFeatureSchema,
  createPrediction,
  evaluateMeasure,
  evaluateMetricSet,
  aggregateMeasure,
  listMeasures,
  createResamplingPlan,
  serializeResamplingPlan,
  deserializeResamplingPlan,
  groupKFold,
  timeSeriesSplit,
  slidingWindowSplit,
  slidingIndexSplit,
  slidingPeriodSplit,
  Archive,
  createTrialRecord
} = require('../src/index.js')
const { ValidationError } = require('../src/errors.js')

const X = { data: new Float64Array([1, 2, 3, 4, 5, 6]), rows: 3, cols: 2 }
const y = new Int32Array([0, 1, 0])

it('Prediction preserves joint sample identity and rejects unsupported dependence declarations', () => {
  const input = { rows: 1, taskKind: 'multioutput', targetCount: 2, sampleCount: 2, sampleKind: 'outcome', samples: [1, 2, 3, 4] }
  const prediction = createPrediction({ ...input, sampleDependence: 'joint', sampleWeights: [1, 2] })
  assert.equal(prediction.sampleDependence, 'joint')
  assert.deepEqual(Array.from(prediction.sampleWeights), [1, 2])
  assert.deepEqual(Array.from(prediction.samples), input.samples)
  assert.throws(() => createPrediction({ ...input, sampleDependence: 'guessed' }), ValidationError)
  assert.throws(() => createPrediction({ response: [1], sampleDependence: 'joint' }), ValidationError)
  for (const sampleWeights of [[0, 0], [-1, 2], [1], [1, NaN]]) {
    assert.throws(() => createPrediction({ ...input, sampleWeights }), ValidationError)
  }
})

it('Prediction rejects invalid probability values, mass and class identity', () => {
  for (const proba of [[NaN, 1, 0, 1], [-0.1, 1.1, 0, 1], [0.2, 0.2, 0, 1]]) {
    assert.throws(() => createPrediction({ truth: [0, 1], proba, classes: [0, 1] }), ValidationError)
  }
  assert.throws(() => createPrediction({ truth: [0, 1], proba: [0.5, 0.5, 0, 1], classes: [0, 0] }), ValidationError)
})

describe('Task primitives', () => {
  it('creates task with inferred kind and feature schema', () => {
    const schema = createFeatureSchema(X, {
      names: ['a', 'b'],
      roles: ['feature', 'offset']
    })
    const task = createTask({ id: 'toy', X, y, featureSchema: schema })

    assert.equal(task.id, 'toy')
    assert.equal(task.kind, 'classification')
    assert.equal(task.featureSchema.features[0].name, 'a')
    assert.equal(task.featureSchema.features[1].role, 'offset')
  })

  it('rejects target length mismatch', () => {
    assert.throws(() => createTask({ X, y: new Int32Array([0, 1]) }), ValidationError)
  })

  it('rejects feature schema row mismatch', () => {
    const schema = createFeatureSchema(X)
    schema.rows = 999
    assert.throws(() => createTask({ X, y, featureSchema: schema }), ValidationError)
  })

  it('rejects out-of-range row roles', () => {
    assert.throws(() => createTask({
      X,
      y,
      rowRoles: { train: new Int32Array([0, 3]) }
    }), ValidationError)
  })
})

describe('Prediction and measure primitives', () => {
  it('scores built-in measures from prediction objects', () => {
    const prediction = createPrediction({
      truth: y,
      response: new Int32Array([0, 1, 1])
    })

    assert.equal(evaluateMeasure('accuracy', prediction), 2 / 3)
    const scores = evaluateMetricSet(['accuracy', 'f1'], prediction)
    assert.equal(scores.accuracy, 2 / 3)
    assert.equal(typeof scores.f1, 'number')
  })

  it('supports probability-only predictions with classes', () => {
    const prediction = createPrediction({
      truth: y,
      proba: new Float64Array([0.9, 0.1, 0.2, 0.8, 0.7, 0.3]),
      classes: new Int32Array([0, 1])
    })

    assert.equal(evaluateMeasure('log_loss', prediction, { nClasses: 2 }) > 0, true)
  })

  it('scores probability predictions using declared class order', () => {
    const prediction = createPrediction({
      truth: new Int32Array([0, 1]),
      proba: new Float64Array([0.1, 0.9, 0.9, 0.1]),
      classes: new Int32Array([1, 0])
    })

    assert(evaluateMeasure('log_loss', prediction) < 0.2)
  })

  it('rejects missing class and non-finite probabilities for log loss', () => {
    const missingClass = createPrediction({
      truth: new Int32Array([0, 2]),
      proba: new Float64Array([0.8, 0.2, 0.1, 0.9]),
      classes: new Int32Array([0, 1])
    })
    assert.throws(() => evaluateMeasure('log_loss', missingClass), ValidationError)

    assert.throws(() => createPrediction({
      truth: new Int32Array([0, 1]),
      proba: new Float64Array([0.8, 0.2, NaN, 0.9]),
      classes: new Int32Array([0, 1])
    }), ValidationError)
  })

  it('rejects probability length that does not match classes', () => {
    assert.throws(() => createPrediction({
      proba: new Float64Array([0.1, 0.9, 0.8]),
      classes: new Int32Array([0, 1])
    }), ValidationError)
  })

  it('rejects probaRows that conflict with inferred rows', () => {
    assert.throws(() => createPrediction({
      truth: new Int32Array([0, 1]),
      proba: new Float64Array([0.7, 0.3, 0.2, 0.8]),
      classes: new Int32Array([0, 1]),
      probaRows: 3
    }), ValidationError)
  })

  it('validates interval and quantile row alignment', () => {
    assert.doesNotThrow(() => createPrediction({
      rowIds: ['a', 'b'],
      interval: new Float64Array([0, 1, 2, 3]), coverageLevels: [0.9]
    }))
    assert.throws(() => createPrediction({
      rowIds: ['a', 'b'],
      quantiles: new Float64Array([0, 1, 2])
    }), ValidationError)
  })

  it('exposes measure registry and aggregation', () => {
    assert(listMeasures().includes('accuracy'))
    assert.equal(aggregateMeasure('accuracy', new Float64Array([1, 0.5, 0.75])), 0.75)
  })

  it('passes sample weights and multiclass AUC through built-in measures', () => {
    const prediction = createPrediction({
      truth: new Int32Array([0, 1, 2, 0, 1, 2]),
      response: new Int32Array([0, 1, 2, 1, 1, 0]),
      proba: new Float64Array([
        0.9, 0.05, 0.05,
        0.05, 0.9, 0.05,
        0.05, 0.05, 0.9,
        0.8, 0.1, 0.1,
        0.1, 0.8, 0.1,
        0.1, 0.1, 0.8
      ]),
      classes: new Int32Array([0, 1, 2])
    })

    assert.equal(evaluateMeasure('accuracy', prediction, {
      sampleWeight: new Float64Array([1, 1, 1, 3, 1, 1])
    }), 0.5)
    assert.equal(evaluateMeasure('roc_auc_ovr', prediction), 1)
    assert.equal(evaluateMeasure('roc_auc_ovo', prediction), 1)
  })

  it('allows undefined AUC warnings through the measure registry', () => {
    const warnings = []
    const prediction = createPrediction({
      truth: new Int32Array([1, 1, 1]),
      score: new Float64Array([0.1, 0.5, 0.9])
    })
    const value = evaluateMeasure('roc_auc', prediction, { undefinedValue: 'warn', warnings })
    assert(Number.isNaN(value))
    assert.equal(warnings.length, 1)
  })
})

describe('Resampling primitives', () => {
  it('creates serializable kfold plans', () => {
    const plan = createResamplingPlan({ strategy: 'kfold', n: 10, k: 5, seed: 7 })
    assert.equal(plan.folds.length, 5)
    const restored = deserializeResamplingPlan(serializeResamplingPlan(plan))
    assert.equal(restored.folds.length, 5)
    assert(restored.folds[0].train instanceof Int32Array)
  })

  it('rejects duplicate indices in supplied folds', () => {
    assert.throws(() => createResamplingPlan({
      strategy: 'kfold',
      n: 4,
      folds: [{
        foldId: 'fold-0',
        train: new Int32Array([0, 0, 1]),
        test: new Int32Array([2, 3])
      }]
    }), ValidationError)
  })

  it('rejects invalid validation indices in supplied folds', () => {
    assert.throws(() => createResamplingPlan({
      strategy: 'kfold',
      n: 5,
      folds: [{
        foldId: 'fold-0',
        train: new Int32Array([0, 1]),
        test: new Int32Array([2]),
        validate: new Int32Array([1, 4])
      }]
    }), ValidationError)

    assert.throws(() => createResamplingPlan({
      strategy: 'kfold',
      n: 5,
      folds: [{
        foldId: 'fold-0',
        train: new Int32Array([0, 1]),
        test: new Int32Array([2]),
        validate: new Int32Array([6])
      }]
    }), ValidationError)
  })

  it('rejects invalid repeated kfold repeats', () => {
    assert.throws(() => createResamplingPlan({ strategy: 'repeated_kfold', n: 10, k: 5, repeats: 0 }), ValidationError)
  })

  it('keeps groups together in group kfold', () => {
    const groups = new Int32Array([1, 1, 2, 2, 3, 3])
    const folds = groupKFold(groups, 3, { shuffle: false })

    for (const fold of folds) {
      const testGroups = new Set([...fold.test].map(i => groups[i]))
      for (const idx of fold.train) {
        assert.equal(testGroups.has(groups[idx]), false)
      }
    }
  })

  it('creates forward-only time-series splits', () => {
    const folds = timeSeriesSplit(8, { initialWindow: 4, horizon: 2, step: 2 })
    assert.equal(folds.length, 2)
    for (const fold of folds) {
      assert(Math.max(...fold.train) < Math.min(...fold.test))
    }
  })

  it('creates sliding row-window plans', () => {
    const folds = slidingWindowSplit(8, { lookback: 3, assessStart: 1, assessStop: 2, step: 2 })
    assert.equal(folds.length, 2)
    assert.deepEqual([...folds[0].train], [0, 1, 2])
    assert.deepEqual([...folds[0].test], [3, 4])
    assert.deepEqual([...folds[1].train], [2, 3, 4])
    assert.deepEqual([...folds[1].test], [5, 6])
  })

  it('creates sliding index splits over sorted numeric indices', () => {
    const folds = slidingIndexSplit(new Float64Array([0, 1, 2, 3, 4, 5]), {
      lookback: 2,
      assessStart: 1,
      assessStop: 1
    })
    assert.deepEqual([...folds[0].train], [0, 1, 2])
    assert.deepEqual([...folds[0].test], [3])
    assert.throws(() => slidingIndexSplit(new Float64Array([0, 2, 1]), { lookback: 1 }), ValidationError)
  })

  it('creates sliding period plans and serializes them', () => {
    const index = ['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04', '2026-01-05']
    const folds = slidingPeriodSplit(index, { period: 'day', lookback: 2, assessStart: 1, assessStop: 1 })
    assert.deepEqual([...folds[0].train], [0, 1, 2])
    assert.deepEqual([...folds[0].test], [3])

    const plan = createResamplingPlan({
      strategy: 'sliding_period',
      index,
      period: 'day',
      lookback: 2,
      assessStart: 1,
      assessStop: 1
    })
    const restored = deserializeResamplingPlan(serializeResamplingPlan(plan))
    assert.equal(restored.strategy, 'sliding_period')
    assert.equal(restored.metadata.period, 'day')
  })
})

describe('Archive primitives', () => {
  it('records trials and builds a leaderboard', () => {
    const archive = new Archive({ measures: ['accuracy'], primaryMeasure: 'accuracy' })
    archive.add({
      trialId: 'a-1',
      candidateId: 'a',
      pipelineSpec: { nodes: [], edges: [], endpoints: {} },
      scores: { accuracy: 0.8 },
      status: 'ok'
    })
    archive.add({
      trialId: 'b-1',
      candidateId: 'b',
      scores: { accuracy: 0.9 },
      status: 'ok'
    })

    const rows = archive.leaderboard()
    assert.equal(rows[0].candidateId, 'b')
    assert.equal(rows[0].rank, 1)
    assert.equal(rows[1].pipelineSpec.nodes.length, 0)
  })

  it('rejects duplicate trial ids', () => {
    const archive = new Archive()
    archive.add({ trialId: 'x', candidateId: 'x' })
    assert.throws(() => archive.add({ trialId: 'x', candidateId: 'x' }), ValidationError)
  })

  it('does not expose mutable records', () => {
    const archive = new Archive()
    archive.add({ trialId: 'x', candidateId: 'x', scores: { accuracy: 1 }, status: 'ok' })
    const records = archive.records()
    records[0].scores.accuracy = 0
    assert.equal(archive.records()[0].scores.accuracy, 1)
  })

  it('snapshots nested record payloads on add and update', () => {
    const archive = new Archive()
    const params = { nested: { depth: 3 } }
    archive.add({ trialId: 'x', candidateId: 'x', params, status: 'running' })
    params.nested.depth = 9
    assert.equal(archive.records()[0].params.nested.depth, 3)

    const patch = { metadata: { nested: { phase: 'fit' } }, status: 'ok' }
    archive.update('x', patch)
    patch.metadata.nested.phase = 'predict'
    assert.equal(archive.records()[0].metadata.nested.phase, 'fit')
  })

  it('normalizes failed trial records', () => {
    const archive = new Archive()
    archive.fail({ trialId: 'bad', candidateId: 'bad' }, new Error('boom'), 'fit')
    assert.equal(archive.size, 1)
    assert.equal(archive.records()[0].status, 'failed')
    assert.equal(archive.records()[0].error.phase, 'fit')
  })

  it('validates trial records', () => {
    const record = createTrialRecord({ candidateId: 'x', foldId: 'fold-1', batch: 1 })
    assert.equal(record.trialId, 'x-fold-1-42')
    assert.equal(record.batch, 1)
  })
})

it('uncertainty fields require level metadata and reject invalid numeric values', () => {
  for (const fields of [
    { quantiles: [1, 2, 3, 4] },
    { interval: [1, 2, 3, 4] },
    { quantiles: [1, 2, 3, 4], quantileLevels: [0.9, 0.1] },
    { quantiles: [2, 1, 3, 4], quantileLevels: [0.1, 0.9] },
    { quantiles: [NaN, 2, 3, 4], quantileLevels: [0.1, 0.9] },
    { interval: [1, 2, 3, NaN], coverageLevels: [0.9] }
  ]) assert.throws(() => createPrediction({ truth: [1, 2], ...fields }), ValidationError)
})

it('uncertainty shapes distinguish rows, targets and quantile or coverage levels', () => {
  const { predictionRows } = require('../src/index.js')
  const p = createPrediction({
    rows: 2, taskKind: 'multioutput', targetCount: 2, targetNames: ['a', 'b'],
    quantiles: [1, 2, 10, 20, 3, 4, 30, 40], quantileLevels: [0.1, 0.9]
  })
  assert.equal(predictionRows(p), 2)
  assert.equal(p.targetCount, 2)
  assert.deepEqual(Array.from(p.quantileLevels), [0.1, 0.9])
  assert.throws(() => createPrediction({ ...p, rows: 3 }), ValidationError)
  assert.throws(() => createPrediction({ ...p, targetNames: ['a', 'a'] }), ValidationError)
  assert.doesNotThrow(() => createPrediction({ rows: 1, interval: [-Infinity, Infinity], coverageLevels: [0.99] }))
})

it('multilabel probabilities use independent columns and explicit target axes', () => {
  const p = createPrediction({ rows: 2, taskKind: 'multilabel', targetCount: 2,
    truth: [0, 1, 1, 1], proba: [0.2, 0.9, 0.8, 0.7] })
  assert.equal(p.taskKind, 'multilabel')
  assert.throws(() => createPrediction({ ...p, truth: [0, 2, 1, 1] }), ValidationError)
})

it('matrix targets require an explicit multioutput or multilabel task', () => {
  const targets = [[1, 2], [3, 4], [5, 6]]
  const task = createTask({ X, y: targets, kind: 'multioutput' })
  assert.equal(task.y.rows, 3)
  assert.equal(task.y.cols, 2)
  assert.deepEqual(Array.from(task.y.data), [1, 2, 3, 4, 5, 6])
  assert.throws(() => createTask({ X, y: targets }), ValidationError)
  assert.throws(() => createTask({ X, y: targets, kind: 'regression' }), ValidationError)
  assert.throws(() => createTask({ X, y: targets, kind: 'multilabel' }), ValidationError)
})

it('multioutput scoring averages target scores without mixing target offsets', () => {
  const { getScorer } = require('../src/index.js')
  const p = createPrediction({ rows: 2, taskKind: 'multioutput', targetCount: 2,
    truth: [0, 100, 2, 104], response: [1, 102, 1, 102] })
  assert.equal(evaluateMeasure('r2', p), 0)
  assert.equal(getScorer('mse')([[0, 100], [2, 104]], [[1, 102], [1, 102]], { taskKind: 'multioutput', sampleWeight: [1, 3] }), 2.5)
})

it('scalar target normalization rejects matrices and object coercion', () => {
  const { normalizeY } = require('../src/index.js')
  for (const y of [[[1, 2], [3, 4]], { data: Float64Array.of(1, 2), rows: 1, cols: 2 }, 3]) {
    assert.throws(() => normalizeY(y), ValidationError)
  }
})

it('Pipeline preserves structured prediction options after the level argument', () => {
  const { Pipeline } = require('../src/index.js')
  const model = { fit() { return this }, predictQuantiles(x, levels, options) {
    assert.deepEqual(options, { bound: 'upper' })
    return createPrediction({ rows: 1, taskKind: 'regression', quantileLevels: levels, quantiles: [2] })
  } }
  const pipeline = new Pipeline([['model', model]])
  pipeline.fit([[1]], [1])
  assert.deepEqual(Array.from(pipeline.predictQuantiles([[1]], [.5], { bound: 'upper' }).quantiles), [2])
})
