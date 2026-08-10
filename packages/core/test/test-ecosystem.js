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

    const nonFinite = createPrediction({
      truth: new Int32Array([0, 1]),
      proba: new Float64Array([0.8, 0.2, NaN, 0.9]),
      classes: new Int32Array([0, 1])
    })
    assert.throws(() => evaluateMeasure('log_loss', nonFinite), ValidationError)
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
      interval: new Float64Array([0, 1, 2, 3])
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
